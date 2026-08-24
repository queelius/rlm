from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rlm import RLM, RLMConfig, RunLimits, TraceArtifact, TraceConfig, TraceError, load_trace
from rlm.errors import RemoteRLMError, UpstreamError
from rlm.json import strict_json_dumps, strict_json_loads
from tests.fakes import ScriptedBackend, controller_code, request, responses_text


def _traced_config(directory: Path, *, max_turns: int = 12, max_depth: int = 0) -> RLMConfig:
    return RLMConfig(
        limits=replace(RunLimits(), max_turns=max_turns, max_depth=max_depth),
        tracing=TraceConfig(enabled=True, directory=directory),
    )


@pytest.fixture
def completed_trace(tmp_path: Path) -> Path:
    result = RLM(
        ScriptedBackend([responses_text("direct", model="public-model")]),
        config=_traced_config(tmp_path),
    ).run_direct(request())
    assert result.trace_directory is not None
    return result.trace_directory


@pytest.fixture
def repaired_trace(tmp_path: Path) -> Path:
    controller = ScriptedBackend([responses_text("prose"), controller_code('FINAL_TEXT("fixed")')])
    result = RLM(controller, config=_traced_config(tmp_path, max_turns=2)).run(request())
    assert result.trace_directory is not None
    return result.trace_directory


@pytest.fixture
def recursive_trace(tmp_path: Path) -> Path:
    controller = ScriptedBackend(
        [
            controller_code("nested = rlm_complete()\nFINAL_RESPONSE(nested)"),
            controller_code('FINAL_TEXT("child")'),
        ]
    )
    result = RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_traced_config(tmp_path, max_depth=1),
    ).run(request())
    assert result.trace_directory is not None
    return result.trace_directory


@pytest.fixture
def failed_trace(tmp_path: Path) -> Path:
    backend = ScriptedBackend(
        [UpstreamError(503, {"error": {"message": "down"}}, endpoint="/v1/responses")]
    )
    with pytest.raises(RemoteRLMError):
        RLM(backend, config=_traced_config(tmp_path)).run_direct(request())
    return next(tmp_path.iterdir())


@pytest.fixture
def post_response_failed_trace(tmp_path: Path) -> Path:
    response = responses_text("not terminal", model="public-model")
    response["status"] = "in_progress"
    with pytest.raises(RemoteRLMError):
        RLM(
            ScriptedBackend([response]),
            config=_traced_config(tmp_path),
        ).run_direct(request())
    return next(tmp_path.iterdir())


def test_load_trace_returns_owned_values_and_exact_byte_hashes(completed_trace: Path) -> None:
    manifest_bytes = (completed_trace / "manifest.json").read_bytes()
    jsonl_bytes = (completed_trace / "trace.jsonl").read_bytes()

    artifact = load_trace(completed_trace)

    assert artifact.directory == completed_trace
    assert artifact.manifest["status"] == "completed"
    assert artifact.events[-1]["type"] == "run.completed"
    assert artifact.manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert artifact.jsonl_sha256 == hashlib.sha256(jsonl_bytes).hexdigest()
    assert load_trace(completed_trace, expected_run_id=artifact.manifest["run_id"]) == artifact


def test_trace_artifact_defensively_owns_nested_inputs(tmp_path: Path) -> None:
    manifest = {"nested": {"value": 1}}
    events = ({"payload": {"response": {"output": [1]}}},)

    artifact = TraceArtifact(tmp_path, manifest, events, "a" * 64, "b" * 64)
    manifest["nested"]["value"] = 2
    events[0]["payload"]["response"]["output"].append(2)

    assert artifact.manifest == {"nested": {"value": 1}}
    assert artifact.events == ({"payload": {"response": {"output": [1]}}},)
    with pytest.raises(AttributeError):
        artifact.directory = Path("changed")  # type: ignore[misc]


def test_trace_artifact_access_cannot_mutate_nested_snapshot_values(tmp_path: Path) -> None:
    artifact = TraceArtifact(
        tmp_path,
        {"nested": {"value": 1}},
        ({"payload": {"response": {"output": [1]}}},),
        "a" * 64,
        "b" * 64,
    )

    manifest = artifact.manifest
    events = artifact.events
    manifest["nested"]["value"] = 2
    events[0]["payload"]["response"]["output"].append(2)

    assert artifact.manifest == {"nested": {"value": 1}}
    assert artifact.events == ({"payload": {"response": {"output": [1]}}},)


def test_load_trace_accepts_repaired_trace(repaired_trace: Path) -> None:
    artifact = load_trace(repaired_trace)

    assert any(event["type"] == "controller.repair" for event in artifact.events)
    assert artifact.events[-1]["type"] == "run.completed"


def test_load_trace_accepts_recursive_trace(recursive_trace: Path) -> None:
    artifact = load_trace(recursive_trace)

    assert sum(event["type"] == "branch.started" for event in artifact.events) == 2
    assert artifact.events[-1]["type"] == "run.completed"


def test_load_trace_accepts_failed_trace(failed_trace: Path) -> None:
    artifact = load_trace(failed_trace)

    assert artifact.manifest["status"] == "failed"
    assert artifact.events[-1]["type"] == "run.failed"


def test_load_trace_accepts_null_stop_reason_for_failed_manifest(failed_trace: Path) -> None:
    manifest, events = _read_trace_values(failed_trace)
    manifest["stop_reason"] = None
    _write_trace_values(failed_trace, manifest, events)

    assert load_trace(failed_trace).manifest["stop_reason"] is None


def test_load_trace_accepts_response_then_failure_for_same_call(
    post_response_failed_trace: Path,
) -> None:
    artifact = load_trace(post_response_failed_trace)
    model_types = [event["type"] for event in artifact.events if event["type"].startswith("model.")]

    assert model_types == ["model.request", "model.response", "model.failed"]


def test_load_trace_rejects_nonadjacent_response_then_failure(
    post_response_failed_trace: Path,
) -> None:
    manifest, events = _read_trace_values(post_response_failed_trace)
    response_index = next(
        index for index, event in enumerate(events) if event["type"] == "model.response"
    )
    response = events[response_index]
    failure = events[response_index + 1]
    terminal = events[response_index + 2]
    intervening = {
        "schema_version": response["schema_version"],
        "run_id": response["run_id"],
        "event_id": f"evt-{response_index + 1:06d}",
        "parent_event_id": response["event_id"],
        "branch_id": response["branch_id"],
        "depth": response["depth"],
        "timestamp": response["timestamp"],
        "type": "controller.observation",
        "payload": {"observation": {}},
    }
    failure["event_id"] = f"evt-{response_index + 2:06d}"
    terminal["event_id"] = f"evt-{response_index + 3:06d}"
    terminal["parent_event_id"] = failure["event_id"]
    events.insert(response_index + 1, intervening)
    _write_trace_values(post_response_failed_trace, manifest, events)

    with pytest.raises(TraceError, match="immediately"):
        load_trace(post_response_failed_trace)


def test_load_trace_rejects_malformed_standalone_model_response(completed_trace: Path) -> None:
    manifest, events = _read_trace_values(completed_trace)
    model_response = next(event for event in events if event["type"] == "model.response")
    model_response["payload"]["response"] = {}
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="model.response"):
        load_trace(completed_trace)


def test_load_trace_rejects_nonterminal_standalone_model_response(completed_trace: Path) -> None:
    manifest, events = _read_trace_values(completed_trace)
    model_response = next(event for event in events if event["type"] == "model.response")
    model_response["payload"]["response"]["status"] = "in_progress"
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="terminal outcome"):
        load_trace(completed_trace)


@pytest.mark.parametrize("missing", ["manifest.json", "trace.jsonl"])
def test_load_trace_rejects_a_missing_artifact_file(tmp_path: Path, missing: str) -> None:
    for name in {"manifest.json", "trace.jsonl"} - {missing}:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    with pytest.raises(TraceError, match=missing):
        load_trace(tmp_path)


def test_load_trace_rejects_expected_run_id_mismatch(completed_trace: Path) -> None:
    with pytest.raises(TraceError, match="expected run ID"):
        load_trace(completed_trace, expected_run_id="other-run")


def test_load_trace_rejects_duplicate_manifest_keys(completed_trace: Path) -> None:
    path = completed_trace / "manifest.json"
    path.write_bytes(path.read_bytes().replace(b"{", b'{"run_id":"duplicate",', 1))

    with pytest.raises(TraceError, match="strict JSON"):
        load_trace(completed_trace)


def test_load_trace_rejects_truncated_jsonl(completed_trace: Path) -> None:
    path = completed_trace / "trace.jsonl"
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(TraceError, match="truncated"):
        load_trace(completed_trace)


@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("manifest", "status"),
        ("model_context", "role"),
    ],
)
def test_load_trace_wraps_unhashable_wrong_types_as_trace_errors(
    completed_trace: Path,
    target: str,
    field: str,
) -> None:
    manifest, events = _read_trace_values(completed_trace)
    if target == "manifest":
        manifest[field] = []
    else:
        model_event = next(event for event in events if event["type"] == "model.request")
        model_event["payload"]["context"][field] = []
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError):
        load_trace(completed_trace)


def _synchronize_harness(
    manifest: dict[str, Any], events: list[dict[str, Any]], harness: dict[str, Any]
) -> None:
    manifest["harness"] = copy.deepcopy(harness)
    manifest["effective_config"]["harness"] = copy.deepcopy(harness)
    started = events[0]["payload"]
    started["harness"] = copy.deepcopy(harness)
    started["effective_config"] = copy.deepcopy(manifest["effective_config"])
    for event in events:
        if event["type"] == "branch.started":
            event["payload"]["harness"] = copy.deepcopy(harness)


def test_load_trace_recomputes_consistently_modified_harness_fingerprint(
    completed_trace: Path,
) -> None:
    manifest, events = _read_trace_values(completed_trace)
    harness = copy.deepcopy(manifest["harness"])
    harness["prompt"]["policy"] = "tampered but synchronized"
    _synchronize_harness(manifest, events, harness)
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="harness"):
        load_trace(completed_trace)


def test_load_trace_recomputes_consistently_modified_environment_abi_digest(
    completed_trace: Path,
) -> None:
    manifest, events = _read_trace_values(completed_trace)
    environment_abi = copy.deepcopy(manifest["environment_abi"])
    environment_abi["version"] = "tampered"
    manifest["environment_abi"] = copy.deepcopy(environment_abi)
    events[0]["payload"]["environment_abi"] = copy.deepcopy(environment_abi)
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="environment ABI"):
        load_trace(completed_trace)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda usage: usage["usage"].__setitem__("input_tokens", -1),
        lambda usage: usage["usage"].pop("output_tokens"),
        lambda usage: usage.pop("model_calls"),
    ],
)
def test_load_trace_rejects_negative_or_missing_typed_usage(
    completed_trace: Path,
    tamper: Any,
) -> None:
    manifest, events = _read_trace_values(completed_trace)
    tamper(manifest["usage"])
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="usage"):
        load_trace(completed_trace)


def test_load_trace_rejects_malformed_effective_config_section(completed_trace: Path) -> None:
    manifest, events = _read_trace_values(completed_trace)
    manifest["effective_config"]["limits"] = []
    events[0]["payload"]["effective_config"] = copy.deepcopy(manifest["effective_config"])
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="effective config"):
        load_trace(completed_trace)


def test_load_trace_rejects_malformed_nested_harness(completed_trace: Path) -> None:
    manifest, events = _read_trace_values(completed_trace)
    harness = copy.deepcopy(manifest["harness"])
    harness["prompt"] = []
    _synchronize_harness(manifest, events, harness)
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="harness"):
        load_trace(completed_trace)


def test_load_trace_rejects_malformed_nested_environment_abi(completed_trace: Path) -> None:
    manifest, events = _read_trace_values(completed_trace)
    environment_abi = copy.deepcopy(manifest["environment_abi"])
    environment_abi["functions"] = {}
    manifest["environment_abi"] = copy.deepcopy(environment_abi)
    events[0]["payload"]["environment_abi"] = copy.deepcopy(environment_abi)
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="environment ABI"):
        load_trace(completed_trace)


def _read_trace_values(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = strict_json_loads((directory / "manifest.json").read_bytes())
    events = [
        strict_json_loads(line) for line in (directory / "trace.jsonl").read_bytes().splitlines()
    ]
    assert isinstance(manifest, dict)
    assert all(isinstance(event, dict) for event in events)
    return manifest, events


def _write_trace_values(
    directory: Path,
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    (directory / "manifest.json").write_text(
        strict_json_dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "trace.jsonl").write_text(
        "".join(strict_json_dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )


def _extra_manifest_field(manifest: dict[str, Any], events: list[dict[str, Any]]) -> None:
    manifest["unexpected"] = True


def _extra_payload_field(manifest: dict[str, Any], events: list[dict[str, Any]]) -> None:
    events[0]["payload"]["unexpected"] = True


def _event_run_id_mismatch(manifest: dict[str, Any], events: list[dict[str, Any]]) -> None:
    events[1]["run_id"] = "other-run"


def _out_of_order_event_id(manifest: dict[str, Any], events: list[dict[str, Any]]) -> None:
    events[1]["event_id"] = "evt-000009"


def _forward_parent(manifest: dict[str, Any], events: list[dict[str, Any]]) -> None:
    events[1]["parent_event_id"] = events[2]["event_id"]


def _manifest_terminal_mismatch(manifest: dict[str, Any], events: list[dict[str, Any]]) -> None:
    manifest["stop_reason"] = "tampered"


def _model_call_id_mismatch(manifest: dict[str, Any], events: list[dict[str, Any]]) -> None:
    response = next(event for event in events if event["type"] == "model.response")
    response["payload"]["context"]["call_id"] = "other-call"


def _second_run_started(manifest: dict[str, Any], events: list[dict[str, Any]]) -> None:
    duplicate = copy.deepcopy(events[0])
    duplicate["event_id"] = f"evt-{len(events) - 1:06d}"
    duplicate["parent_event_id"] = events[-2]["event_id"]
    events.insert(-1, duplicate)
    events[-1]["event_id"] = f"evt-{len(events) - 1:06d}"


@pytest.mark.parametrize(
    "tamper",
    [
        _extra_manifest_field,
        _extra_payload_field,
        _event_run_id_mismatch,
        _out_of_order_event_id,
        _forward_parent,
        _manifest_terminal_mismatch,
        _model_call_id_mismatch,
        _second_run_started,
    ],
)
def test_load_trace_rejects_schema_causality_and_provenance_tampering(
    completed_trace: Path,
    tamper: Any,
) -> None:
    manifest, events = _read_trace_values(completed_trace)
    tamper(manifest, events)
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError):
        load_trace(completed_trace)


def test_load_trace_rejects_branch_harness_provenance_tampering(repaired_trace: Path) -> None:
    manifest, events = _read_trace_values(repaired_trace)
    branch = next(event for event in events if event["type"] == "branch.started")
    branch["payload"]["harness_fingerprint"] = "c" * 64
    _write_trace_values(repaired_trace, manifest, events)

    with pytest.raises(TraceError, match="branch.started"):
        load_trace(repaired_trace)


def test_load_trace_rejects_more_than_one_final_event(completed_trace: Path) -> None:
    manifest, events = _read_trace_values(completed_trace)
    duplicate = copy.deepcopy(events[-1])
    duplicate["event_id"] = f"evt-{len(events):06d}"
    duplicate["parent_event_id"] = events[-1]["event_id"]
    events.append(duplicate)
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="exactly one final"):
        load_trace(completed_trace)


def test_load_trace_rejects_trace_without_a_final_event(completed_trace: Path) -> None:
    manifest, events = _read_trace_values(completed_trace)
    events.pop()
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="exactly one final"):
        load_trace(completed_trace)


def test_load_trace_rejects_model_request_without_an_outcome(completed_trace: Path) -> None:
    manifest, events = _read_trace_values(completed_trace)
    response_index = next(
        index for index, event in enumerate(events) if event["type"] == "model.response"
    )
    response_id = events[response_index]["event_id"]
    events.pop(response_index)
    for index, event in enumerate(events):
        event["event_id"] = f"evt-{index:06d}"
        if event["parent_event_id"] == response_id:
            event["parent_event_id"] = events[index - 1]["event_id"]
    _write_trace_values(completed_trace, manifest, events)

    with pytest.raises(TraceError, match="model request"):
        load_trace(completed_trace)
