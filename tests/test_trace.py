import json
from dataclasses import replace
from pathlib import Path

import pytest

from rlm import RLMConfig, TraceConfig
from rlm.abi import ENVIRONMENT_ABI
from rlm.config import ControllerConfig
from rlm.engine import RLM
from rlm.errors import (
    RecoveryAbortedError,
    RemoteRLMError,
    TraceError,
    UpstreamError,
)
from rlm.json import strict_json_dumps, strict_json_loads
from rlm.prompts import default_harness_spec
from rlm.protocol import extract_text
from rlm.specs import RecoveryMode, RecoverySpec
from rlm.trace import (
    TRACE_ENVELOPE_FIELDS,
    TRACE_MANIFEST_FIELDS,
    TRACE_SCHEMA_VERSION,
    NullTraceSink,
    RunStartedPayload,
    TraceEventKind,
    TraceManifest,
    TraceRecorder,
    trace_payload_schema,
)
from tests.fakes import ScriptedBackend, controller_code, request, responses_text
from tests.trace_helpers import (
    completed_manifest,
    failed_manifest,
    read_events,
    read_trace_contract,
    run_started_payload,
    traced_config,
)


class NoCopy:
    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise AssertionError("disabled tracing copied a payload")


def test_disabled_trace_does_no_payload_work() -> None:
    def unexpected_manifest() -> TraceManifest:
        raise AssertionError("disabled tracing built a manifest")

    payload = run_started_payload(request={"opaque": NoCopy()})
    trace = NullTraceSink()
    assert trace.event(payload, branch_id="root", depth=0) is None
    trace.close(manifest_factory=unexpected_manifest)


def test_jsonl_only_trace_does_not_retain_events(tmp_path: Path) -> None:
    trace = TraceRecorder(
        TraceConfig(enabled=True, directory=tmp_path, markdown=False), run_id="run"
    )
    trace.event(run_started_payload(), branch_id="root", depth=0)
    trace.close(manifest_factory=lambda: completed_manifest("run"))

    assert trace.events == []
    event = json.loads((trace.directory / "trace.jsonl").read_text().splitlines()[0])
    assert event["schema_version"] == TRACE_SCHEMA_VERSION


def test_enabled_trace_rejects_unknown_python_objects(tmp_path: Path) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")
    with pytest.raises(TraceError, match="strict JSON"):
        trace.event(
            run_started_payload(request={"value": object()}),
            branch_id="root",
            depth=0,
        )


def test_enabled_trace_rejects_events_after_close(tmp_path: Path) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")
    trace.close(manifest_factory=lambda: completed_manifest("run"))

    with pytest.raises(TraceError, match="closed"):
        trace.event(run_started_payload(), branch_id="root", depth=0)


def test_trace_finalization_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")

    def fail_write(name: str, value: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(trace, "write_json", fail_write)
    with pytest.raises(TraceError, match="finalize"):
        trace.close(manifest_factory=lambda: completed_manifest("run"))


def test_trace_failure_preserves_an_active_run_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")
    original = UpstreamError(503, {"error": {"message": "down"}}, endpoint="/responses")

    def fail_write(name: str, value: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr(trace, "write_json", fail_write)

    with pytest.raises(TraceError) as caught:
        trace.close(
            manifest_factory=lambda: failed_manifest("run", original.to_record()),
            cause=original,
        )

    assert caught.value.__cause__ is original


def test_trace_schema_matches_reviewed_golden_contract() -> None:
    fixture = Path(__file__).parent / "fixtures/trace-schema-v1.json"
    contract = json.loads(fixture.read_text())

    assert contract["schema_version"] == TRACE_SCHEMA_VERSION
    assert contract["envelope_fields"] == list(TRACE_ENVELOPE_FIELDS)
    assert contract["manifest_fields"] == list(TRACE_MANIFEST_FIELDS)
    assert contract["event_types"] == [kind.value for kind in TraceEventKind]
    assert contract["required_payload_fields"] == trace_payload_schema()


def test_enabled_trace_rejects_a_dangling_causal_parent(tmp_path: Path) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")
    trace.event(run_started_payload(), branch_id="root", depth=0)

    with pytest.raises(TraceError, match="parent_event_id"):
        trace.event(
            run_started_payload(),
            branch_id="root",
            depth=0,
            parent_event_id="missing",
        )


def test_enabled_trace_rejects_payload_shape_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = run_started_payload()
    monkeypatch.setattr(RunStartedPayload, "to_dict", lambda self: {"request": self.request})
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")

    with pytest.raises(TraceError, match="payload fields"):
        trace.event(payload, branch_id="root", depth=0)


def test_repair_trace_has_the_reviewed_causal_sequence(tmp_path: Path) -> None:
    controller = ScriptedBackend([responses_text("prose"), controller_code('FINAL_TEXT("fixed")')])
    result = RLM(controller, config=traced_config(tmp_path, max_turns=2)).run(request())

    events = read_events(result.trace_directory)
    contract = read_trace_contract()

    assert extract_text(result.response) == "fixed"
    assert [event["type"] for event in events] == contract["canonical_repair_sequence"]
    for index, event in enumerate(events):
        if index:
            assert event["parent_event_id"] in {item["event_id"] for item in events[:index]}


def test_controller_model_request_trace_contains_bootstrap_not_caller_content(
    tmp_path: Path,
) -> None:
    caller_content = "TRACE-CALLER-PAYLOAD-92a781"
    config = RLMConfig(
        controller=ControllerConfig(model="controller"),
        tracing=TraceConfig(enabled=True, directory=tmp_path),
    )
    result = RLM(
        ScriptedBackend([controller_code('FINAL_TEXT("done")')]),
        config=config,
    ).run({"model": "public-model", "input": caller_content})

    events = read_events(result.trace_directory)
    event = next(item for item in events if item["type"] == "model.request")
    traced_request = event["payload"]["request"]
    bootstrap = strict_json_loads(traced_request["input"][0]["content"])

    assert bootstrap["type"] == "rlm.controller_bootstrap"
    assert bootstrap["schema_version"] == "2"
    assert bootstrap["request_binding"] == ENVIRONMENT_ABI.request_name
    assert bootstrap["content_in_message"] is False
    assert bootstrap["submission"] == {
        "required": True,
        "allowed_kinds": ["text", "response"],
    }
    assert caller_content not in strict_json_dumps(traced_request)


def test_next_controller_request_traces_a_typed_observation_runtime_item(
    tmp_path: Path,
) -> None:
    caller_content = "TRACE-CALLER-PAYLOAD-e15545"
    controller = ScriptedBackend(
        [controller_code('print("ready")'), controller_code('FINAL_TEXT("done")')]
    )
    result = RLM(controller, config=traced_config(tmp_path, max_turns=2)).run(
        {"model": "public-model", "input": caller_content}
    )

    events = read_events(result.trace_directory)
    requests = [
        event["payload"]["request"]
        for event in events
        if event["type"] == "model.request" and event["payload"]["context"]["role"] == "controller"
    ]
    observation_text = requests[1]["input"][-1]["content"]
    observation = strict_json_loads(observation_text)

    assert observation["type"] == "rlm.controller_observation"
    assert observation["request_binding"] == ENVIRONMENT_ABI.request_name
    assert observation["submission"] == {"status": "absent", "required": True}
    assert observation["execution"]["stdout"] == "ready\n"
    assert caller_content not in observation_text
    assert "Continue." not in observation_text


def test_fatal_model_failure_links_request_failed_and_run_failed(tmp_path: Path) -> None:
    backend = ScriptedBackend([UpstreamError(503, {"error": {"message": "down"}}, endpoint="x")])

    with pytest.raises(RemoteRLMError) as caught:
        RLM(backend, config=traced_config(tmp_path)).run(request())

    assert caught.value.status_code == 503
    assert caught.value.to_body() == {"error": {"message": "down"}}
    events = read_events(next(tmp_path.iterdir()))
    request_event = next(event for event in events if event["type"] == "model.request")
    failed_event = next(event for event in events if event["type"] == "model.failed")
    terminal = events[-1]
    assert failed_event["parent_event_id"] == request_event["event_id"]
    assert terminal["type"] == "run.failed"
    assert terminal["parent_event_id"] == failed_event["event_id"]
    assert not any(event["type"] == "controller.repair" for event in events)


def test_post_response_fatal_validation_links_response_failed_and_run_failed(
    tmp_path: Path,
) -> None:
    invalid = responses_text("x")
    invalid["status"] = "in_progress"

    with pytest.raises(RemoteRLMError):
        RLM(ScriptedBackend([invalid]), config=traced_config(tmp_path)).run(request())

    events = read_events(next(tmp_path.iterdir()))
    response_event = next(event for event in events if event["type"] == "model.response")
    failed_event = next(event for event in events if event["type"] == "model.failed")
    assert failed_event["parent_event_id"] == response_event["event_id"]
    assert events[-1]["parent_event_id"] == failed_event["event_id"]
    assert not any(event["type"] == "controller.repair" for event in events)


def test_abort_links_decision_directly_to_terminal_failure_without_repair(tmp_path: Path) -> None:
    harness = replace(
        default_harness_spec(),
        recovery=RecoverySpec(mode=RecoveryMode.ABORT),
    )
    config = replace(traced_config(tmp_path, max_turns=1), harness=harness)

    with pytest.raises(RecoveryAbortedError):
        RLM(
            ScriptedBackend([responses_text("prose")]),
            config=config,
        ).run(request())

    events = read_events(next(tmp_path.iterdir()))
    decision = next(event for event in events if event["type"] == "controller.recovery_decision")
    assert events[-1]["type"] == "run.failed"
    assert events[-1]["parent_event_id"] == decision["event_id"]
    assert not any(event["type"] == "controller.repair" for event in events)
