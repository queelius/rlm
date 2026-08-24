from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from rlm import (
    RLM,
    ControllerConfig,
    RLMConfig,
    RunAttestation,
    TokenUsage,
    TraceArtifact,
    TraceConfig,
    TraceError,
    load_trace,
    validate_run_artifacts,
)
from tests.fakes import ScriptedBackend, controller_code, request, responses_text


def _run_direct(tmp_path: Path):
    result = RLM(
        ScriptedBackend(
            [responses_text("direct", model="public-model", input_tokens=3, output_tokens=2)]
        ),
        config=RLMConfig(tracing=TraceConfig(enabled=True, directory=tmp_path)),
    ).run_direct(request())
    assert result.trace_directory is not None
    return result, RunAttestation.from_result(result), load_trace(result.trace_directory)


def test_normal_run_and_run_direct_artifacts_agree_exactly(tmp_path: Path) -> None:
    direct, direct_attestation, direct_trace = _run_direct(tmp_path / "direct")
    validate_run_artifacts(direct.response, direct_attestation, direct_trace)
    assert direct.duration_seconds == direct_attestation.duration_seconds
    assert direct.duration_seconds == direct_trace.manifest["duration_seconds"]
    assert RunAttestation.from_headers(direct_attestation.to_headers()) == direct_attestation

    controller = ScriptedBackend(
        [controller_code('FINAL_TEXT("done")', input_tokens=4, output_tokens=1)]
    )
    result = RLM(
        controller,
        config=RLMConfig(
            controller=ControllerConfig(model="controller"),
            tracing=TraceConfig(enabled=True, directory=tmp_path / "run"),
        ),
    ).run(request())
    assert result.trace_directory is not None
    attestation = RunAttestation.from_result(result)
    trace = load_trace(result.trace_directory)
    validate_run_artifacts(result.response, attestation, trace)
    assert result.duration_seconds == attestation.duration_seconds
    assert result.duration_seconds == trace.manifest["duration_seconds"]


def test_validator_accepts_explicit_unreported_usage_evidence(tmp_path: Path) -> None:
    response = responses_text("direct", model="public-model")
    response.pop("usage")
    result = RLM(
        ScriptedBackend([response]),
        config=RLMConfig(tracing=TraceConfig(enabled=True, directory=tmp_path)),
    ).run_direct(request())
    assert result.trace_directory is not None
    attestation = RunAttestation.from_result(result)

    assert attestation.usage.unreported_calls == 1
    validate_run_artifacts(result.response, attestation, load_trace(result.trace_directory))


def test_validator_rejects_a_valid_failed_trace_as_successful_provenance(tmp_path: Path) -> None:
    result, direct_attestation, _ = _run_direct(tmp_path / "success")
    with pytest.raises(RuntimeError, match="failed"):
        RLM(
            ScriptedBackend([RuntimeError("failed")]),
            config=RLMConfig(tracing=TraceConfig(enabled=True, directory=tmp_path / "failed")),
        ).run_direct(request())
    failed_trace = load_trace(next((tmp_path / "failed").iterdir()))
    manifest = failed_trace.manifest
    usage = TokenUsage(**manifest["usage"]["usage"])
    attestation = RunAttestation(
        run_id=manifest["run_id"],
        stop_reason=manifest["stop_reason"],
        turns=manifest["turns"],
        duration_seconds=manifest["duration_seconds"],
        usage=usage,
        controller_identity=direct_attestation.controller_identity,
        harness_fingerprint=manifest["harness_fingerprint"],
    )

    with pytest.raises(TraceError, match="completed trace"):
        validate_run_artifacts(result.response, attestation, failed_trace)


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, run_id="other-run"),
        lambda value: replace(value, stop_reason="other-stop"),
        lambda value: replace(value, turns=1),
        lambda value: replace(value, duration_seconds=value.duration_seconds + 1),
        lambda value: replace(value, usage=replace(value.usage, input_tokens=4)),
        lambda value: replace(value, usage=replace(value.usage, output_tokens=3)),
        lambda value: replace(value, usage=replace(value.usage, calls=2)),
        lambda value: replace(value, usage=replace(value.usage, unreported_calls=1)),
        lambda value: replace(
            value,
            controller_identity=replace(value.controller_identity, model="other-model"),
        ),
        lambda value: replace(
            value,
            controller_identity=replace(value.controller_identity, options_sha256="0" * 64),
        ),
        lambda value: replace(value, harness_fingerprint="0" * 64),
    ],
    ids=[
        "run-id",
        "stop-reason",
        "turns",
        "duration",
        "input-tokens",
        "output-tokens",
        "calls",
        "unreported-calls",
        "controller-model",
        "controller-options",
        "harness",
    ],
)
def test_validator_rejects_every_attestation_trace_disagreement(tmp_path: Path, change) -> None:
    result, attestation, trace = _run_direct(tmp_path)

    with pytest.raises(TraceError):
        validate_run_artifacts(result.response, change(attestation), trace)


def test_validator_rejects_terminal_response_disagreement(tmp_path: Path) -> None:
    result, attestation, trace = _run_direct(tmp_path)
    changed = copy.deepcopy(result.response)
    changed["output"][0]["content"][0]["text"] = "changed"

    with pytest.raises(TraceError, match="terminal response"):
        validate_run_artifacts(changed, attestation, trace)

    malformed = copy.deepcopy(result.response)
    malformed["status"] = "in_progress"
    with pytest.raises(TraceError, match="terminal response"):
        validate_run_artifacts(malformed, attestation, trace)


@pytest.mark.parametrize("snapshot", ["manifest", "events", "manifest-digest", "jsonl-digest"])
def test_validator_reloads_exact_trace_bytes_instead_of_trusting_direct_construction(
    tmp_path: Path, snapshot: str
) -> None:
    result, attestation, trace = _run_direct(tmp_path)
    values = {
        "directory": trace.directory,
        "manifest": trace.manifest,
        "events": trace.events,
        "manifest_sha256": trace.manifest_sha256,
        "jsonl_sha256": trace.jsonl_sha256,
    }
    if snapshot == "manifest":
        values["manifest"] = {**trace.manifest, "duration_seconds": 999}
    elif snapshot == "events":
        events = list(trace.events)
        events[-1] = {**events[-1], "payload": {**events[-1]["payload"], "stop_reason": "x"}}
        values["events"] = tuple(events)
    elif snapshot == "manifest-digest":
        values["manifest_sha256"] = "0" * 64
    else:
        values["jsonl_sha256"] = "0" * 64
    bypass = TraceArtifact(**values)

    with pytest.raises(TraceError, match="snapshot"):
        validate_run_artifacts(result.response, attestation, bypass)
