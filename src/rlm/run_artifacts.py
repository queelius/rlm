"""Agreement validation for successful public run artifacts."""

from __future__ import annotations

from typing import Any

from rlm.attestation import RunAttestation
from rlm.errors import TraceError
from rlm.json import strict_json_sha256
from rlm.response import validate_terminal_response
from rlm.trace_reader import TraceArtifact, load_trace
from rlm.types import _canonical_duration_seconds


def validate_run_artifacts(
    response: dict[str, Any],
    attestation: RunAttestation,
    trace: TraceArtifact,
) -> None:
    """Require one successful response, attestation, and exact trace to agree."""

    if not isinstance(attestation, RunAttestation):
        raise TypeError("attestation must be a RunAttestation")
    if not isinstance(trace, TraceArtifact):
        raise TypeError("trace must be a TraceArtifact")
    response_error = validate_terminal_response(response)
    if response_error is not None:
        raise TraceError(f"terminal response is invalid: {response_error}")

    loaded = load_trace(trace.directory, expected_run_id=attestation.run_id)
    if (
        loaded.manifest != trace.manifest
        or loaded.events != trace.events
        or loaded.manifest_sha256 != trace.manifest_sha256
        or loaded.jsonl_sha256 != trace.jsonl_sha256
    ):
        raise TraceError("trace artifact snapshot disagrees with its exact directory bytes")

    manifest = loaded.manifest
    events = loaded.events
    if manifest["status"] != "completed":
        raise TraceError("successful attestation requires a completed trace")
    if manifest["run_id"] != attestation.run_id:
        raise TraceError("attestation run ID disagrees with trace")
    if manifest["harness_fingerprint"] != attestation.harness_fingerprint:
        raise TraceError("attestation harness fingerprint disagrees with trace")
    if manifest["stop_reason"] != attestation.stop_reason:
        raise TraceError("attestation stop reason disagrees with trace")
    if manifest["turns"] != attestation.turns:
        raise TraceError("attestation turns disagree with trace")
    if manifest["duration_seconds"] != attestation.duration_seconds or manifest[
        "duration_seconds"
    ] != _canonical_duration_seconds(manifest["duration_seconds"]):
        raise TraceError("attestation duration disagrees with canonical trace duration")
    if manifest["usage"]["usage"] != attestation.usage.to_dict():
        raise TraceError("attestation usage disagrees with complete trace usage")

    effective_controller = manifest["effective_config"]["controller"]
    model = effective_controller["model"] or events[0]["payload"]["request"]["model"]
    if attestation.controller_identity.model != model:
        raise TraceError("attestation controller model disagrees with trace")
    options_sha256 = strict_json_sha256(effective_controller["options"])
    if attestation.controller_identity.options_sha256 != options_sha256:
        raise TraceError("attestation controller options disagree with trace")

    terminal = events[-1]
    if terminal["type"] != "run.completed" or terminal["payload"]["response"] != response:
        raise TraceError("trace terminal response disagrees with successful response")


__all__ = ["validate_run_artifacts"]
