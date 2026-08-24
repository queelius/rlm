"""Typed strict-JSON readers and trace fixtures shared by trace tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rlm.abi import ENVIRONMENT_ABI
from rlm.config import RLMConfig, RunLimits, TraceConfig
from rlm.errors import ErrorRecord
from rlm.json import strict_json_loads
from rlm.prompts import default_harness_spec
from rlm.trace import (
    TRACE_ENVELOPE_FIELDS,
    TRACE_MANIFEST_FIELDS,
    TRACE_SCHEMA_VERSION,
    HarnessFingerprint,
    RunStartedPayload,
    RunTraceStatus,
    TraceManifest,
)
from rlm.types import TokenUsage


def traced_config(directory: Path, *, max_turns: int = 12) -> RLMConfig:
    return RLMConfig(
        limits=replace(RunLimits(), max_turns=max_turns),
        tracing=TraceConfig(enabled=True, directory=directory),
    )


def read_events(directory: Path | None) -> list[dict[str, Any]]:
    if directory is None:
        raise AssertionError("expected an enabled trace directory")
    lines = (directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    values: list[dict[str, Any]] = []
    for line in lines:
        value = strict_json_loads(line)
        if not isinstance(value, dict) or set(value) != set(TRACE_ENVELOPE_FIELDS):
            raise AssertionError("trace event does not have the strict envelope shape")
        values.append(value)
    return values


def read_manifest(directory: Path | None) -> dict[str, Any]:
    if directory is None:
        raise AssertionError("expected an enabled trace directory")
    value = strict_json_loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(TRACE_MANIFEST_FIELDS):
        raise AssertionError("trace manifest does not have the strict terminal shape")
    return value


def read_trace_contract() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "trace-schema-v1.json"
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("trace contract must be an object")
    return value


def run_started_payload(*, request: dict[str, Any] | None = None) -> RunStartedPayload:
    config = RLMConfig()
    return RunStartedPayload(
        request={} if request is None else request,
        effective_config=config,
        harness=config.harness,
        harness_fingerprint=HarnessFingerprint(config.harness),
        trace_schema_version=TRACE_SCHEMA_VERSION,
        environment_abi=ENVIRONMENT_ABI.to_dict(),
        environment_abi_digest=ENVIRONMENT_ABI.digest(),
    )


def completed_manifest(run_id: str) -> TraceManifest:
    harness = default_harness_spec()
    return TraceManifest(
        run_id=run_id,
        status=RunTraceStatus.COMPLETED,
        stop_reason="complete",
        turns=1,
        duration_seconds=0.0,
        usage=TokenUsage().to_dict(),
        error=None,
        harness=harness.to_dict(),
        harness_fingerprint=harness.fingerprint(),
        effective_config=RLMConfig().to_dict(),
        environment_abi=ENVIRONMENT_ABI.to_dict(),
        environment_abi_digest=ENVIRONMENT_ABI.digest(),
    )


def failed_manifest(run_id: str, error: ErrorRecord) -> TraceManifest:
    manifest = completed_manifest(run_id)
    return TraceManifest(
        run_id=manifest.run_id,
        status=RunTraceStatus.FAILED,
        stop_reason=None,
        turns=manifest.turns,
        duration_seconds=manifest.duration_seconds,
        usage=manifest.usage,
        error=error,
        harness=manifest.harness,
        harness_fingerprint=manifest.harness_fingerprint,
        effective_config=manifest.effective_config,
        environment_abi=manifest.environment_abi,
        environment_abi_digest=manifest.environment_abi_digest,
    )
