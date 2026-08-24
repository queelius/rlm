"""Strict post-hoc reading for canonical RLM trace artifacts."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from rlm.errors import ErrorRecord, TraceError
from rlm.json import StrictJSONError, strict_json_loads
from rlm.response import validate_terminal_response

_CONTEXT_FIELDS = {"run_id", "branch_id", "call_id", "role", "depth"}
_MODEL_ROLES = {"controller", "public", "subcall"}
_FINAL_EVENT_TYPES = {"run.completed", "run.failed"}


@dataclass(frozen=True, slots=True)
class TraceArtifact:
    """An owned, integrity-checked manifest and event stream."""

    directory: Path
    manifest: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    manifest_sha256: str
    jsonl_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            raise TypeError("trace artifact directory must be a Path")
        if not isinstance(self.manifest, Mapping):
            raise TypeError("trace artifact manifest must be an object")
        if not isinstance(self.events, tuple) or not all(
            isinstance(event, Mapping) for event in self.events
        ):
            raise TypeError("trace artifact events must be a tuple of objects")
        _require_sha256(self.manifest_sha256, name="manifest_sha256")
        _require_sha256(self.jsonl_sha256, name="jsonl_sha256")
        object.__setattr__(self, "manifest", copy.deepcopy(dict(self.manifest)))
        object.__setattr__(
            self,
            "events",
            tuple(copy.deepcopy(dict(event)) for event in self.events),
        )


def load_trace(directory: Path, *, expected_run_id: str | None = None) -> TraceArtifact:
    """Load one complete canonical trace or raise :class:`TraceError`."""

    if not isinstance(directory, Path):
        raise TypeError("trace directory must be a Path")
    if expected_run_id is not None and (
        not isinstance(expected_run_id, str) or not expected_run_id
    ):
        raise ValueError("expected_run_id must be a non-empty string or None")

    manifest_bytes = _read_file_once(directory / "manifest.json")
    jsonl_bytes = _read_file_once(directory / "trace.jsonl")
    manifest = _decode_manifest(manifest_bytes)
    events = _decode_events(jsonl_bytes)
    contract = _trace_contract()
    _validate_trace(
        manifest,
        events,
        contract=contract,
        expected_run_id=expected_run_id,
    )
    return TraceArtifact(
        directory=directory,
        manifest=manifest,
        events=tuple(events),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        jsonl_sha256=hashlib.sha256(jsonl_bytes).hexdigest(),
    )


def _read_file_once(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise TraceError(f"cannot read trace artifact {path.name}: {exc}") from exc


def _decode_manifest(value: bytes) -> dict[str, Any]:
    try:
        manifest = strict_json_loads(value)
    except StrictJSONError as exc:
        raise TraceError(f"manifest.json requires strict JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TraceError("manifest.json must contain an object")
    return manifest


def _decode_events(value: bytes) -> list[dict[str, Any]]:
    if not value or not value.endswith(b"\n"):
        raise TraceError("trace.jsonl is empty or truncated")
    lines = value.split(b"\n")[:-1]
    if not lines or any(not line for line in lines):
        raise TraceError("trace.jsonl contains an empty event line")
    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            event = strict_json_loads(line)
        except StrictJSONError as exc:
            raise TraceError(f"trace.jsonl line {index} requires strict JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise TraceError(f"trace.jsonl line {index} must contain an object")
        events.append(event)
    return events


@lru_cache(maxsize=1)
def _trace_contract() -> dict[str, Any]:
    resource = files("rlm").joinpath("schemas/trace-v1.json")
    try:
        value = strict_json_loads(resource.read_bytes())
    except (OSError, StrictJSONError) as exc:  # pragma: no cover - packaging invariant
        raise TraceError(f"packaged trace v1 contract is unreadable: {exc}") from exc
    if not isinstance(value, dict):  # pragma: no cover - packaging invariant
        raise TraceError("packaged trace v1 contract must be an object")
    return value


def _validate_trace(
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    contract: dict[str, Any],
    expected_run_id: str | None,
) -> None:
    version = contract.get("schema_version")
    manifest_fields = contract.get("manifest_fields")
    envelope_fields = contract.get("envelope_fields")
    event_types = contract.get("event_types")
    payload_fields = contract.get("required_payload_fields")
    if (
        not isinstance(version, str)
        or not isinstance(manifest_fields, list)
        or not isinstance(envelope_fields, list)
        or not isinstance(event_types, list)
        or not isinstance(payload_fields, dict)
    ):  # pragma: no cover - packaging invariant
        raise TraceError("packaged trace v1 contract has an invalid structure")

    _validate_manifest(manifest, fields=set(manifest_fields), version=version)
    run_id = manifest["run_id"]
    if expected_run_id is not None and run_id != expected_run_id:
        raise TraceError(
            f"trace run ID {run_id!r} does not match expected run ID {expected_run_id!r}"
        )

    seen_event_ids: set[str] = set()
    finals: list[dict[str, Any]] = []
    model_calls: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        _validate_event(
            event,
            index=index,
            run_id=run_id,
            version=version,
            envelope_fields=set(envelope_fields),
            event_types=set(event_types),
            payload_fields=payload_fields,
            seen_event_ids=seen_event_ids,
        )
        seen_event_ids.add(event["event_id"])
        if event["type"] in _FINAL_EVENT_TYPES:
            finals.append(event)
        if event["type"] in {"model.request", "model.response", "model.failed"}:
            _record_model_event(event, model_calls=model_calls)

    if sum(event["type"] == "run.started" for event in events) != 1:
        raise TraceError("trace must contain exactly one run.started event")
    if len(finals) != 1:
        raise TraceError("trace must contain exactly one final run event")
    if not events or events[-1] is not finals[0]:
        raise TraceError("the final run event must be the last trace event")
    _validate_manifest_provenance(manifest, events)
    _validate_terminal_agreement(manifest, finals[0])
    _validate_model_outcomes(model_calls)


def _validate_manifest(manifest: dict[str, Any], *, fields: set[str], version: str) -> None:
    if set(manifest) != fields:
        raise TraceError("trace manifest fields do not match the v1 contract")
    _require_nonempty_text(manifest["run_id"], name="manifest run_id")
    if manifest["schema_version"] != version:
        raise TraceError("trace manifest schema_version does not match the v1 contract")
    if manifest["status"] not in {"completed", "failed"}:
        raise TraceError("trace manifest status is invalid")
    stop_reason = manifest["stop_reason"]
    if manifest["status"] == "completed" or stop_reason is not None:
        _require_nonempty_text(stop_reason, name="manifest stop_reason")
    _require_nonnegative_integer(manifest["turns"], name="manifest turns")
    _require_nonnegative_number(manifest["duration_seconds"], name="manifest duration_seconds")
    for name in ("usage", "harness", "effective_config", "environment_abi"):
        if not isinstance(manifest[name], dict):
            raise TraceError(f"trace manifest {name} must be an object")
    _require_sha256(manifest["harness_fingerprint"], name="manifest harness_fingerprint")
    _require_sha256(manifest["environment_abi_digest"], name="manifest environment_abi_digest")
    error = manifest["error"]
    if manifest["status"] == "completed" and error is not None:
        raise TraceError("completed trace manifest error must be null")
    if manifest["status"] == "failed":
        _validate_error_record(error, name="trace manifest error")


def _validate_event(
    event: dict[str, Any],
    *,
    index: int,
    run_id: str,
    version: str,
    envelope_fields: set[str],
    event_types: set[str],
    payload_fields: dict[str, Any],
    seen_event_ids: set[str],
) -> None:
    if set(event) != envelope_fields:
        raise TraceError(f"trace event {index} fields do not match the v1 contract")
    if event["schema_version"] != version:
        raise TraceError(f"trace event {index} schema_version does not match the v1 contract")
    if event["run_id"] != run_id:
        raise TraceError(f"trace event {index} run_id does not match the manifest")
    expected_event_id = f"evt-{index:06d}"
    if event["event_id"] != expected_event_id or event["event_id"] in seen_event_ids:
        raise TraceError("trace event IDs must be unique and ordered")
    parent = event["parent_event_id"]
    if index == 0:
        if parent is not None or event["type"] != "run.started":
            raise TraceError("the first trace event must be run.started with no parent")
    elif not isinstance(parent, str) or parent not in seen_event_ids:
        raise TraceError("trace causal parents must refer only to earlier events")
    _require_nonempty_text(event["branch_id"], name="trace event branch_id")
    _require_nonnegative_integer(event["depth"], name="trace event depth")
    _require_nonnegative_number(event["timestamp"], name="trace event timestamp")
    event_type = event["type"]
    if not isinstance(event_type, str) or event_type not in event_types:
        raise TraceError(f"trace event {index} type is not in the v1 contract")
    payload = event["payload"]
    expected_payload_fields = payload_fields.get(event_type)
    if not isinstance(payload, dict) or not isinstance(expected_payload_fields, list):
        raise TraceError(f"trace event {index} payload is invalid")
    if set(payload) != set(expected_payload_fields):
        raise TraceError(f"trace event {index} payload fields do not match {event_type}")
    _validate_payload(event)


def _validate_payload(event: dict[str, Any]) -> None:
    event_type = event["type"]
    payload = event["payload"]
    if event_type == "run.started":
        for name in ("request", "effective_config", "harness", "environment_abi"):
            if not isinstance(payload[name], dict):
                raise TraceError(f"run.started {name} must be an object")
        _require_nonempty_text(
            payload["trace_schema_version"], name="run.started trace_schema_version"
        )
        _require_sha256(payload["harness_fingerprint"], name="run.started harness_fingerprint")
        _require_sha256(
            payload["environment_abi_digest"], name="run.started environment_abi_digest"
        )
    elif event_type in {"run.completed", "branch.completed"}:
        _require_nonempty_text(payload["stop_reason"], name=f"{event_type} stop_reason")
        error = validate_terminal_response(payload["response"])
        if error is not None:
            raise TraceError(f"{event_type} response is invalid: {error}")
    elif event_type == "run.failed":
        _validate_error_record(payload["error"], name="run.failed error")
    elif event_type == "branch.started":
        for name in ("request", "harness"):
            if not isinstance(payload[name], dict):
                raise TraceError(f"branch.started {name} must be an object")
        for name in ("controller_model", "prompt"):
            _require_nonempty_text(payload[name], name=f"branch.started {name}")
        _require_sha256(payload["harness_fingerprint"], name="branch.started fingerprint")
    elif event_type == "model.request":
        if not isinstance(payload["request"], dict):
            raise TraceError("model.request request must be an object")
    elif event_type == "model.response":
        _require_nonnegative_number(
            payload["duration_seconds"], name="model.response duration_seconds"
        )
        if not isinstance(payload["response"], dict):
            raise TraceError("model.response response must be an object")
    elif event_type == "model.failed":
        _require_nonnegative_number(
            payload["duration_seconds"], name="model.failed duration_seconds"
        )
        _validate_error_record(payload["error"], name="model.failed error")
    elif event_type == "controller.action":
        _require_nonempty_text(payload["code"], name="controller.action code")
    elif event_type == "controller.execution" and not isinstance(payload["execution"], dict):
        raise TraceError("controller.execution execution must be an object")
    elif event_type == "controller.recovery_decision":
        _require_nonnegative_integer(payload["turn"], name="recovery decision turn")
        if not isinstance(payload["decision"], dict):
            raise TraceError("controller recovery decision must be an object")
    elif event_type == "controller.repair":
        _require_nonnegative_integer(payload["turn"], name="controller repair turn")
        if not isinstance(payload["fault"], dict):
            raise TraceError("controller repair fault must be an object")


def _record_model_event(
    event: dict[str, Any],
    *,
    model_calls: dict[str, dict[str, Any]],
) -> None:
    context = event["payload"].get("context")
    _validate_model_context(context, event=event)
    call_id = context["call_id"]
    event_type = event["type"]
    if event_type == "model.request":
        if call_id in model_calls:
            raise TraceError(f"duplicate model request call_id {call_id!r}")
        model_calls[call_id] = {
            "context": context,
            "request_event_id": event["event_id"],
            "response_event_id": None,
            "failed_event_id": None,
        }
        return
    call = model_calls.get(call_id)
    if call is None:
        raise TraceError(f"{event_type} has no earlier model request for call_id {call_id!r}")
    if context != call["context"]:
        raise TraceError(f"{event_type} context does not match its model request")
    if event_type == "model.response":
        if call["response_event_id"] is not None or call["failed_event_id"] is not None:
            raise TraceError(f"model call {call_id!r} has multiple outcomes")
        if event["parent_event_id"] != call["request_event_id"]:
            raise TraceError("model.response must be caused by its model.request")
        call["response_event_id"] = event["event_id"]
    else:
        if call["failed_event_id"] is not None:
            raise TraceError(f"model call {call_id!r} has multiple failures")
        expected_parent = call["response_event_id"] or call["request_event_id"]
        if event["parent_event_id"] != expected_parent:
            raise TraceError("model.failed must follow its response or request")
        call["failed_event_id"] = event["event_id"]


def _validate_model_context(value: Any, *, event: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != _CONTEXT_FIELDS:
        raise TraceError(f"{event['type']} context fields are invalid")
    if value["run_id"] != event["run_id"]:
        raise TraceError(f"{event['type']} context run_id does not match the event")
    if value["branch_id"] != event["branch_id"]:
        raise TraceError(f"{event['type']} context branch_id does not match the event")
    if value["depth"] != event["depth"]:
        raise TraceError(f"{event['type']} context depth does not match the event")
    _require_nonempty_text(value["call_id"], name=f"{event['type']} context call_id")
    if value["role"] not in _MODEL_ROLES:
        raise TraceError(f"{event['type']} context role is invalid")


def _validate_model_outcomes(model_calls: dict[str, dict[str, Any]]) -> None:
    for call_id, call in model_calls.items():
        if call["response_event_id"] is None and call["failed_event_id"] is None:
            raise TraceError(f"model request {call_id!r} has no response or failure")


def _validate_manifest_provenance(manifest: dict[str, Any], events: list[dict[str, Any]]) -> None:
    run_started = events[0]
    if run_started["type"] != "run.started":
        raise TraceError("trace is missing its initial run.started event")
    payload = run_started["payload"]
    comparisons = {
        "effective_config": "effective_config",
        "harness": "harness",
        "harness_fingerprint": "harness_fingerprint",
        "environment_abi": "environment_abi",
        "environment_abi_digest": "environment_abi_digest",
    }
    for manifest_name, payload_name in comparisons.items():
        if manifest[manifest_name] != payload[payload_name]:
            raise TraceError(f"manifest {manifest_name} disagrees with run.started")
    if payload["trace_schema_version"] != manifest["schema_version"]:
        raise TraceError("run.started trace schema version disagrees with the manifest")
    for event in events:
        if event["type"] != "branch.started":
            continue
        branch_payload = event["payload"]
        if (
            branch_payload["harness"] != manifest["harness"]
            or branch_payload["harness_fingerprint"] != manifest["harness_fingerprint"]
        ):
            raise TraceError("branch.started harness provenance disagrees with the manifest")


def _validate_terminal_agreement(manifest: dict[str, Any], final_event: dict[str, Any]) -> None:
    if final_event["type"] == "run.completed":
        if manifest["status"] != "completed":
            raise TraceError("manifest status disagrees with run.completed")
        if manifest["stop_reason"] != final_event["payload"]["stop_reason"]:
            raise TraceError("manifest stop_reason disagrees with run.completed")
    else:
        if manifest["status"] != "failed":
            raise TraceError("manifest status disagrees with run.failed")
        if manifest["error"] != final_event["payload"]["error"]:
            raise TraceError("manifest error disagrees with run.failed")


def _validate_error_record(value: Any, *, name: str) -> None:
    try:
        ErrorRecord.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise TraceError(f"{name} is invalid: {exc}") from exc


def _require_nonempty_text(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise TraceError(f"{name} must be a non-empty string")


def _require_nonnegative_integer(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceError(f"{name} must be a nonnegative integer")


def _require_nonnegative_number(value: Any, *, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise TraceError(f"{name} must be a finite nonnegative number")


def _require_sha256(value: Any, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TraceError(f"{name} must be a SHA-256 digest")
