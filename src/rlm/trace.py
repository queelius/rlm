"""Strict, causal, opt-in trace recording."""

from __future__ import annotations

import copy
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Protocol, get_args

from rlm.config import RLMConfig, TraceConfig
from rlm.errors import ErrorRecord, TraceError
from rlm.json import json_compatibility_error, strict_json_dumps
from rlm.recovery import ControllerFault, RecoveryDecision
from rlm.specs import HarnessSpec
from rlm.types import ExecutionResult, ModelCallContext

TRACE_SCHEMA_VERSION = "1"
EventRef = str | None


@dataclass(frozen=True, slots=True)
class HarnessFingerprint:
    """Lazily serialize a harness identity only when enabled tracing needs it."""

    harness: HarnessSpec

    def value(self) -> str:
        return self.harness.fingerprint()


class TraceEventKind(str, Enum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    BRANCH_STARTED = "branch.started"
    BRANCH_COMPLETED = "branch.completed"
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    MODEL_FAILED = "model.failed"
    CONTROLLER_ACTION = "controller.action"
    CONTROLLER_EXECUTION = "controller.execution"
    CONTROLLER_OBSERVATION = "controller.observation"
    CONTROLLER_RECOVERY_DECISION = "controller.recovery_decision"
    CONTROLLER_REPAIR = "controller.repair"


@dataclass(frozen=True, slots=True)
class RunStartedPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.RUN_STARTED
    request: dict[str, Any]
    effective_config: RLMConfig
    harness: HarnessSpec
    harness_fingerprint: HarnessFingerprint
    trace_schema_version: str
    environment_abi: dict[str, Any]
    environment_abi_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "effective_config": self.effective_config.to_dict(),
            "harness": self.harness.to_dict(),
            "harness_fingerprint": self.harness_fingerprint.value(),
            "trace_schema_version": self.trace_schema_version,
            "environment_abi": self.environment_abi,
            "environment_abi_digest": self.environment_abi_digest,
        }


@dataclass(frozen=True, slots=True)
class RunCompletedPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.RUN_COMPLETED
    stop_reason: str
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"stop_reason": self.stop_reason, "response": self.response}


@dataclass(frozen=True, slots=True)
class RunFailedPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.RUN_FAILED
    error: ErrorRecord

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.error.to_dict()}


@dataclass(frozen=True, slots=True)
class BranchStartedPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.BRANCH_STARTED
    request: dict[str, Any]
    controller_model: str
    prompt: str
    harness: HarnessSpec
    harness_fingerprint: HarnessFingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "controller_model": self.controller_model,
            "prompt": self.prompt,
            "harness": self.harness.to_dict(),
            "harness_fingerprint": self.harness_fingerprint.value(),
        }


@dataclass(frozen=True, slots=True)
class BranchCompletedPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.BRANCH_COMPLETED
    stop_reason: str
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"stop_reason": self.stop_reason, "response": self.response}


@dataclass(frozen=True, slots=True)
class ModelRequestPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.MODEL_REQUEST
    context: ModelCallContext
    request: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"context": _model_context(self.context), "request": self.request}


@dataclass(frozen=True, slots=True)
class ModelResponsePayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.MODEL_RESPONSE
    context: ModelCallContext
    duration_seconds: float
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": _model_context(self.context),
            "duration_seconds": self.duration_seconds,
            "response": self.response,
        }


@dataclass(frozen=True, slots=True)
class ModelFailedPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.MODEL_FAILED
    context: ModelCallContext
    duration_seconds: float
    error: ErrorRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": _model_context(self.context),
            "duration_seconds": self.duration_seconds,
            "error": self.error.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ControllerActionPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.CONTROLLER_ACTION
    code: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code}


@dataclass(frozen=True, slots=True)
class ControllerExecutionPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.CONTROLLER_EXECUTION
    execution: ExecutionResult

    def to_dict(self) -> dict[str, Any]:
        return {"execution": self.execution.to_dict()}


@dataclass(frozen=True, slots=True)
class ControllerObservationPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.CONTROLLER_OBSERVATION
    observation: Any

    def to_dict(self) -> dict[str, Any]:
        return {"observation": self.observation}


@dataclass(frozen=True, slots=True)
class RecoveryDecisionPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.CONTROLLER_RECOVERY_DECISION
    turn: int
    decision: RecoveryDecision

    def to_dict(self) -> dict[str, Any]:
        return {"turn": self.turn, "decision": self.decision.to_dict()}


@dataclass(frozen=True, slots=True)
class RepairPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.CONTROLLER_REPAIR
    turn: int
    fault: ControllerFault
    observation: Any

    def to_dict(self) -> dict[str, Any]:
        return {"turn": self.turn, "fault": self.fault.to_dict(), "observation": self.observation}


TracePayload = (
    RunStartedPayload
    | RunCompletedPayload
    | RunFailedPayload
    | BranchStartedPayload
    | BranchCompletedPayload
    | ModelRequestPayload
    | ModelResponsePayload
    | ModelFailedPayload
    | ControllerActionPayload
    | ControllerExecutionPayload
    | ControllerObservationPayload
    | RecoveryDecisionPayload
    | RepairPayload
)
_TRACE_PAYLOAD_TYPES = get_args(TracePayload)


class RunTraceStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TraceManifest:
    run_id: str
    status: RunTraceStatus
    stop_reason: str | None
    turns: int
    duration_seconds: float
    usage: dict[str, Any]
    error: ErrorRecord | None
    harness: dict[str, Any]
    harness_fingerprint: str
    effective_config: dict[str, Any]
    environment_abi: dict[str, Any]
    environment_abi_digest: str
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "stop_reason": self.stop_reason,
            "turns": self.turns,
            "duration_seconds": self.duration_seconds,
            "usage": self.usage,
            "error": None if self.error is None else self.error.to_dict(),
            "harness": self.harness,
            "harness_fingerprint": self.harness_fingerprint,
            "effective_config": self.effective_config,
            "environment_abi": self.environment_abi,
            "environment_abi_digest": self.environment_abi_digest,
            "schema_version": self.schema_version,
        }


ManifestFactory = Callable[[], TraceManifest]
TRACE_MANIFEST_FIELDS = tuple(field.name for field in fields(TraceManifest))


@dataclass(frozen=True, slots=True)
class TraceEvent:
    schema_version: str
    run_id: str
    event_id: str
    parent_event_id: EventRef
    branch_id: str
    depth: int
    timestamp: float
    type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "parent_event_id": self.parent_event_id,
            "branch_id": self.branch_id,
            "depth": self.depth,
            "timestamp": self.timestamp,
            "type": self.type,
            "payload": self.payload,
        }


TRACE_ENVELOPE_FIELDS = tuple(field.name for field in fields(TraceEvent))


def trace_payload_schema() -> dict[str, list[str]]:
    """Derive the reviewed payload key contract from the closed payload union."""

    schema: dict[str, list[str]] = {}
    payload_kinds: set[TraceEventKind] = set()
    for payload_type in _TRACE_PAYLOAD_TYPES:
        kind = payload_type.kind
        if kind in payload_kinds:
            raise RuntimeError(f"duplicate trace payload type for {kind.value}")
        payload_kinds.add(kind)
        schema[kind.value] = [field.name for field in fields(payload_type)]
    enum_kinds = set(TraceEventKind)
    if payload_kinds != enum_kinds:
        missing = sorted(kind.value for kind in enum_kinds - payload_kinds)
        extra = sorted(kind.value for kind in payload_kinds - enum_kinds)
        raise RuntimeError(f"trace payload coverage mismatch: missing={missing}, extra={extra}")
    return schema


_TRACE_PAYLOAD_SCHEMA = trace_payload_schema()


class TraceSink(Protocol):
    directory: Path | None

    def event(
        self,
        payload: TracePayload,
        *,
        branch_id: str,
        depth: int,
        parent_event_id: EventRef = None,
    ) -> EventRef: ...

    def close(
        self,
        *,
        manifest_factory: ManifestFactory,
        cause: BaseException | None = None,
    ) -> None: ...


class NullTraceSink:
    """A stateless trace sink that deliberately does not inspect its inputs."""

    directory: Path | None = None

    def event(
        self,
        payload: TracePayload,
        *,
        branch_id: str,
        depth: int,
        parent_event_id: EventRef = None,
    ) -> EventRef:
        return None

    def close(
        self,
        *,
        manifest_factory: ManifestFactory,
        cause: BaseException | None = None,
    ) -> None:
        return None


def make_trace_sink(config: TraceConfig, *, run_id: str) -> TraceSink:
    if not config.enabled:
        return NullTraceSink()
    return TraceRecorder(config, run_id=run_id)


class TraceRecorder:
    """The enabled trace sink; disabled recording is represented by ``NullTraceSink``."""

    def __init__(self, config: TraceConfig, *, run_id: str) -> None:
        if not config.enabled:
            raise ValueError("TraceRecorder requires enabled TraceConfig")
        self.config = config
        self.run_id = run_id
        self._lock = threading.Lock()
        self._next_id = 0
        self._closed = False
        self._event_ids: set[str] = set()
        self.events: list[dict[str, Any]] = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        directory = Path(config.directory).expanduser().resolve()
        self.directory: Path | None = directory / f"{stamp}-{run_id}"
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._file = (self.directory / "trace.jsonl").open("a", encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise TraceError(f"trace initialization failed: {exc}") from exc

    def event(
        self,
        payload: TracePayload,
        *,
        branch_id: str,
        depth: int,
        parent_event_id: EventRef = None,
    ) -> EventRef:
        with self._lock:
            if self._closed:
                raise TraceError("trace sink is closed")
            try:
                self._validate_event(payload, parent_event_id=parent_event_id)
                payload_value = payload.to_dict()
                expected_fields = _TRACE_PAYLOAD_SCHEMA[payload.kind.value]
                if not isinstance(payload_value, dict) or set(payload_value) != set(
                    expected_fields
                ):
                    raise TraceError(f"trace payload fields do not match {payload.kind.value}")
                event_id = f"evt-{self._next_id:06d}"
                event = TraceEvent(
                    schema_version=TRACE_SCHEMA_VERSION,
                    run_id=self.run_id,
                    event_id=event_id,
                    parent_event_id=parent_event_id,
                    branch_id=branch_id,
                    depth=depth,
                    timestamp=time.time(),
                    type=payload.kind.value,
                    payload=_paths_to_strings(copy.deepcopy(payload_value)),
                ).to_dict()
                error = json_compatibility_error(event)
                if error is not None:
                    raise TraceError(f"trace event requires strict JSON: {error}")
                encoded = strict_json_dumps(event, separators=(",", ":"))
                self._file.write(encoded + "\n")
                self._file.flush()
            except TraceError:
                raise
            except Exception as exc:
                raise TraceError(f"trace event failed: {exc}") from exc
            self._next_id += 1
            self._event_ids.add(event_id)
            if self.config.markdown:
                self.events.append(event)
            return event_id

    def close(
        self,
        *,
        manifest_factory: ManifestFactory,
        cause: BaseException | None = None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                manifest = manifest_factory()
                if type(manifest) is not TraceManifest:
                    raise TypeError("trace manifest factory must return exactly TraceManifest")
                value = manifest.to_dict()
                if set(value) != set(TRACE_MANIFEST_FIELDS):
                    raise TypeError("trace manifest fields do not match the schema")
                error = json_compatibility_error(value)
                if error is not None:
                    raise TypeError(f"trace manifest requires strict JSON: {error}")
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                if self.config.markdown:
                    self._write_markdown()
                self.write_json("manifest.json", value)
                self._fsync_directory()
            except TraceError as exc:
                if cause is not None:
                    raise exc from cause
                raise
            except Exception as exc:
                error = TraceError(f"trace finalize failed: {exc}")
                if cause is not None:
                    raise error from cause
                raise error from exc

    def write_json(self, name: str, value: object) -> None:
        """Atomically write a strict JSON trace artifact and fsync its contents."""

        if self.directory is None:  # pragma: no cover - enabled recorder always has a directory
            raise TraceError("trace recorder has no directory")
        encoded = strict_json_dumps(value, indent=2) + "\n"
        target = self.directory / name
        temporary = self.directory / f".{name}.{uuid.uuid4().hex}.tmp"
        with temporary.open("x", encoding="utf-8") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)

    def _validate_event(self, payload: object, *, parent_event_id: EventRef) -> None:
        if type(payload) not in _TRACE_PAYLOAD_TYPES:
            raise TraceError("trace event payload is not a closed TracePayload member")
        if not self._event_ids:
            if payload.kind is not TraceEventKind.RUN_STARTED or parent_event_id is not None:
                raise TraceError(
                    "run.started must be the first trace event with no parent_event_id"
                )
            return
        if parent_event_id not in self._event_ids:
            raise TraceError("trace event parent_event_id does not name a written event")

    def _write_markdown(self) -> None:
        if self.directory is None:  # pragma: no cover - enabled recorder always has a directory
            raise TraceError("trace recorder has no directory")
        lines = [f"# RLM trace `{self.run_id}`", ""]
        for event in self.events:
            lines.extend(
                [
                    f"## {event['event_id']} · {event['type']}",
                    "",
                    f"Branch `{event['branch_id']}`, depth {event['depth']}",
                    "",
                    "````json",
                    strict_json_dumps(event["payload"], indent=2),
                    "````",
                    "",
                ]
            )
        target = self.directory / "trace.md"
        target.write_text("\n".join(lines), encoding="utf-8")
        with target.open("r+", encoding="utf-8") as file:
            file.flush()
            os.fsync(file.fileno())

    def _fsync_directory(self) -> None:
        if self.directory is None:  # pragma: no cover - enabled recorder always has a directory
            raise TraceError("trace recorder has no directory")
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _model_context(context: ModelCallContext) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "branch_id": context.branch_id,
        "call_id": context.call_id,
        "role": context.role.value,
        "depth": context.depth,
    }


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_paths_to_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    return value
