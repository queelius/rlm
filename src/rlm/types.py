"""Small public and internal data types."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

from rlm.errors import ErrorRecord, ModelOutputFault

_COPY_ON_ACCESS = "rlm.copy_on_access"


def _snapshot_field() -> Any:
    """Mark a dataclass field as an owned value copied on every public access."""

    return field(metadata={_COPY_ON_ACCESS: True})


class _SnapshotAccess:
    """Keep frozen public snapshots from exposing their mutable stored values."""

    __slots__ = ()

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        dataclass_fields = object.__getattribute__(self, "__dataclass_fields__")
        descriptor = dataclass_fields.get(name)
        if descriptor is not None and descriptor.metadata.get(_COPY_ON_ACCESS, False):
            return copy.deepcopy(value)
        return value


class ModelRole(str, Enum):
    CONTROLLER = "controller"
    PUBLIC = "public"
    SUBCALL = "subcall"


@dataclass(frozen=True, slots=True)
class ModelCallContext:
    run_id: str
    branch_id: str
    call_id: str
    role: ModelRole
    depth: int


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    unreported_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.unreported_calls += other.unreported_calls

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class HostFailureKind(str, Enum):
    RLM_ERROR = "rlm_error"
    INFRASTRUCTURE = "infrastructure"


class ExecutionFaultKind(str, Enum):
    """Closed worker-side classification of recoverable cell exceptions."""

    EXECUTION = "execution"
    MODEL_OUTPUT = "model_output"
    FINAL_SUBMISSION = "final_submission"


@dataclass(frozen=True, slots=True)
class HostFailure:
    kind: HostFailureKind
    error: ErrorRecord
    causal_parent_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionException:
    type: str
    message: str
    fault_kind: ExecutionFaultKind = ExecutionFaultKind.EXECUTION
    code: str | None = None
    traceback: str | None = None
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fault_kind, ExecutionFaultKind):
            raise TypeError("execution exception fault_kind must be an ExecutionFaultKind")
        if self.details is not None:
            object.__setattr__(self, "details", copy.deepcopy(self.details))


class OutputChannel(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"
    DISPLAY = "display"
    TRACEBACK = "traceback"


@dataclass(frozen=True, slots=True)
class OutputTruncation:
    channel: OutputChannel
    original_chars: int
    retained_chars: int
    omitted_chars: int


@dataclass(frozen=True, slots=True)
class TextSubmission:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text submission text must be a string")


@dataclass(frozen=True, slots=True)
class ResponseSubmission:
    response: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.response, Mapping):
            raise TypeError("response submission response must be a mapping")
        value = copy.deepcopy(dict(self.response))
        from rlm.json import json_compatibility_error

        json_error = json_compatibility_error(value)
        if json_error is not None:
            raise TypeError(f"response submission must be strict JSON: {json_error}")
        object.__setattr__(self, "response", value)


ExecutionSubmission = TextSubmission | ResponseSubmission


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    display: str = ""
    exception: ExecutionException | None = None
    host_failure: HostFailure | None = None
    submission: ExecutionSubmission | None = None
    namespace: dict[str, str] = field(default_factory=dict)
    truncations: tuple[OutputTruncation, ...] = ()
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.submission is not None and not isinstance(
            self.submission,
            (TextSubmission, ResponseSubmission),
        ):
            raise TypeError(
                "execution submission must be TextSubmission, ResponseSubmission, or None"
            )
        if self.submission is not None and (
            self.exception is not None or self.host_failure is not None
        ):
            raise ValueError("execution submission cannot accompany an exception or host failure")
        if self.exception is not None and self.host_failure is not None:
            raise ValueError("execution result cannot contain both exception and host failure")

    @property
    def output(self) -> str:
        parts = [part.rstrip() for part in (self.stdout, self.stderr, self.display) if part]
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "display": self.display,
            "exception": None
            if self.exception is None
            else {
                "type": self.exception.type,
                "message": self.exception.message,
                "fault_kind": self.exception.fault_kind.value,
                "code": self.exception.code,
                "traceback": self.exception.traceback,
                "details": self.exception.details,
            },
            "host_failure": None
            if self.host_failure is None
            else {
                "kind": self.host_failure.kind.value,
                "error": self.host_failure.error.to_dict(),
                "causal_parent_event_id": self.host_failure.causal_parent_event_id,
            },
            "submission": None
            if self.submission is None
            else {"text": self.submission.text}
            if isinstance(self.submission, TextSubmission)
            else {"response": self.submission.response},
            "namespace": self.namespace,
            "truncations": [
                {
                    "channel": item.channel.value,
                    "original_chars": item.original_chars,
                    "retained_chars": item.retained_chars,
                    "omitted_chars": item.omitted_chars,
                }
                for item in self.truncations
            ],
            "duration_seconds": self.duration_seconds,
        }


class TextFailureKind(str, Enum):
    EMPTY_TEXT = "empty_text"
    INCOMPLETE_RESPONSE = "incomplete_response"


@dataclass(frozen=True, slots=True)
class TextFailure:
    index: int
    kind: TextFailureKind
    response: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "response", copy.deepcopy(self.response))

    def summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind.value,
            "status": self.response.get("status"),
            "incomplete_details": self.response.get("incomplete_details"),
            "usage": self.response.get("usage"),
        }


@dataclass(frozen=True, slots=True)
class AskBatchResult:
    texts: tuple[str | None, ...]
    responses: tuple[dict[str, Any], ...]
    failures: tuple[TextFailure, ...]

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.responses):
            raise ValueError("batch texts and responses must have equal lengths")
        failed = tuple(item.index for item in self.failures)
        if len(failed) != len(set(failed)) or any(
            index < 0 or index >= len(self.texts) for index in failed
        ):
            raise ValueError("batch failures must have unique in-range indexes")
        missing = tuple(index for index, text in enumerate(self.texts) if text is None)
        if failed != missing:
            raise ValueError("batch failures must exactly identify missing texts")
        object.__setattr__(self, "responses", tuple(copy.deepcopy(item) for item in self.responses))

    @property
    def failed_indexes(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.failures)

    def require_texts(self) -> list[str]:
        if self.failures:
            raise ModelOutputFault(
                f"ask_batch has unusable text at indexes {self.failed_indexes}",
                details={"failures": [item.summary() for item in self.failures]},
            )
        if any(text is None for text in self.texts):
            raise ValueError("batch result violates its alignment invariant")
        return [cast(str, text) for text in self.texts]


@dataclass(frozen=True, slots=True)
class ControllerRunIdentity:
    model: str
    options_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("controller identity model must be non-empty")
        if (
            not isinstance(self.options_sha256, str)
            or len(self.options_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.options_sha256)
        ):
            raise ValueError("controller identity options_sha256 must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class RunResult(_SnapshotAccess):
    response: dict[str, Any] = _snapshot_field()
    run_id: str
    stop_reason: str
    usage: TokenUsage = _snapshot_field()
    turns: int
    duration_seconds: float
    controller_identity: ControllerRunIdentity
    harness_fingerprint: str
    trace_directory: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.response, Mapping):
            raise TypeError("run result response must be an object")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("run result usage must be TokenUsage")
        if not isinstance(self.controller_identity, ControllerRunIdentity):
            raise TypeError("run result requires a controller identity")
        duration = _canonical_duration_seconds(self.duration_seconds)
        if (
            not isinstance(self.harness_fingerprint, str)
            or len(self.harness_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.harness_fingerprint)
        ):
            raise ValueError("run result requires a harness SHA-256 fingerprint")
        object.__setattr__(self, "response", copy.deepcopy(dict(self.response)))
        object.__setattr__(self, "usage", copy.deepcopy(self.usage))
        object.__setattr__(self, "duration_seconds", duration)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.trace_directory is not None:
            result["trace_directory"] = str(self.trace_directory)
        return result


def _canonical_duration_seconds(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("duration_seconds must be a finite nonnegative number")
    return float(f"{value:.6f}")
