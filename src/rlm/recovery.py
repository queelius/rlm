"""Typed, versioned recovery decisions for controller faults."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from rlm.json import json_compatibility_error, strict_json_dumps, strict_json_loads
from rlm.specs import RecoveryMode, RecoverySpec


class FaultKind(str, Enum):
    CONTROLLER_PROTOCOL = "controller_protocol"
    EXECUTION = "execution"
    MODEL_OUTPUT = "model_output"
    FINAL_SUBMISSION = "final_submission"


@dataclass(frozen=True, slots=True)
class ControllerFault:
    kind: FaultKind
    message: str
    details: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FaultKind):
            raise TypeError("kind must be a FaultKind")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string")
        if self.details is not None:
            if not isinstance(self.details, dict):
                raise TypeError("details must be an object or None")
            json_error = json_compatibility_error(self.details)
            if json_error is not None:
                raise ValueError(f"details must contain strict JSON: {json_error}")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind.value, "message": self.message}
        if self.details is not None:
            value["details"] = strict_json_loads(strict_json_dumps(self.details))
        return value


@dataclass(frozen=True, slots=True)
class Repair:
    fault: ControllerFault

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "repair", "fault": self.fault.to_dict()}


@dataclass(frozen=True, slots=True)
class Abort:
    fault: ControllerFault

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "abort", "fault": self.fault.to_dict()}


RecoveryDecision = Repair | Abort


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """The sealed interpreter for a run's effective recovery specification."""

    spec: RecoverySpec

    def decide(self, fault: ControllerFault, *, turn: int) -> RecoveryDecision:
        if turn < 1:
            raise ValueError("turn must be positive")
        if self.spec.mode is RecoveryMode.REPAIR:
            return Repair(fault)
        if self.spec.mode is RecoveryMode.ABORT:
            return Abort(fault)
        raise AssertionError(f"unhandled recovery mode: {self.spec.mode!r}")
