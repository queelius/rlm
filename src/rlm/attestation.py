"""Typed run metadata shared by Python and HTTP consumers."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass

from rlm.types import (
    ControllerRunIdentity,
    RunResult,
    TokenUsage,
    _snapshot_field,
    _SnapshotAccess,
)

_HEADER_NAMES = {
    "run_id": "X-RLM-Run-ID",
    "stop_reason": "X-RLM-Stop-Reason",
    "turns": "X-RLM-Turns",
    "duration_seconds": "X-RLM-Duration-Seconds",
    "calls": "X-RLM-Model-Calls",
    "input_tokens": "X-RLM-Input-Tokens",
    "output_tokens": "X-RLM-Output-Tokens",
    "unreported_calls": "X-RLM-Usage-Unreported-Calls",
    "controller_model": "X-RLM-Controller-Model",
    "controller_options_sha256": "X-RLM-Controller-Options-SHA256",
    "harness_fingerprint": "X-RLM-Harness-Fingerprint",
}


@dataclass(frozen=True, slots=True)
class RunAttestation(_SnapshotAccess):
    """The complete execution identity carried by a successful run."""

    run_id: str
    stop_reason: str
    turns: int
    duration_seconds: float
    usage: TokenUsage = _snapshot_field()
    controller_identity: ControllerRunIdentity
    harness_fingerprint: str

    def __post_init__(self) -> None:
        _require_nonempty_text(self.run_id, name="run_id")
        _require_nonempty_text(self.stop_reason, name="stop_reason")
        _require_nonnegative_integer(self.turns, name="turns")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be a finite nonnegative number")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        for name in ("input_tokens", "output_tokens", "unreported_calls"):
            _require_nonnegative_integer(getattr(self.usage, name), name=name)
        _require_nonnegative_integer(self.usage.calls, name="calls")
        if self.usage.calls == 0:
            raise ValueError("calls must be positive")
        if not isinstance(self.controller_identity, ControllerRunIdentity):
            raise TypeError("controller_identity must be ControllerRunIdentity")
        _require_sha256(self.harness_fingerprint, name="harness_fingerprint")
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))
        object.__setattr__(self, "usage", copy.deepcopy(self.usage))

    @classmethod
    def from_result(cls, result: RunResult) -> RunAttestation:
        """Build an owned attestation from one completed run result."""

        if not isinstance(result, RunResult):
            raise TypeError("result must be RunResult")
        return cls(
            run_id=result.run_id,
            stop_reason=result.stop_reason,
            turns=result.turns,
            duration_seconds=result.duration_seconds,
            usage=result.usage,
            controller_identity=result.controller_identity,
            harness_fingerprint=result.harness_fingerprint,
        )

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> RunAttestation:
        """Decode the exact case-insensitive ``X-RLM-*`` wire contract."""

        if not isinstance(headers, Mapping):
            raise TypeError("headers must be a mapping")
        normalized: dict[str, str] = {}
        for name, value in headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ValueError("header names and values must be strings")
            lowered = name.lower()
            if lowered in normalized:
                raise ValueError(f"duplicate case-insensitive header {name!r}")
            normalized[lowered] = value

        values: dict[str, str] = {}
        missing: list[str] = []
        for field_name, header_name in _HEADER_NAMES.items():
            value = normalized.get(header_name.lower())
            if value is None:
                missing.append(header_name)
            else:
                values[field_name] = value
        if missing:
            raise ValueError(f"missing required RLM headers: {', '.join(missing)}")

        duration = _parse_duration(values["duration_seconds"])
        return cls(
            run_id=values["run_id"],
            stop_reason=values["stop_reason"],
            turns=_parse_nonnegative_integer(values["turns"], name="turns"),
            duration_seconds=duration,
            usage=TokenUsage(
                input_tokens=_parse_nonnegative_integer(
                    values["input_tokens"], name="input_tokens"
                ),
                output_tokens=_parse_nonnegative_integer(
                    values["output_tokens"], name="output_tokens"
                ),
                calls=_parse_nonnegative_integer(values["calls"], name="calls"),
                unreported_calls=_parse_nonnegative_integer(
                    values["unreported_calls"], name="unreported_calls"
                ),
            ),
            controller_identity=ControllerRunIdentity(
                values["controller_model"],
                values["controller_options_sha256"],
            ),
            harness_fingerprint=values["harness_fingerprint"],
        )

    def to_headers(self) -> dict[str, str]:
        """Encode the stable wire representation used by the HTTP adapter."""

        usage = self.usage
        return {
            _HEADER_NAMES["run_id"]: self.run_id,
            _HEADER_NAMES["stop_reason"]: self.stop_reason,
            _HEADER_NAMES["turns"]: str(self.turns),
            _HEADER_NAMES["duration_seconds"]: f"{self.duration_seconds:.6f}",
            _HEADER_NAMES["calls"]: str(usage.calls),
            _HEADER_NAMES["input_tokens"]: str(usage.input_tokens),
            _HEADER_NAMES["output_tokens"]: str(usage.output_tokens),
            _HEADER_NAMES["unreported_calls"]: str(usage.unreported_calls),
            _HEADER_NAMES["controller_model"]: self.controller_identity.model,
            _HEADER_NAMES["controller_options_sha256"]: self.controller_identity.options_sha256,
            _HEADER_NAMES["harness_fingerprint"]: self.harness_fingerprint,
        }


def _require_nonempty_text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_nonnegative_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _parse_nonnegative_integer(value: str, *, name: str) -> int:
    if not value or any(character not in "0123456789" for character in value):
        raise ValueError(f"{name} header must be a nonnegative integer")
    return int(value)


def _parse_duration(value: str) -> float:
    try:
        duration = float(value)
    except ValueError as exc:
        raise ValueError("duration header must be a finite nonnegative number") from exc
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration header must be a finite nonnegative number")
    return duration


def _require_sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
