"""Validated, immutable runtime configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlm.json import json_compatibility_error, strict_json_dumps, strict_json_loads
from rlm.observation import MIN_OBSERVATION_CHARS
from rlm.prompts import default_harness_spec
from rlm.protocol import PROTOCOL_OWNED_FIELDS, UNSUPPORTED_CONTROLLER_FIELDS
from rlm.specs import HarnessSpec, harness_spec_from_dict


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _positive_number(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_turns: int = 12
    max_model_calls: int = 64
    max_subcalls: int = 48
    max_parallel_model_calls: int = 8
    max_depth: int = 0
    max_total_tokens: int | None = None
    deadline_seconds: float = 600.0
    execution_timeout_seconds: float = 120.0
    max_execution_output_chars: int = 1_000_000
    max_observation_chars: int = 6_000

    def __post_init__(self) -> None:
        for name in (
            "max_turns",
            "max_model_calls",
            "max_parallel_model_calls",
            "max_execution_output_chars",
        ):
            _positive_int(getattr(self, name), name=name)
        _positive_int(
            self.max_observation_chars,
            name="max_observation_chars",
            minimum=MIN_OBSERVATION_CHARS,
        )
        _positive_int(self.max_subcalls, name="max_subcalls", minimum=0)
        _positive_int(self.max_depth, name="max_depth", minimum=0)
        if self.max_total_tokens is not None:
            _positive_int(self.max_total_tokens, name="max_total_tokens")
        _positive_number(self.deadline_seconds, name="deadline_seconds")
        _positive_number(self.execution_timeout_seconds, name="execution_timeout_seconds")


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    model: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("controller model must be a non-empty string or None")
        validate_controller_options(self.options)
        # A frozen dataclass does not freeze nested mappings. Store an owned,
        # strict JSON round-trip so a caller cannot change a running profile.
        object.__setattr__(self, "options", _strict_json_object(self.options))


def _strict_json_object(options: dict[str, Any]) -> dict[str, Any]:
    encoded = strict_json_dumps(options, ensure_ascii=False, sort_keys=True)
    decoded = strict_json_loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded by caller
        raise TypeError("controller options must be an object")
    return decoded


def validate_controller_options(options: dict[str, Any]) -> None:
    """Validate controller sampling without permitting protocol replacement."""

    if not isinstance(options, dict):
        raise TypeError("controller options must be an object")
    json_error = json_compatibility_error(options)
    if json_error:
        raise ValueError(f"controller options must be strict JSON: {json_error}")
    unsupported = UNSUPPORTED_CONTROLLER_FIELDS.intersection(options)
    if unsupported:
        raise ValueError(
            "controller options contain unsupported fields: " + ", ".join(sorted(unsupported))
        )
    protected = PROTOCOL_OWNED_FIELDS.intersection(options)
    if protected:
        raise ValueError(
            "controller options cannot replace protocol fields: " + ", ".join(sorted(protected))
        )


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    working_directory: str | Path | None = None

    def __post_init__(self) -> None:
        if self.working_directory is not None and not isinstance(
            self.working_directory, (str, Path)
        ):
            raise TypeError("working_directory must be a string, Path, or None")


@dataclass(frozen=True, slots=True)
class TraceConfig:
    enabled: bool = False
    directory: str | Path = "runs"
    markdown: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.markdown, bool):
            raise TypeError("trace enabled and markdown values must be bool")
        if not isinstance(self.directory, (str, Path)):
            raise TypeError("trace directory must be a string or Path")


@dataclass(frozen=True, slots=True)
class RLMConfig:
    harness: HarnessSpec = field(default_factory=default_harness_spec)
    limits: RunLimits = field(default_factory=RunLimits)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    tracing: TraceConfig = field(default_factory=TraceConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.harness, HarnessSpec):
            raise TypeError("harness must be a HarnessSpec")
        if not isinstance(self.limits, RunLimits):
            raise TypeError("limits must be a RunLimits")
        if not isinstance(self.controller, ControllerConfig):
            raise TypeError("controller must be a ControllerConfig")
        if not isinstance(self.execution, ExecutionConfig):
            raise TypeError("execution must be an ExecutionConfig")
        if not isinstance(self.tracing, TraceConfig):
            raise TypeError("tracing must be a TraceConfig")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "harness": self.harness.to_dict(),
            "limits": {
                "max_turns": self.limits.max_turns,
                "max_model_calls": self.limits.max_model_calls,
                "max_subcalls": self.limits.max_subcalls,
                "max_parallel_model_calls": self.limits.max_parallel_model_calls,
                "max_depth": self.limits.max_depth,
                "max_total_tokens": self.limits.max_total_tokens,
                "deadline_seconds": self.limits.deadline_seconds,
                "execution_timeout_seconds": self.limits.execution_timeout_seconds,
                "max_execution_output_chars": self.limits.max_execution_output_chars,
                "max_observation_chars": self.limits.max_observation_chars,
            },
            "controller": {"model": self.controller.model, "options": self.controller.options},
            "execution": {
                "working_directory": (
                    str(self.execution.working_directory)
                    if self.execution.working_directory is not None
                    else None
                )
            },
            "tracing": {
                "enabled": self.tracing.enabled,
                "directory": str(self.tracing.directory),
                "markdown": self.tracing.markdown,
            },
        }
        encoded = strict_json_dumps(value, ensure_ascii=False, sort_keys=True)
        decoded = strict_json_loads(encoded)
        if not isinstance(decoded, dict):  # pragma: no cover
            raise TypeError("runtime config must encode an object")
        json_error = json_compatibility_error(decoded)
        if json_error:
            raise TypeError(f"runtime config must be strict JSON: {json_error}")
        return decoded

    def snapshot(self) -> RLMConfig:
        value = self.to_dict()
        return RLMConfig(
            harness=harness_spec_from_dict(value["harness"]),
            limits=RunLimits(**value["limits"]),
            controller=ControllerConfig(**value["controller"]),
            execution=ExecutionConfig(**value["execution"]),
            tracing=TraceConfig(**value["tracing"]),
        )
