"""Strict, bounded controller observations derived from typed execution state."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from rlm.abi import ENVIRONMENT_ABI
from rlm.json import strict_json_dumps, strict_json_loads
from rlm.recovery import ControllerFault
from rlm.specs import ObservationSpec, SubmissionStatus
from rlm.types import ExecutionResult

# Runtime configurations below this floor cannot reliably carry the complete
# typed identity of every built-in repair path. The encoder may still be used
# directly with a smaller budget when a particular value demonstrably fits.
MIN_OBSERVATION_CHARS = 512


def controller_observation(
    *,
    spec: ObservationSpec,
    fault: ControllerFault | None,
    execution: ExecutionResult | None,
    max_chars: int,
) -> dict[str, Any]:
    """Encode one deterministic model-visible observation within ``max_chars``.

    Structural identity is retained before optional text.  The bounded value is
    assembled as data and encoded only for measurement; no serialized JSON is
    sliced, so it always remains valid strict JSON.
    """

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    status = "error" if fault is not None else "ok"
    source = execution or ExecutionResult()
    execution_value: dict[str, Any] = {
        "status": status,
        "truncation": {"omitted_chars": 0, "fields": {}},
    }
    value: dict[str, Any] = {
        "type": spec.type.value,
        "schema_version": spec.schema_version,
        "request_binding": ENVIRONMENT_ABI.request_name,
        "submission": {"status": SubmissionStatus.ABSENT.value, "required": True},
        "execution": execution_value,
    }
    if fault is not None:
        execution_value["fault"] = {"kind": fault.kind.value}
    if source.exception is not None:
        execution_value["exception"] = {"type": source.exception.type}

    candidates: list[tuple[str, str]] = []
    if fault is not None:
        candidates.append(("fault.message", fault.message))
        if fault.details is not None:
            candidates.append(
                ("fault.details", strict_json_dumps(fault.details, separators=(",", ":")))
            )
    if source.exception is not None:
        candidates.append(("exception.message", source.exception.message))
        if source.exception.code is not None:
            candidates.append(("exception.code", source.exception.code))
        if source.exception.details is not None:
            candidates.append(
                (
                    "exception.details",
                    strict_json_dumps(source.exception.details, separators=(",", ":")),
                )
            )
    candidates.extend(
        [
            ("stdout", source.stdout),
            ("stderr", source.stderr),
            ("display", source.display),
            ("namespace", strict_json_dumps(source.namespace, separators=(",", ":"))),
            ("duration_seconds", _duration_text(source.duration_seconds)),
        ]
    )
    structured = {"fault.details", "exception.details", "namespace", "duration_seconds"}

    retained: dict[str, str] = {}
    omitted = {name: len(text) for name, text in candidates if text}
    _set_truncation(execution_value, omitted)
    if _encoded_length(value) > max_chars:
        raise ValueError("observation structural metadata exceeds max_chars")

    for name, text in candidates:
        if not text:
            continue
        if name in structured:
            proposed_retained = dict(retained)
            proposed_retained[name] = text
            proposed_omitted = dict(omitted)
            proposed_omitted.pop(name, None)
            accepted = (
                len(text)
                if _encoded_length(_with_retained(value, proposed_retained, proposed_omitted))
                <= max_chars
                else 0
            )
        else:
            accepted = _max_prefix_that_fits(value, retained, omitted, name, text, max_chars)
        if accepted:
            retained[name] = text[:accepted]
            remaining = len(text) - accepted
            if remaining:
                omitted[name] = remaining
            else:
                omitted.pop(name, None)
            _apply_retained(execution_value, retained)
            _set_truncation(execution_value, omitted)

    if _encoded_length(value) > max_chars:  # pragma: no cover - guarded by construction
        raise ValueError("observation encoding exceeds max_chars")
    return value


def _duration_text(seconds: float) -> str:
    return format(seconds, ".6f")


def _max_prefix_that_fits(
    value: dict[str, Any],
    retained: Mapping[str, str],
    omitted: Mapping[str, int],
    name: str,
    text: str,
    max_chars: int,
) -> int:
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        proposed_retained = dict(retained)
        proposed_retained[name] = text[:middle]
        proposed_omitted = dict(omitted)
        remaining = len(text) - middle
        if remaining:
            proposed_omitted[name] = remaining
        else:
            proposed_omitted.pop(name, None)
        candidate = _with_retained(value, proposed_retained, proposed_omitted)
        if _encoded_length(candidate) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return low


def _with_retained(
    value: Mapping[str, Any], retained: Mapping[str, str], omitted: Mapping[str, int]
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(value))
    execution = candidate.get("execution")
    if not isinstance(execution, dict):  # pragma: no cover - internal envelope invariant
        raise TypeError("controller observation execution must be an object")
    _apply_retained(execution, retained)
    _set_truncation(execution, omitted)
    return candidate


def _apply_retained(value: dict[str, Any], retained: Mapping[str, str]) -> None:
    for name, text in retained.items():
        if name == "fault.details":
            assert isinstance(value.get("fault"), dict)
            value["fault"]["details"] = strict_json_loads(text)
        elif name.startswith("fault."):
            assert isinstance(value.get("fault"), dict)
            value["fault"][name.removeprefix("fault.")] = text
        elif name.startswith("exception."):
            assert isinstance(value.get("exception"), dict)
            field = name.removeprefix("exception.")
            value["exception"][field] = strict_json_loads(text) if field == "details" else text
        elif name == "namespace":
            value[name] = strict_json_loads(text)
        elif name == "duration_seconds":
            value[name] = float(text)
        else:
            value[name] = text


def _set_truncation(value: dict[str, Any], omitted: Mapping[str, int]) -> None:
    value["truncation"] = {
        "omitted_chars": sum(omitted.values()),
        "fields": {name: omitted[name] for name in sorted(omitted)},
    }


def _encoded_length(value: Mapping[str, Any]) -> int:
    return len(strict_json_dumps(value, sort_keys=True, separators=(",", ":")))
