"""One strict JSON contract for every RLM boundary."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


class StrictJSONError(ValueError):
    """Raised when a value or document is outside the JSON data model."""


def json_compatibility_error(value: Any) -> str | None:
    """Return why ``value`` is not strict JSON, or ``None`` when it is valid."""

    try:
        return _json_compatibility_error(value, path="$", active_containers=set())
    except RecursionError:
        return "value exceeds the supported JSON nesting depth"


def _json_compatibility_error(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return f"{path} contains an invalid Unicode surrogate"
        return None
    if isinstance(value, int):
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return f"{path} contains a non-finite number"
        return None
    if isinstance(value, list):
        return _container_error(value, path=path, active_containers=active_containers)
    if isinstance(value, dict):
        return _object_error(value, path=path, active_containers=active_containers)
    return f"{path} contains a non-JSON value ({type(value).__name__})"


def _container_error(value: list[Any], *, path: str, active_containers: set[int]) -> str | None:
    identity = id(value)
    if identity in active_containers:
        return f"{path} contains a reference cycle"
    active_containers.add(identity)
    try:
        for index, item in enumerate(value):
            error = _json_compatibility_error(
                item,
                path=f"{path}[{index}]",
                active_containers=active_containers,
            )
            if error is not None:
                return error
    finally:
        active_containers.remove(identity)
    return None


def _object_error(value: dict[Any, Any], *, path: str, active_containers: set[int]) -> str | None:
    identity = id(value)
    if identity in active_containers:
        return f"{path} contains a reference cycle"
    active_containers.add(identity)
    try:
        for key, item in value.items():
            if not isinstance(key, str):
                return f"{path} contains a non-string object key ({type(key).__name__})"
            try:
                key.encode("utf-8")
            except UnicodeEncodeError:
                return f"{path} contains an object key with an invalid Unicode surrogate"
            error = _json_compatibility_error(
                item,
                path=f"{path}[{key!r}]",
                active_containers=active_containers,
            )
            if error is not None:
                return error
    finally:
        active_containers.remove(identity)
    return None


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Decode a JSON document while rejecting duplicate keys and extensions."""

    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise StrictJSONError(f"invalid strict JSON: {exc}") from exc
    error = json_compatibility_error(decoded)
    if error is not None:
        raise StrictJSONError(f"invalid strict JSON: {error}")
    return decoded


def strict_json_dumps(
    value: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
    indent: int | None = None,
) -> str:
    """Validate and encode one JSON value without Python encoder extensions."""

    error = json_compatibility_error(value)
    if error is not None:
        raise StrictJSONError(f"value must be strict JSON: {error}")
    try:
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
            sort_keys=sort_keys,
            separators=separators,
            indent=indent,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded above
        raise StrictJSONError(f"value must be strict JSON: {exc}") from exc


def strict_json_sha256(value: Any) -> str:
    """Hash the canonical compact, sorted strict JSON encoding of ``value``."""

    encoded = strict_json_dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise StrictJSONError(f"non-finite JSON constant {value!r}")
