"""Versioned, inspected ABI for the controller IPython environment."""

from __future__ import annotations

import copy
import inspect
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, get_type_hints

from rlm.errors import HostProtocolError
from rlm.json import json_compatibility_error, strict_json_dumps, strict_json_sha256
from rlm.types import AskBatchResult


@dataclass(frozen=True, slots=True)
class HostRequestPayload:
    request: dict[str, Any] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", copy.deepcopy(self.request))


@dataclass(frozen=True, slots=True)
class HostBatchPayload:
    requests: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(copy.deepcopy(item) for item in self.requests))


HostPayload = HostRequestPayload | HostBatchPayload


class HostOperation(str, Enum):
    MODEL_COMPLETE = "model_complete"
    MODEL_COMPLETE_BATCH = "model_complete_batch"
    RLM_COMPLETE = "rlm_complete"
    RLM_COMPLETE_BATCH = "rlm_complete_batch"

    def decode_payload(self, raw: object) -> HostPayload:
        """Validate one strict wire payload and return its typed representation."""

        json_error = json_compatibility_error(raw)
        if json_error is not None:
            raise HostProtocolError(f"{self.value} payload is not strict JSON: {json_error}")
        if not isinstance(raw, dict):
            raise HostProtocolError(f"{self.value} payload must be an object")
        if self in (HostOperation.MODEL_COMPLETE, HostOperation.RLM_COMPLETE):
            if set(raw) != {"request"}:
                raise HostProtocolError(f"{self.value} payload requires only 'request'")
            request = raw["request"]
            if request is not None and not isinstance(request, dict):
                raise HostProtocolError(f"{self.value} request must be an object or null")
            return HostRequestPayload(request)
        if set(raw) != {"requests"}:
            raise HostProtocolError(f"{self.value} payload requires only 'requests'")
        requests = raw["requests"]
        if not isinstance(requests, list) or not all(isinstance(item, dict) for item in requests):
            raise HostProtocolError(f"{self.value} requests must be a list of objects")
        return HostBatchPayload(tuple(requests))


@dataclass(frozen=True, slots=True)
class FunctionMetadata:
    description: str
    operation: HostOperation | None = None
    requires_recursion: bool = False


def environment_function(
    *,
    description: str,
    operation: HostOperation | None = None,
    requires_recursion: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach immutable ABI metadata to a protocol function declaration."""

    metadata = FunctionMetadata(description, operation, requires_recursion)

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        function.__rlm_environment_metadata__ = metadata
        return function

    return decorate


class EnvironmentNamespace(Protocol):
    @staticmethod
    @environment_function(
        description="Call the public model and return its exact Responses object.",
        operation=HostOperation.MODEL_COMPLETE,
    )
    def model_complete(request_obj: dict[str, Any] | None = None) -> dict[str, Any]: ...

    @staticmethod
    @environment_function(
        description="Call the public model for each full Responses request.",
        operation=HostOperation.MODEL_COMPLETE_BATCH,
    )
    def model_complete_batch(requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]: ...

    @staticmethod
    @environment_function(description="Call a text model helper and return non-empty text.")
    def ask(
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str: ...

    @staticmethod
    @environment_function(description="Call text helpers and preserve index-aligned failures.")
    def ask_batch(
        prompts: Sequence[str],
        *,
        system: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AskBatchResult: ...

    @staticmethod
    @environment_function(
        description="Run one isolated recursive RLM child.",
        operation=HostOperation.RLM_COMPLETE,
        requires_recursion=True,
    )
    def rlm_complete(request_obj: dict[str, Any] | None = None) -> dict[str, Any]: ...

    @staticmethod
    @environment_function(
        description="Run isolated recursive RLM children for full Responses requests.",
        operation=HostOperation.RLM_COMPLETE_BATCH,
        requires_recursion=True,
    )
    def rlm_complete_batch(requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]: ...

    @staticmethod
    @environment_function(description="Submit exactly one plain-text final answer.")
    def FINAL_TEXT(text: str) -> None: ...

    @staticmethod
    @environment_function(description="Submit exactly one strict-JSON Responses final answer.")
    def FINAL_RESPONSE(response: dict[str, Any]) -> None: ...

    @staticmethod
    @environment_function(description="Show compact summaries of user-created variables.")
    def SHOW_VARS() -> dict[str, str]: ...


@dataclass(frozen=True, slots=True)
class FunctionSpec:
    name: str
    signature: inspect.Signature
    metadata: FunctionMetadata

    def render(self) -> str:
        return f"{self.name}{self.signature}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signature": str(self.signature),
            "description": self.metadata.description,
            "operation": None if self.metadata.operation is None else self.metadata.operation.value,
            "requires_recursion": self.metadata.requires_recursion,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentABI:
    version: str
    request_name: str
    functions: tuple[FunctionSpec, ...]

    @classmethod
    def from_protocol(
        cls,
        protocol: type[EnvironmentNamespace],
        *,
        version: str,
        request_name: str,
    ) -> EnvironmentABI:
        functions: list[FunctionSpec] = []
        module_globals = vars(sys.modules[protocol.__module__])
        for name, descriptor in protocol.__dict__.items():
            if not isinstance(descriptor, staticmethod):
                continue
            function = descriptor.__func__
            metadata = getattr(function, "__rlm_environment_metadata__", None)
            if not isinstance(metadata, FunctionMetadata):
                continue
            hints = get_type_hints(function, globalns=module_globals, localns=dict(vars(protocol)))
            signature = inspect.signature(function, eval_str=True, globals=module_globals)
            parameters = [
                parameter.replace(annotation=hints.get(parameter.name, parameter.annotation))
                for parameter in signature.parameters.values()
            ]
            signature = signature.replace(
                parameters=parameters,
                return_annotation=hints.get("return", signature.return_annotation),
            )
            functions.append(FunctionSpec(name, signature, metadata))
        abi = cls(version=version, request_name=request_name, functions=tuple(functions))
        abi._validate_structure()
        return abi

    def _validate_structure(self) -> None:
        if not self.version or not self.request_name:
            raise ValueError("environment ABI version and request name must be non-empty")
        names = [item.name for item in self.functions]
        if len(names) != len(set(names)):
            raise ValueError("environment ABI function names must be unique")
        operations = [item.metadata.operation for item in self.functions if item.metadata.operation]
        if len(operations) != len(set(operations)):
            raise ValueError("environment ABI host operations must be unique")

    def enabled_functions(self, *, allow_recursion: bool) -> tuple[FunctionSpec, ...]:
        return tuple(
            item
            for item in self.functions
            if allow_recursion or not item.metadata.requires_recursion
        )

    def canonical_json(self) -> str:
        return strict_json_dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "request_name": self.request_name,
            "functions": [item.to_dict() for item in self.functions],
        }

    def digest(self) -> str:
        return strict_json_sha256(self.to_dict())


def validate_environment_bindings(
    abi: EnvironmentABI,
    bindings: dict[str, Callable[..., Any]],
    *,
    allow_recursion: bool = True,
) -> None:
    """Require the installed helper set to exactly match the inspected ABI."""

    expected = {item.name: item for item in abi.enabled_functions(allow_recursion=allow_recursion)}
    actual = set(bindings)
    missing = set(expected).difference(actual)
    extra = actual.difference(expected)
    if missing:
        raise HostProtocolError(
            f"environment bindings missing documented callable(s): {sorted(missing)}"
        )
    if extra:
        raise HostProtocolError(
            f"environment bindings include undocumented callable(s): {sorted(extra)}"
        )
    for name, spec in expected.items():
        value = bindings[name]
        if not callable(value):
            raise HostProtocolError(f"environment binding {name!r} is not callable")
        module = sys.modules.get(value.__module__)
        module_globals = vars(module) if module is not None else None
        hints = get_type_hints(value, globalns=module_globals)
        raw_signature = inspect.signature(value, eval_str=True, globals=module_globals)
        signature = raw_signature.replace(
            parameters=[
                parameter.replace(annotation=hints.get(parameter.name, parameter.annotation))
                for parameter in raw_signature.parameters.values()
            ],
            return_annotation=hints.get("return", raw_signature.return_annotation),
        )
        if signature != spec.signature:
            raise HostProtocolError(
                f"environment binding {name!r} has signature {signature}, expected {spec.signature}"
            )


ENVIRONMENT_ABI = EnvironmentABI.from_protocol(
    EnvironmentNamespace,
    version="2",
    request_name="request",
)
