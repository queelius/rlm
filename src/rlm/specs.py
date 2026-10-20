"""Versioned, content-addressed runtime harness specifications."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from rlm.abi import ENVIRONMENT_ABI
from rlm.json import strict_json_dumps, strict_json_loads, strict_json_sha256


def _nonempty(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class HistoryMode(str, Enum):
    LATEST_TURN = "latest_turn"


class BootstrapType(str, Enum):
    CONTROLLER = "rlm.controller_bootstrap"


class ObservationType(str, Enum):
    CONTROLLER = "rlm.controller_observation"


class SubmissionKind(str, Enum):
    TEXT = "text"
    RESPONSE = "response"


class SubmissionStatus(str, Enum):
    ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class BootstrapSpec:
    """Versioned identity of the metadata-only first controller item."""

    type: BootstrapType = BootstrapType.CONTROLLER
    schema_version: str = "2"
    content_in_message: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.type, BootstrapType):
            raise TypeError("type must be a BootstrapType")
        _nonempty(self.schema_version, field_name="schema_version")
        if not isinstance(self.content_in_message, bool):
            raise TypeError("content_in_message must be a bool")
        if self.content_in_message:
            raise ValueError("controller bootstrap cannot contain caller content")


@dataclass(frozen=True, slots=True)
class PromptSpec:
    name: str
    version: str
    protocol: str
    policy: str

    def __post_init__(self) -> None:
        for name in ("name", "version", "protocol", "policy"):
            _nonempty(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class PromptDigests:
    """Content identities for both controller prompt variants."""

    nonrecursive: str
    recursive: str

    def __post_init__(self) -> None:
        for name in ("nonrecursive", "recursive"):
            value = getattr(self, name)
            if not _is_lowercase_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _is_lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class RecoveryMode(str, Enum):
    REPAIR = "repair"
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class RecoverySpec:
    mode: RecoveryMode = RecoveryMode.REPAIR
    version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RecoveryMode):
            raise TypeError("mode must be a RecoveryMode")
        _nonempty(self.version, field_name="version")


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    type: ObservationType = ObservationType.CONTROLLER
    schema_version: str = "2"

    def __post_init__(self) -> None:
        if not isinstance(self.type, ObservationType):
            raise TypeError("type must be an ObservationType")
        _nonempty(self.schema_version, field_name="schema_version")


@dataclass(frozen=True, slots=True)
class ContextSpec:
    history: HistoryMode = HistoryMode.LATEST_TURN
    preserve_reasoning: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.history, HistoryMode):
            raise TypeError("history must be a HistoryMode")
        if not isinstance(self.preserve_reasoning, bool):
            raise TypeError("preserve_reasoning must be a bool")


@dataclass(frozen=True, slots=True)
class PreludeSpec:
    """Python source the environment executes once before the first controller cell.

    The names it defines are part of the harness, not of the controller's work: they are listed
    in the rendered prompt with one-line descriptions, hidden from ``SHOW_VARS`` and restored
    before every cell. The source digest is part of the harness identity.
    """

    source: str
    exports: tuple[tuple[str, str], ...]
    version: str = "1"

    def __post_init__(self) -> None:
        _nonempty(self.source, field_name="source")
        _nonempty(self.version, field_name="version")
        if not isinstance(self.exports, tuple) or not self.exports:
            raise TypeError("exports must be a non-empty tuple of (name, description) pairs")
        for item in self.exports:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(part, str) for part in item)
            ):
                raise TypeError("each export must be a (name, description) pair of strings")
            if not item[0].isidentifier():
                raise ValueError(f"export name is not an identifier: {item[0]!r}")
            _nonempty(item[1], field_name="export description")

    def digest(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HarnessSpec:
    prompt: PromptSpec
    rendered_prompts: PromptDigests
    schema_version: str = "1"
    abi_version: str = ENVIRONMENT_ABI.version
    abi_digest: str = ENVIRONMENT_ABI.digest()
    bootstrap: BootstrapSpec = field(default_factory=BootstrapSpec)
    recovery: RecoverySpec = field(default_factory=RecoverySpec)
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    context: ContextSpec = field(default_factory=ContextSpec)
    prelude: PreludeSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, PromptSpec):
            raise TypeError("prompt must be a PromptSpec")
        if not isinstance(self.rendered_prompts, PromptDigests):
            raise TypeError("rendered_prompts must be a PromptDigests")
        for name in ("schema_version", "abi_version", "abi_digest"):
            _nonempty(getattr(self, name), field_name=name)
        if not isinstance(self.bootstrap, BootstrapSpec):
            raise TypeError("bootstrap must be a BootstrapSpec")
        if not isinstance(self.recovery, RecoverySpec):
            raise TypeError("recovery must be a RecoverySpec")
        if not isinstance(self.observation, ObservationSpec):
            raise TypeError("observation must be an ObservationSpec")
        if not isinstance(self.context, ContextSpec):
            raise TypeError("context must be a ContextSpec")
        if self.prelude is not None and not isinstance(self.prelude, PreludeSpec):
            raise TypeError("prelude must be a PreludeSpec or None")

    def _snapshot(self) -> dict[str, Any]:
        value = _jsonable(asdict(self))
        if value.get("prelude") is None:
            # A harness without a prelude keeps the identity it had before preludes existed.
            value.pop("prelude", None)
        return value

    def canonical_json(self) -> str:
        return strict_json_dumps(self._snapshot(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        value = strict_json_loads(self.canonical_json())
        if not isinstance(value, dict):
            raise TypeError("canonical HarnessSpec must encode an object")
        return value

    def fingerprint(self) -> str:
        return strict_json_sha256(self._snapshot())


def _jsonable(value: Any) -> Any:
    """Turn dataclass tuples (prelude exports) into lists so the snapshot stays strict JSON."""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def harness_spec_from_dict(value: dict[str, Any]) -> HarnessSpec:
    """Reconstruct a validated harness after its strict JSON snapshot."""
    prompt = value["prompt"]
    rendered_prompts = value["rendered_prompts"]
    bootstrap = value["bootstrap"]
    recovery = value["recovery"]
    observation = value["observation"]
    context = value["context"]
    if not all(
        isinstance(item, dict)
        for item in (prompt, rendered_prompts, bootstrap, recovery, observation, context)
    ):
        raise TypeError("harness snapshot sections must be objects")
    return HarnessSpec(
        prompt=PromptSpec(**prompt),
        rendered_prompts=PromptDigests(**rendered_prompts),
        schema_version=value["schema_version"],
        abi_version=value["abi_version"],
        abi_digest=value["abi_digest"],
        bootstrap=BootstrapSpec(
            type=BootstrapType(bootstrap["type"]),
            schema_version=bootstrap["schema_version"],
            content_in_message=bootstrap["content_in_message"],
        ),
        recovery=RecoverySpec(mode=RecoveryMode(recovery["mode"]), version=recovery["version"]),
        observation=ObservationSpec(
            type=ObservationType(observation["type"]),
            schema_version=observation["schema_version"],
        ),
        prelude=(
            PreludeSpec(
                source=value["prelude"]["source"],
                exports=tuple(tuple(item) for item in value["prelude"]["exports"]),
                version=value["prelude"]["version"],
            )
            if value.get("prelude") is not None
            else None
        ),
        context=ContextSpec(
            history=HistoryMode(context["history"]),
            preserve_reasoning=context["preserve_reasoning"],
        ),
    )
