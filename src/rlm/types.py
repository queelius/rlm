"""Small public and internal data types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

API = Literal["chat.completions", "responses"]
ActionMode = Literal["tool", "code"]


def normalize_api(value: str) -> API:
    normalized = value.strip().lower().removeprefix("/v1/").strip("/")
    aliases = {
        "chat/completions": "chat.completions",
        "chat.completion": "chat.completions",
        "chat.completions": "chat.completions",
        "responses": "responses",
        "response": "responses",
    }
    try:
        return aliases[normalized]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(f"unsupported API family: {value!r}") from exc


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    display: str = ""
    duration_seconds: float = 0.0
    final_kind: str | None = None
    final_value: Any = None
    namespace: dict[str, str] = field(default_factory=dict)
    host_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def output(self) -> str:
        parts = [part.rstrip() for part in (self.stdout, self.stderr, self.display) if part]
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunResult:
    response: dict[str, Any]
    run_id: str
    stop_reason: str
    usage: TokenUsage
    turns: int
    duration_seconds: float
    trace_directory: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.trace_directory is not None:
            result["trace_directory"] = str(self.trace_directory)
        return result


def copy_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a JSON-like mapping without imposing a provider schema."""
    import copy

    return copy.deepcopy(dict(value))
