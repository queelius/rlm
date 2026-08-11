"""Runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlm.prompts import Prompt
from rlm.types import API, ActionMode, normalize_api


@dataclass(frozen=True, slots=True)
class DebugConfig:
    enabled: bool = False
    directory: str | Path = "runs"
    markdown: bool = True


@dataclass(slots=True)
class RLMConfig:
    controller_model: str | None = None
    controller_api: API | str = "chat.completions"
    controller_options: dict[str, Any] = field(default_factory=dict)
    worker_model: str | None = None
    worker_api: API | str | None = None
    worker_options: dict[str, Any] = field(default_factory=dict)
    action_mode: ActionMode = "tool"
    max_turns: int = 12
    max_model_calls: int = 64
    max_subcalls: int = 48
    max_parallel_subcalls: int = 8
    max_depth: int = 0
    deadline_seconds: float = 600.0
    execution_timeout: float = 120.0
    max_execution_output_chars: int = 1_000_000
    max_observation_chars: int = 6_000
    max_consecutive_errors: int = 3
    working_directory: str | Path | None = None
    prompt: Prompt = field(default_factory=Prompt)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self) -> None:
        self.controller_api = normalize_api(str(self.controller_api))
        if self.worker_api is not None:
            self.worker_api = normalize_api(str(self.worker_api))
        if self.action_mode not in ("tool", "code"):
            raise ValueError("action_mode must be 'tool' or 'code'")
        for name in (
            "max_turns",
            "max_model_calls",
            "max_parallel_subcalls",
            "max_execution_output_chars",
            "max_observation_chars",
            "max_consecutive_errors",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_subcalls < 0:
            raise ValueError("max_subcalls cannot be negative")
        if self.max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        if self.deadline_seconds <= 0 or self.execution_timeout <= 0:
            raise ValueError("timeouts must be positive")
        protected = {
            "audio",
            "function_call",
            "functions",
            "input",
            "instructions",
            "messages",
            "modalities",
            "model",
            "n",
            "parallel_tool_calls",
            "response_format",
            "stream",
            "stream_options",
            "text",
            "tool_choice",
            "tools",
        }
        overlap = protected.intersection(self.controller_options)
        if overlap:
            raise ValueError(
                "controller_options cannot replace protocol fields: " + ", ".join(sorted(overlap))
            )
