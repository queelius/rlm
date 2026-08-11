"""A small, request-preserving Recursive Language Model runtime."""

from rlm.backend import ModelBackend, OpenAIEndpoint
from rlm.codex_backend import CodexAgentBackend
from rlm.config import DebugConfig, RLMConfig
from rlm.engine import RLM
from rlm.errors import (
    ExecutionTimeoutError,
    InvalidRequestError,
    LimitExceededError,
    ProtocolError,
    RLMError,
    UpstreamError,
)
from rlm.prompts import Prompt
from rlm.server import create_app
from rlm.types import API, ActionMode, RunResult, TokenUsage

__version__ = "0.1.0"

__all__ = [
    "API",
    "ActionMode",
    "CodexAgentBackend",
    "DebugConfig",
    "ExecutionTimeoutError",
    "InvalidRequestError",
    "LimitExceededError",
    "ModelBackend",
    "OpenAIEndpoint",
    "Prompt",
    "ProtocolError",
    "RLM",
    "RLMConfig",
    "RLMError",
    "RunResult",
    "TokenUsage",
    "UpstreamError",
    "__version__",
    "create_app",
]
