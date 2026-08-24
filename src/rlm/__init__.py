"""A small, request-preserving Recursive Language Model runtime."""

from rlm.backend import ModelBackend, OpenAIEndpoint
from rlm.config import ControllerConfig, ExecutionConfig, RLMConfig, RunLimits, TraceConfig
from rlm.engine import RLM
from rlm.errors import (
    ExecutionTimeoutError,
    InvalidRequestError,
    LimitExceededError,
    ProtocolError,
    RLMError,
    UpstreamError,
)
from rlm.recovery import Abort, ControllerFault, FaultKind, RecoveryPolicy, Repair
from rlm.server import create_app
from rlm.specs import (
    BootstrapSpec,
    BootstrapType,
    ContextSpec,
    HarnessSpec,
    HistoryMode,
    ObservationSpec,
    ObservationType,
    PromptDigests,
    PromptSpec,
    RecoveryMode,
    RecoverySpec,
    SubmissionKind,
    SubmissionStatus,
)
from rlm.types import ModelCallContext, ModelRole, RunResult, TokenUsage

__version__ = "0.1.0"

__all__ = [
    "Abort",
    "BootstrapSpec",
    "BootstrapType",
    "ContextSpec",
    "ControllerConfig",
    "ControllerFault",
    "ExecutionConfig",
    "ExecutionTimeoutError",
    "InvalidRequestError",
    "LimitExceededError",
    "FaultKind",
    "HarnessSpec",
    "HistoryMode",
    "ModelBackend",
    "ModelCallContext",
    "ModelRole",
    "ObservationSpec",
    "ObservationType",
    "OpenAIEndpoint",
    "PromptSpec",
    "PromptDigests",
    "ProtocolError",
    "RLM",
    "RLMConfig",
    "RLMError",
    "RecoveryMode",
    "RecoveryPolicy",
    "RecoverySpec",
    "Repair",
    "RunLimits",
    "RunResult",
    "SubmissionKind",
    "SubmissionStatus",
    "TokenUsage",
    "TraceConfig",
    "UpstreamError",
    "__version__",
    "create_app",
]
