"""Errors with an OpenAI-compatible public representation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rlm.json import json_compatibility_error

if TYPE_CHECKING:
    from rlm.recovery import ControllerFault


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """The complete public and diagnostic representation of an RLM exception."""

    exception_type: str
    message: str
    code: str
    status_code: int
    public_error_type: str
    parameter: str | None
    body: dict[str, Any]
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "exception_type": self.exception_type,
            "message": self.message,
            "code": self.code,
            "status_code": self.status_code,
            "public_error_type": self.public_error_type,
            "parameter": self.parameter,
            "body": self.body,
            "details": self.details,
        }
        error = json_compatibility_error(value)
        if error is not None:
            raise TypeError(f"error record must be strict JSON: {error}")
        return value

    @classmethod
    def from_dict(cls, value: Any) -> ErrorRecord:
        """Decode one complete strict-JSON error record from an IPC frame."""

        if not isinstance(value, Mapping):
            raise ValueError("error record must be an object")
        fields = {
            "exception_type",
            "message",
            "code",
            "status_code",
            "public_error_type",
            "parameter",
            "body",
            "details",
        }
        if set(value) != fields:
            raise ValueError("error record has an invalid field set")
        if not all(
            isinstance(value[name], str) and value[name]
            for name in ("exception_type", "code", "public_error_type")
        ) or not isinstance(value["message"], str):
            raise ValueError("error record requires non-empty string identity fields")
        if isinstance(value["status_code"], bool) or not isinstance(value["status_code"], int):
            raise ValueError("error record status_code must be an integer")
        if value["parameter"] is not None and not isinstance(value["parameter"], str):
            raise ValueError("error record parameter must be a string or null")
        if not isinstance(value["body"], dict):
            raise ValueError("error record body must be an object")
        if value["details"] is not None and not isinstance(value["details"], dict):
            raise ValueError("error record details must be an object or null")
        record = cls(
            exception_type=value["exception_type"],
            message=value["message"],
            code=value["code"],
            status_code=value["status_code"],
            public_error_type=value["public_error_type"],
            parameter=value["parameter"],
            body=value["body"],
            details=value["details"],
        )
        record.to_dict()
        return record


@dataclass(frozen=True, slots=True)
class TraceCausalParent:
    """The event that directly caused one propagated run failure."""

    event_id: str | None

    def __post_init__(self) -> None:
        if self.event_id is not None and (not isinstance(self.event_id, str) or not self.event_id):
            raise ValueError("trace causal parent event_id must be a non-empty string or None")


class RLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "rlm_error",
        status_code: int = 500,
        error_type: str = "server_error",
        param: str | None = None,
        causal_parent: TraceCausalParent | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.causal_parent = causal_parent

    def to_body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }

    def to_record(self) -> ErrorRecord:
        details = self._record_details()
        return ErrorRecord(
            exception_type=type(self).__name__,
            message=self.message,
            code=self.code,
            status_code=self.status_code,
            public_error_type=self.error_type,
            parameter=self.param,
            body=self.to_body(),
            details=details,
        )

    def _record_details(self) -> dict[str, Any] | None:
        return None


def error_record_for_exception(error: BaseException, *, source: str) -> ErrorRecord:
    """Return the one strict diagnostic record for any internal exception."""

    if isinstance(error, RLMError):
        return error.to_record()
    message = str(error)
    return ErrorRecord(
        exception_type=type(error).__name__,
        message=message,
        code="internal_error",
        status_code=500,
        public_error_type="server_error",
        parameter=None,
        body={
            "error": {
                "message": message,
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            }
        },
        details={"source": source},
    )


class RemoteRLMError(RLMError):
    """Reconstituted RLM failure received over the executor boundary."""

    def __init__(self, record: ErrorRecord, *, causal_parent: TraceCausalParent | None = None):
        super().__init__(
            record.message,
            code=record.code,
            status_code=record.status_code,
            error_type=record.public_error_type,
            param=record.parameter,
            causal_parent=causal_parent,
        )
        self._record = record

    @classmethod
    def from_record(
        cls, record: ErrorRecord, *, causal_parent: TraceCausalParent | None = None
    ) -> RemoteRLMError:
        return cls(record, causal_parent=causal_parent)

    def to_body(self) -> dict[str, Any]:
        return self._record.body

    def to_record(self) -> ErrorRecord:
        return self._record


class InvalidRequestError(RLMError):
    def __init__(self, message: str, *, code: str = "invalid_request", param: str | None = None):
        super().__init__(
            message,
            code=code,
            status_code=400,
            error_type="invalid_request_error",
            param=param,
        )


class UpstreamError(RLMError):
    def __init__(self, status_code: int, body: Any, *, endpoint: str):
        message = _message_from_body(body) or f"upstream request failed with HTTP {status_code}"
        super().__init__(
            message,
            code="upstream_error",
            status_code=status_code,
            error_type="upstream_error",
        )
        self.body = body
        self.endpoint = endpoint

    def to_body(self) -> dict[str, Any]:
        if isinstance(self.body, dict) and "error" in self.body:
            return self.body
        return super().to_body()

    def _record_details(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint, "body": self.body}


class BackendProtocolError(RLMError):
    """A successful provider response violated the Responses protocol."""

    def __init__(self, message: str):
        super().__init__(message, code="backend_protocol_error", status_code=502)


class BackendResponseError(RLMError):
    """A provider returned a valid terminal failure or cancellation."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message, code="backend_response_error", status_code=502)
        self.details = details

    def _record_details(self) -> dict[str, Any] | None:
        return self.details


@dataclass(frozen=True, slots=True)
class BackendCloseFailure:
    backend_name: str
    exception_type: str
    message: str


class BackendCloseError(RLMError):
    """One or more optional backend cleanup calls failed."""

    def __init__(self, failures: Sequence[BackendCloseFailure]):
        if not failures:
            raise ValueError("BackendCloseError requires at least one failure")
        self.failures = tuple(failures)
        message = "; ".join(
            f"{failure.backend_name}: {failure.exception_type}: {failure.message}"
            for failure in self.failures
        )
        super().__init__(f"backend cleanup failed: {message}", code="backend_close_error")

    def _record_details(self) -> dict[str, Any]:
        return {
            "failures": [
                {
                    "backend_name": failure.backend_name,
                    "exception_type": failure.exception_type,
                    "message": failure.message,
                }
                for failure in self.failures
            ]
        }


class TraceError(RLMError):
    def __init__(self, message: str):
        super().__init__(message, code="trace_error")


class ProtocolError(RLMError):
    def __init__(self, message: str, *, causal_parent: TraceCausalParent | None = None):
        super().__init__(message, code="controller_protocol_error", causal_parent=causal_parent)


class RecoveryAbortedError(RLMError):
    """A sealed recovery policy deliberately declined a controller repair."""

    def __init__(self, fault: ControllerFault, *, causal_parent: TraceCausalParent | None = None):
        self.fault = fault
        super().__init__(
            f"controller recovery aborted: {fault.message}",
            code="recovery_aborted",
            causal_parent=causal_parent,
        )

    def _record_details(self) -> dict[str, Any]:
        return {"fault": self.fault.to_dict()}


class HarnessContractError(RLMError):
    """The configured controller harness does not match the runtime ABI."""

    def __init__(self, message: str):
        super().__init__(message, code="harness_contract_error")


class HostProtocolError(RLMError):
    """The private executor/host IPC protocol was violated."""

    def __init__(self, message: str):
        super().__init__(message, code="host_protocol_error")


class ModelOutputFault(Exception):
    """A strict text helper received a valid response without non-empty text."""

    def __init__(self, message: str, *, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class FinalSubmissionFault(Exception):
    """A controller final cannot cross the strict final-submission boundary."""

    def __init__(self, message: str, *, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class LimitExceededError(RLMError):
    def __init__(self, message: str, *, code: str):
        super().__init__(message, code=code)


class ExecutionTimeoutError(RLMError):
    def __init__(self, seconds: float):
        super().__init__(
            f"IPython execution exceeded {seconds:.1f} seconds",
            code="execution_timeout",
        )


def _message_from_body(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    if isinstance(body.get("message"), str):
        return body["message"]
    return None
