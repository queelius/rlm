"""Errors with an OpenAI-compatible public representation."""

from __future__ import annotations

from typing import Any


class RLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "rlm_error",
        status_code: int = 500,
        error_type: str = "server_error",
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.error_type = error_type
        self.param = param

    def to_body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


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


class ProtocolError(RLMError):
    def __init__(self, message: str):
        super().__init__(message, code="controller_protocol_error")


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
