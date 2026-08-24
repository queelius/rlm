"""Deterministic Responses backends used by the tests."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlm.types import ModelCallContext

Response = dict[str, Any]
Step = Response | BaseException | Callable[[dict[str, Any]], Response]


def request() -> dict[str, Any]:
    """Return a fresh representative public request for each test."""
    return {
        "model": "public-model",
        "input": [{"role": "user", "content": "Preserve this request."}],
        "temperature": 0,
    }


@dataclass(frozen=True)
class BackendCall:
    request: dict[str, Any]


class ScriptedBackend:
    def __init__(self, steps: Sequence[Step] = (), *, name: str = "backend") -> None:
        self.name = name
        self._steps = list(steps)
        self._lock = threading.Lock()
        self.calls: list[BackendCall] = []
        self.timeouts: list[float] = []
        self.contexts: list[ModelCallContext] = []

    def complete(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float,
        context: ModelCallContext,
    ) -> Response:
        request_copy = copy.deepcopy(dict(request))
        with self._lock:
            self.calls.append(BackendCall(request_copy))
            self.timeouts.append(timeout)
            self.contexts.append(context)
            if not self._steps:
                raise AssertionError(f"{self.name} received unexpected call: {request_copy!r}")
            step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        response = step(copy.deepcopy(request_copy)) if callable(step) else step
        if not isinstance(response, dict):
            raise AssertionError(f"{self.name} returned a non-object: {response!r}")
        return copy.deepcopy(response)

    def assert_exhausted(self) -> None:
        assert not self._steps, f"{self.name} has {len(self._steps)} unused scripted step(s)"


class FunctionBackend:
    def __init__(
        self,
        responder: Callable[[dict[str, Any]], Response],
        *,
        name: str = "backend",
    ) -> None:
        self.name = name
        self.responder = responder
        self._lock = threading.Lock()
        self.calls: list[BackendCall] = []
        self.timeouts: list[float] = []
        self.contexts: list[ModelCallContext] = []

    def complete(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float,
        context: ModelCallContext,
    ) -> Response:
        request_copy = copy.deepcopy(dict(request))
        with self._lock:
            self.calls.append(BackendCall(request_copy))
            self.timeouts.append(timeout)
            self.contexts.append(context)
        response = self.responder(copy.deepcopy(request_copy))
        if not isinstance(response, dict):
            raise AssertionError(f"{self.name} returned a non-object: {response!r}")
        return copy.deepcopy(response)


def responses_text(
    text: str,
    *,
    model: str = "controller",
    response_id: str = "resp_scripted",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Response:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1_700_000_000,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": "msg_scripted",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "text": text,
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def controller_code(code: str, **kwargs: Any) -> Response:
    return responses_text(f"```python\n{code}\n```", **kwargs)


class RecordingCloseBackend(ScriptedBackend):
    def __init__(self, name: str = "recording") -> None:
        super().__init__(name=name)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class FailingCloseBackend(RecordingCloseBackend):
    def close(self) -> None:
        super().close()
        raise RuntimeError(f"{self.name} close failed")
