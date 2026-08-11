"""Deterministic OpenAI-compatible backends used by the test suite."""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlm.types import API

Response = dict[str, Any]
Step = Response | BaseException | Callable[[API, dict[str, Any]], Response]


@dataclass(frozen=True)
class BackendCall:
    api: API
    request: dict[str, Any]


class ScriptedBackend:
    """Record calls and consume one deterministic response step per call.

    A callable step receives a detached copy of the API and request. Responses
    are copied on return so tests cannot accidentally share mutable provider
    objects with the engine.
    """

    def __init__(self, steps: Sequence[Step] = (), *, name: str = "backend") -> None:
        self.name = name
        self._steps = list(steps)
        self._lock = threading.Lock()
        self.calls: list[BackendCall] = []

    def complete(self, api: API, request: Mapping[str, Any]) -> Response:
        request_copy = copy.deepcopy(dict(request))
        with self._lock:
            self.calls.append(BackendCall(api=api, request=request_copy))
            if not self._steps:
                raise AssertionError(
                    f"{self.name} received unexpected call {api}: {request_copy!r}"
                )
            step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        response = step(api, copy.deepcopy(request_copy)) if callable(step) else step
        if not isinstance(response, dict):
            raise AssertionError(f"{self.name} scripted response is not an object: {response!r}")
        return copy.deepcopy(response)

    def assert_exhausted(self) -> None:
        assert not self._steps, f"{self.name} has {len(self._steps)} unused scripted step(s)"


class FunctionBackend:
    """Thread-safe recorder backed by a request-dependent response function."""

    def __init__(
        self,
        responder: Callable[[API, dict[str, Any]], Response],
        *,
        name: str = "backend",
    ) -> None:
        self.name = name
        self.responder = responder
        self._lock = threading.Lock()
        self.calls: list[BackendCall] = []

    def complete(self, api: API, request: Mapping[str, Any]) -> Response:
        request_copy = copy.deepcopy(dict(request))
        with self._lock:
            self.calls.append(BackendCall(api=api, request=request_copy))
        response = self.responder(api, copy.deepcopy(request_copy))
        if not isinstance(response, dict):
            raise AssertionError(f"{self.name} response is not an object: {response!r}")
        return copy.deepcopy(response)


def chat_text(
    text: str,
    *,
    model: str = "controller",
    response_id: str = "chatcmpl-scripted",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Response:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chat_tool(
    code: str,
    *,
    call_id: str = "call-ipython-1",
    model: str = "controller",
    response_id: str = "chatcmpl-tool-scripted",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Response:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "ipython",
                                "arguments": json.dumps({"code": code}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


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


def responses_tool(
    code: str,
    *,
    call_id: str = "call-ipython-response-1",
    model: str = "controller",
    response_id: str = "resp_tool_scripted",
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
                "id": "fc_scripted",
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": "ipython",
                "arguments": json.dumps({"code": code}),
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
