"""Minimal valid response construction for controller-authored plain text."""

from __future__ import annotations

import copy
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from rlm.types import API, TokenUsage


def text_response(
    api: API,
    request: Mapping[str, Any],
    text: str,
    *,
    usage: TokenUsage,
) -> dict[str, Any]:
    if api == "chat.completions":
        return _chat_text_response(request, text, usage)
    return _responses_text_response(request, text, usage)


def validate_response(api: API, response: Any) -> str | None:
    if not isinstance(response, dict):
        return f"FINAL_RESPONSE requires an object, got {type(response).__name__}"
    if api == "chat.completions":
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return "Chat Completions response must contain a non-empty choices list"
    else:
        output = response.get("output")
        if not isinstance(output, list):
            return "Responses response must contain an output list"

    json_error = json_compatibility_error(response)
    if json_error is not None:
        return f"FINAL_RESPONSE requires strict JSON: {json_error}"
    return None


def json_compatibility_error(value: Any) -> str | None:
    """Return why ``value`` is not strict JSON, or ``None`` when it is valid.

    Python's ``json`` encoder accepts conveniences that are not part of the
    JSON data model, including non-string object keys, tuples, and non-finite
    floats.  Public response objects deliberately use the narrower wire-safe
    representation: null, booleans, finite numbers, Unicode strings, lists,
    and dictionaries whose keys are Unicode strings.
    """

    try:
        return _json_compatibility_error(value, path="$", active_containers=set())
    except RecursionError:
        return "value exceeds the supported JSON nesting depth"


def _json_compatibility_error(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return f"{path} contains an invalid Unicode surrogate"
        return None
    if isinstance(value, int):
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return f"{path} contains a non-finite number"
        return None

    if isinstance(value, list):
        identity = id(value)
        if identity in active_containers:
            return f"{path} contains a reference cycle"
        active_containers.add(identity)
        try:
            for index, item in enumerate(value):
                error = _json_compatibility_error(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                if error is not None:
                    return error
        finally:
            active_containers.remove(identity)
        return None

    if isinstance(value, dict):
        identity = id(value)
        if identity in active_containers:
            return f"{path} contains a reference cycle"
        active_containers.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    return f"{path} contains a non-string object key ({type(key).__name__})"
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError:
                    return f"{path} contains an object key with an invalid Unicode surrogate"
                error = _json_compatibility_error(
                    item,
                    path=f"{path}[{key!r}]",
                    active_containers=active_containers,
                )
                if error is not None:
                    return error
        finally:
            active_containers.remove(identity)
        return None

    return f"{path} contains a non-JSON value ({type(value).__name__})"


def _chat_text_response(request: Mapping[str, Any], text: str, usage: TokenUsage) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-rlm-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(request.get("model") or "rlm"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "refusal": None},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
    }


def _responses_text_response(
    request: Mapping[str, Any], text: str, usage: TokenUsage
) -> dict[str, Any]:
    response_id = f"resp_rlm_{uuid.uuid4().hex}"
    message_id = f"msg_rlm_{uuid.uuid4().hex}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": copy.deepcopy(request.get("instructions")),
        "max_output_tokens": request.get("max_output_tokens"),
        "model": str(request.get("model") or "rlm"),
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": text,
                    }
                ],
            }
        ],
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", True)),
        "previous_response_id": request.get("previous_response_id"),
        "reasoning": copy.deepcopy(request.get("reasoning") or {}),
        "store": bool(request.get("store", True)),
        "temperature": request.get("temperature", 1.0),
        "text": copy.deepcopy(request.get("text") or {"format": {"type": "text"}}),
        "tool_choice": copy.deepcopy(request.get("tool_choice") or "auto"),
        "tools": copy.deepcopy(request.get("tools") or []),
        "top_logprobs": request.get("top_logprobs", 0),
        "top_p": request.get("top_p", 1.0),
        "truncation": request.get("truncation", "disabled"),
        "usage": {
            "input_tokens": usage.input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.total_tokens,
        },
    }
