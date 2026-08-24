"""Responses wire validation and safe synthetic text construction."""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Mapping
from typing import Any

from rlm.json import json_compatibility_error
from rlm.types import TokenUsage

RESPONSE_STATUSES = frozenset(
    {"cancelled", "completed", "failed", "in_progress", "incomplete", "queued"}
)
OUTPUT_ITEM_STATUSES = frozenset({"completed", "in_progress", "incomplete"})

# This deliberately accepts only the fields a minimal response can represent.
SYNTHETIC_TEXT_REQUEST_FIELDS = frozenset(
    {"model", "input", "instructions", "temperature", "top_p", "max_output_tokens", "store"}
)


def requires_complete_response(request: Mapping[str, Any]) -> bool:
    """Whether a request cannot be represented by a minimal text response."""

    return any(name not in SYNTHETIC_TEXT_REQUEST_FIELDS for name in request)


def text_response(request: Mapping[str, Any], text: str, *, usage: TokenUsage) -> dict[str, Any]:
    """Build a checked minimal completed Responses object for plain text."""

    response: dict[str, Any] = {
        "id": f"resp_rlm_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": request["model"],
        "output": [
            {
                "id": f"msg_rlm_{uuid.uuid4().hex}",
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
        "error": None,
        "usage": None if usage.unreported_calls else _usage_payload(usage),
    }
    for name in ("instructions", "max_output_tokens", "store", "temperature", "top_p"):
        if name in request:
            response[name] = copy.deepcopy(request[name])
    error = validate_terminal_response(response)
    if error is not None:  # pragma: no cover - internal contract invariant
        raise AssertionError(f"synthetic response violated Responses contract: {error}")
    return response


def validate_response_envelope(value: Any) -> str | None:
    """Validate a complete Responses object without imposing terminal state."""

    if not isinstance(value, dict):
        return f"Responses response requires an object, got {type(value).__name__}"
    json_error = json_compatibility_error(value)
    if json_error is not None:
        return f"Responses response requires strict JSON: {json_error}"
    if not _nonempty_string(value.get("id")):
        return "Responses response requires a non-empty string id"
    if value.get("object") != "response":
        return "Responses response object must equal 'response'"
    if not _nonempty_string(value.get("model")):
        return "Responses response requires a non-empty string model"
    status = value.get("status")
    if not isinstance(status, str) or status not in RESPONSE_STATUSES:
        return "Responses response has an unrecognized status"
    if status == "failed":
        error = value.get("error")
        if not isinstance(error, dict) or not _nonempty_string(error.get("message")):
            return "failed Responses response requires an error with a non-empty message"
        if "code" in error and not isinstance(error["code"], str):
            return "failed Responses response error code must be a string"
    output = value.get("output")
    if not isinstance(output, list):
        return "Responses response must contain an output list"
    for index, item in enumerate(output):
        error = _validate_output_item(item)
        if error is not None:
            return f"Responses output[{index}] {error}"
    return None


def validate_terminal_response(value: Any) -> str | None:
    """Validate a completed response usable as a FINAL_RESPONSE submission."""

    error = validate_response_envelope(value)
    if error is not None:
        return error
    assert isinstance(value, dict)
    if value["status"] != "completed":
        return f"Responses response has nonterminal status {value['status']!r}"
    if value.get("error") is not None:
        return "completed Responses response must not contain an error"
    output = value["output"]
    assert isinstance(output, list)
    if not output or not any(
        isinstance(item, dict) and item.get("type") != "reasoning" for item in output
    ):
        return "Responses response contains no terminal output"
    return None


def response_output_text(response: Mapping[str, Any]) -> str:
    """Return concatenated assistant text without changing the Responses value."""

    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct
    texts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, Mapping)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])
    return "".join(texts)


def _validate_output_item(value: Any) -> str | None:
    if not isinstance(value, dict):
        return "item must be an object"
    if not _nonempty_string(value.get("id")):
        return "item requires a non-empty string id"
    item_type = value.get("type")
    if not _nonempty_string(item_type):
        return "item requires a non-empty string type"
    if "status" in value and (
        not isinstance(value["status"], str) or value["status"] not in OUTPUT_ITEM_STATUSES
    ):
        return "item has an unrecognized status"
    if item_type == "message":
        return _validate_message_item(value)
    if item_type not in {
        "reasoning",
        "function_call",
        "computer_call",
        "file_search_call",
    } and not set(value).difference({"id", "type", "status"}):
        return "unknown item requires a non-empty payload"
    return None


def _validate_message_item(value: Mapping[str, Any]) -> str | None:
    if value.get("role") != "assistant":
        return "message item role must be 'assistant'"
    if not isinstance(value.get("status"), str) or value["status"] not in OUTPUT_ITEM_STATUSES:
        return "message item has an unrecognized status"
    content = value.get("content")
    if not isinstance(content, list):
        return "message item content must be a list"
    for index, part in enumerate(content):
        if not isinstance(part, dict) or not _nonempty_string(part.get("type")):
            return f"message content[{index}] must be a typed object"
        if part["type"] == "output_text" and not isinstance(part.get("text"), str):
            return f"message content[{index}] output_text requires a string text"
        if part["type"] == "refusal" and not isinstance(part.get("refusal"), str):
            return f"message content[{index}] refusal requires a string refusal"
    return None


def _usage_payload(usage: TokenUsage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.total_tokens,
    }


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)
