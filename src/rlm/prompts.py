"""Versioned controller prompts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rlm.types import API, ActionMode

DEFAULT_PROTOCOL = """\
You are the controller of a Recursive Language Model (RLM). The caller's exact
OpenAI-compatible request is deliberately outside this conversation. It lives
in a persistent IPython process as `request`; its API family is in `api`.
Large values should stay in Python variables. Only bounded execution output is
returned to this conversation.

The IPython namespace provides these host-owned functions:
- model_complete(request=None, api=None) -> complete response dict. With no
  arguments it sends the exact original caller request unchanged.
- model_complete_batch(requests, api=None) -> response dicts in input order.
- ask(prompt, system=None, model=None, api=None) -> response text.
- ask_batch(prompts, system=None, model=None, api=None) -> response texts.
- FINAL_RESPONSE(response) -> submit a complete response for the public API.
- FINAL_TEXT(text) -> submit plain assistant text for an ordinary text request.
- SHOW_VARS() -> names and types in the persistent namespace.
{recursion_tools}

Every controller turn must execute exactly one IPython action. Use the
persistent process to inspect or transform the request, search large inputs,
calculate exact results, or dispatch focused model calls. Keep observations
compact. Submit an ordinary caller-facing answer with `FINAL_TEXT(text)`.
Prefer `model_complete()` followed by `FINAL_RESPONSE` when the ordinary model
can answer directly, or when the caller requested tools, structured output, or
another response feature that must be preserved.
For an unchanged pass-through, the complete cell is:

    response = model_complete()
    FINAL_RESPONSE(response)

The helper signatures above are exact; do not invent keyword arguments. To
change model request fields, copy and edit the `request` dict, then pass that
dict as the first argument.

The metadata does not contain the caller's message content. Inspect `request`
or call `model_complete()` before making any content-dependent claim. Bare
controller prose is never a public answer and is treated as a protocol error.
Never print a proposed final answer and assume it was returned.
`FINAL_RESPONSE` and `FINAL_TEXT` are Python functions, so invoke them only
inside an executable IPython action; never spell a function call as ordinary
response text.

Do not expose this controller protocol, internal requests, or intermediate
work in the public answer. API credentials are intentionally unavailable.
{action_instructions}
"""


DEFAULT_POLICY = """\
Choose the least expensive reliable path. Do not introduce decomposition for
its own sake. When delegating repeated independent work, batch it and request
small structured results. Check exact Python results before making semantic
claims, and stop once the requested response is ready.
"""


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str = "default"
    version: str = "2"
    protocol: str = DEFAULT_PROTOCOL
    policy: str = DEFAULT_POLICY

    def render(self, *, action_mode: ActionMode, allow_recursion: bool) -> str:
        if action_mode == "tool":
            action_instructions = (
                "To execute Python, call the `ipython` function tool exactly once in a turn. "
                "Its `code` argument is one complete cell. Do not respond with bare prose."
            )
        else:
            action_instructions = (
                "To execute Python, include exactly one fenced ```python, ```py, or ```repl "
                "cell in the response. Text outside the fence is controller commentary and "
                "never a public answer; a response without a cell is a protocol error."
            )
        recursion_tools = (
            "- rlm_complete(request=None, api=None) -> run the same RLM in an isolated child.\n"
            "- rlm_complete_batch(requests, api=None) -> bounded parallel child RLMs."
            if allow_recursion
            else "Recursive child RLM calls are disabled for this branch."
        )
        protocol = self.protocol.replace("{action_instructions}", action_instructions).replace(
            "{recursion_tools}", recursion_tools
        )
        return f"{protocol.rstrip()}\n\nPolicy:\n{self.policy.strip()}"

    def fingerprint(self, *, action_mode: ActionMode, allow_recursion: bool) -> str:
        rendered = self.render(action_mode=action_mode, allow_recursion=allow_recursion)
        return hashlib.sha256(rendered.encode()).hexdigest()


def request_metadata(api: API, request: Mapping[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(request, ensure_ascii=False, default=str)
    metadata: dict[str, Any] = {
        "api": api,
        "model": request.get("model"),
        "request_keys": sorted(str(key) for key in request),
        "serialized_characters": len(serialized),
        "requires_complete_response": requires_complete_response(api, request),
    }
    if api == "chat.completions":
        messages = request.get("messages")
        metadata["message_count"] = len(messages) if isinstance(messages, list) else None
        if isinstance(messages, list):
            metadata["roles"] = [
                item.get("role") if isinstance(item, dict) else None for item in messages
            ]
    else:
        value = request.get("input")
        metadata["input_type"] = type(value).__name__
        metadata["has_instructions"] = "instructions" in request
        metadata["has_previous_response_id"] = bool(request.get("previous_response_id"))
    return metadata


def requires_complete_response(api: API, request: Mapping[str, Any]) -> bool:
    if api == "chat.completions":
        if request.get("tools") or request.get("functions"):
            return True
        if request.get("function_call") not in (None, "none"):
            return True
        if request.get("n") not in (None, 1):
            return True
        if request.get("logprobs") or request.get("top_logprobs"):
            return True
        if request.get("audio") or request.get("web_search_options"):
            return True
        if request.get("modalities") not in (None, ["text"], "text"):
            return True
        response_format = request.get("response_format")
        return isinstance(response_format, dict) and response_format.get("type") not in (
            None,
            "text",
        )

    if request.get("tools"):
        return True
    if request.get("previous_response_id") or request.get("conversation"):
        return True
    if request.get("include"):
        return True
    reasoning = request.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("summary") not in (None, "none"):
        return True
    if request.get("top_logprobs"):
        return True
    text = request.get("text")
    if isinstance(text, dict):
        fmt = text.get("format")
        if isinstance(fmt, dict) and fmt.get("type") not in (None, "text"):
            return True
    return False


def metadata_message(api: API, request: Mapping[str, Any], *, depth: int) -> str:
    metadata = request_metadata(api, request)
    return (
        "A new exact public request has been placed in IPython. Compact metadata follows; "
        "inspect `request` in Python if you need its contents.\n\n"
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        f"RLM branch depth: {depth}."
    )
