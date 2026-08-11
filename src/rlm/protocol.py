"""Controller conversation codecs and action parsing."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from rlm.types import API, ActionMode

IPYTHON_TOOL_NAME = "ipython"
IPYTHON_DESCRIPTION = (
    "Execute one cell in the branch's persistent IPython process. Variables and imports "
    "persist. Put large values in variables and print only compact observations."
)
IPYTHON_PARAMETERS = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "A complete Python/IPython cell to execute.",
        }
    },
    "required": ["code"],
    "additionalProperties": False,
}
CHAT_IPYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": IPYTHON_TOOL_NAME,
        "description": IPYTHON_DESCRIPTION,
        "parameters": IPYTHON_PARAMETERS,
    },
}
RESPONSES_IPYTHON_TOOL = {
    "type": "function",
    "name": IPYTHON_TOOL_NAME,
    "description": IPYTHON_DESCRIPTION,
    "parameters": IPYTHON_PARAMETERS,
    "strict": True,
}

_CODE_FENCE = re.compile(
    r"```(?:python|py|repl)[ \t]*\r?\n(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_BARE_FINAL_CALL = re.compile(r"^\s*(?:FINAL_RESPONSE|FINAL_TEXT)\s*\(")


ActionKind = Literal["execute", "protocol_error"]


@dataclass(slots=True)
class ControllerAction:
    kind: ActionKind
    code: str | None = None
    text: str | None = None
    call_id: str | None = None
    error: str | None = None


class ControllerConversation:
    """One private controller conversation in Chat or Responses form."""

    def __init__(self, api: API, *, system: str, first_user: str) -> None:
        self.api = api
        self.system = system
        if api == "chat.completions":
            self.items: list[dict[str, Any]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": first_user},
            ]
        else:
            self.items = [{"role": "user", "content": first_user}]

    def request(
        self,
        *,
        model: str,
        options: Mapping[str, Any],
        action_mode: ActionMode,
    ) -> dict[str, Any]:
        body = copy.deepcopy(dict(options))
        body["model"] = model
        if self.api == "chat.completions":
            body["messages"] = copy.deepcopy(self.items)
            if action_mode == "tool":
                body["tools"] = [copy.deepcopy(CHAT_IPYTHON_TOOL)]
                body.setdefault("parallel_tool_calls", False)
        else:
            body["instructions"] = self.system
            body["input"] = copy.deepcopy(self.items)
            if action_mode == "tool":
                body["tools"] = [copy.deepcopy(RESPONSES_IPYTHON_TOOL)]
                body.setdefault("parallel_tool_calls", False)
        return body

    def parse(self, response: Mapping[str, Any], *, action_mode: ActionMode) -> ControllerAction:
        if self.api == "chat.completions":
            return _parse_chat(response, action_mode=action_mode)
        return _parse_responses(response, action_mode=action_mode)

    def append_response(self, response: Mapping[str, Any]) -> None:
        if self.api == "chat.completions":
            message = _chat_message(response)
            if message is not None:
                self.items.append(copy.deepcopy(message))
            return
        output = response.get("output")
        if isinstance(output, list):
            self.items.extend(copy.deepcopy(item) for item in output if isinstance(item, dict))

    def append_observation(self, action: ControllerAction, observation: str) -> None:
        if self.api == "chat.completions" and action.call_id:
            self.items.append(
                {"role": "tool", "tool_call_id": action.call_id, "content": observation}
            )
        elif self.api == "responses" and action.call_id:
            self.items.append(
                {
                    "type": "function_call_output",
                    "call_id": action.call_id,
                    "output": observation,
                }
            )
        else:
            self.items.append({"role": "user", "content": _observation_message(observation)})

    def append_feedback(self, feedback: str) -> None:
        self.items.append({"role": "user", "content": feedback})


def extract_text(api: API, response: Mapping[str, Any]) -> str:
    if api == "chat.completions":
        message = _chat_message(response)
        return _content_text(message.get("content") if message else None)

    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct
    texts: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text") and isinstance(
                    part.get("text"), str
                ):
                    texts.append(part["text"])
    return "".join(texts)


def _parse_chat(response: Mapping[str, Any], *, action_mode: ActionMode) -> ControllerAction:
    message = _chat_message(response)
    if message is None:
        return ControllerAction(kind="protocol_error", error="missing choices[0].message")
    if action_mode == "tool":
        tool_calls = message.get("tool_calls")
        if tool_calls:
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                count = len(tool_calls) if isinstance(tool_calls, list) else "invalid"
                return ControllerAction(
                    kind="protocol_error", error=f"expected one ipython tool call, got {count}"
                )
            return _parse_chat_tool_call(tool_calls[0])
    return _parse_text_action(_content_text(message.get("content")), action_mode=action_mode)


def _parse_chat_tool_call(value: Any) -> ControllerAction:
    if not isinstance(value, dict):
        return ControllerAction(kind="protocol_error", error="tool call is not an object")
    function = value.get("function")
    if not isinstance(function, dict) or function.get("name") != IPYTHON_TOOL_NAME:
        return ControllerAction(kind="protocol_error", error="controller called an unknown tool")
    arguments = _parse_arguments(function.get("arguments"))
    if isinstance(arguments, str):
        return ControllerAction(kind="protocol_error", error=arguments)
    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        return ControllerAction(kind="protocol_error", error="ipython.code must be non-empty")
    return ControllerAction(
        kind="execute", code=code, call_id=str(value.get("id") or "ipython-call")
    )


def _parse_responses(response: Mapping[str, Any], *, action_mode: ActionMode) -> ControllerAction:
    if action_mode == "tool":
        calls = (
            [
                item
                for item in response.get("output", [])
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]
            if isinstance(response.get("output"), list)
            else []
        )
        if calls:
            if len(calls) != 1:
                return ControllerAction(
                    kind="protocol_error", error=f"expected one ipython tool call, got {len(calls)}"
                )
            call = calls[0]
            if call.get("name") != IPYTHON_TOOL_NAME:
                return ControllerAction(
                    kind="protocol_error",
                    error="controller called an unknown tool",
                )
            arguments = _parse_arguments(call.get("arguments"))
            if isinstance(arguments, str):
                return ControllerAction(kind="protocol_error", error=arguments)
            code = arguments.get("code")
            if not isinstance(code, str) or not code.strip():
                return ControllerAction(
                    kind="protocol_error", error="ipython.code must be non-empty"
                )
            call_id = call.get("call_id") or call.get("id") or "ipython-call"
            return ControllerAction(kind="execute", code=code, call_id=str(call_id))
    return _parse_text_action(extract_text("responses", response), action_mode=action_mode)


def _parse_text_action(text: str, *, action_mode: ActionMode) -> ControllerAction:
    if action_mode == "code":
        blocks = [match.group("code").strip() for match in _CODE_FENCE.finditer(text)]
        if len(blocks) == 1:
            if not blocks[0]:
                return ControllerAction(
                    kind="protocol_error",
                    text=text,
                    error="the fenced IPython cell must be non-empty",
                )
            return ControllerAction(kind="execute", code=blocks[0])
        if len(blocks) > 1:
            return ControllerAction(
                kind="protocol_error",
                text=text,
                error=f"expected at most one Python cell, got {len(blocks)}",
            )
    if text.strip():
        if _BARE_FINAL_CALL.match(text):
            return ControllerAction(
                kind="protocol_error",
                text=text,
                error=(
                    "FINAL functions exist only inside IPython; execute the call in a Python action"
                ),
            )
        return ControllerAction(
            kind="protocol_error",
            text=text,
            error=(
                "controller prose is not a terminal action; execute FINAL_TEXT(...) or "
                "FINAL_RESPONSE(...) inside an IPython action"
            ),
        )
    return ControllerAction(kind="protocol_error", error="controller returned no action or text")


def _parse_arguments(value: Any) -> dict[str, Any] | str:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return "tool arguments must be a JSON object"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return f"invalid tool arguments: {exc.msg} at line {exc.lineno} column {exc.colno}"
    if not isinstance(parsed, dict):
        return f"tool arguments must decode to an object, got {type(parsed).__name__}"
    return parsed


def _chat_message(response: Mapping[str, Any]) -> dict[str, Any] | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    return message if isinstance(message, dict) else None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)


def _observation_message(observation: str) -> str:
    return f"<ipython_output>\n{observation}\n</ipython_output>\nContinue or give the final answer."
