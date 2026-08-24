"""The exact Responses conversation used by the private controller."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol

from rlm.errors import ProtocolError
from rlm.json import strict_json_dumps
from rlm.specs import ContextSpec


@dataclass(frozen=True, slots=True)
class ControllerRequestCore:
    """The controller request fields owned exclusively by the runtime."""

    model: str
    instructions: str
    input: list[dict[str, Any]]


PROTOCOL_OWNED_FIELDS = frozenset(field.name for field in fields(ControllerRequestCore))
UNSUPPORTED_CONTROLLER_FIELDS = frozenset(
    {"background", "stream", "modalities", "response_format", "text", "tools", "tool_choice"}
)


@dataclass(frozen=True, slots=True)
class ControllerAction:
    code: str


class ControllerProtocol(Protocol):
    """Internal boundary for the single controller-output grammar."""

    def parse(self, response: Mapping[str, Any]) -> ControllerAction: ...


class PythonCellControllerProtocol:
    """Parse the controller's one permitted executable action."""

    def parse(self, response: Mapping[str, Any]) -> ControllerAction:
        return ControllerAction(code=_python_cell(controller_output_text(response)))


_CONTROLLER_PROTOCOL = PythonCellControllerProtocol()


class ControllerConversation:
    """Build a compact private Responses transcript and parse Python actions."""

    def __init__(
        self,
        *,
        system: str,
        first_user: str,
        context: ContextSpec | None = None,
    ) -> None:
        self.system = system
        self.context = ContextSpec() if context is None else context
        self.items: list[dict[str, Any]] = [{"role": "user", "content": first_user}]

    def request(self, *, model: str, options: Mapping[str, Any]) -> dict[str, Any]:
        body = copy.deepcopy(dict(options))
        core = ControllerRequestCore(
            model=model,
            instructions=self.system,
            input=copy.deepcopy(self.items),
        )
        body.update({field.name: getattr(core, field.name) for field in fields(core)})
        return body

    def parse(self, response: Mapping[str, Any]) -> ControllerAction:
        return _CONTROLLER_PROTOCOL.parse(response)

    def append_response(self, response: Mapping[str, Any]) -> None:
        """Keep the latest permitted controller reasoning and message items."""

        output = response.get("output")
        retained: list[dict[str, Any]] = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping):
                    continue
                item_type = item.get("type")
                if item_type == "message" or (
                    item_type == "reasoning" and self.context.preserve_reasoning
                ):
                    retained.append(copy.deepcopy(dict(item)))
        self.items = [self.items[0], *retained]

    def append_observation(self, observation: Mapping[str, Any]) -> None:
        """Serialize the canonical observation once for the next model turn."""

        serialized = strict_json_dumps(dict(observation), separators=(",", ":"))
        self.items.append(
            {
                "role": "user",
                "content": serialized,
            }
        )


def controller_output_text(response: Mapping[str, Any]) -> str:
    """Extract the sole executable text part from a structured Responses output."""

    output = response.get("output")
    if not isinstance(output, list):
        raise ProtocolError("controller response output must be a list")
    if any(
        not isinstance(item, Mapping) or item.get("type") not in {"reasoning", "message"}
        for item in output
    ):
        raise ProtocolError("controller output may contain only reasoning and one message")
    messages = [item for item in output if item.get("type") == "message"]
    if len(messages) != 1:
        raise ProtocolError("controller must return exactly one assistant message")
    message = messages[0]
    if message.get("role") != "assistant":
        raise ProtocolError("controller message must have assistant role")
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        raise ProtocolError("controller message must contain exactly one output_text part")
    part = content[0]
    text = part.get("text") if isinstance(part, Mapping) else None
    if (
        not isinstance(part, Mapping)
        or part.get("type") != "output_text"
        or not isinstance(text, str)
    ):
        raise ProtocolError("controller message must contain exactly one output_text part")
    return text


def _python_cell(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) < 3 or lines[0] != "```python" or lines[-1] != "```":
        raise ProtocolError("controller must return exactly one fenced python cell")
    if any(line.strip().startswith("```") for line in lines[1:-1]):
        raise ProtocolError("controller returned more than one fenced cell")
    code = "\n".join(lines[1:-1]).strip()
    if not code:
        raise ProtocolError("controller returned an empty Python cell")
    return code


def extract_text(response: Mapping[str, Any]) -> str:
    """Return concatenated assistant text from a Responses object.

    This permissive public-response helper is intentionally not part of the
    controller protocol; executable controller text is extracted only above.
    """

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
                and part.get("type") in ("output_text", "text")
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])
    return "".join(texts)
