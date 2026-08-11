"""A distinct, local Codex CLI agent backend for controller evaluation.

This adapter deliberately does not claim that ``codex exec`` is an
OpenAI-compatible model endpoint.  It translates one controller request into
an agent instruction, then wraps the agent's final observable message in the
requested response family so it can be evaluated as an RLM controller.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rlm.errors import InvalidRequestError, UpstreamError
from rlm.response import text_response
from rlm.types import API, TokenUsage

CODEX_PROVIDER_EXTENSION = "x_rlm_codex_agent"
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Small subprocess result type used by the injectable command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    """Run one argv-only command without a shell."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None,
        cwd: Path,
        timeout: float,
    ) -> CommandResult: ...


def subprocess_runner(
    argv: Sequence[str],
    *,
    input_text: str | None,
    cwd: Path,
    timeout: float,
) -> CommandResult:
    """Default argv-only subprocess runner.

    The process inherits its ordinary environment so the installed CLI can use
    its supported login state.  This adapter does not inspect, extract, copy,
    or reconstruct any authentication value.
    """

    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class CodexAgentBackend:
    """Use authenticated local ``codex exec`` as a private RLM controller.

    The backend is intended only as ``RLM(..., controller_backend=...)`` with
    ``RLMConfig(action_mode="code")``.  Codex is an agent with its own system
    behavior and observable event stream, so this must not be presented as a
    raw GPT endpoint, public model backend, or worker model.

    The adapter never reads, copies, or injects authentication.  ``codex exec``
    discovers its existing login in the ordinary CLI-supported way.
    """

    controller_only = True
    required_action_mode = "code"

    def __init__(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        timeout: float = 600.0,
        command: Sequence[str] = ("codex", "exec"),
        runner: CommandRunner | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            allowed = ", ".join(sorted(REASONING_EFFORTS))
            raise ValueError(f"reasoning_effort must be one of: {allowed}")
        if (
            isinstance(command, (str, bytes))
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise ValueError("command must be a non-empty argv sequence")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.command = tuple(command)
        self.runner = runner or subprocess_runner

    def complete(self, api: API, request: Mapping[str, Any]) -> dict[str, Any]:
        body = dict(request)
        self._validate_request(api, body)
        prompt = _controller_prompt(api, body)

        with tempfile.TemporaryDirectory(prefix="rlm-codex-agent-") as temporary:
            workspace = Path(temporary)
            self._run_git_init(workspace)
            command = self._command(workspace, request_model=body.get("model"))
            result = self._invoke(command, input_text=prompt, cwd=workspace, operation="codex exec")

        events = _parse_events(result.stdout)
        if result.returncode != 0:
            message = _event_error(events) or result.stderr.strip()
            if not message:
                message = f"codex exec exited with status {result.returncode}"
            raise _codex_error(message)

        text = _agent_message(events)
        if text is None:
            raise _codex_error("codex exec completed without an agent_message event")

        usage = _usage(events)
        response_request = dict(body)
        response_request["model"] = self.model or body.get("model") or "codex-agent"
        response = text_response(api, response_request, text, usage=usage)
        response[CODEX_PROVIDER_EXTENSION] = {
            "semantics": "codex_agent_trajectory",
            "events": events,
            "stderr": result.stderr,
        }
        return response

    def _validate_request(self, api: API, request: Mapping[str, Any]) -> None:
        if request.get("stream") is True:
            raise InvalidRequestError(
                "CodexAgentBackend supports only non-streaming controller requests",
                code="streaming_not_supported",
                param="stream",
            )
        if api == "responses" and request.get("background") is True:
            raise InvalidRequestError(
                "CodexAgentBackend does not support background controller requests",
                code="background_not_supported",
                param="background",
            )
        tool_param = _native_tool_parameter(api, request)
        if tool_param is not None:
            raise InvalidRequestError(
                "CodexAgentBackend cannot preserve native model tool-call semantics; "
                "configure the RLM controller with action_mode='code'",
                code="codex_native_tools_not_supported",
                param=tool_param,
            )

    def _run_git_init(self, workspace: Path) -> None:
        result = self._invoke(
            ("git", "init", "--quiet", "."),
            input_text=None,
            cwd=workspace,
            operation="git init",
            timeout=min(self.timeout, 30.0),
        )
        if result.returncode != 0:
            message = result.stderr.strip() or "could not initialize isolated Git workspace"
            raise _codex_error(message)

    def _command(self, workspace: Path, *, request_model: Any) -> tuple[str, ...]:
        command = [
            *self.command,
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--cd",
            str(workspace),
        ]
        model = self.model or (request_model if isinstance(request_model, str) else None)
        if model:
            command.extend(("--model", model))
        if self.reasoning_effort is not None:
            command.extend(
                (
                    "--config",
                    f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
                )
            )
        command.append("-")
        return tuple(command)

    def _invoke(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None,
        cwd: Path,
        operation: str,
        timeout: float | None = None,
    ) -> CommandResult:
        try:
            result = self.runner(
                tuple(argv),
                input_text=input_text,
                cwd=cwd,
                timeout=self.timeout if timeout is None else timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise _codex_error(f"{operation} exceeded its timeout", status_code=504) from exc
        except OSError as exc:
            raise _codex_error(f"could not start {operation}: {exc}") from exc
        if not isinstance(result, CommandResult):
            raise TypeError("Codex command runner must return CommandResult")
        return result


def _controller_prompt(api: API, request: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(request, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as exc:
        raise InvalidRequestError(
            f"controller request is not JSON serializable: {exc}",
            param="request",
        ) from exc
    return (
        "You are being evaluated as the private controller model inside an RLM runtime.\n"
        "The JSON below is one complete OpenAI-style model request. Follow its system and "
        "conversation instructions in role order and produce only the assistant message content "
        "that should answer it. Do not add an API response envelope. The request's RLM protocol "
        "may require exactly one fenced Python cell; preserve that code protocol literally. "
        "Do not inspect or change the empty workspace.\n\n"
        f"API family: {api}\n"
        "<controller_request_json>\n"
        f"{encoded}\n"
        "</controller_request_json>"
    )


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _codex_error(
                f"codex exec emitted invalid JSONL on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(event, dict):
            raise _codex_error(
                f"codex exec emitted {type(event).__name__} on JSONL line {line_number}; "
                "expected an object"
            )
        events.append(event)
    return events


def _agent_message(events: Sequence[Mapping[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if (
            isinstance(item, Mapping)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
        ):
            return item["text"]
    return None


def _usage(events: Sequence[Mapping[str, Any]]) -> TokenUsage:
    for event in reversed(events):
        if event.get("type") != "turn.completed" or not isinstance(event.get("usage"), Mapping):
            continue
        raw = event["usage"]
        return TokenUsage(
            input_tokens=_nonnegative_int(raw.get("input_tokens")),
            output_tokens=_nonnegative_int(raw.get("output_tokens")),
            calls=1,
        )
    return TokenUsage(calls=1)


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _event_error(events: Sequence[Mapping[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("type") != "error":
            continue
        for key in ("message", "error"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _native_tool_parameter(api: API, request: Mapping[str, Any]) -> str | None:
    for parameter in ("tools", "functions"):
        value = request.get(parameter)
        if value not in (None, []):
            return parameter
    for parameter in ("tool_choice", "function_call"):
        value = request.get(parameter)
        if value not in (None, "none"):
            return parameter

    history_name = "messages" if api == "chat.completions" else "input"
    history = request.get(history_name)
    if not isinstance(history, list):
        return None
    for item in history:
        if not isinstance(item, Mapping):
            continue
        if item.get("role") == "tool" or item.get("type") in {
            "function_call",
            "function_call_output",
        }:
            return history_name
        if item.get("tool_calls") or item.get("function_call"):
            return history_name
    return None


def _codex_error(message: str, *, status_code: int = 502) -> UpstreamError:
    return UpstreamError(status_code, {"message": message}, endpoint="codex exec")
