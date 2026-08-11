from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rlm import RLM, CodexAgentBackend, InvalidRequestError, RLMConfig, UpstreamError
from rlm.codex_backend import CODEX_PROVIDER_EXTENSION, CommandResult
from rlm.config import DebugConfig

from .fakes import ScriptedBackend


class RecordingRunner:
    def __init__(self, codex_result: CommandResult | BaseException) -> None:
        self.codex_result = codex_result
        self.calls: list[dict[str, Any]] = []
        self.workspace: Path | None = None

    def __call__(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None,
        cwd: Path,
        timeout: float,
    ) -> CommandResult:
        self.calls.append(
            {
                "argv": tuple(argv),
                "input_text": input_text,
                "cwd": cwd,
                "timeout": timeout,
            }
        )
        assert cwd.is_dir()
        if tuple(argv[:2]) == ("git", "init"):
            (cwd / ".git").mkdir()
            self.workspace = cwd
            return CommandResult(0)
        assert (cwd / ".git").is_dir()
        if isinstance(self.codex_result, BaseException):
            raise self.codex_result
        return self.codex_result


def event_stream(message: str = "```python\nprint(request['model'])\n```") -> str:
    events = [
        {"type": "thread.started", "thread_id": "thread-test"},
        {
            "type": "item.completed",
            "item": {"id": "item-reason", "type": "reasoning", "text": "summary"},
        },
        {
            "type": "item.completed",
            "item": {"id": "item-message", "type": "agent_message", "text": message},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 31, "cached_input_tokens": 7, "output_tokens": 11},
        },
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


@pytest.mark.parametrize("api", ["chat.completions", "responses"])
def test_codex_agent_backend_wraps_final_message_and_retains_events(api: str) -> None:
    runner = RecordingRunner(CommandResult(0, event_stream(), "observable warning\n"))
    backend = CodexAgentBackend(model="gpt-test", reasoning_effort="max", timeout=42, runner=runner)
    request = (
        {
            "model": "request-model",
            "messages": [
                {"role": "system", "content": "controller protocol"},
                {"role": "user", "content": "keep this exact: λ"},
            ],
        }
        if api == "chat.completions"
        else {
            "model": "request-model",
            "instructions": "controller protocol",
            "input": [{"role": "user", "content": "keep this exact: λ"}],
        }
    )

    response = backend.complete(api, request)  # type: ignore[arg-type]

    if api == "chat.completions":
        text = response["choices"][0]["message"]["content"]
        assert response["usage"]["prompt_tokens"] == 31
        assert response["usage"]["completion_tokens"] == 11
    else:
        text = response["output"][0]["content"][0]["text"]
        assert response["usage"]["input_tokens"] == 31
        assert response["usage"]["output_tokens"] == 11
    assert text == "```python\nprint(request['model'])\n```"
    assert response["model"] == "gpt-test"
    extension = response[CODEX_PROVIDER_EXTENSION]
    assert extension["semantics"] == "codex_agent_trajectory"
    assert extension["events"][1]["item"]["type"] == "reasoning"
    assert extension["stderr"] == "observable warning\n"

    assert len(runner.calls) == 2
    codex_call = runner.calls[1]
    argv = codex_call["argv"]
    assert argv[:2] == ("codex", "exec")
    assert "--json" in argv
    assert "--ephemeral" in argv
    assert "--ignore-user-config" in argv
    assert "--ignore-rules" in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("--model") + 1] == "gpt-test"
    assert argv[argv.index("--config") + 1] == 'model_reasoning_effort="max"'
    assert argv[-1] == "-"
    assert "keep this exact: λ" in codex_call["input_text"]
    workspace = runner.workspace
    assert workspace is not None
    assert not workspace.exists()


def test_codex_agent_backend_uses_last_agent_message_and_usage() -> None:
    events = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "draft"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "final"}},
        {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 4}},
    ]
    stdout = "\n".join(json.dumps(event) for event in events)
    backend = CodexAgentBackend(runner=RecordingRunner(CommandResult(0, stdout)))

    response = backend.complete(
        "chat.completions",
        {"model": "controller", "messages": [{"role": "user", "content": "go"}]},
    )

    assert response["choices"][0]["message"]["content"] == "final"
    assert response["usage"]["prompt_tokens"] == 3
    assert response["usage"]["completion_tokens"] == 4


def test_codex_agent_backend_validates_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        CodexAgentBackend(reasoning_effort='max" --sandbox danger-full-access')


@pytest.mark.parametrize(
    ("api", "body", "parameter"),
    [
        (
            "chat.completions",
            {"model": "x", "messages": [], "tools": [{"type": "function"}]},
            "tools",
        ),
        (
            "responses",
            {"model": "x", "input": [{"type": "function_call_output"}]},
            "input",
        ),
    ],
)
def test_codex_agent_backend_rejects_native_tool_semantics(
    api: str, body: dict[str, Any], parameter: str
) -> None:
    runner = RecordingRunner(CommandResult(0, event_stream()))
    backend = CodexAgentBackend(runner=runner)

    with pytest.raises(InvalidRequestError) as caught:
        backend.complete(api, body)  # type: ignore[arg-type]

    assert caught.value.code == "codex_native_tools_not_supported"
    assert caught.value.param == parameter
    assert runner.calls == []


@pytest.mark.parametrize(
    "result",
    [
        CommandResult(9, '{"type":"error","message":"agent failed"}\n'),
        subprocess.TimeoutExpired(("codex", "exec"), timeout=2),
    ],
)
def test_codex_agent_backend_surfaces_process_failures(
    result: CommandResult | BaseException,
) -> None:
    backend = CodexAgentBackend(timeout=2, runner=RecordingRunner(result))

    with pytest.raises(UpstreamError) as caught:
        backend.complete(
            "chat.completions",
            {"model": "controller", "messages": [{"role": "user", "content": "go"}]},
        )

    assert caught.value.endpoint == "codex exec"
    if isinstance(result, subprocess.TimeoutExpired):
        assert caught.value.status_code == 504
    else:
        assert "agent failed" in str(caught.value)


def test_codex_agent_backend_rejects_invalid_jsonl() -> None:
    backend = CodexAgentBackend(runner=RecordingRunner(CommandResult(0, "not-json\n")))

    with pytest.raises(UpstreamError, match="invalid JSONL"):
        backend.complete(
            "chat.completions",
            {"model": "controller", "messages": [{"role": "user", "content": "go"}]},
        )


def test_rlm_enforces_codex_controller_only_code_mode() -> None:
    codex = CodexAgentBackend(runner=RecordingRunner(CommandResult(0, event_stream())))
    ordinary = RecordingRunner(CommandResult(0, event_stream()))

    with pytest.raises(ValueError, match="public backend"):
        RLM(codex, config=RLMConfig(action_mode="code"))

    with pytest.raises(ValueError, match="requires action_mode='code'"):
        RLM(ordinary, controller_backend=codex)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="worker backend"):
        RLM(
            ordinary,  # type: ignore[arg-type]
            worker_backend=codex,
            config=RLMConfig(action_mode="code"),
        )


def test_codex_observable_events_reach_rlm_debug_trace(tmp_path: Path) -> None:
    runner = RecordingRunner(CommandResult(0, event_stream("```python\nFINAL_TEXT('hello')\n```")))
    model = RLM(
        ScriptedBackend(),
        controller_backend=CodexAgentBackend(model="codex-test", runner=runner),
        config=RLMConfig(
            controller_model="codex-test",
            action_mode="code",
            debug=DebugConfig(enabled=True, directory=tmp_path),
        ),
    )

    result = model.run(
        "chat.completions",
        {"model": "public-test", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert result.response["choices"][0]["message"]["content"] == "hello"
    assert result.trace_directory is not None
    events = [
        json.loads(line)
        for line in (result.trace_directory / "trace.jsonl").read_text().splitlines()
    ]
    controller_responses = [
        event["payload"]["response"]
        for event in events
        if event["type"] == "model.response" and event["payload"]["role"] == "controller"
    ]
    assert (
        controller_responses[0][CODEX_PROVIDER_EXTENSION]["events"][1]["item"]["type"]
        == "reasoning"
    )
