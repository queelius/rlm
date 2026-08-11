from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import pytest

from rlm.config import DebugConfig, RLMConfig
from rlm.engine import RLM
from rlm.errors import InvalidRequestError, LimitExceededError, RLMError
from rlm.prompts import Prompt, metadata_message
from rlm.protocol import CHAT_IPYTHON_TOOL, extract_text
from rlm.types import API
from tests.fakes import (
    FunctionBackend,
    ScriptedBackend,
    chat_text,
    chat_tool,
    responses_text,
    responses_tool,
)


def _chat_request() -> dict[str, Any]:
    return {
        "model": "public-model",
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
        "temperature": 0,
    }


def _multi_turn_request() -> dict[str, Any]:
    return {
        "model": "public-model",
        "messages": [
            {"role": "system", "content": "Be exact."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check the earlier result."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA==", "detail": "low"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-original",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"key":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-original", "content": "41"},
            {"role": "assistant", "content": "The earlier value was 41."},
            {"role": "user", "content": "Increment it."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "temperature": 0.25,
        "seed": 17,
        "provider_extension": {"nested": [1, {"preserve": True}]},
    }


@pytest.mark.parametrize("api", ["chat.completions", "responses"])
def test_explicit_final_text_terminates_plain_request(api: API) -> None:
    public = ScriptedBackend(name="public")
    controller = ScriptedBackend(
        [
            chat_tool(
                """FINAL_TEXT("I'm fine!")""",
                prompt_tokens=3,
                completion_tokens=2,
            )
        ],
        name="controller",
    )
    request = (
        _chat_request()
        if api == "chat.completions"
        else {"model": "public-model", "input": "Hi, how are you?", "temperature": 0}
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(controller_model="controller"),
    ).run(api, request)

    assert extract_text(api, result.response) == "I'm fine!"
    assert result.stop_reason == "final_text"
    assert result.turns == 1
    assert result.usage.to_dict() == {"input_tokens": 3, "output_tokens": 2, "calls": 1}
    assert public.calls == []
    controller.assert_exhausted()


@pytest.mark.parametrize("controller_api", ["chat.completions", "responses"])
def test_planning_prose_is_rejected_then_repaired_with_explicit_final(
    controller_api: API,
) -> None:
    planning = "I should inspect the request before answering."
    if controller_api == "chat.completions":
        planning_response = chat_text(planning)
        final_response = chat_tool('FINAL_TEXT("done")', call_id="call-final")
    else:
        planning_response = responses_text(planning)
        final_response = responses_tool('FINAL_TEXT("done")', call_id="call-final")
    controller = ScriptedBackend(
        [planning_response, final_response],
        name="controller",
    )
    public = ScriptedBackend(name="public")

    result = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(controller_model="controller", controller_api=controller_api),
    ).run("chat.completions", _chat_request())

    assert extract_text("chat.completions", result.response) == "done"
    assert result.stop_reason == "final_text"
    assert result.turns == 2
    history_key = "messages" if controller_api == "chat.completions" else "input"
    repair_history = controller.calls[1].request[history_key]
    if controller_api == "chat.completions":
        assert repair_history[-2] == {"role": "assistant", "content": planning}
    else:
        assert repair_history[-2] == planning_response["output"][0]
    assert repair_history[-1]["role"] == "user"
    assert "FINAL_TEXT" in repair_history[-1]["content"]
    assert public.calls == []
    controller.assert_exhausted()


def test_ipython_no_arg_model_complete_is_exact_identity_and_final_response() -> None:
    request = _multi_turn_request()
    untouched = copy.deepcopy(request)
    public_response = chat_text(
        "42",
        model="public-model",
        response_id="chatcmpl-public",
        prompt_tokens=23,
        completion_tokens=1,
    )
    code = "response = model_complete()\nFINAL_RESPONSE(response)"
    public = ScriptedBackend([public_response], name="public")
    controller = ScriptedBackend(
        [chat_tool(code, prompt_tokens=11, completion_tokens=5)],
        name="controller",
    )
    worker = ScriptedBackend(name="worker")

    result = RLM(
        public,
        controller_backend=controller,
        worker_backend=worker,
        config=RLMConfig(controller_model="controller"),
    ).run("chat.completions", request)

    assert result.response == public_response
    assert result.response is not public_response
    assert result.stop_reason == "final_response"
    assert result.turns == 1
    assert request == untouched
    assert len(public.calls) == 1
    assert public.calls[0].api == "chat.completions"
    assert public.calls[0].request == untouched
    assert public.calls[0].request is not request
    assert worker.calls == []
    public.assert_exhausted()
    controller.assert_exhausted()


def test_structured_request_rejects_final_text_then_repairs_with_identity_call() -> None:
    request = {
        "model": "public-model",
        "messages": [{"role": "user", "content": "Return a number."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "number"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    }
    public_response = chat_text('{"answer":42}', model="public-model")
    code = "response = model_complete()\nFINAL_RESPONSE(response)"
    public = ScriptedBackend([public_response], name="public")
    rejected = chat_tool(
        'FINAL_TEXT("The answer is 42.")',
        call_id="call-text-rejected",
    )
    controller = ScriptedBackend(
        [rejected, chat_tool(code, call_id="call-repair")],
        name="controller",
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(controller_model="controller"),
    ).run("chat.completions", request)

    assert result.response == public_response
    assert result.stop_reason == "final_response"
    assert result.turns == 2
    assert public.calls[0].request == request
    assert len(controller.calls) == 2
    repair_messages = controller.calls[1].request["messages"]
    assert repair_messages[-2] == rejected["choices"][0]["message"]
    assert repair_messages[-1]["role"] == "tool"
    assert repair_messages[-1]["tool_call_id"] == "call-text-rejected"
    assert "requires a complete response object" in repair_messages[-1]["content"]
    public.assert_exhausted()
    controller.assert_exhausted()


def test_bare_final_function_text_is_repaired_instead_of_leaking_to_caller() -> None:
    controller = ScriptedBackend(
        [
            chat_text('FINAL_TEXT("hello")'),
            chat_text('```python\nFINAL_TEXT("hello")\n```'),
        ],
        name="controller",
    )
    public = ScriptedBackend(name="public")

    result = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(controller_model="controller", action_mode="code"),
    ).run("chat.completions", _chat_request())

    assert extract_text("chat.completions", result.response) == "hello"
    assert result.stop_reason == "final_text"
    assert result.turns == 2
    repair_messages = controller.calls[1].request["messages"]
    assert repair_messages[-2] == {
        "role": "assistant",
        "content": 'FINAL_TEXT("hello")',
    }
    assert repair_messages[-1]["role"] == "user"
    assert "IPython" in repair_messages[-1]["content"]
    assert "Python action" in repair_messages[-1]["content"]
    assert public.calls == []
    controller.assert_exhausted()


def test_empty_fenced_cell_is_repaired_instead_of_executed() -> None:
    controller = ScriptedBackend(
        [
            chat_text("```python\n\n```"),
            chat_text('```python\nFINAL_TEXT("hello")\n```'),
        ],
        name="controller",
    )
    public = ScriptedBackend(name="public")

    result = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(controller_model="controller", action_mode="code"),
    ).run("chat.completions", _chat_request())

    assert extract_text("chat.completions", result.response) == "hello"
    assert result.stop_reason == "final_text"
    assert result.turns == 2
    repair_messages = controller.calls[1].request["messages"]
    assert repair_messages[-2]["content"] == "```python\n\n```"
    assert "must be non-empty" in repair_messages[-1]["content"]
    assert public.calls == []
    controller.assert_exhausted()


def test_controller_prose_uses_exact_direct_fallback_at_error_limit() -> None:
    request = _multi_turn_request()
    public_response = chat_text("fallback", model="public-model")
    public = ScriptedBackend([public_response], name="public")
    controller = ScriptedBackend(
        [chat_text("This cannot preserve a tool call.")],
        name="controller",
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(
            controller_model="controller",
            max_consecutive_errors=1,
        ),
    ).run("chat.completions", request)

    assert result.response == public_response
    assert result.stop_reason == "consecutive_protocol_errors_direct_fallback"
    assert result.turns == 1
    assert public.calls[0].request == request
    public.assert_exhausted()
    controller.assert_exhausted()


def test_code_mode_executes_one_fenced_cell_without_advertising_tools() -> None:
    code = "value = 6 * 7\nFINAL_TEXT(str(value))"
    controller = ScriptedBackend(
        [chat_text(f"I will calculate it.\n```python\n{code}\n```")],
        name="controller",
    )
    public = ScriptedBackend(name="public")
    config = RLMConfig(controller_model="controller", action_mode="code")

    result = RLM(public, controller_backend=controller, config=config).run(
        "chat.completions", _chat_request()
    )

    assert extract_text("chat.completions", result.response) == "42"
    assert result.stop_reason == "final_text"
    assert result.turns == 1
    sent = controller.calls[0].request
    assert "tools" not in sent
    assert "parallel_tool_calls" not in sent
    assert sent["messages"][0] == {
        "role": "system",
        "content": config.prompt.render(action_mode="code", allow_recursion=False),
    }
    assert public.calls == []
    controller.assert_exhausted()


def test_responses_controller_preserves_function_call_history() -> None:
    request = {"model": "public-model", "input": "hello"}
    code = 'print(request["input"])'
    controller = ScriptedBackend(
        [
            responses_tool(code, call_id="call-response", input_tokens=5, output_tokens=2),
            responses_tool(
                'FINAL_TEXT("done")',
                call_id="call-final",
                input_tokens=7,
                output_tokens=1,
            ),
        ],
        name="controller",
    )
    public = ScriptedBackend(name="public")
    config = RLMConfig(controller_model="controller", controller_api="responses")

    result = RLM(public, controller_backend=controller, config=config).run("responses", request)

    assert extract_text("responses", result.response) == "done"
    assert result.stop_reason == "final_text"
    assert result.usage.to_dict() == {"input_tokens": 12, "output_tokens": 3, "calls": 2}
    second = controller.calls[1].request
    assert second["instructions"] == config.prompt.render(action_mode="tool", allow_recursion=False)
    assert second["input"][-2]["type"] == "function_call"
    assert second["input"][-2]["call_id"] == "call-response"
    assert second["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-response",
        "output": "stdout:\nhello\n\nvariables: {}",
    }
    assert public.calls == []
    controller.assert_exhausted()


@pytest.mark.parametrize("entrypoint", ["run", "direct"])
@pytest.mark.parametrize("api", ["chat.completions", "responses"])
def test_stream_true_is_rejected_before_any_backend_call(entrypoint: str, api: API) -> None:
    backend = ScriptedBackend(name="public")
    rlm = RLM(backend, config=RLMConfig(controller_model="controller"))
    request: dict[str, Any] = {"model": "public-model", "stream": True}
    request["messages" if api == "chat.completions" else "input"] = (
        [] if api == "chat.completions" else "hello"
    )

    with pytest.raises(InvalidRequestError) as caught:
        getattr(rlm, entrypoint)(api, request)

    assert caught.value.status_code == 400
    assert caught.value.code == "streaming_not_supported"
    assert caught.value.param == "stream"
    assert caught.value.to_body() == {
        "error": {
            "message": "RLM supports only non-streaming requests; set stream to false or omit it",
            "type": "invalid_request_error",
            "param": "stream",
            "code": "streaming_not_supported",
        }
    }
    assert backend.calls == []


@pytest.mark.parametrize("entrypoint", ["run", "direct"])
def test_responses_background_is_rejected_before_any_backend_call(entrypoint: str) -> None:
    backend = ScriptedBackend(name="public")
    rlm = RLM(backend, config=RLMConfig(controller_model="controller"))

    with pytest.raises(InvalidRequestError) as caught:
        getattr(rlm, entrypoint)(
            "responses",
            {"model": "public-model", "input": "hello", "background": True},
        )

    assert caught.value.code == "background_not_supported"
    assert caught.value.param == "background"
    assert backend.calls == []


@pytest.mark.parametrize("entrypoint", ["run", "direct"])
def test_non_json_request_values_are_rejected_before_backend_call(entrypoint: str) -> None:
    backend = ScriptedBackend(name="public")
    rlm = RLM(backend, config=RLMConfig(controller_model="controller"))

    with pytest.raises(InvalidRequestError) as caught:
        getattr(rlm, entrypoint)(
            "chat.completions",
            {"model": "public-model", "messages": [], "bad": {1, 2}},
        )

    assert caught.value.code == "invalid_json_value"
    assert caught.value.param == "request"
    assert "non-JSON value (set)" in str(caught.value)
    assert backend.calls == []


def test_controller_failure_uses_exact_public_fallback() -> None:
    request = _multi_turn_request()
    public_response = chat_text("fallback", model="public-model")
    public = ScriptedBackend([public_response], name="public")
    controller = ScriptedBackend(
        [RLMError("controller endpoint rejected its action protocol", code="protocol_transport")],
        name="controller",
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(controller_model="controller"),
    ).run("chat.completions", request)

    assert result.response == public_response
    assert result.stop_reason == "controller_protocol_transport_direct_fallback"
    assert public.calls[0].request == request
    public.assert_exhausted()
    controller.assert_exhausted()


def test_model_complete_batch_preserves_result_order_and_counts_budget() -> None:
    prompts = ["slow", "fast", "medium"]
    requests_literal = repr(
        [
            {"model": "worker", "messages": [{"role": "user", "content": prompt}]}
            for prompt in prompts
        ]
    )
    code = (
        f"batch_requests = {requests_literal}\n"
        "batch_responses = model_complete_batch(batch_requests)\n"
        "texts = [item['choices'][0]['message']['content'] for item in batch_responses]\n"
        "FINAL_TEXT('|'.join(texts))"
    )
    delays = {"slow": 0.04, "fast": 0.0, "medium": 0.02}

    def respond(_api: API, request: dict[str, Any]) -> dict[str, Any]:
        prompt = request["messages"][0]["content"]
        time.sleep(delays[prompt])
        return chat_text(
            f"done:{prompt}",
            model="worker",
            response_id=f"chatcmpl-{prompt}",
            prompt_tokens=1,
            completion_tokens=1,
        )

    worker = FunctionBackend(respond, name="worker")
    controller = ScriptedBackend([chat_tool(code)], name="controller")
    public = ScriptedBackend(name="public")

    result = RLM(
        public,
        controller_backend=controller,
        worker_backend=worker,
        config=RLMConfig(
            controller_model="controller",
            worker_model="worker",
            max_subcalls=3,
            max_parallel_subcalls=3,
        ),
    ).run("chat.completions", _chat_request())

    assert extract_text("chat.completions", result.response) == ("done:slow|done:fast|done:medium")
    assert result.stop_reason == "final_text"
    assert result.usage.calls == 4
    assert sorted(call.request["messages"][0]["content"] for call in worker.calls) == sorted(
        prompts
    )
    assert public.calls == []
    controller.assert_exhausted()


def test_over_budget_batch_starts_no_worker_items_and_falls_back_unchanged() -> None:
    batch_requests = [
        {"model": "worker", "messages": [{"role": "user", "content": str(index)}]}
        for index in range(3)
    ]
    code = f"model_complete_batch({batch_requests!r})"
    request = _multi_turn_request()
    fallback_response = chat_text("fallback", model="public-model")
    public = ScriptedBackend([fallback_response], name="public")
    controller = ScriptedBackend([chat_tool(code)], name="controller")
    worker = ScriptedBackend(name="worker")

    result = RLM(
        public,
        controller_backend=controller,
        worker_backend=worker,
        config=RLMConfig(
            controller_model="controller",
            max_turns=1,
            max_subcalls=2,
            max_parallel_subcalls=2,
        ),
    ).run("chat.completions", request)

    assert result.response == fallback_response
    assert result.stop_reason == "subcall_limit_direct_fallback"
    assert result.usage.calls == 2
    assert worker.calls == []
    assert len(public.calls) == 1
    assert public.calls[0].request == request
    public.assert_exhausted()
    controller.assert_exhausted()


def test_caught_model_call_limit_cannot_be_replaced_with_a_final() -> None:
    code = (
        "try:\n"
        "    model_complete()\n"
        "except RuntimeError:\n"
        '    FINAL_TEXT("must not bypass the model-call budget")'
    )
    public = ScriptedBackend(name="public")
    controller = ScriptedBackend([chat_tool(code)], name="controller")

    with pytest.raises(LimitExceededError) as caught:
        RLM(
            public,
            controller_backend=controller,
            config=RLMConfig(
                controller_model="controller",
                max_model_calls=1,
            ),
        ).run("chat.completions", _chat_request())

    assert caught.value.code == "model_call_limit"
    assert public.calls == []
    controller.assert_exhausted()


def test_executor_startup_failure_uses_deadline_bound_exact_fallback(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FailingExecutor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def __enter__(self) -> FailingExecutor:
            raise RLMError("kernel unavailable", code="executor_startup")

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr("rlm.engine.IPythonExecutor", FailingExecutor)
    request = _multi_turn_request()
    fallback_response = chat_text("fallback", model="public-model")
    public = ScriptedBackend([fallback_response], name="public")
    controller = ScriptedBackend(name="controller")

    result = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(controller_model="controller", deadline_seconds=0.5),
    ).run("chat.completions", request)

    assert 0 < captured["startup_timeout"] <= 0.5
    assert result.response == fallback_response
    assert result.stop_reason == "executor_startup_direct_fallback"
    assert result.turns == 0
    assert public.calls[0].request == request
    assert controller.calls == []


def test_debug_trace_is_complete_and_markdown_is_exact_rendering(tmp_path: Path) -> None:
    request = _chat_request()
    code = 'value = 6 * 7\nprint(f"answer={value}")'
    final_code = 'FINAL_TEXT("The answer is 42.")'
    tool_response = chat_tool(
        code,
        call_id="call-debug",
        response_id="chatcmpl-debug-tool",
        prompt_tokens=7,
        completion_tokens=3,
    )
    final_response = chat_tool(
        final_code,
        call_id="call-debug-final",
        response_id="chatcmpl-debug-final",
        prompt_tokens=11,
        completion_tokens=2,
    )
    controller = ScriptedBackend([tool_response, final_response], name="controller")
    public = ScriptedBackend(name="public")
    config = RLMConfig(
        controller_model="controller",
        debug=DebugConfig(enabled=True, directory=tmp_path, markdown=True),
    )

    result = RLM(public, controller_backend=controller, config=config).run(
        "chat.completions", request
    )

    assert result.trace_directory is not None
    trace_dir = result.trace_directory
    events = [json.loads(line) for line in (trace_dir / "trace.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "run.started",
        "branch.started",
        "model.request",
        "model.response",
        "controller.action",
        "ipython.started",
        "ipython.completed",
        "controller.observation",
        "model.request",
        "model.response",
        "controller.action",
        "ipython.started",
        "ipython.completed",
        "branch.completed",
        "run.completed",
    ]
    assert [event["event_id"] for event in events] == [
        f"evt-{index:06d}" for index in range(len(events))
    ]
    assert {event["run_id"] for event in events} == {result.run_id}

    expected_prompt = Prompt().render(action_mode="tool", allow_recursion=False)
    branch_started = _only_event(events, "branch.started")
    assert branch_started["payload"]["prompt"] == expected_prompt
    assert branch_started["payload"]["request"] == request

    expected_first_controller_request = {
        "model": "controller",
        "messages": [
            {"role": "system", "content": expected_prompt},
            {
                "role": "user",
                "content": metadata_message("chat.completions", request, depth=0),
            },
        ],
        "tools": [copy.deepcopy(CHAT_IPYTHON_TOOL)],
        "parallel_tool_calls": False,
    }
    model_requests = [event for event in events if event["type"] == "model.request"]
    assert model_requests[0]["payload"] == {
        "call_id": model_requests[0]["payload"]["call_id"],
        "role": "controller",
        "api": "chat.completions",
        "request": expected_first_controller_request,
    }
    started_cells = [event for event in events if event["type"] == "ipython.started"]
    assert [event["payload"] for event in started_cells] == [
        {"code": code},
        {"code": final_code},
    ]

    completed_cells = [event for event in events if event["type"] == "ipython.completed"]
    assert len(completed_cells) == 2
    completed = completed_cells[0]["payload"]
    assert {key: value for key, value in completed.items() if key != "duration_seconds"} == {
        "stdout": "answer=42\n",
        "stderr": "",
        "display": "",
        "final_kind": None,
        "final_value": None,
        "namespace": {"value": "int"},
        "host_errors": [],
    }
    assert completed["duration_seconds"] >= 0
    final_completed = completed_cells[1]["payload"]
    assert {key: value for key, value in final_completed.items() if key != "duration_seconds"} == {
        "stdout": "",
        "stderr": "",
        "display": "",
        "final_kind": "text",
        "final_value": "The answer is 42.",
        "namespace": {"value": "int"},
        "host_errors": [],
    }
    assert final_completed["duration_seconds"] >= 0

    observation = "stdout:\nanswer=42\n\nvariables: {'value': 'int'}"
    assert _only_event(events, "controller.observation")["payload"] == {
        "observation": observation,
        "raw_characters": len("answer=42"),
        "consecutive_errors": 0,
    }
    expected_second_controller_request = copy.deepcopy(expected_first_controller_request)
    expected_second_controller_request["messages"].extend(
        [
            copy.deepcopy(tool_response["choices"][0]["message"]),
            {
                "role": "tool",
                "tool_call_id": "call-debug",
                "content": observation,
            },
        ]
    )
    assert model_requests[1]["payload"]["request"] == expected_second_controller_request

    expected_markdown = _render_markdown(result.run_id, events)
    assert (trace_dir / "trace.md").read_text() == expected_markdown
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["stop_reason"] == "final_text"
    assert manifest["turns"] == 2
    assert manifest["ledger"]["model_calls"] == 2
    assert manifest["ledger"]["subcalls"] == 0
    assert manifest["ledger"]["usage"] == {
        "input_tokens": 18,
        "output_tokens": 5,
        "calls": 2,
    }
    assert public.calls == []
    controller.assert_exhausted()


def _only_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    matches = [event for event in events if event["type"] == event_type]
    assert len(matches) == 1
    return matches[0]


def _render_markdown(run_id: str, events: list[dict[str, Any]]) -> str:
    lines = [f"# RLM trace `{run_id}`", ""]
    for event in events:
        lines.extend(
            [
                f"## {event['event_id']} · {event['type']}",
                "",
                f"Branch `{event['branch_id']}`, depth {event['depth']}",
                "",
                "````json",
                json.dumps(event["payload"], ensure_ascii=False, indent=2),
                "````",
                "",
            ]
        )
    return "\n".join(lines)
