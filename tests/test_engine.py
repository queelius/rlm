from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rlm.abi import ENVIRONMENT_ABI
from rlm.config import ControllerConfig, RLMConfig, RunLimits, TraceConfig
from rlm.engine import RLM
from rlm.errors import (
    BackendCloseError,
    InvalidRequestError,
    LimitExceededError,
    RecoveryAbortedError,
    RemoteRLMError,
    RLMError,
    TraceCausalParent,
    UpstreamError,
)
from rlm.json import strict_json_loads
from rlm.prompts import default_harness_spec
from rlm.protocol import extract_text
from rlm.recovery import Abort, ControllerFault, FaultKind
from rlm.specs import RecoveryMode, RecoverySpec
from rlm.trace import RecoveryDecisionPayload, RunStartedPayload
from rlm.types import ModelCallContext, ModelRole
from tests.fakes import (
    FailingCloseBackend,
    FunctionBackend,
    RecordingCloseBackend,
    ScriptedBackend,
    controller_code,
    request,
    responses_text,
)
from tests.trace_helpers import read_events, read_manifest, read_trace_contract, traced_config


def _request() -> dict[str, Any]:
    return request()


def test_disabled_tracing_does_not_convert_payloads_or_build_a_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("disabled tracing performed trace-only work")

    monkeypatch.setattr(RunStartedPayload, "to_dict", unexpected)
    monkeypatch.setattr("rlm.engine._trace_manifest", unexpected)
    rlm = RLM(ScriptedBackend([controller_code('FINAL_TEXT("done")')]))
    result = rlm.run(request())
    assert extract_text(result.response) == "done"
    assert result.harness_fingerprint == rlm.config.harness.fingerprint()


def test_explicit_falsey_controller_backend_is_not_replaced() -> None:
    class FalseyBackend(ScriptedBackend):
        def __bool__(self) -> bool:
            return False

    public = ScriptedBackend([])
    controller = FalseyBackend([controller_code('FINAL_TEXT("done")')])
    result = RLM(public, controller_backend=controller).run(request())

    assert extract_text(result.response) == "done"
    assert len(controller.calls) == 1
    assert public.calls == []


def test_run_configuration_snapshot_detaches_mutable_options() -> None:
    options = {"temperature": 0, "nested": {"seed": 7}}
    config = RLMConfig(controller=ControllerConfig(options=options))
    snapshot = config.snapshot()

    options["temperature"] = 1
    config.controller.options["nested"]["seed"] = 9

    assert snapshot.controller.options == {"temperature": 0, "nested": {"seed": 7}}


def test_run_uses_one_snapshot_when_caller_options_mutate_between_turns() -> None:
    options = {"temperature": 0}

    class MutatingBackend(ScriptedBackend):
        def complete(
            self,
            model_request: Mapping[str, Any],
            *,
            timeout: float,
            context: ModelCallContext,
        ) -> dict[str, Any]:
            response = super().complete(model_request, timeout=timeout, context=context)
            options["temperature"] = 1
            return response

    controller = MutatingBackend(
        [controller_code("print('continue')"), controller_code('FINAL_TEXT("done")')]
    )
    config = RLMConfig(controller=ControllerConfig(options=options))

    RLM(ScriptedBackend(), controller_backend=controller, config=config).run(request())

    assert [call.request["temperature"] for call in controller.calls] == [0, 0]


def config_with(**limit_overrides: Any) -> RLMConfig:
    return RLMConfig(limits=replace(RunLimits(), **limit_overrides))


def _controller_config(**limit_overrides: Any) -> RLMConfig:
    return RLMConfig(
        controller=ControllerConfig(model="controller"),
        limits=replace(RunLimits(), **limit_overrides),
    )


def test_final_text_ends_the_loop_without_calling_public_model() -> None:
    public = ScriptedBackend(name="public")
    controller = ScriptedBackend(
        [controller_code('FINAL_TEXT("done")', input_tokens=3, output_tokens=2)],
        name="controller",
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    assert extract_text(result.response) == "done"
    assert result.stop_reason == "final_text"
    assert result.turns == 1
    assert result.usage.to_dict() == {
        "input_tokens": 3,
        "output_tokens": 2,
        "calls": 1,
        "unreported_calls": 0,
    }
    assert public.calls == []


def test_first_controller_item_is_typed_bootstrap_without_caller_content() -> None:
    caller_content = "CALLER-PAYLOAD-7f3c2b"
    request = {"model": "public-model", "input": caller_content}
    controller = ScriptedBackend([controller_code('FINAL_TEXT("done")')])

    RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(request)

    first_message = controller.calls[0].request["input"][0]
    bootstrap = strict_json_loads(first_message["content"])

    assert first_message["role"] == "user"
    assert bootstrap == {
        "type": "rlm.controller_bootstrap",
        "schema_version": "2",
        "branch_depth": 0,
        "request_binding": ENVIRONMENT_ABI.request_name,
        "content_in_message": False,
        "submission": {
            "required": True,
            "allowed_kinds": ["text", "response"],
        },
        "request_metadata": {
            "model": "public-model",
            "request_keys": ["input", "model"],
            "serialized_characters": 59,
            "input_type": "str",
        },
    }
    assert caller_content not in first_message["content"]


def test_bootstrap_requires_response_submission_for_rich_responses_requests() -> None:
    request_value = {**_request(), "seed": 7}
    response = responses_text("done", model="public-model")
    controller = ScriptedBackend([controller_code(f"FINAL_RESPONSE({response!r})")])

    RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(request_value)

    bootstrap = strict_json_loads(controller.calls[0].request["input"][0]["content"])
    assert bootstrap["submission"] == {
        "required": True,
        "allowed_kinds": ["response"],
    }
    assert "requires_complete_response" not in bootstrap["request_metadata"]


def test_model_complete_without_arguments_preserves_the_exact_request() -> None:
    request = _request()
    public_response = responses_text("public answer", model="public-model")
    public = ScriptedBackend([public_response], name="public")
    controller = ScriptedBackend(
        [controller_code("response = model_complete()\nFINAL_RESPONSE(response)")],
        name="controller",
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(request)

    assert result.response == public_response
    assert result.stop_reason == "final_response"
    assert public.calls[0].request == request
    assert result.usage.calls == 2


def test_empty_completed_public_response_is_a_structurally_valid_final_response() -> None:
    public = ScriptedBackend([responses_text("", model="public-model")], name="public")
    controller = ScriptedBackend(
        [controller_code("response = model_complete()\nFINAL_RESPONSE(response)")],
        name="controller",
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    assert extract_text(result.response) == ""
    assert len(public.calls) == 1


def test_successful_execution_observation_is_returned_for_another_turn() -> None:
    controller = ScriptedBackend(
        [
            controller_code('print(request["model"])'),
            controller_code('FINAL_TEXT("observed")'),
        ],
        name="controller",
    )

    result = RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    second_input = controller.calls[1].request["input"]
    observation = strict_json_loads(second_input[-1]["content"])
    assert observation["type"] == "rlm.controller_observation"
    assert observation["schema_version"] == "2"
    assert observation["request_binding"] == ENVIRONMENT_ABI.request_name
    assert observation["submission"] == {"status": "absent", "required": True}
    assert observation["execution"]["status"] == "ok"
    assert observation["execution"]["stdout"] == "public-model\n"
    assert "Continue." not in second_input[-1]["content"]
    assert extract_text(result.response) == "observed"
    assert result.turns == 2


def test_controller_reasoning_is_replayed_in_private_context() -> None:
    first = controller_code('print(request["model"])')
    first["output"].insert(
        0,
        {
            "id": "reasoning_scripted",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "large hidden reasoning"}],
        },
    )
    controller = ScriptedBackend([first, controller_code('FINAL_TEXT("done")')])

    RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    second_input = controller.calls[1].request["input"]
    assert any(item.get("type") == "reasoning" for item in second_input)
    assert "large hidden reasoning" in json.dumps(second_input)
    assert any(item.get("type") == "message" for item in second_input)


def test_private_context_keeps_only_the_latest_action_and_observation() -> None:
    controller = ScriptedBackend(
        [
            controller_code('print("first observation")'),
            controller_code('print("second observation")'),
            controller_code('FINAL_TEXT("done")'),
        ]
    )

    RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    third_input = controller.calls[2].request["input"]
    serialized = json.dumps(third_input)
    assert "first observation" not in serialized
    assert "second observation" in serialized


def test_controller_metadata_excludes_labeled_data_payload() -> None:
    request = {
        "model": "public-model",
        "input": "Compute the exact total. DATA=" + "private-payload" * 100,
    }
    controller = ScriptedBackend([controller_code('FINAL_TEXT("done")')])

    RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(request)

    first_message = controller.calls[0].request["input"][0]["content"]
    assert "Compute the exact total." not in first_message
    assert "private-payload" not in first_message


def test_controller_protocol_error_can_be_repaired_without_fallback() -> None:
    public = ScriptedBackend(name="public")
    controller = ScriptedBackend(
        [responses_text("I will answer normally."), controller_code('FINAL_TEXT("repaired")')]
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    assert '"kind":"controller_protocol"' in controller.calls[1].request["input"][-1]["content"]
    assert extract_text(result.response) == "repaired"
    assert result.turns == 2
    assert public.calls == []


@pytest.mark.parametrize(
    ("controller_steps", "fault_kind"),
    [
        ([responses_text("prose"), controller_code('FINAL_TEXT("fixed")')], "controller_protocol"),
        ([controller_code("1 / 0"), controller_code('FINAL_TEXT("fixed")')], "execution"),
        (
            [
                controller_code('FINAL_RESPONSE({"output": []})'),
                controller_code('FINAL_TEXT("fixed")'),
            ],
            "final_submission",
        ),
    ],
)
def test_controller_faults_have_one_structured_repair_path(
    tmp_path: Path, controller_steps: list[dict[str, Any]], fault_kind: str
) -> None:
    """Bypasses of the recovery controller must fail this fault matrix."""

    result = RLM(
        ScriptedBackend(controller_steps), config=traced_config(tmp_path, max_turns=2)
    ).run(request())

    assert extract_text(result.response) == "fixed"
    events = read_events(result.trace_directory)
    decisions = [event for event in events if event["type"] == "controller.recovery_decision"]
    repairs = [event for event in events if event["type"] == "controller.repair"]
    assert len(decisions) == len(repairs) == 1
    decision = decisions[0]
    repair = repairs[0]
    source_type = (
        "model.response" if fault_kind == "controller_protocol" else "controller.execution"
    )
    source = next(event for event in events if event["type"] == source_type)
    assert decision["parent_event_id"] == source["event_id"]
    assert repair["parent_event_id"] == decision["event_id"]
    assert repair["payload"]["fault"]["kind"] == fault_kind
    observation = repair["payload"]["observation"]
    assert observation["type"] == "rlm.controller_observation"
    assert observation["schema_version"] == "2"
    assert observation["submission"] == {"status": "absent", "required": True}
    assert observation["execution"]["status"] == "error"
    assert observation["execution"]["fault"]["kind"] == fault_kind


def test_non_string_final_text_is_repairable_without_silent_coercion() -> None:
    controller = ScriptedBackend(
        [
            controller_code('FINAL_TEXT({"alpha": 43})'),
            controller_code("FINAL_TEXT('{\"alpha\":43}')"),
        ]
    )

    result = RLM(controller, config=config_with(max_turns=2)).run(request())

    assert extract_text(result.response) == '{"alpha":43}'
    observation = strict_json_loads(controller.calls[1].request["input"][-1]["content"])
    assert observation["submission"] == {"status": "absent", "required": True}
    assert observation["execution"]["status"] == "error"
    assert observation["execution"]["fault"]["kind"] == "final_submission"
    assert observation["execution"]["exception"]["details"] == {"received_type": "dict"}


@pytest.mark.parametrize(
    ("first_step", "status", "fault_kind"),
    [
        (controller_code('print("ready")'), "ok", None),
        (responses_text("prose"), "error", "controller_protocol"),
        (controller_code("1 / 0"), "error", "execution"),
        (controller_code('FINAL_RESPONSE({"output": []})'), "error", "final_submission"),
    ],
)
def test_minimum_observation_budget_supports_every_turn_path(
    first_step: dict[str, Any], status: str, fault_kind: str | None
) -> None:
    controller = ScriptedBackend([first_step, controller_code('FINAL_TEXT("fixed")')])

    result = RLM(
        controller,
        config=config_with(max_turns=2, max_observation_chars=512),
    ).run(request())

    assert extract_text(result.response) == "fixed"
    encoded = controller.calls[1].request["input"][-1]["content"]
    observation = strict_json_loads(encoded)
    assert len(encoded) <= 512
    assert observation["execution"]["status"] == status
    assert observation["submission"] == {"status": "absent", "required": True}
    if fault_kind is None:
        assert "fault" not in observation["execution"]
    else:
        assert observation["execution"]["fault"]["kind"] == fault_kind


def test_identical_cells_execute_again_after_state_changes() -> None:
    """Exact text equality must not reject a stateful controller action."""

    cell = "counter = globals().get('counter', 0) + 1\nprint(counter)"
    controller = ScriptedBackend(
        [controller_code(cell), controller_code(cell), controller_code("FINAL_TEXT(str(counter))")]
    )

    result = RLM(controller, config=config_with(max_turns=3)).run(request())

    assert extract_text(result.response) == "2"


def test_subcall_timeout_is_bounded_by_the_active_cell_deadline() -> None:
    """Dropping the executor callback cap must make this action timeout too large."""

    public = ScriptedBackend([responses_text("leaf")])
    controller = ScriptedBackend([controller_code('text = ask("leaf")\nFINAL_TEXT(text)')])
    config = RLMConfig(
        limits=RunLimits(deadline_seconds=1, execution_timeout_seconds=0.2),
    )

    RLM(public, controller_backend=controller, config=config).run(request())

    assert 0 < public.timeouts[0] <= 0.2


def test_queued_batch_member_uses_the_remaining_action_deadline() -> None:
    """A fresh relative callback cap for queued work makes the second timeout too large."""

    def complete(request_value: dict[str, Any]) -> dict[str, Any]:
        time.sleep(0.06)
        return responses_text(request_value["input"], model=request_value["model"])

    public = FunctionBackend(complete)
    controller = ScriptedBackend(
        [
            controller_code(
                "responses = model_complete_batch(["
                "{'model': 'm', 'input': 'first'}, {'model': 'm', 'input': 'second'}"
                "])\nFINAL_TEXT('done')"
            )
        ]
    )
    config = RLMConfig(
        limits=RunLimits(
            deadline_seconds=5,
            execution_timeout_seconds=0.2,
            max_parallel_model_calls=1,
        )
    )

    result = RLM(public, controller_backend=controller, config=config).run(request())

    assert extract_text(result.response) == "done"
    assert len(public.timeouts) == 2
    assert public.timeouts[1] < public.timeouts[0] - 0.03


def test_recursive_branch_inherits_the_parent_action_deadline() -> None:
    controller = ScriptedBackend(
        [
            controller_code("nested = rlm_complete()\nFINAL_RESPONSE(nested)"),
            controller_code("answer = ask('leaf')\nFINAL_TEXT(answer)"),
        ]
    )
    public = ScriptedBackend([responses_text("leaf")])
    config = RLMConfig(
        limits=RunLimits(deadline_seconds=5, execution_timeout_seconds=1, max_depth=1)
    )

    result = RLM(public, controller_backend=controller, config=config).run(request())

    assert extract_text(result.response) == "leaf"
    assert controller.timeouts[1] <= 1
    assert public.timeouts[0] <= 1


def test_environment_call_roles_come_from_typed_host_operations() -> None:
    public = ScriptedBackend([responses_text("original"), responses_text("leaf")])
    controller = ScriptedBackend(
        [
            controller_code(
                "original = model_complete()\n"
                "leaf = model_complete({'model': 'm', 'input': 'leaf'})\n"
                "FINAL_RESPONSE(original)"
            )
        ]
    )

    RLM(public, controller_backend=controller).run(request())

    assert [context.role for context in public.contexts] == [ModelRole.PUBLIC, ModelRole.SUBCALL]


def test_incomplete_controller_response_is_repaired_as_model_output() -> None:
    incomplete = controller_code("partial =")
    incomplete["status"] = "incomplete"
    incomplete["incomplete_details"] = {"reason": "max_output_tokens"}
    controller = ScriptedBackend([incomplete, controller_code('FINAL_TEXT("fixed")')])

    result = RLM(controller, config=config_with(max_turns=2)).run(request())

    assert extract_text(result.response) == "fixed"
    assert '"kind":"model_output"' in controller.calls[1].request["input"][-1]["content"]


@pytest.mark.parametrize(
    ("controller_steps", "public_steps", "source_type"),
    [
        (
            [
                {**controller_code("partial ="), "status": "incomplete"},
                controller_code('FINAL_TEXT("fixed")'),
            ],
            [],
            "model.response",
        ),
        (
            [
                controller_code('answer = ask("empty")'),
                controller_code('FINAL_TEXT("fixed")'),
            ],
            [responses_text("")],
            "controller.execution",
        ),
    ],
)
def test_each_model_output_source_has_one_causal_repair_transition(
    tmp_path: Path,
    controller_steps: list[dict[str, Any]],
    public_steps: list[dict[str, Any]],
    source_type: str,
) -> None:
    result = RLM(
        ScriptedBackend(public_steps),
        controller_backend=ScriptedBackend(controller_steps),
        config=traced_config(tmp_path, max_turns=2),
    ).run(request())

    assert extract_text(result.response) == "fixed"
    events = read_events(result.trace_directory)
    source = next(event for event in events if event["type"] == source_type)
    decisions = [event for event in events if event["type"] == "controller.recovery_decision"]
    repairs = [event for event in events if event["type"] == "controller.repair"]
    assert len(decisions) == len(repairs) == 1
    assert decisions[0]["payload"]["decision"]["fault"]["kind"] == "model_output"
    assert decisions[0]["parent_event_id"] == source["event_id"]
    assert repairs[0]["parent_event_id"] == decisions[0]["event_id"]


def test_shared_parallel_backend_exception_has_call_local_causal_wrappers(tmp_path: Path) -> None:
    shared = UpstreamError(503, {"error": {"message": "shared"}}, endpoint="public")

    def fail(_: dict[str, Any]) -> dict[str, Any]:
        raise shared

    controller = ScriptedBackend(
        [
            controller_code(
                "model_complete_batch(["
                "{'model': 'm', 'input': 'first'}, {'model': 'm', 'input': 'second'}"
                "])"
            )
        ]
    )
    config = RLMConfig(
        limits=RunLimits(max_parallel_model_calls=2),
        tracing=TraceConfig(enabled=True, directory=tmp_path),
    )

    with pytest.raises(RemoteRLMError) as caught:
        RLM(FunctionBackend(fail), controller_backend=controller, config=config).run(request())

    assert caught.value.causal_parent is not None
    assert shared.causal_parent is None
    events = read_events(next(tmp_path.iterdir()))
    failures = [event for event in events if event["type"] == "model.failed"]
    assert len(failures) == 2
    assert len({event["parent_event_id"] for event in failures}) == 2
    requests = {event["event_id"]: event for event in events if event["type"] == "model.request"}
    first_failure = next(
        event
        for event in failures
        if requests[event["parent_event_id"]]["payload"]["request"]["input"] == "first"
    )
    assert events[-1]["parent_event_id"] == first_failure["event_id"]


def test_explicit_abort_has_a_distinct_typed_fatal_error() -> None:
    harness = replace(default_harness_spec(), recovery=RecoverySpec(mode=RecoveryMode.ABORT))

    with pytest.raises(RecoveryAbortedError) as caught:
        RLM(ScriptedBackend([responses_text("prose")]), config=RLMConfig(harness=harness)).run(
            request()
        )

    assert caught.value.code == "recovery_aborted"


def test_abort_harness_policy_stops_on_a_controller_protocol_fault() -> None:
    harness = replace(
        default_harness_spec(),
        recovery=RecoverySpec(mode=RecoveryMode.ABORT),
    )
    controller = ScriptedBackend([responses_text("not a Python cell")])

    with pytest.raises(RecoveryAbortedError):
        RLM(
            ScriptedBackend(name="public"),
            controller_backend=controller,
            config=RLMConfig(harness=harness),
        ).run(_request())


def test_python_fence_literal_does_not_truncate_the_controller_action() -> None:
    controller = ScriptedBackend(
        [controller_code('marker = "```"\nFINAL_TEXT(str(marker == "```"))')]
    )

    result = RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    assert extract_text(result.response) == "True"


def test_repeated_controller_action_executes_again_when_the_state_allows_it() -> None:
    repeated = 'print("same cell")'
    controller = ScriptedBackend(
        [
            controller_code(repeated),
            controller_code(repeated),
            controller_code('FINAL_TEXT("repaired")'),
        ]
    )

    result = RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    feedback = controller.calls[2].request["input"][-1]["content"]
    assert '"status":"ok"' in feedback
    assert extract_text(result.response) == "repaired"


def test_abort_harness_policy_stops_on_a_controller_execution_fault() -> None:
    harness = replace(
        default_harness_spec(),
        recovery=RecoverySpec(mode=RecoveryMode.ABORT),
    )
    controller = ScriptedBackend([controller_code('raise ValueError("broken")')])

    with pytest.raises(RecoveryAbortedError) as caught:
        RLM(
            ScriptedBackend(name="public"),
            controller_backend=controller,
            config=RLMConfig(harness=harness),
        ).run(_request())

    assert caught.value.code == "recovery_aborted"


def test_controller_backend_failure_is_not_replaced_by_public_answer() -> None:
    public = ScriptedBackend(name="public")
    controller = ScriptedBackend(
        [UpstreamError(503, {"message": "controller unavailable"}, endpoint="controller")]
    )

    with pytest.raises(RemoteRLMError, match="controller unavailable") as caught:
        RLM(
            public,
            controller_backend=controller,
            config=_controller_config(),
        ).run(_request())

    assert caught.value.code == "upstream_error"
    assert public.calls == []


def test_invalid_model_authored_subcall_can_be_repaired() -> None:
    public = ScriptedBackend(name="public")
    controller = ScriptedBackend(
        [
            controller_code('model_complete({"input": "missing model"})'),
            controller_code('FINAL_TEXT("repaired")'),
        ]
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    observation = controller.calls[1].request["input"][-1]["content"]
    assert "a model is required" in observation
    assert extract_text(result.response) == "repaired"
    assert public.calls == []


def test_failed_controller_python_can_be_repaired_on_the_next_turn() -> None:
    controller = ScriptedBackend(
        [
            controller_code('raise ValueError("broken cell")'),
            controller_code('FINAL_TEXT("recovered")'),
        ]
    )

    result = RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    repair_observation = controller.calls[1].request["input"][-1]["content"]
    assert "broken cell" in repair_observation
    assert "Traceback" not in repair_observation
    assert '"kind":"execution"' in repair_observation
    assert extract_text(result.response) == "recovered"
    assert result.turns == 2


def test_execution_recovery_events_are_causal(tmp_path: Path) -> None:
    controller = ScriptedBackend(
        [
            controller_code('raise ValueError("broken cell")'),
            controller_code('FINAL_TEXT("recovered")'),
        ]
    )
    result = RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=RLMConfig(tracing=TraceConfig(enabled=True, directory=tmp_path)),
    ).run(_request())

    events = read_events(result.trace_directory)
    execution_index = next(
        index for index, event in enumerate(events) if event["type"] == "controller.execution"
    )
    execution = events[execution_index]
    decision = events[execution_index + 1]
    repair = events[execution_index + 2]

    assert decision["type"] == "controller.recovery_decision"
    assert decision["parent_event_id"] == execution["event_id"]
    assert repair["type"] == "controller.repair"
    assert repair["parent_event_id"] == decision["event_id"]


@pytest.mark.parametrize("_attempt", range(3))
def test_parallel_children_settle_before_selected_failure_preserves_causal_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _attempt: int,
) -> None:
    """Every submitted child reaches its terminal decision before input-order failure escapes."""

    all_children_submitted = threading.Event()
    child_a_terminal = threading.Event()
    child_b_terminal = threading.Event()
    first_failure_set = threading.Event()
    second_outcome_requested = threading.Event()
    second_future_cancelled = threading.Event()
    release_first_worker = threading.Event()
    run_finished = threading.Event()
    signals = threading.Condition()
    submitted_futures: list[Future[Any]] = []
    submissions_lock = threading.Lock()

    class GatedThreadPoolExecutor(ThreadPoolExecutor):
        def submit(self, function: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
            future = super().submit(function, *args, **kwargs)
            with submissions_lock:
                submitted_futures.append(future)
                if len(submitted_futures) == 2:
                    all_children_submitted.set()
            return future

    def second_future() -> Future[Any] | None:
        with submissions_lock:
            return submitted_futures[1] if len(submitted_futures) == 2 else None

    def signal(event: threading.Event) -> None:
        with signals:
            event.set()
            signals.notify_all()

    original_set_exception = Future.set_exception
    original_result = Future.result
    original_cancel = Future.cancel

    def gate_first_failure(future: Future[Any], exception: BaseException) -> None:
        original_set_exception(future, exception)
        if isinstance(exception, RLMError) and exception.message == "child-a":
            signal(first_failure_set)
            assert release_first_worker.wait(timeout=5), "test did not release the first worker"

    def record_second_outcome(future: Future[Any], timeout: float | None = None) -> Any:
        if future is second_future():
            signal(second_outcome_requested)
        return original_result(future, timeout=timeout)

    def record_second_cancellation(future: Future[Any]) -> bool:
        cancelled = original_cancel(future)
        if future is second_future():
            signal(second_future_cancelled)
        return cancelled

    monkeypatch.setattr("rlm.engine.ThreadPoolExecutor", GatedThreadPoolExecutor)
    monkeypatch.setattr(Future, "set_exception", gate_first_failure)
    monkeypatch.setattr(Future, "result", record_second_outcome)
    monkeypatch.setattr(Future, "cancel", record_second_cancellation)

    class ConcurrentChildFailureRLM(RLM):
        def _run_branch(self, **kwargs: Any) -> Any:
            if kwargs["depth"] == 0:
                return super()._run_branch(**kwargs)

            request_value = kwargs["request"]["input"]
            assert request_value in {"child-a", "child-b"}
            if request_value == "child-a":
                assert all_children_submitted.wait(timeout=5), "both children were not submitted"
            else:
                assert child_a_terminal.is_set(), (
                    "child-b ran before child-a reached its terminal trace"
                )
            fault = ControllerFault(FaultKind.EXECUTION, request_value)
            decision_event_id = kwargs["trace"].event(
                RecoveryDecisionPayload(turn=1, decision=Abort(fault)),
                branch_id=kwargs["branch_id"],
                depth=kwargs["depth"],
                parent_event_id=kwargs["parent_event_id"],
            )
            if request_value == "child-a":
                child_a_terminal.set()
            else:
                child_b_terminal.set()
            raise RLMError(
                request_value,
                causal_parent=TraceCausalParent(decision_event_id),
            )

    harness = replace(
        default_harness_spec(),
        recovery=RecoverySpec(mode=RecoveryMode.ABORT),
    )
    rlm = ConcurrentChildFailureRLM(
        ScriptedBackend(
            [
                controller_code(
                    "rlm_complete_batch(["
                    "{'model': 'child-a', 'input': 'child-a'}, "
                    "{'model': 'child-b', 'input': 'child-b'}"
                    "])"
                )
            ]
        ),
        config=RLMConfig(
            harness=harness,
            limits=RunLimits(max_depth=1, max_parallel_model_calls=1),
            tracing=TraceConfig(enabled=True, directory=tmp_path),
        ),
    )
    failures: list[BaseException] = []

    def run() -> None:
        try:
            rlm.run(_request())
        except BaseException as exc:
            failures.append(exc)
        finally:
            signal(run_finished)

    thread = threading.Thread(target=run, name="parallel-settlement-test")
    thread.start()
    with signals:
        assert signals.wait_for(first_failure_set.is_set, timeout=5), "child-a did not fail"
        assert signals.wait_for(
            lambda: second_outcome_requested.is_set() or second_future_cancelled.is_set(),
            timeout=5,
        ), "parallel operation neither awaited nor cancelled child-b"
    release_first_worker.set()
    with signals:
        assert signals.wait_for(run_finished.is_set, timeout=5), "run did not finish"
    thread.join()

    assert len(failures) == 1
    assert isinstance(failures[0], RLMError)
    assert failures[0].message == "child-a"
    assert child_a_terminal.is_set()
    assert child_b_terminal.is_set()
    assert second_outcome_requested.is_set()
    assert not second_future_cancelled.is_set()

    run_directory = next(tmp_path.iterdir())
    events = read_events(run_directory)
    child_a_decision = next(
        event
        for event in events
        if event["type"] == "controller.recovery_decision"
        and "child-a" in event["payload"]["decision"]["fault"]["message"]
    )
    child_b_decision = next(
        event
        for event in events
        if event["type"] == "controller.recovery_decision"
        and "child-b" in event["payload"]["decision"]["fault"]["message"]
    )
    terminal = events[-1]

    assert child_a_decision["event_id"] != child_b_decision["event_id"]
    assert terminal["type"] == "run.failed"
    assert terminal["parent_event_id"] == child_a_decision["event_id"]


def test_parallel_keyboard_interrupt_cancels_queued_siblings_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An infrastructure interrupt must not be delayed behind a queued sibling."""

    all_children_submitted = threading.Event()
    first_worker_release = threading.Event()
    queued_sibling_cancelled = threading.Event()
    sibling_started = threading.Event()
    run_finished = threading.Event()
    signals = threading.Condition()
    submitted_futures: list[Future[Any]] = []
    submissions_lock = threading.Lock()
    failures: list[BaseException] = []

    class GatedThreadPoolExecutor(ThreadPoolExecutor):
        def submit(self, function: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
            future = super().submit(function, *args, **kwargs)
            with submissions_lock:
                submitted_futures.append(future)
                if len(submitted_futures) == 2:
                    all_children_submitted.set()
            return future

    def second_future() -> Future[Any] | None:
        with submissions_lock:
            return submitted_futures[1] if len(submitted_futures) == 2 else None

    def signal(event: threading.Event) -> None:
        with signals:
            event.set()
            signals.notify_all()

    original_set_exception = Future.set_exception
    original_cancel = Future.cancel

    def gate_keyboard_interrupt(future: Future[Any], exception: BaseException) -> None:
        original_set_exception(future, exception)
        if isinstance(exception, KeyboardInterrupt):
            first_worker_release.wait()

    def record_queued_cancellation(future: Future[Any]) -> bool:
        cancelled = original_cancel(future)
        if future is second_future() and cancelled:
            signal(queued_sibling_cancelled)
        return cancelled

    monkeypatch.setattr("rlm.engine.ThreadPoolExecutor", GatedThreadPoolExecutor)
    monkeypatch.setattr(Future, "set_exception", gate_keyboard_interrupt)
    monkeypatch.setattr(Future, "cancel", record_queued_cancellation)

    def interrupt_or_wait(value: dict[str, Any]) -> str:
        if value["input"] == "interrupt":
            assert all_children_submitted.wait(timeout=5), "both children were not submitted"
            raise KeyboardInterrupt("stop now")
        signal(sibling_started)
        return "sibling result"

    rlm = RLM(ScriptedBackend())

    def run() -> None:
        try:
            rlm._parallel_map(
                interrupt_or_wait,
                [{"input": "interrupt"}, {"input": "sibling"}],
                config=RLMConfig(limits=RunLimits(max_parallel_model_calls=1)),
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            signal(run_finished)

    thread = threading.Thread(target=run, name="parallel-interrupt-test")
    thread.start()
    try:
        with signals:
            assert signals.wait_for(queued_sibling_cancelled.is_set, timeout=5), (
                "queued sibling was not cancelled before infrastructure interrupt propagation"
            )
    finally:
        first_worker_release.set()
        thread.join(timeout=5)

    assert not thread.is_alive(), "executor work was orphaned"
    assert run_finished.is_set()
    assert not sibling_started.is_set()
    assert len(failures) == 1
    assert isinstance(failures[0], KeyboardInterrupt)


def test_max_turns_is_an_explicit_limit_error() -> None:
    controller = ScriptedBackend([controller_code('print("not final")')])

    with pytest.raises(LimitExceededError, match="did not submit") as caught:
        RLM(
            ScriptedBackend(name="public"),
            controller_backend=controller,
            config=_controller_config(max_turns=1),
        ).run(_request())

    assert caught.value.code == "max_turns_exceeded"


def test_ask_uses_the_public_backend_and_request_model() -> None:
    public = ScriptedBackend([responses_text("helper result", model="public-model")])
    controller = ScriptedBackend(
        [controller_code('answer = ask("small task")\nFINAL_TEXT(answer)')]
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    assert public.calls[0].request == {"model": "public-model", "input": "small task"}
    assert extract_text(result.response) == "helper result"


def test_ask_batch_forwards_explicit_response_options() -> None:
    public = ScriptedBackend([responses_text("label", model="public-model")])
    controller = ScriptedBackend(
        [
            controller_code(
                'labels = ask_batch(["classify"], options={"think": False, '
                '"temperature": 0, "max_output_tokens": 64})\n'
                "FINAL_TEXT(labels.require_texts()[0])"
            )
        ]
    )

    RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    assert public.calls[0].request == {
        "model": "public-model",
        "input": "classify",
        "think": False,
        "temperature": 0,
        "max_output_tokens": 64,
    }


def test_empty_ask_text_requires_a_successful_retry() -> None:
    public = ScriptedBackend(
        [
            responses_text("", model="public-model", output_tokens=64),
            responses_text("recovered"),
        ]
    )
    controller = ScriptedBackend(
        [
            controller_code('answer = ask("classify", options={"max_output_tokens": 64})'),
            controller_code('answer = ask("classify again")\nFINAL_TEXT(answer)'),
        ]
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    observation = controller.calls[1].request["input"][-1]["content"]
    assert "ask returned no usable text" in observation
    assert '"kind":"model_output"' in observation
    assert "empty_text" in observation
    assert extract_text(result.response) == "recovered"


def test_uncapped_empty_reasoning_asks_for_a_smaller_retry() -> None:
    empty = responses_text("", model="public-model", output_tokens=3884)
    empty["output"].insert(0, {"id": "rs_scripted", "type": "reasoning", "summary": []})
    public = ScriptedBackend([empty, responses_text("recovered")])
    controller = ScriptedBackend(
        [
            controller_code('answer = ask("classify four questions")'),
            controller_code('answer = ask("classify one question")\nFINAL_TEXT(answer)'),
        ]
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    observation = controller.calls[1].request["input"][-1]["content"]
    assert '"kind":"model_output"' in observation
    assert "empty_text" in observation
    assert extract_text(result.response) == "recovered"


def test_partial_ask_batch_results_survive_required_retry() -> None:
    def respond(request: dict[str, Any]) -> dict[str, Any]:
        text = "" if request["input"] == "empty" else f"answer:{request['input']}"
        return responses_text(text, model=request["model"])

    public = FunctionBackend(respond)
    controller = ScriptedBackend(
        [
            controller_code(
                'batch = ask_batch(["kept", "empty"])\nFINAL_TEXT("|".join(batch.require_texts()))'
            ),
            controller_code('FINAL_TEXT(batch.texts[0] + "|" + ask("retry"))'),
        ]
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    observation = controller.calls[1].request["input"][-1]["content"]
    assert '"kind":"model_output"' in observation
    assert "empty_text" in observation
    assert extract_text(result.response) == "answer:kept|answer:retry"


def test_ask_can_select_an_explicit_model() -> None:
    public = ScriptedBackend([responses_text("helper result", model="small-model")])
    controller = ScriptedBackend(
        [controller_code('answer = ask("small task", model="small-model")\nFINAL_TEXT(answer)')]
    )

    RLM(
        public,
        controller_backend=controller,
        config=_controller_config(),
    ).run(_request())

    assert public.calls[0].request["model"] == "small-model"


def test_batch_subcalls_run_in_parallel_and_preserve_order() -> None:
    delays = {"slow": 0.04, "fast": 0.0, "medium": 0.02}

    def respond(request: dict[str, Any]) -> dict[str, Any]:
        prompt = request["input"]
        time.sleep(delays[prompt])
        return responses_text(f"done:{prompt}", model=request["model"])

    public = FunctionBackend(respond, name="public")
    controller = ScriptedBackend(
        [
            controller_code(
                'texts = ask_batch(["slow", "fast", "medium"])\n'
                'FINAL_TEXT("|".join(texts.require_texts()))'
            )
        ]
    )

    result = RLM(
        public,
        controller_backend=controller,
        config=_controller_config(max_parallel_model_calls=3),
    ).run(_request())

    assert extract_text(result.response) == "done:slow|done:fast|done:medium"
    assert result.usage.calls == 4


def test_subcall_limit_is_not_hidden() -> None:
    controller = ScriptedBackend([controller_code("model_complete()")])

    with pytest.raises(RLMError) as caught:
        RLM(
            ScriptedBackend(name="public"),
            controller_backend=controller,
            config=_controller_config(max_subcalls=0),
        ).run(_request())

    assert caught.value.code == "subcall_limit"


def test_recursive_call_runs_the_same_loop() -> None:
    child_request = {"model": "child-model", "input": "child"}
    controller = ScriptedBackend(
        [
            controller_code(
                f"response = rlm_complete({child_request!r})\nFINAL_RESPONSE(response)"
            ),
            controller_code('FINAL_TEXT("child answer")'),
        ]
    )

    result = RLM(
        ScriptedBackend(name="public"),
        controller_backend=controller,
        config=_controller_config(max_depth=1),
    ).run(_request())

    assert extract_text(result.response) == "child answer"
    assert result.usage.calls == 2


def test_invalid_final_text_for_tool_request_can_be_repaired() -> None:
    request = {**_request(), "tools": [{"type": "function", "name": "lookup"}]}
    public_response = responses_text("tool-capable response", model="public-model")
    controller = ScriptedBackend(
        [
            controller_code('FINAL_TEXT("invalid")'),
            controller_code("response = model_complete()\nFINAL_RESPONSE(response)"),
        ]
    )

    result = RLM(
        ScriptedBackend([public_response], name="public"),
        controller_backend=controller,
        config=_controller_config(),
    ).run(request)

    assert "requires FINAL_RESPONSE" in controller.calls[1].request["input"][-1]["content"]
    assert result.response == public_response


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("stream", True, "streaming_not_supported"),
        ("background", True, "background_not_supported"),
    ],
)
def test_unsupported_request_modes_are_rejected(field: str, value: Any, code: str) -> None:
    request = {**_request(), field: value}

    with pytest.raises(InvalidRequestError) as caught:
        RLM(ScriptedBackend()).run(request)

    assert caught.value.code == code


def test_direct_sends_the_exact_request_without_controller() -> None:
    request = _request()
    response = responses_text("direct", model="public-model")
    public = ScriptedBackend([response])

    assert RLM(public).direct(request) == response
    assert public.calls[0].request == request


def test_failed_run_is_recorded_without_a_fallback_event(tmp_path: Path) -> None:
    controller = ScriptedBackend([responses_text("not code")])
    config = RLMConfig(
        controller=ControllerConfig(model="controller"),
        limits=RunLimits(max_turns=1),
        tracing=TraceConfig(enabled=True, directory=tmp_path),
    )

    with pytest.raises(LimitExceededError):
        RLM(
            ScriptedBackend(name="public"),
            controller_backend=controller,
            config=config,
        ).run(_request())

    run_directory = next(tmp_path.iterdir())
    events = [json.loads(line) for line in (run_directory / "trace.jsonl").read_text().splitlines()]
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["turns"] == 1
    assert events[-1]["type"] == "run.failed"
    assert not any(event["type"].startswith("fallback") for event in events)


def test_successful_trace_manifest_reports_completed_turns(tmp_path: Path) -> None:
    result = RLM(
        ScriptedBackend([controller_code('FINAL_TEXT("done")')]),
        config=RLMConfig(tracing=TraceConfig(enabled=True, directory=tmp_path)),
    ).run(_request())

    assert read_manifest(result.trace_directory)["turns"] == 1


def test_missing_usage_is_reported_not_silently_counted_as_complete() -> None:
    response = controller_code('FINAL_TEXT("done")')
    response.pop("usage")
    result = RLM(ScriptedBackend([response])).run(request())
    assert result.usage.unreported_calls == 1


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"input_tokens": True, "output_tokens": 0, "total_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1, "total_tokens": 0},
        {"input_tokens": 1.5, "output_tokens": 1, "total_tokens": 2.5},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 3},
    ],
)
def test_present_usage_must_be_complete_nonnegative_integers_with_exact_total(
    usage: dict[str, Any],
) -> None:
    response = controller_code('FINAL_TEXT("done")')
    response["usage"] = usage

    with pytest.raises(RemoteRLMError, match="usage") as caught:
        RLM(ScriptedBackend([response])).run(request())
    assert caught.value.code == "backend_protocol_error"


def test_configured_token_budget_requires_reported_usage() -> None:
    response = controller_code('FINAL_TEXT("done")')
    response.pop("usage")
    config = RLMConfig(limits=RunLimits(max_total_tokens=100))

    with pytest.raises(RemoteRLMError, match="token budget requires usage") as caught:
        RLM(ScriptedBackend([response]), config=config).run(request())
    assert caught.value.code == "backend_protocol_error"


def test_aggregate_token_budget_is_fatal_when_exceeded() -> None:
    response = controller_code('FINAL_TEXT("done")', input_tokens=3, output_tokens=2)
    config = RLMConfig(limits=RunLimits(max_total_tokens=4))

    with pytest.raises(RemoteRLMError, match="token limit") as caught:
        RLM(ScriptedBackend([response]), config=config).run(request())
    assert caught.value.code == "token_limit"


def test_failed_backend_response_is_fatal() -> None:
    response = responses_text("")
    response.update(status="failed", error={"code": "provider_failed", "message": "boom"})

    with pytest.raises(RemoteRLMError, match="boom") as caught:
        RLM(ScriptedBackend([response])).run(request())
    assert caught.value.code == "backend_response_error"


def test_nonterminal_backend_response_is_a_protocol_error() -> None:
    response = responses_text("")
    response["status"] = "in_progress"

    with pytest.raises(RemoteRLMError, match="nonterminal status") as caught:
        RLM(ScriptedBackend([response])).run(request())
    assert caught.value.code == "backend_protocol_error"


def test_final_text_fails_closed_for_unknown_request_semantics() -> None:
    public_request = {**request(), "future_provider_mode": {"enabled": True}}
    controller = ScriptedBackend(
        [
            controller_code('FINAL_TEXT("unsafe")'),
            controller_code(f"FINAL_RESPONSE({responses_text('safe')!r})"),
        ]
    )

    result = RLM(controller, config=config_with(max_turns=2)).run(public_request)
    assert extract_text(result.response) == "safe"


def test_synthetic_response_does_not_claim_complete_usage_when_unreported() -> None:
    response = controller_code('FINAL_TEXT("done")')
    response.pop("usage")
    result = RLM(ScriptedBackend([response])).run(request())

    assert result.response["usage"] is None


def test_direct_baseline_has_canonical_public_trace(tmp_path: Path) -> None:
    public = ScriptedBackend([responses_text("direct", model="public-model")])
    result = RLM(public, config=traced_config(tmp_path)).run_direct(request())

    assert extract_text(result.response) == "direct"
    assert result.stop_reason == "direct"
    assert result.turns == 0
    assert public.contexts[0].role is ModelRole.PUBLIC
    events = read_events(result.trace_directory)
    contract = read_trace_contract()
    assert [event["type"] for event in events] == contract["canonical_direct_sequence"]
    model_response = next(event for event in events if event["type"] == "model.response")
    assert model_response["payload"]["context"]["role"] == "public"
    assert extract_text(model_response["payload"]["response"]) == "direct"


def test_close_attempts_every_distinct_backend_and_reports_all_failures() -> None:
    first = FailingCloseBackend("first")
    second = FailingCloseBackend("second")
    rlm = RLM(first, controller_backend=second)

    with pytest.raises(BackendCloseError) as caught:
        rlm.close()

    assert first.close_calls == second.close_calls == 1
    assert [record.backend_name for record in caught.value.failures] == ["first", "second"]


def test_same_backend_is_closed_exactly_once() -> None:
    backend = RecordingCloseBackend()
    RLM(backend, controller_backend=backend).close()
    assert backend.close_calls == 1
