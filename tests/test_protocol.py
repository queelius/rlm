from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from rlm import RLM, ContextSpec, RLMConfig
from rlm.errors import HarnessContractError, ProtocolError
from rlm.json import strict_json_loads
from rlm.prompts import default_harness_spec, render_prompt
from rlm.protocol import ControllerConversation
from tests.fakes import ScriptedBackend, controller_code, request, responses_text


def parse(text: str) -> str:
    response = responses_text(text)
    conversation = ControllerConversation(system="system", first_user="start")
    return conversation.parse(response).code


@pytest.mark.parametrize(
    "text",
    [
        "prose\n```python\nprint(1)\n```",
        "```python\nprint(1)\n```\nprose",
        "```py\nprint(1)\n```",
        "```python\nprint(1)\n```\n```python\nprint(2)\n```",
    ],
)
def test_parser_rejects_anything_except_one_exact_python_cell(text: str) -> None:
    with pytest.raises(ProtocolError):
        parse(text)


def test_parser_accepts_one_exact_python_cell() -> None:
    assert parse("```python\nvalue = 1\nprint(value)\n```") == "value = 1\nprint(value)"


def test_parser_rejects_multiple_controller_messages() -> None:
    response = responses_text("```python\nprint(1)\n```")
    response["output"].append(copy.deepcopy(response["output"][0]))
    conversation = ControllerConversation(system="system", first_user="start")

    with pytest.raises(ProtocolError, match="one assistant message"):
        conversation.parse(response)


def test_latest_controller_reasoning_is_replayed() -> None:
    response = responses_text("```python\nprint(1)\n```")
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": []}
    response["output"].insert(0, reasoning)
    conversation = ControllerConversation(system="system", first_user="start")

    conversation.append_response(response)
    conversation.append_observation({"status": "ok"})

    assert conversation.items[1] == reasoning


def test_observation_is_a_strict_json_runtime_item_without_prompt_suffixes() -> None:
    observation = {
        "type": "rlm.controller_observation",
        "schema_version": "2",
        "request_binding": "request",
        "submission": {"status": "absent", "required": True},
        "execution": {
            "status": "ok",
            "truncation": {"omitted_chars": 0, "fields": {}},
            "stdout": "ready\n",
        },
    }
    conversation = ControllerConversation(system="system", first_user="start")

    conversation.append_observation(observation)

    item = conversation.items[-1]
    assert item["role"] == "user"
    assert strict_json_loads(item["content"]) == observation
    assert "<ipython_output>" not in item["content"]
    assert "Continue." not in item["content"]


def test_context_spec_can_mask_reasoning_without_changing_message_history() -> None:
    response = responses_text("```python\nprint(1)\n```")
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": []}
    response["output"].insert(0, reasoning)
    conversation = ControllerConversation(
        system="system",
        first_user="start",
        context=ContextSpec(preserve_reasoning=False),
    )

    conversation.append_response(response)

    assert reasoning not in conversation.items
    assert any(item.get("type") == "message" for item in conversation.items)


def test_default_prompt_contains_no_benchmark_recipe() -> None:
    rendered = render_prompt(default_harness_spec(), allow_recursion=True)
    flat = rendered.replace("\n", " ")
    forbidden = ("determinant", "DATA=", "Oolong", "mandatory inspection")
    assert not any(value in rendered for value in forbidden)
    assert "Do not answer the bootstrap envelope." in flat
    assert "model_complete(request)" in rendered
    assert "Work privately until the actual answer itself is ready." in rendered
    assert "Never submit an acknowledgement, readiness or progress report" in flat
    assert "does not mean the caller's request is complete." in flat
    assert "no valid final was accepted" in flat
    assert "bootstrap's `submission.allowed_kinds`" in flat
    assert "ABI-provided globals are deliberately omitted from `SHOW_VARS()`" in flat
    assert "Prefer short, direct cells" in flat
    assert "Never repeat an unchanged failing cell." in flat
    assert "including JSON text" in flat
    assert "never synthesize that envelope" in flat
    assert "submit that value unchanged" in flat
    assert "Continue." not in rendered
    assert "must inspect" not in rendered.lower()


def test_default_harness_matches_reviewed_golden_file() -> None:
    expected = (Path(__file__).parent / "fixtures/harness-v1.json").read_text().strip()
    spec = default_harness_spec()
    assert spec.canonical_json() == expected
    assert spec.fingerprint() == hashlib.sha256(expected.encode()).hexdigest()


@pytest.mark.parametrize(
    ("allow_recursion", "fixture_name", "digest_field"),
    [
        (False, "controller-prompt-v24-nonrecursive.txt", "nonrecursive"),
        (True, "controller-prompt-v24-recursive.txt", "recursive"),
    ],
)
def test_exact_rendered_prompt_matches_golden_and_harness_identity(
    allow_recursion: bool,
    fixture_name: str,
    digest_field: str,
) -> None:
    spec = default_harness_spec()
    rendered = render_prompt(spec, allow_recursion=allow_recursion)
    expected = (Path(__file__).parent / "fixtures" / fixture_name).read_text().rstrip("\n")

    assert rendered == expected
    assert (
        getattr(spec.rendered_prompts, digest_field)
        == hashlib.sha256(rendered.encode()).hexdigest()
    )


@pytest.mark.parametrize("digest_field", ["nonrecursive", "recursive"])
def test_stale_rendered_prompt_digest_fails_before_model_call(digest_field: str) -> None:
    backend = ScriptedBackend([controller_code('FINAL_TEXT("unused")')])
    spec = default_harness_spec()
    stale = replace(
        spec,
        rendered_prompts=replace(spec.rendered_prompts, **{digest_field: "0" * 64}),
    )

    with pytest.raises(HarnessContractError, match="rendered prompt"):
        RLM(backend, config=RLMConfig(harness=stale)).run(request())

    assert backend.calls == []
