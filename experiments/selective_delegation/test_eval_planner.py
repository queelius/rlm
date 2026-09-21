"""Matched planner evaluation: public inputs, role isolation and failure accounting."""

import importlib.util
import json
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("eval_planner.py")
    assert path.exists(), "planner evaluator implementation is missing"
    spec = importlib.util.spec_from_file_location("matched_planner_eval", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


CASE = {
    "id": "opaque",
    "split": "validation",
    "question": "Who built it?",
    "documents": [{"id": "p0", "title": "Machine", "text": "PUBLIC_SOURCE", "is_supporting": True}],
    "answer": "GOLD_SECRET",
    "metadata": {
        "source_id": "SOURCE_SECRET",
        "hops": 3,
        "question_decomposition": [{"question": "REFERENCE_SECRET", "answer": "STEP_SECRET"}],
    },
}
PLAN = {"subquestions": ["Who designed it?", "Did #1 build it?"]}


def test_all_evaluation_prompts_exclude_host_labels_and_keep_public_evidence():
    m = module()
    root = m.planner_prompt(CASE)
    helper = m.helper_prompt(CASE, PLAN)
    final = m.final_prompt(CASE, PLAN, "ACTUAL_HELPER_REPORT")
    assert "PUBLIC_SOURCE" not in root
    assert "PUBLIC_SOURCE" in helper and "PUBLIC_SOURCE" in final
    assert "ACTUAL_HELPER_REPORT" in final
    assert "#1" in helper and "own inferred answer" in helper
    for secret in (
        "GOLD_SECRET",
        "SOURCE_SECRET",
        "REFERENCE_SECRET",
        "STEP_SECRET",
        "is_supporting",
        "initial_attempt",
    ):
        assert secret not in root + helper + final


def test_only_sft_root_uses_adapter_and_native_receipts_match_actual_generation(tmp_path):
    import torch

    m = module()

    class Model:
        device = "cpu"
        disabled = False

        def __init__(self):
            self.observed = []

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def generate(self, input_ids, **kwargs):
            self.observed.append((self.disabled, kwargs["max_time"], kwargs["max_new_tokens"]))
            assert kwargs["do_sample"] and kwargs["temperature"] == 0.5
            assert kwargs["top_p"] == 1 and kwargs["top_k"] == 0
            return torch.cat((input_ids, torch.tensor([[7, 2]])), dim=1)

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return [10, 11]

        def decode(self, ids, **kwargs):
            return '{"subquestions":["Question?"]}'

    model = Model()
    client = m.HFClient(model, Tokenizer(), tmp_path, time.time() + 1000, "adapter-hash")
    for condition in ("base", "sft"):
        for role in ("root", "helper", "final"):
            row = client.call(condition + role, "prompt", condition, role, 42)
            assert row["input_token_ids"] == [10, 11]
            assert row["output_token_ids"] == [7, 2]
            assert row["usage"] == {"prompt_tokens": 2, "completion_tokens": 2}
            assert row["available"] and row["terminated"]
    assert [disabled for disabled, _, _ in model.observed] == [True, True, True, False, True, True]
    assert all(0 < cap <= 90 for _, cap, _ in model.observed)
    cached = client.call("sftroot", "prompt", "sft", "root", 42)
    assert cached["available"] and len(model.observed) == 6
    with pytest.raises(ValueError, match="request"):
        client.call("sftroot", "changed", "sft", "root", 42)
    restricted = client.call("isolated-step", "prompt", "sft", "helper", 42, max_new_tokens=192)
    assert model.observed[-1][0] and model.observed[-1][2] == 192
    assert restricted["request"]["sampling"]["max_new_tokens"] == 192


def test_malformed_plan_is_recorded_without_helper_fallback_and_keeps_denominator(tmp_path):
    m = module()

    class Client:
        output = tmp_path

        def __init__(self):
            self.roles = []

        def call(self, identity, prompt, condition, role, seed):
            self.roles.append(role)
            return {
                "call_id": identity,
                "available": True,
                "text": '{"answer":"bad"}',
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }

    client = Client()
    result = m.collect_episode(client, CASE, "base", 0)
    assert result["status"] == "invalid_plan"
    assert result["plan_valid"] is False and result["correct"] is False
    assert client.roles == ["root"]
    summary = m.summarize(tmp_path, {"case_ids": ["opaque"], "repeats": 1})
    assert summary["conditions"]["base"]["planned_episodes"] == 1
    assert summary["conditions"]["base"]["invalid_plans"] == 1
    assert summary["conditions"]["base"]["em"] == 0
    assert summary["conditions"]["sft"]["missing_episodes"] == 1


def test_failed_generation_retains_unknown_cost(tmp_path):
    import torch

    m = module()

    class BrokenModel:
        device = "cpu"

        @contextmanager
        def disable_adapter(self):
            yield

        def generate(self, input_ids, **kwargs):
            assert torch.is_tensor(input_ids)
            raise RuntimeError("observed model failure")

    class Tokenizer:
        eos_token_id = pad_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2]

    client = m.HFClient(BrokenModel(), Tokenizer(), tmp_path, time.time() + 1000, "hash")
    row = client.call("failure", "prompt", "base", "root", 1)
    assert not row["available"]
    assert m.cost([row])["unknown_usage_calls"] == 1
    assert "completion_tokens" not in row["usage"]


def test_successful_matched_episodes_use_paired_seeds_and_official_scoring(tmp_path):
    m = module()

    class Client:
        output = tmp_path

        def __init__(self):
            self.requests = []

        def call(self, identity, prompt, condition, role, seed):
            self.requests.append((condition, role, seed, prompt))
            return {
                "call_id": identity,
                "available": True,
                "text": json.dumps(PLAN)
                if role == "root"
                else "ACTUAL_REPORT"
                if role == "helper"
                else '{"answer":"GOLD_SECRET"}',
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }

    client = Client()
    for condition in ("base", "sft"):
        result = m.collect_episode(client, CASE, condition, 0)
        assert result["status"] == "scored"
        assert result["correct"] and result["f1"] == 1
        assert result["deployed_cost"]["calls"] == 3
    first, second = client.requests[:3], client.requests[3:]
    assert [(role, seed, prompt) for _, role, seed, prompt in first] == [
        (role, seed, prompt) for _, role, seed, prompt in second
    ]
    assert "ACTUAL_REPORT" in first[-1][-1]


def test_isolated_binds_actual_previous_prediction_and_hides_composed_question(tmp_path):
    m = module()
    requests = []
    questions = {"subquestions": ["Who designed the machine?", "Where was #1 born?"]}

    class Client:
        output = tmp_path

        def call(self, identity, prompt, condition, role, seed, **kwargs):
            requests.append((role, prompt, kwargs.get("max_new_tokens")))
            text = (
                json.dumps(questions)
                if role == "root"
                else (
                    '{"answer":"MODEL_PERSON"}'
                    if role == "helper" and len(requests) == 2
                    else '{"answer":"MODEL_PLACE"}'
                    if role == "helper"
                    else '{"answer":"GOLD_SECRET"}'
                )
            )
            return {
                "call_id": identity,
                "available": True,
                "text": text,
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }

    result = m.collect_episode(Client(), CASE, "sft", 0, execution="isolated")
    assert result["correct"] and result["execution"] == "isolated"
    assert result["deployed_cost"]["calls"] == 4
    assert result["expected_calls_for_valid_plan"] == 4
    helpers = [row for row in requests if row[0] == "helper"]
    assert [row[2] for row in helpers] == [192, 192]
    assert "Where was MODEL_PERSON born?" in helpers[1][1]
    for _, prompt, _ in helpers:
        assert CASE["question"] not in prompt
        assert "model_plan" not in prompt and "subquestions" not in prompt
        assert "REFERENCE_SECRET" not in prompt and "GOLD_SECRET" not in prompt
        assert "PUBLIC_SOURCE" in prompt
    assert "MODEL_PERSON" in requests[-1][1] and "MODEL_PLACE" in requests[-1][1]
    assert questions["subquestions"][0] in requests[-1][1]
    assert "call_id" not in requests[-1][1]


@pytest.mark.parametrize("question", ["Where is #1 from?", "Who is #2?", "What is #0?"])
def test_isolated_rejects_unavailable_dependencies_before_helper_call(tmp_path, question):
    m = module()

    class Client:
        output = tmp_path

        def __init__(self):
            self.roles = []

        def call(self, identity, prompt, condition, role, seed, **kwargs):
            self.roles.append(role)
            return {
                "call_id": identity,
                "available": True,
                "text": json.dumps({"subquestions": [question, "Another question?"]}),
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }

    client = Client()
    result = m.collect_episode(client, CASE, "base", 0, execution="isolated")
    assert result["status"] == "invalid_dependency"
    assert not result["correct"]
    assert client.roles == ["root"]


def test_isolated_rejects_malformed_helper_without_answer_fallback(tmp_path):
    m = module()

    class Client:
        output = tmp_path

        def __init__(self):
            self.roles = []

        def call(self, identity, prompt, condition, role, seed, **kwargs):
            self.roles.append(role)
            return {
                "call_id": identity,
                "available": True,
                "text": json.dumps(PLAN) if role == "root" else "raw answer without JSON",
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }

    client = Client()
    result = m.collect_episode(client, CASE, "base", 0, execution="isolated")
    assert result["status"] == "invalid_helper"
    assert not result["correct"]
    assert client.roles == ["root", "helper"]
