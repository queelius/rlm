"""Plan-only final replay: strict visibility, fixed controls and honest source accounting."""

import importlib.util
import json
import time
from pathlib import Path

import pytest
from test_eval_planner import CASE, PLAN


def module():
    path = Path(__file__).with_name("plan_only_probe.py")
    assert path.exists(), "plan-only collector not implemented"
    spec = importlib.util.spec_from_file_location("plan_only_probe", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_empty_steps_changes_only_helper_trace_without_leaking_secrets():
    m = module()
    trace = [
        {
            "step": 1,
            "question": "Who?",
            "resolved_question": "RESOLVED_SECRET",
            "answer": "HELPER_SECRET",
        }
    ]
    factual = m.evaluation.final_prompt(CASE, PLAN, {"execution": "isolated", "steps": trace})
    empty = m.plan_only_prompt(CASE, PLAN)
    a, b = [json.loads(p.rsplit("\n", 1)[1]) for p in (factual, empty)]
    a["helper_report"]["steps"] = []
    assert a == b and factual.rsplit("\n", 1)[0] == empty.rsplit("\n", 1)[0]
    for secret in (
        "HELPER_SECRET",
        "RESOLVED_SECRET",
        "GOLD_SECRET",
        "SOURCE_SECRET",
        "REFERENCE_SECRET",
        "STEP_SECRET",
    ):
        assert secret not in empty


def test_control_selection_is_sorted_outcome_independent_and_not_replenished():
    m = module()
    parents = [f"p{i:02}" for i in reversed(range(64))]
    assert m.control_parents(parents) == ["p00", "p01", "p02", "p03"]
    assert len(m.control_parents(parents)) * 2 * len(m.POLICIES) == 16


def test_dependency_zero_is_observed_but_missing_source_is_not():
    m = module()
    assert m.source_grade({"status": "invalid_dependency"}, None, CASE)["observed"]
    assert not m.source_grade({"status": "invalid_dependency"}, None, CASE)["correct"]
    with pytest.raises(ValueError, match="missing"):
        m.source_grade({"status": "scored"}, None, CASE)
    with pytest.raises(ValueError, match="unavailable"):
        m.source_grade({"status": "scored"}, {"available": False}, CASE)


def test_missing_factual_control_not_replaced_but_plan_only_retains_dependency_slot(tmp_path):
    m = module()

    class Client:
        output = tmp_path
        calls = 0

        def call(self, identity, prompt, condition, role, seed, *, max_new_tokens):
            self.calls += 1
            assert prompt == m.plan_only_prompt(CASE, PLAN)
            return dict(
                call_id=identity,
                available=True,
                text='{"answer":"GOLD_SECRET"}',
                usage={"prompt_tokens": 20, "completion_tokens": 4},
            )

    root = dict(
        call_id="saved-root", available=True, usage={"prompt_tokens": 10, "completion_tokens": 5}
    )
    job = dict(
        case_id=CASE["id"],
        repeat=0,
        policy="sft",
        source_episode_id="saved",
        root=root,
        factual=None,
        seed=1,
        old=m.source_grade({"status": "invalid_dependency"}, None, CASE),
        plan_only_prompt=m.plan_only_prompt(CASE, PLAN),
    )
    client = Client()
    control = m.collect(client, job, CASE, "factual_replay")
    assert client.calls == 0 and control["status"] == "control_source_unavailable"
    row = m.collect(client, job, CASE, "plan_only")
    assert client.calls == 1 and row["observed"] and not row["source_final_available"]
    assert row["new"]["correct"] and not row["old"]["correct"]
    assert row["new_physical_cost"]["calls"] == 1
    assert row["hypothetical_deployed_cost"]["calls"] == 2
    assert row["hypothetical_deployed_cost"]["prompt_tokens"] == 30
    m.collect(client, job, CASE, "plan_only")
    assert client.calls == 1  # Completed episode is immutable; no implicit generation on resume.


def test_native_final_path_rejects_other_roles_and_disables_adapter(tmp_path):
    torch = pytest.importorskip("torch")
    from contextlib import contextmanager

    m = module()

    class Model:
        device = "cpu"
        disabled = False

        def parameters(self):
            return []

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def generate(self, input_ids, **kwargs):
            assert self.disabled and kwargs["temperature"] == 0.5
            assert kwargs["max_new_tokens"] == 128
            return torch.cat([input_ids, torch.tensor([[7, 2]])], dim=1)

    class Tokenizer:
        eos_token_id = pad_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            return [10, 11]

        def decode(self, *args, **kwargs):
            return '{"answer":"actual"}'

    client = m.FinalClient(Model(), Tokenizer(), tmp_path, time.time() + 100, "inert-sha")
    call = client.call("final", "prompt", "base", "final", 7, max_new_tokens=128)
    assert call["available"] and call["adapter_sha256"] is None
    assert call["request"]["helper_contract"]["mode"] == "base"
    with pytest.raises(ValueError, match="final-only"):
        client.call("root", "prompt", "sft", "root", 7, max_new_tokens=128)
