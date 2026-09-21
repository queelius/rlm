"""Frozen-plan native routing and leakage-free held-seed selection, CPU only."""

import json
import time

import analyze_frozen_execution as analysis
import eval_helper
import frozen_execution_probe as collector
import pytest
from test_eval_planner import CASE


def test_selector_is_order_independent_excludes_held_seed_and_rejects_null_parent():
    matrix = [[1, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    rows = [
        {"case_id": "a", "candidate": c, "execution_repeat": r, "reward": matrix[c][r]}
        for c in range(4)
        for r in range(4)
    ]
    result = analysis.held_seed_selection(rows)
    assert result == analysis.held_seed_selection(list(reversed(rows)))
    held = next(r for r in result if r["held_seed"] == 0)
    assert held["one_training_seed"] == 1
    assert held["one_candidate"] == held["three_candidate"] == 1
    assert held["one"] == held["three"] == 0
    variability = analysis.reward_variability(rows)
    assert variability["success_count_histogram"] == {"0": 2, "1": 1, "3": 1}
    assert variability["same_plan_seed_pairs"] == 24
    assert variability["same_plan_disagreeing_seed_pairs"] == 6
    assert variability["advantage_sign_counts"]["sign_flip"] == 6
    assert variability["rank_order_counts"]["reversed_order"] == 3
    assert variability["rank_order_counts"]["tied_both"] == 12
    all_zero = [{**row, "reward": 0} for row in rows]
    assert all(
        r["one_candidate"] == r["three_candidate"] == 0
        for r in analysis.held_seed_selection(all_zero)
    )
    broken = [
        {**row, "reward": None} if row["candidate"] == 0 and row["execution_repeat"] == 0 else row
        for row in rows
    ]
    assert analysis.held_seed_selection(broken) == []
    partial = analysis.reward_variability(broken, parents=["a", "missing"])
    assert len(partial["per_plan"]) == 8
    assert partial["rank_order_comparisons"] == 0
    assert partial["advantage_sign_comparisons"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        analysis.held_seed_selection(rows + [rows[0]])


def test_native_helper_only_adapter_and_base_final_preserve_training_trace(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                eos_token_id=2,
                pad_token_id=2,
            )
        ),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
        adapter_name="helper_sft",
    )
    prompts, observed = [], []
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    handle = layer.register_forward_pre_hook(
        lambda *_: observed.append((len(prompts), layer.disable_adapters))
    )

    class Tokenizer:
        eos_token_id = pad_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return [10, 11]

        def decode(self, *args, **kwargs):
            return '{"answer":"actual prediction"}'

    case = {**CASE, "answer": "actual prediction"}
    client = eval_helper.HelperClient(
        model, Tokenizer(), tmp_path, time.time() + 1000, "helper-sha"
    )
    slot = {
        "case_id": case["id"],
        "candidate": 2,
        "parent_index": 3,
        "plan": {"subquestions": ["Who made it?", "Where was #1 born?"]},
    }
    row = collector.execute_slot(client, case, slot, 1)
    assert row["reward"] == 1 and row["status"] == "scored"
    assert row["seed"] == collector.SEED + 31000
    calls = [
        json.loads((tmp_path / "calls" / (cid + ".json")).read_text()) for cid in row["call_ids"]
    ]
    assert [r["role"] for r in calls] == ["helper", "helper", "final"]
    assert [r["adapter_enabled"] for r in calls] == [True, True, False]
    assert {index for index, _ in observed} == {1, 2, 3}
    assert all(disabled == (index == 3) for index, disabled in observed)
    handle.remove()
    assert [r["request"]["seed"] for r in calls] == [
        row["seed"] + 1,
        row["seed"] + 2,
        row["seed"] + 100,
    ]
    assert [r["request"]["sampling"]["max_new_tokens"] for r in calls] == [192, 192, 128]
    assert all(r["request"]["sampling"]["temperature"] == 0.5 for r in calls)
    assert "Where was actual prediction born?" in prompts[1]
    assert '"step": 1' in prompts[-1] and '"step": 2' in prompts[-1]
    assert all(
        secret not in "".join(prompts)
        for secret in ("GOLD_SECRET", "REFERENCE_SECRET", "STEP_SECRET")
    )
    assert not any(p.requires_grad for p in model.parameters())


def test_unavailable_is_null_but_invalid_dependency_is_observed_zero():
    class Client:
        calls = 0

        def call(self, identity, *args, **kwargs):
            self.calls += 1
            return {"call_id": identity, "available": False, "usage": {}}

    client = Client()
    slot = {
        "case_id": CASE["id"],
        "candidate": 0,
        "parent_index": 0,
        "plan": {"subquestions": ["Question?"]},
    }
    missing = collector.execute_slot(client, CASE, slot, 0)
    assert (
        missing["reward"] is None
        and missing["status"] == "missing_generation"
        and client.calls == 1
    )
    invalid = collector.execute_slot(
        client, CASE, {**slot, "plan": {"subquestions": ["Where is #2?"]}}, 0
    )
    assert (
        invalid["reward"] == 0 and invalid["status"] == "invalid_dependency" and client.calls == 1
    )
