"""CPU contracts for on-policy root-only RLOO, sampling, and frozen downstream weights."""

import importlib.util
from pathlib import Path

import pytest
import torch


def implementation():
    path = Path(__file__).with_name("rl_planner.py")
    assert path.exists(), "planner RL implementation missing"
    spec = importlib.util.spec_from_file_location("rl_planner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rloo_advantages_and_root_sequence_loss_have_correct_gradient_direction():
    mod = implementation()
    assert mod.rloo([0, 1, 0, 1]) == pytest.approx([-2 / 3, 2 / 3, -2 / 3, 2 / 3])
    assert mod.rloo([1, 1, 1, 1]) == [0, 0, 0, 0]
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    tokens = torch.tensor([[1, 1]])
    logps = mod.token_logps(logits, tokens)
    loss = mod.policy_loss(logps, 2 / 3)
    loss.backward()
    # Both emitted positions (including a final EOS when present) have equal credit.
    assert torch.allclose(logits.grad[0, 0], logits.grad[0, 1])
    assert logits.grad[0, 0, 1] < 0 and logits.grad[0, 0, 0] > 0
    assert float(loss.detach()) == pytest.approx(
        (2 / 3) * 2 * torch.log(torch.tensor(2.0)).item() / 64
    )


def test_seeds_vary_only_root_candidate_and_keep_downstream_common():
    mod = implementation()
    rows = [mod.seed_schedule(1, 3, candidate) for candidate in range(4)]
    assert len({r["root"] for r in rows}) == 4
    assert len({r["downstream"] for r in rows}) == 1
    assert mod.seed_schedule(2, 3, 0) != rows[0]


def test_lora_only_update_changes_planner_but_disabled_helpers_stay_frozen():
    mod = implementation()
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(13)
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_dropout=0.0,
        )
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.eval()
    params = mod.planner_parameters(model)
    ids = torch.tensor([[1, 2, 3]])
    with model.disable_adapter(), torch.no_grad():
        before_helper = model(input_ids=ids, use_cache=False).logits.clone()
    before_planner = model(input_ids=ids, use_cache=False).logits.detach().clone()
    optimizer = torch.optim.SGD(params, lr=0.5)
    loss = mod.policy_loss(
        mod.token_logps(model(input_ids=ids, use_cache=False).logits[:, -1:], torch.tensor([[4]])),
        1.0,
    )
    loss.backward()
    assert all(p.grad is None for name, p in model.named_parameters() if "lora_" not in name)
    optimizer.step()
    with model.disable_adapter(), torch.no_grad():
        after_helper = model(input_ids=ids, use_cache=False).logits
    after_planner = model(input_ids=ids, use_cache=False).logits.detach()
    assert torch.equal(before_helper, after_helper)
    assert not torch.equal(before_planner, after_planner)


def test_admission_requires_two_diverse_mixed_groups_without_dropping_other_groups():
    mod = implementation()
    groups = [
        [
            {"reward": 0, "plan_valid": True, "plan": {"subquestions": ["A"]}},
            {"reward": 1, "plan_valid": True, "plan": {"subquestions": ["B"]}},
            {"reward": 0, "plan_valid": False},
            {"reward": 0, "plan_valid": False},
        ]
    ]
    assert not mod.batch_diagnostics(groups + [groups[0][:1] * 4] * 15)["admitted"]
    result = mod.batch_diagnostics(groups * 2 + [groups[0][:1] * 4] * 14)
    assert result["admitted"] and result["qualifying_groups"] == 2
    assert result["episodes"] == 64 and len(result["groups"]) == 16


def test_valid_vs_invalid_plan_variation_alone_does_not_admit_update():
    mod = implementation()
    group = [
        {"reward": 1, "plan_valid": True, "plan": {"subquestions": ["A"]}},
        {"reward": 1, "plan_valid": True, "plan": {"subquestions": ["B"]}},
        {"reward": 0, "plan_valid": False},
        {"reward": 0, "plan_valid": False},
    ]
    result = mod.batch_diagnostics([group] * 16)
    assert not result["admitted"]
    assert result["groups"][0]["root_protocol_only_variation"]


def test_root_logprob_alignment_includes_every_emitted_token_including_eos():
    mod = implementation()
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_dropout=0.0,
        )
    ).eval()
    record = {"input_token_ids": [1, 5], "output_token_ids": [7, 9, 2]}
    selected = mod.root_logps(model, record)
    full = model(input_ids=torch.tensor([[1, 5, 7, 9, 2]]), use_cache=False).logits[:, 1:4]
    expected = (
        torch.log_softmax(full.float() / 0.8, -1)
        .gather(-1, torch.tensor([[[7], [9], [2]]]))
        .squeeze(-1)
    )
    assert selected.shape == (1, 3)
    assert torch.allclose(selected, expected, atol=1e-6)
