import copy

import pytest


def test_additive_keeps_malformed_marginal_zero_but_unknown_is_not_zero():
    import analyze_sufficiency_additive_training as audit

    case = {"answer": "Paris", "answer_aliases": []}
    good = {"answerable": False, "answer": ""}
    row = audit.grade_pair(case, [None, good], observed=True)
    assert row == {
        "positive_success": 0,
        "negative_success": 1,
        "reward": 0,
        "training_reward": 0.5,
        "all_valid": False,
    }
    wrong = {"answerable": True, "answer": "London"}
    assert audit.grade_pair(case, [wrong, good], observed=True)["positive_success"] == 0
    assert audit.grade_pair(case, [wrong, good], observed=False) is None


def test_additive_credits_negative_variation_when_product_is_uniform_zero():
    import analyze_sufficiency_additive_training as audit

    groups = [
        [
            {
                "reward": 0,
                "training_reward": n / 2,
                "all_valid": True,
                "positive_success": 0,
                "negative_success": n,
            }
            for n in (0, 1, 0, 1)
        ]
    ]
    groups += [
        [
            {
                "reward": 0,
                "training_reward": 0,
                "all_valid": True,
                "positive_success": 0,
                "negative_success": 0,
            }
            for _ in range(4)
        ]
        for _ in range(15)
    ]
    stats, credits = audit.credit_statistics(groups)
    assert stats["product_effective_groups"] == 0
    assert stats["additive_effective_groups"] == 1
    assert stats["nonzero_response_coefficients"] == 8
    assert credits["p00-k0-v0"] == pytest.approx(-1 / 3)
    assert credits["p00-k1-v1"] == pytest.approx(1 / 3)
    assert stats["paired_denominator"] == 64
    with pytest.raises(ValueError):
        audit.credit_statistics(groups[:-1])


def test_likelihood_loss_uses_emitted_tokens_and_full64_denominator():
    import analyze_sufficiency_additive_training as audit

    row = {
        "call_id": "p00-k0-v0",
        "advantage": 0.5,
        "generation": [-0.2, -0.3],
        "before": [-0.25, -0.35],
        "after": [-0.2, -0.25],
    }
    native = {"p00-k0-v0": {"generation_logps": [-0.2, -0.3], "output_token_ids": [17, 151645]}}
    update = {
        "likelihoods": [row],
        "loss": 0.3 / 64,
        "credited_responses": 1,
        "generation_replay_max_abs_gap": 0.05,
    }
    result = audit.likelihood_statistics(update, {row["call_id"]: 0.5}, native)
    assert result["credited_tokens"] == 2
    assert result["loss_recomputed"] == pytest.approx(0.3 / 64)
    assert result["advantage_weighted_logp_movement"] == pytest.approx(0.075)
    bad = copy.deepcopy(update)
    bad["likelihoods"][0]["before"].pop()
    with pytest.raises(ValueError):
        audit.likelihood_statistics(bad, {row["call_id"]: 0.5}, native)


def test_actual_saved_adapter_delta_not_only_recorded_scalar(tmp_path):
    import analyze_sufficiency_additive_training as audit
    import torch
    from safetensors.torch import save_file

    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file({"model.q_proj.lora_A.weight": torch.tensor([0.0, 0.0])}, a)
    save_file({"model.q_proj.lora_A.weight": torch.tensor([3.0, 4.0])}, b)
    assert audit.adapter_delta(a, b) == {"l2": 5.0, "tensors": 1, "scalars": 2}
    save_file({"model.q_proj.base_layer.weight": torch.tensor([3.0, 4.0])}, b)
    with pytest.raises(ValueError):
        audit.adapter_delta(a, b)


def test_actual_adam_step_is_not_just_checkpoint_label(tmp_path):
    import analyze_sufficiency_additive_training as audit
    import torch

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5, weight_decay=0)
    path = tmp_path / "optimizer.pt"
    torch.save(optimizer.state_dict(), path)
    assert audit.optimizer_steps(path, 0)["parameter_states"] == 0
    parameter.square().sum().backward()
    optimizer.step()
    torch.save(optimizer.state_dict(), path)
    assert audit.optimizer_steps(path, 1)["step_values"] == [1]
    with pytest.raises(ValueError, match="Adam"):
        audit.optimizer_steps(path, 2)
