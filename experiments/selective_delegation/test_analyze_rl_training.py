"""Tiny CPU fixtures for RLOO diagnostic interpretation."""

import importlib.util
from pathlib import Path


def implementation():
    path = Path(__file__).with_name("analyze_rl_training.py")
    assert path.exists(), "RL training analyzer missing"
    spec = importlib.util.spec_from_file_location("rl_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_separates_scored_variation_from_protocol_only_variation():
    module = implementation()
    rows = [
        dict(reward=r, plan_valid=True, plan={"subquestions": [str(i)]}, status=s)
        for i, (r, s) in enumerate(
            [(1, "scored"), (0, "scored"), (0, "invalid_helper"), (1, "scored")]
        )
    ]
    result = module.group_summary(rows)
    assert result["advantages"] == [2 / 3, -2 / 3, -2 / 3, 2 / 3]
    assert result["qualifies"] is True
    assert result["fully_scored_reward_variation"] is True
    assert result["variation_includes_protocol_zeros"] is True
    rows[1]["status"] = "invalid_helper"
    result = module.group_summary(rows)
    assert result["fully_scored_reward_variation"] is False
    assert result["protocol_only_reward_variation"] is True


def test_likelihood_movement_uses_token_sum_and_fixed_denominator():
    module = implementation()
    result = module.likelihood_movement([[-2, -3], [-2]], [[-1, -2], [-4]], [1, -1])
    assert result["advantage_weighted_delta_per_64"] == 4 / 64
    assert result["positive_advantage_mean_sum_logp_change"] == 2
    assert result["negative_advantage_mean_sum_logp_change"] == -2
