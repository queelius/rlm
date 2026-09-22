import analyze_sufficiency_pairing_mean as analysis
import pytest


def test_two_arm_profile_requires_256_calls_and_pairing_mean_only_difference():
    plan = {
        "conditions": ["rl_terminal", "pairing_mean_terminal"],
        "planned_calls": 256,
        "control_binding": {
            "rl_terminal_reward": "product",
            "pairing_mean_terminal_reward": "product",
            "rl_terminal_estimator": "diagonal",
            "pairing_mean_terminal_estimator": "pairing_mean",
            "equal_actual_steps_and_sample_cursors": True,
        },
    }
    analysis.validate_condition_contract(plan)
    plan["control_binding"]["pairing_mean_terminal_reward"] = "additive"
    with pytest.raises(ValueError, match="pairing-mean contract"):
        analysis.validate_condition_contract(plan)
