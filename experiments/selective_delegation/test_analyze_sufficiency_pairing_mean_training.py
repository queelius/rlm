import importlib.util
from pathlib import Path

import pytest


def test_coefficients_use_json_canonical_nested_lists():
    source = Path(__file__).with_name("analyze_sufficiency_pairing_mean_training.py")
    spec = importlib.util.spec_from_file_location("audit003", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def pair(p, n):
        return {
            "positive_success": p,
            "negative_success": n,
            "reward": p * n,
            "training_reward": p * n,
            "all_valid": True,
        }

    groups = [[pair(0, 1), pair(1, 1), pair(0, 1), pair(1, 1)]]
    groups += [[pair(0, 0) for _ in range(4)] for _ in range(15)]
    _, advantages = module.coefficient_map(groups)
    assert [value for pair in advantages[0] for value in pair] == pytest.approx(
        [-2 / 3, 0.0, 2 / 3, 0.0, -2 / 3, 0.0, 2 / 3, 0.0]
    )
    assert all(isinstance(value, list) for pair in advantages for value in pair)
