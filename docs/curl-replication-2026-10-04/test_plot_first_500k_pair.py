"""The guide figure must select the fixed seed-123 pair, never other branches."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def plotting():
    path = Path(__file__).with_name("plot_first_500k_pair.py")
    assert path.exists(), "First 500k pair plotting script is not implemented"
    spec = importlib.util.spec_from_file_location("first_500k_plot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot():
    path = Path(__file__).parent / "extension-data" / "first-500k-pair-summary.json"
    return json.loads(path.read_text())


def test_selects_only_completed_seed123_joined_curves_and_recorded_fixed_endpoints():
    selected = plotting().select_pair(snapshot())
    expected = {
        "curl": (678.0208181129025, 842.7435524938227),
        "no_curl": (454.4710285971863, 866.9394162304535),
    }
    for arm, (at_100k, at_500k) in expected.items():
        run = selected[arm]
        assert run["seed"] == 123 and run["status"] == "completed"
        assert len(run["curve"]) == 127
        assert next(p["mean_return"] for p in run["curve"] if p["env_steps"] == 100000) == at_100k
        assert run["curve"][-1]["mean_return"] == at_500k
        assert run["endpoint_return"] == at_500k
    assert selected["curl"]["abandoned_training_env_steps"] == 102000
    assert selected["no_curl"]["abandoned_training_env_steps"] == 0


def test_ambiguous_completed_seed123_arm_is_rejected_instead_of_selecting_best():
    data = snapshot()
    duplicate = copy.deepcopy(data["runs"][0])
    duplicate["endpoint_return"] = 999
    data["runs"].append(duplicate)
    with pytest.raises(ValueError, match="one completed"):
        plotting().select_pair(data)


def test_endpoint_inconsistent_with_recorded_pair_is_rejected():
    data = snapshot()
    data["runs"][0]["curve"][-1]["mean_return"] = 999
    with pytest.raises(ValueError, match="endpoint"):
        plotting().select_pair(data)
