"""Focused accounting checks for the correlated TRAIN direct control."""

import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).with_name("analyze_training_direct_control.py")
    spec = importlib.util.spec_from_file_location("analyze_training_direct_control", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_parent_deltas_average_five_settings_and_four_correlated_candidates():
    m = module()
    direct = {
        ("parent-a", setting, candidate): int(setting == 0)
        for setting in range(5)
        for candidate in range(4)
    }
    executed = {
        ("parent-a", setting, candidate): int(candidate == 0)
        for setting in range(5)
        for candidate in range(4)
    }
    plan_only = {
        ("parent-a", setting, candidate): 0 for setting in range(5) for candidate in range(4)
    }
    result = m.parent_deltas(["parent-a"], direct, executed, plan_only)
    assert result["executed_minus_direct"] == {"parent-a": 0.05}
    assert result["direct_minus_plan_only"] == {"parent-a": 0.2}


def test_direct_availability_is_not_conflated_with_invalid_json():
    m = module()
    assert m.direct_status({"available": False}, {"valid": False}) == "unavailable"
    assert m.direct_status({"available": True}, {"valid": False}) == "invalid_json"
    assert m.direct_status({"available": True}, {"valid": True}) == "scored"


def test_candidate_independent_seed_reads_episode_metadata_not_grade_fields():
    m = module()
    slots = {
        ("a", 0, candidate): {"seed": 7, "executed": {"correct": False}} for candidate in range(4)
    }
    assert m.candidate_independent_seed(slots, "a", 0) == 7
