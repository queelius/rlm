"""The TRAIN direct control must share each setting execution across four candidates."""

import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).with_name("training_direct_control.py")
    assert path.exists(), "TRAIN direct-control collector is missing"
    spec = importlib.util.spec_from_file_location("training_direct_control", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_logical_map_expands_each_unique_parent_setting_call_to_four_candidate_slots():
    m = module()
    unique = [
        {"case_id": "a", "repeat": 0, "seed": 11},
        {"case_id": "a", "repeat": 1, "seed": 12},
    ]
    slots = m.logical_slots(unique)
    assert len(slots) == 8
    assert {(row["case_id"], row["repeat"], row["candidate"]) for row in slots} == {
        ("a", repeat, candidate) for repeat in range(2) for candidate in range(4)
    }
    assert {row["physical_identity"] for row in slots if row["repeat"] == 0} == {"a-s0-direct"}


def test_unique_jobs_rejects_candidate_dependent_final_seeds():
    m = module()
    jobs = [
        {"case_id": "a", "repeat": 0, "policy": "c0", "seed": 11},
        {"case_id": "a", "repeat": 0, "policy": "c1", "seed": 12},
        {"case_id": "a", "repeat": 0, "policy": "c2", "seed": 11},
        {"case_id": "a", "repeat": 0, "policy": "c3", "seed": 11},
    ]
    try:
        m.unique_jobs(jobs)
    except ValueError as exc:
        assert "candidate-independent" in str(exc)
    else:
        raise AssertionError("candidate-dependent seed was accepted")
