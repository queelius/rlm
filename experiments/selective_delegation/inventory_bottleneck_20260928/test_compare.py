"""Paired pilot denominators must retain missing and unknown planned outcomes."""

import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).with_name("compare.py")
    assert path.exists(), "paired readout not implemented"
    spec = importlib.util.spec_from_file_location("quantity_paired_fixture", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_unknown_is_not_a_loss_and_bounds_use_all_planned_pairs():
    c = module()
    jobs = [dict(task_id="task", repeat=i, episode_id=f"e{i}") for i in range(2)]
    masked = {"e0": dict(observed=True, native_score=0), "e1": dict(observed=True, native_score=1)}
    demand = {"e0": dict(observed=True, native_score=1)}
    result = c.paired(jobs, masked, demand)
    assert result["wins"] == 1 and result["losses"] == 0
    assert result["unknown_pairs"] == 1
    assert result["difference_bounds"] == [0.0, 1.0]
    assert result["difference"] is None and result["task_cluster_95"] is None


def test_complete_all_ties_have_zero_difference():
    c = module()
    jobs = [dict(task_id=f"task{i}", repeat=0, episode_id=f"e{i}") for i in range(2)]
    rows = {f"e{i}": dict(observed=True, native_score=i) for i in range(2)}
    result = c.paired(jobs, rows, rows)
    assert result["ties"] == 2 and result["unknown_pairs"] == 0
    assert result["difference"] == 0 and result["task_cluster_95"] == [0.0, 0.0]
