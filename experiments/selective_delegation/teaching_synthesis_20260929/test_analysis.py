"""Small checks for synthesis-only pairing and cluster units."""

import importlib.util
from pathlib import Path

import pytest


def analysis():
    path = Path(__file__).with_name("analyze.py")
    assert path.exists(), "synthesis not implemented"
    spec = importlib.util.spec_from_file_location("teaching_synthesis_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows():
    return [
        dict(task_id=t, panel=0, fit_seed=f, world=w, rollout_seed=s, score=0)
        for t in ("a", "b")
        for f in (1, 2)
        for w in (42, 50)
        for s in (4, 5)
    ]


def test_world_fit_and_rollout_repeats_do_not_multiply_root_clusters():
    left = rows()
    right = [dict(r, score=int(r["task_id"] == "a")) for r in reversed(left)]
    result = analysis().paired(left, right)
    assert result["planned_pairs"] == 16
    assert result["root_clusters"] == 2
    assert result["wins"] == 8
    assert result["difference"] == 0.5
    assert result["root_cluster_95"] == [0, 1]


def test_incomplete_attempt_remains_unknown_not_failure():
    left = rows()
    right = [dict(r, score=1) for r in left]
    right[0]["score"] = None
    result = analysis().paired(left, right)
    assert result["unknown_pairs"] == 1
    assert result["planned_pairs"] == 16
    assert result["difference"] is None
    assert result["root_cluster_95"] is None
    assert result["difference_bounds"] == [14 / 16, 1]


def test_wrong_fit_seed_cannot_pair_despite_same_task_world():
    left = rows()
    right = [dict(r, fit_seed=r["fit_seed"] + 10) for r in left]
    with pytest.raises(ValueError, match="paired identities"):
        analysis().paired(left, right)


def test_two_panels_are_fixed_strata_not_more_world_clusters():
    left = [
        dict(task_id="a", panel=0, fit_seed=1, world=42, rollout_seed=4, score=0),
        dict(task_id="b", panel=1, fit_seed=2, world=42, rollout_seed=4, score=0),
    ]
    right = [dict(left[0], score=1), dict(left[1], score=0)]
    result = analysis().paired(left, right)
    assert result["strata"] == {"0": 1, "1": 1}
    assert result["root_clusters"] == 2
    assert result["difference"] == 0.5
    assert result["root_cluster_95"] == [0.5, 0.5]


def test_missing_native_cost_remains_unknown():
    cells = {
        "pending": dict(
            artifact="pending",
            attempts=rows(),
            observed=0,
            successes=0,
            unknown=16,
            cost=None,
        )
    }
    result = analysis().cohort(cells)
    assert result["cost"] is None
    assert result["unknown"] == 16


def test_compact_report_retains_attempt_scores_and_root_pairing():
    module = analysis()
    left = rows()
    right = [dict(r, score=1) for r in left]
    report = dict(
        cells={"a": dict(attempts=left)},
        cohorts={"a": dict(attempts=left, successes=0)},
        comparisons={"b-a": module.paired(left, right)},
    )
    result = module.compact(report)
    assert len(result["cells"]["a"]["attempts"]) == 16
    assert result["cells"]["a"]["attempts"][0] == ["a", 4, 0]
    assert result["comparisons"]["b-a"]["root_pairs"] == [
        [0, "a", 8, 0, 8, 0],
        [0, "b", 8, 0, 8, 0],
    ]
    assert "attempts" not in result["cohorts"]["a"]
