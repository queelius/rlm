"""Focused schedule, identity and paired-evidence seams."""

import importlib.util
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).with_name(name + ".py")
    assert path.exists(), "replication component not implemented"
    spec = importlib.util.spec_from_file_location("test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_schedule_covers_both_orderings_and_two_orthogonal_slices():
    schedule = load("replication").schedule()
    assert len(schedule) == 8
    assert {(c["seed"], c["panel"], c["world"]) for c in schedule} == {
        (2026092291, 0, 42),
        (2026092291, 0, 50),
        (2026092208, 1, 42),
        (2026092208, 1, 50),
    }
    assert {c["mode"] for c in schedule} == {"stable_visible", "random_visible"}


def test_baseline_selection_uses_saved_fit_seed_convention_not_reference_teacher():
    study = load("replication")
    assert study.baseline(2026092291, 0, 42, "known").name == (
        "textcraft-breadth-p00-w42-s2291-corrected-raw-001"
    )
    assert study.baseline(2026092208, 1, 50, "discovery").name == (
        "textcraft-breadth-p01-w50-soriginal-raw-001"
    )


def test_first_seed_uses_existing_actor_second_seed_uses_new_training_root(tmp_path):
    study = load("replication")
    assert study.training("stable_visible", 2026092208, tmp_path) == (
        study.ORIGINAL / "train-stable_visible-seed2026092208"
    )
    assert study.training("random_visible", 2026092291, tmp_path) == (
        tmp_path / "train-random_visible-seed2026092291"
    )


def test_endpoint_gate_rejects_partial_or_mismatched_training():
    study = load("replication")
    valid = dict(
        path="actor/checkpoint-0023",
        state={"step": 23, "epoch": 1, "cursor": 0},
        training_plan_sha256="plan",
        training_rows_sha256="rows",
    )
    expected = dict(checkpoint="actor/checkpoint-0023", plan_sha256="plan", rows_sha256="rows")
    study.check_binding(valid, expected)
    for field, value in (
        ("state", {"step": 22, "epoch": 0, "cursor": 352}),
        ("training_plan_sha256", "other"),
        ("path", "wrong/checkpoint-0023"),
    ):
        with pytest.raises(ValueError):
            study.check_binding(dict(valid, **{field: value}), expected)


def test_clustered_pairing_joins_task_world_seed_not_row_order():
    report = load("report")
    left = [
        dict(task_id=t, world=w, seed=s, score=0)
        for t in ("a", "b")
        for w in (42, 50)
        for s in (4, 5)
    ]
    right = [dict(r, score=int(r["task_id"] == "a")) for r in reversed(left)]
    result = report.pair(left, right)
    assert (
        result["planned_pairs"],
        result["task_identities"],
        result["wins"],
        result["losses"],
        result["difference"],
    ) == (8, 2, 4, 0, 0.5)
    assert result["task_cluster_95"] == [0, 1]


def test_partial_pair_retains_denominator_and_no_point_estimate():
    report = load("report")
    left = [dict(task_id="a", world=42, seed=s, score=0) for s in (4, 5)]
    right = [dict(left[0], score=1), dict(left[1], score=None)]
    result = report.pair(left, right)
    assert result["planned_pairs"] == 2
    assert result["unknown_pairs"] == 1
    assert result["difference"] is None
    assert result["difference_bounds"] == [0, 1]


def test_pair_rejects_different_task_seed_slots():
    report = load("report")
    with pytest.raises(ValueError, match="slots"):
        report.pair(
            [dict(task_id="a", world=42, seed=4, score=0)],
            [dict(task_id="b", world=42, seed=4, score=1)],
        )
