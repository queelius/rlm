"""Tiny credit math fixtures, including unknown rather than scientific zero."""

import analyze_training_execution_credit as analysis


def test_action_dependent_credit_can_reverse_terminal_advantage():
    result = analysis.group_credit([1, 0, 1, 0], [1, 0, 0, 1])
    assert result["execution_benefit"] == [0, 0, 1, -1]
    assert result["alignment"] == ["terminal_only", "terminal_only", "agree", "agree"]
    reversal = analysis.group_credit([1, 1, 1, 0], [1, 1, 0, 0])
    assert reversal["alignment"][:2] == ["opposite", "opposite"]


def test_unknown_is_not_zero_and_ties_are_explicit():
    assert analysis.group_credit([1, 0, 1, 0], [None, 0, 1, 0]) is None
    result = analysis.group_credit([1, 1, 1, 1], [1, 1, 1, 1])
    assert result["alignment"] == ["both_zero"] * 4
    assert result["terminal_advantages"] == [0] * 4


def test_empty_planned_inventory_keeps_unknowns_and_full_denominator(tmp_path):
    import pytest

    collector = analysis.collector
    if not (collector.ROOT / "frozen-execution-001/PLAN.json").exists():
        pytest.skip("external completed source unavailable")
    plan, _, _ = collector.prepare(collector.ROOT, tmp_path, 1 / 3)
    collector.runtime.immutable(tmp_path / "PLAN.json", plan)
    report = analysis.analyze(tmp_path)
    assert report["planned_attempts"] == len(report["missing_episodes"]) == 320
    assert report["observed"] == report["complete_groups"] == 0
    assert report["unavailable_groups"] == 80
    assert report["execution_minus_plan_only"] is None
    assert report["new_physical_cost"]["calls"] == 0
    assert report["method"]["parent_count"] == 16
    assert report["method"]["component_count"] == 15
    assert "independent_parent_count" not in report["method"]


def test_unavailable_control_is_unknown_without_output_tokens():
    original = {"output_token_ids": [7], "text": "answer"}
    old = {"valid": True, "correct": True, "f1": 1.0}
    result = analysis.control_comparison("control", {"available": False}, original, {}, old)
    assert result["observed"] is False
    assert result["same_output_tokens"] is None
    assert result["same_text"] is None
    assert result["same_grade"] is None
    assert sum(c["same_output_tokens"] is False for c in [result]) == 0
