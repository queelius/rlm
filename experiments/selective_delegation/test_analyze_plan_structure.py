"""Small fixtures for executor-compatible graphs and scalar RLOO credit accounting."""

import pytest


def test_structure_uses_actual_binder_syntax_and_separates_invalid_references():
    import analyze_plan_structure as analysis

    first = analysis.structure({"subquestions": ["Who?", "Where is #1 or #01?"]})
    second = analysis.structure({"subquestions": ["Different entity?", "When did #1 work?"]})
    assert first["canonical"] == second["canonical"] == [2, [[], [1]]]
    assert first["valid"] and first["reference_occurrences"] == 2
    literal = analysis.structure({"subquestions": ["#word #-1 #1st"]})
    assert literal["valid"] and literal["canonical"] == [1, [[]]]
    bad = analysis.structure({"subquestions": ["#0 #1 #2 #9", "ordinary"]})
    assert not bad["valid"]
    assert [r["kind"] for r in bad["invalid_references"]] == [
        "invalid_index",
        "self",
        "forward",
        "forward",
    ]
    assert sum(r["outside_plan"] for r in bad["invalid_references"]) == 2


def test_mixed_groups_distinguish_text_graph_and_scalar_advantage_mass():
    import analyze_plan_structure as analysis

    def row(candidate, reward, question, independent=False):
        return {
            "candidate": candidate,
            "reward": reward,
            "plan_valid": True,
            "plan": {"subquestions": [question, "Independent?" if independent else "#1?"]},
            "root_tokens": 10,
            "status": "scored",
        }

    same = analysis.summarize_group([row(c, int(c == 0), str(c)) for c in range(4)])
    assert same["category"] == "same_valid_structure"
    assert same["distinct_valid_text_lists"] == 4
    assert same["nonzero_advantages"] == 4
    assert same["absolute_advantage_mass"] == pytest.approx(2)
    varied = analysis.summarize_group([row(c, int(c < 2), str(c), c == 0) for c in range(4)])
    assert varied["category"] == "multiple_valid_structures"
    assert varied["step_count_varies"] is False
    assert varied["same_length_dependency_varies"] is True
    assert varied["absolute_advantage_mass"] == pytest.approx(8 / 3)
    assert varied["absolute_advantage_token_mass"] == pytest.approx(80 / 3)
    mixed_invalid = [row(c, int(c == 0), str(c)) for c in range(4)]
    mixed_invalid[3]["plan"] = {"subquestions": ["#1?"]}
    result = analysis.summarize_group(mixed_invalid)
    assert result["category"] == "same_valid_structure"
    assert result["invalid_structure_plans"] == 1
