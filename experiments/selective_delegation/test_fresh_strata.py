"""Focused subgroup accounting, not a second native-evaluation framework."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("analyze_fresh_strata.py")
    assert path.exists()
    spec = importlib.util.spec_from_file_location("fresh_strata", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_frozen_strata_preserve_every_parent_and_exact_exposure_definition():
    m = module()
    cases = [{"id": str(i), "metadata": {"hops": 2 if i < 32 else 3}} for i in range(64)]
    exposure = {
        "parents_detail": [
            {
                "parent_id": str(i),
                "exact_title_text": {"match_occurrences": int(i < 14)},
                "title": {"shared_unique_titles": 1},
            }
            for i in range(64)
        ]
    }
    strata = m.strata(cases, exposure)
    assert {k: len(v) for k, v in strata.items()} == {
        "all64": 64,
        "two_hop": 32,
        "three_hop": 32,
        "exact_train_document_overlap": 14,
        "no_exact_train_document_overlap": 50,
    }
    assert set(strata["two_hop"]) | set(strata["three_hop"]) == set(strata["all64"])
    assert "63" in strata["no_exact_train_document_overlap"]  # title-only is not exact
    exposure["parents_detail"].pop()
    with pytest.raises(ValueError, match="inventory"):
        m.strata(cases, exposure)


def test_parent_weighting_components_missing_and_protocol_are_separate():
    m = module()
    values = {}
    for parent in ("a", "b", "c"):
        for repeat in range(2):
            for policy in ("planner_sft", "planner_rl"):
                missing = parent == "c" and policy == "planner_sft"
                em = float(policy == "planner_rl")
                values[parent, repeat, policy] = dict(
                    em=em,
                    f1=em,
                    valid=bool(em),
                    missing=missing,
                    status="missing_episode" if missing else "scored",
                )
    values["a", 0, "planner_sft"]["valid"] = True
    result = m.summarize_stratum(
        ["a", "b", "c"],
        values,
        ["planner_sft", "planner_rl"],
        [["a", "b"], ["c"]],
        draws=25,
        seed=1,
    )
    assert result["groups"]["planner_sft"]["planned"] == 6
    assert result["groups"]["planner_sft"]["missing"] == 2
    pair = result["contrasts"]["planner_rl_minus_planner_sft"]
    assert pair["observed_wins"] == {"both_valid": 1, "protocol_involved": 3}
    assert pair["unobserved_pairs"] == 2
    assert pair["em"]["estimate"] == 1  # explicitly labeled lower-bound difference
    assert pair["incomplete"] is True


def test_stratum_keeps_full_panel_transitive_component_connections():
    m = module()
    assert m.restrict_clusters([["a", "bridge", "c"], ["d"]], ["a", "c", "d"]) == [
        ["a", "c"],
        ["d"],
    ]
