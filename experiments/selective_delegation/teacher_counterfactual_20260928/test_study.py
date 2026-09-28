"""Focused safeguards: protect root evidence and require exact prompt collisions."""

from types import SimpleNamespace

import pytest
import study


def recipe(depth, ingredient):
    return SimpleNamespace(depth=depth, ingredients={ingredient: 1}, result_count=1)


def test_hybrid_replaces_only_unqueried_lower_same_tier_recipes():
    first = SimpleNamespace(
        recipes={
            "root": [recipe(3, "low")],
            "low": [recipe(1, "base_a")],
            "peer": [recipe(3, "low")],
        },
        item_depths={"root": 3, "low": 1, "peer": 3},
    )
    second = SimpleNamespace(
        recipes={
            "root": [recipe(3, "other")],
            "low": [recipe(1, "base_b")],
            "peer": [recipe(3, "other")],
        },
        item_depths=dict(first.item_depths),
    )
    hybrid, changed = study.hybrid_world(first, second, {"root": 1})
    assert changed == ["low"]
    assert hybrid.recipes["root"][0].ingredients == {"low": 1}
    assert hybrid.recipes["peer"][0].ingredients == {"low": 1}
    assert hybrid.recipes["low"][0].ingredients == {"base_b": 1}
    assert first.recipes["low"][0].ingredients == {"base_a": 1}
    second.item_depths["low"] = 2
    with pytest.raises(ValueError, match="tier"):
        study.hybrid_world(first, second, {"root": 1})


def test_collision_requires_both_complete_bytes_and_input_ids():
    study.exact_collision(["same", "same"], [[1, 2], [1, 2]])
    with pytest.raises(ValueError, match="prompt"):
        study.exact_collision(["same", "same "], [[1, 2], [1, 2]])
    with pytest.raises(ValueError, match="token"):
        study.exact_collision(["same", "same"], [[1, 2], [1, 3]])


def test_entropy_counts_only_actual_exact_groups_and_keeps_singletons_separate():
    records = [("x", "a"), ("x", "b"), ("y", "c"), ("y", "c"), ("z", "d")]
    result = study.label_statistics(records)
    assert result["conditional_entropy_bits"] == pytest.approx(0.4)
    assert result["maximum_empirical_exact_label_accuracy"] == pytest.approx(0.8)
    assert result["conflicting_groups"] == 1
    assert result["singleton_groups"] == 1
