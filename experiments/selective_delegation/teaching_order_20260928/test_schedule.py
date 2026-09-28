"""Protect the public-information and original-craft-order boundaries."""

import random

from prepare import choose_index

ACTIONS = [
    {"action": "get_info", "items": ["a"]},
    {"action": "craft", "target_item": "a", "ingredients": {"raw": 1}, "output_count": 1},
    {"action": "get_info", "items": ["b"]},
    {"action": "craft", "target_item": "b", "ingredients": {"raw": 1}, "output_count": 1},
    {"action": "get_info", "items": ["goal"]},
    {"action": "craft", "target_item": "goal", "ingredients": {"a": 1, "b": 1}, "output_count": 1},
    {"action": "finish", "message": "done"},
]


def test_hidden_leaf_query_moves_after_visible_goal_query():
    # Choosing the first gold action would teach an unobserved name.
    assert choose_index(ACTIONS, list(range(7)), {"goal", "raw"}, {"raw": 2}, set()) == 4


def test_stable_frontier_keeps_original_craft_order_and_requires_queried_recipe():
    assert (
        choose_index(ACTIONS, [0, 1, 2, 3, 5, 6], {"goal", "raw", "a", "b"}, {"raw": 2}, {"goal"})
        == 0
    )
    assert (
        choose_index(ACTIONS, [1, 2, 3, 5, 6], {"goal", "raw", "a", "b"}, {"raw": 2}, {"goal", "a"})
        == 1
    )


def test_random_visible_frontier_is_distinct_without_moving_crafts():
    assert (
        choose_index(
            ACTIONS,
            [0, 1, 2, 3, 5, 6],
            {"goal", "raw", "a", "b"},
            {"raw": 2},
            {"goal"},
            random.Random(0),
        )
        == 2
    )
    # Even when b is known, its feasible craft cannot skip the original first craft a.
    assert (
        choose_index(ACTIONS, [0, 1, 3, 5, 6], {"goal", "raw", "a", "b"}, {"raw": 2}, {"goal", "b"})
        == 0
    )


def test_native_inventory_shortage_does_not_select_a_craft():
    assert (
        choose_index(ACTIONS, [1, 2, 3, 5, 6], {"goal", "raw", "a", "b"}, {"raw": 0}, {"goal", "a"})
        == 2
    )
