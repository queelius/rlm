"""Focused selection, matching and public-state seam tests; no model/GPU."""

import json

import prepare
import pytest


def row(step, action, tokens=10, task="A"):
    return {
        "step": step,
        "task_id": task,
        "target": json.dumps({"action": action}),
        "target_tokens": tokens,
        "row_id": f"{task}-{step}",
    }


def test_first_action_craft_is_not_selected_twice():
    rows = [row(i, a) for i, a in enumerate(["craft", "craft", "craft", "craft", "finish"])]
    assert [r["step"] for r in prepare.select_rows(rows)] == [0, 1, 3, 4]


def test_inadequate_distinct_crafts_reject_without_substitution():
    with pytest.raises(ValueError, match="distinct"):
        prepare.select_rows([row(0, "get_info"), row(1, "craft"), row(2, "finish")])


def test_matching_is_task_and_type_bound_no_replacement_with_earliest_tie():
    recovery = [row(5, "craft", 12), row(7, "craft", 12), row(9, "finish", 8)]
    clean = [
        row(0, "craft", 12, "B"),
        row(1, "craft", 10),
        row(2, "craft", 14),
        row(3, "finish", 8),
    ]
    matched, pairs = prepare.match_rows(recovery, clean)
    assert [r["step"] for r in matched] == [1, 2, 3]
    assert [p["target_token_difference"] for p in pairs] == [2, 2, 0]


def test_restore_retains_original_baseline_and_spent_budget():
    state = {
        "target_items": {"t9_i3": 2},
        "inventory_at_task_start": {"t9_i3": 5},
        "current_inventory": {"t9_i3": 6},
        "agent_depth": 0,
        "max_agent_depth": 0,
        "global_calls_remaining": 89,
        "global_output_tokens_remaining": 8161,
        "delegated_context": "",
    }
    frame = prepare.restore_frame(state, prepare.bridge.load_world())
    assert frame.initial_inventory == {"t9_i3": 5}
    assert frame.inventory == {"t9_i3": 6}
    assert (frame.budget.calls, frame.budget.output_tokens) == (7, 31)
    frame.apply({"action": "finish", "message": "test"})
    assert frame.score()[0] == 0


def test_teacher_recipe_map_discards_native_metadata():
    history = [
        {
            "action": {"action": "get_info", "items": ["x"]},
            "feedback": [
                {
                    "item": "x",
                    "is_base": False,
                    "crafting_depth": 999,
                    "can_craft": True,
                    "in_inventory": 999,
                    "recipes": [{"ingredients": {"y": 2}, "result_count": 3}],
                }
            ],
        }
    ]
    assert prepare.public_recipes(history) == {
        "x": {"is_base": False, "recipes": [{"ingredients": {"y": 2}, "result_count": 3}]}
    }
