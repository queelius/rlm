import importlib.util
import json

import pytest


def public_teacher():
    assert importlib.util.find_spec("prepare_textcraft_public") is not None, (
        "public teacher missing"
    )
    import prepare_textcraft_public

    return prepare_textcraft_public


def recipe(ingredients, count=1):
    return {"is_base": False, "recipes": [{"ingredients": ingredients, "result_count": count}]}


def test_queries_root_then_unseen_needed_prerequisite_not_hidden_recipe():
    teacher = public_teacher()
    assert teacher.next_action({"root": 1}, {"raw": 2}, {"raw": 2}, {}) == {
        "action": "get_info",
        "items": ["root"],
    }
    assert teacher.next_action(
        {"root": 1}, {"raw": 2}, {"raw": 2}, {"root": recipe({"mid": 1})}
    ) == {"action": "get_info", "items": ["mid"]}


def test_shared_demand_and_batch_yield_include_available_stock_once():
    teacher = public_teacher()
    known = {
        "root": recipe({"left": 1, "right": 1}),
        "left": recipe({"shared": 2}),
        "right": recipe({"shared": 3}),
        "shared": recipe({"raw": 1}, 3),
    }
    action = teacher.next_action(
        {"root": 1}, {"raw": 1, "shared": 2}, {"raw": 1, "shared": 2}, known
    )
    assert action == {
        "action": "craft",
        "ingredients": {"raw": 1},
        "target_item": "shared",
        "output_count": 3,
    }


def test_simultaneous_unknown_prerequisites_use_ascending_item_ids():
    teacher = public_teacher()
    assert teacher.next_action({"root": 1}, {}, {}, {"root": recipe({"a": 1, "b": 1})}) == {
        "action": "get_info",
        "items": ["a"],
    }


def test_finish_uses_net_goal_not_absolute_stock_and_cycles_fail():
    teacher = public_teacher()
    assert teacher.next_action({"root": 1}, {"root": 2}, {"root": 3}, {}) == {
        "action": "finish",
        "message": "done",
    }
    assert teacher.next_action({"root": 1}, {"root": 2}, {"root": 2}, {}) == {
        "action": "get_info",
        "items": ["root"],
    }
    with pytest.raises(ValueError, match="cycle"):
        teacher.next_action(
            {"root": 1}, {}, {}, {"root": recipe({"mid": 1}), "mid": recipe({"root": 1})}
        )
    with pytest.raises(ValueError, match="unavailable"):
        teacher.next_action({"raw": 1}, {}, {}, {"raw": {"is_base": True, "recipes": []}})


def test_real_native_teacher_uses_public_projection_and_preserves_capped_failure():
    teacher = public_teacher()
    world = teacher.bridge.load_world()
    native = next(
        rs[0]
        for rs in world.recipes.values()
        if all(world.is_base_item(i) for i in rs[0].ingredients)
    )

    class GuardedTask(dict):
        def __getitem__(self, key):
            assert key in ("id", "goal", "misc"), "teacher attempted host field"
            return super().__getitem__(key)

    class PublicMisc(dict):
        def __getitem__(self, key):
            assert key in ("target_items", "initial_inventory"), "gold/depth access forbidden"
            return super().__getitem__(key)

    class Tokenizer:
        eos_token_id = 999

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2]

        def encode(self, text, **kwargs):
            return [3, 4]

    task = GuardedTask(
        id="fixture-host-id",
        goal="Craft the public target.",
        misc=PublicMisc(
            target_items={native.result_item: 1}, initial_inventory=dict(native.ingredients)
        ),
    )
    receipt, rows = teacher.trajectory(task, world, Tokenizer())
    assert receipt["eligible"] and receipt["native_score"] == 1
    assert [json.loads(r["target"])["action"] for r in rows] == ["get_info", "craft", "finish"]
    assert all("fixture-host-id" not in r["prompt"] for r in rows)
    assert rows[0]["labels"][:2] == [-100, -100] and rows[0]["labels"][-1] == 999
    assert native.result_item in rows[1]["public_recipe_provenance"]
    assert '"recipes"' not in rows[0]["prompt"] and '"recipes"' in rows[1]["prompt"]
    capped, partial = teacher.trajectory(task, world, Tokenizer(), max_calls=1)
    assert capped["status"] == "global_call_cap" and not capped["eligible"] and len(partial) == 1
