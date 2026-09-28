"""Public-only quantity analysis must aggregate shared demand and not use future recipes."""

import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).with_name("analyze.py")
    assert path.exists(), "public inventory analysis not implemented"
    spec = importlib.util.spec_from_file_location("inventory_analysis_fixture", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def info(item, ingredients, count=1):
    return dict(
        item=item, is_base=False, recipes=[dict(ingredients=ingredients, result_count=count)]
    )


def test_shared_child_demand_is_summed_before_batch_rounding():
    a = module()
    known = {
        i["item"]: i
        for i in [
            info("root", {"a": 1, "b": 1}),
            info("a", {"x": 1}),
            info("b", {"x": 1}),
            info("x", {"ore": 1}, 2),
        ]
    }
    result = a.public_needs({"root": 1}, {"ore": 1}, {"ore": 1}, known)
    assert result["completion_feasible"] is True
    assert result["needed"]["x"]["required"] == 2
    assert result["needed"]["x"]["batches"] == 1
    assert result["needed"]["ore"]["required"] == 1
    assert result["ready_prerequisites"] == ["x"]
    assert result["shared_items"] == ["x"]


def test_missing_recipe_is_unknown_until_publicly_returned():
    a = module()
    known = {"root": info("root", {"mid": 2})}
    result = a.public_needs({"root": 1}, {"ore": 1}, {"ore": 1}, known)
    assert result["completion_feasible"] is False
    assert result["unknown_recipe_names"] == ["mid"]
    assert result["frontier_deficits"] == {"mid": 2}
    known["mid"] = info("mid", {"ore": 1}, 2)
    result = a.public_needs({"root": 1}, {"ore": 1}, {"ore": 1}, known)
    assert result["completion_feasible"] is True
    assert result["ready_prerequisites"] == ["mid"]
    assert result["goal_completing_craft_ready"] is False


def test_one_goal_completing_batch_is_visible_without_its_consumed_subrecipes():
    a = module()
    known = {"root": info("root", {"left": 1, "right": 1}, 2)}
    result = a.public_needs({"root": 1}, {}, {"left": 1, "right": 2}, known)
    assert result["completion_feasible"] is True
    assert result["goal_completing_craft_ready"] is True
    assert result["recipe_closure_complete"] is False


def test_repeated_inventory_is_not_mislabeled_as_identical_full_public_state():
    a = module()
    root = dict(action="craft", target_item="root", output_count=1, ingredients={"mid": 2})
    mid = dict(action="craft", target_item="mid", output_count=1, ingredients={"ore": 1})
    history = [
        dict(action=dict(action="get_info", items=["root"]), feedback=[info("root", {"mid": 2})]),
        dict(
            action=root,
            feedback="Error: Insufficient ingredients in inventory: mid: need 2, have 0",
        ),
        dict(action=dict(action="get_info", items=["mid"]), feedback=[info("mid", {"ore": 1})]),
        dict(
            action=root,
            feedback="Error: Insufficient ingredients in inventory: mid: need 2, have 0",
        ),
        dict(action=mid, feedback="Successfully crafted 1 mid(s)"),
        dict(
            action=root,
            feedback="Error: Insufficient ingredients in inventory: mid: need 2, have 1",
        ),
        dict(
            action=root,
            feedback="Error: Insufficient ingredients in inventory: mid: need 2, have 1",
        ),
        dict(action=dict(action="finish", message="done"), feedback="Finished."),
    ]
    node = dict(
        targets={"root": 1},
        initial_inventory={"ore": 1},
        final_inventory={"mid": 1},
        public_history=history,
        call_ids=[f"c{i}" for i in range(8)],
        status="finished",
        native_score=0,
        started=0,
        ended=1,
    )
    result = a.analyze_node(node)
    assert result["counts"]["insufficient_stock_crafts"] == 4
    assert result["counts"]["repeat_insufficient_same_action_and_stock"] == 2
    assert result["counts"]["consecutive_repeat_insufficient_same_action_and_stock"] == 1
    assert result["max_identical_insufficient_run"] == 2


def test_binder_history_contains_executed_action_not_requested_ingredients():
    a = module()
    executed = dict(action="craft", target_item="root", output_count=1, ingredients={"ore": 1})
    requested = dict(executed, ingredients={"wrong_ore": 1})
    node = dict(
        targets={"root": 1},
        initial_inventory={"ore": 1},
        final_inventory={"root": 1},
        public_history=[dict(action=executed, feedback="Successfully crafted 1 root(s)")],
        call_ids=["c0"],
        execution_assists=[
            dict(call_id="c0", requested_action=requested, executed_action=executed)
        ],
        status="finished",
        native_score=1,
        started=0,
        ended=1,
    )
    result = a.analyze_node(node)
    assert result["trace"][0]["requested_action"] == requested
    assert result["trace"][0]["executed_action"] == executed
