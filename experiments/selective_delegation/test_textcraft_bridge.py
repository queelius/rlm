import ast
import hashlib
import json
from pathlib import Path

import pytest

INPUTS = Path(
    "/project/alex_phd/runs/rlm-research-r4/sidecars/"
    "selective-delegation-20260921/textcraft-inputs-001"
)


def test_actual_frozen_gold_native_quantity_finish_and_method_ast():
    import textcraft_bridge as bridge

    selected = json.loads((INPUTS / "SELECTION.json").read_text())
    tasks = [json.loads(x) for x in (INPUTS / "tasks.jsonl").read_text().splitlines()]
    assert selected["selected_before_replay"]
    world = bridge.load_world()
    for task in tasks:
        result = bridge.replay_gold(task, world)
        assert result["native_score"] == 1, result
    native = bridge.native_functions()
    source = ast.parse(bridge.ENV.read_text())
    method = next(
        n
        for c in source.body
        if isinstance(c, ast.ClassDef) and c.name == "TextCraftCodeExecutor"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "craft"
    )
    expected = hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()
    assert bridge.trusted_provenance()["method_ast_sha256"]["craft"] == expected
    assert callable(native["craft"])
    # Actual official recipe producing >1: wrong repetitions are not valid output quantity.
    recipe = next(
        r[0]
        for r in world.recipes.values()
        if r[0].result_count > 1 and all(world.is_base_item(i) for i in r[0].ingredients)
    )
    frame = bridge.Frame(
        world, dict(recipe.ingredients), {recipe.result_item: 1}, bridge.Budget(), max_depth=2
    )
    before = dict(frame.inventory)
    response = frame.apply(
        {
            "action": "craft",
            "ingredients": recipe.ingredients,
            "target_item": recipe.result_item,
            "output_count": 1,
        }
    )
    assert response.startswith("Error:") and frame.inventory == before
    frame.apply(
        {
            "action": "craft",
            "ingredients": recipe.ingredients,
            "target_item": recipe.result_item,
            "output_count": recipe.result_count,
        }
    )
    assert frame.score()[0] == 0  # target alone is not finish
    frame.apply({"action": "finish", "message": ""})
    assert frame.score()[0] == 0
    frame.finished, frame.message = False, None
    frame.apply({"action": "finish", "message": "done"})
    assert frame.score()[0] == 1
    preexisting = bridge.Frame(
        world, {recipe.result_item: 3}, {recipe.result_item: 1}, bridge.Budget(), max_depth=2
    )
    preexisting.apply({"action": "finish", "message": "done"})
    assert preexisting.score()[0] == 0  # exact net-growth rule


def test_nested_frames_share_global_budget_inventory_and_strict_json():
    import textcraft_bridge as bridge

    root = bridge.Frame(bridge.load_world(), {}, {"m0_i1": 1}, bridge.Budget(3, 10), max_depth=2)
    child = root.delegate({"m0_i1": 1})
    grandchild = child.delegate({"m0_i1": 1})
    assert child.inventory is root.inventory and grandchild.budget is root.budget
    with pytest.raises(ValueError, match="depth"):
        grandchild.delegate({"m0_i1": 1})
    child.budget.charge(2)
    grandchild.budget.charge(3)
    root.budget.charge(5)
    assert (root.budget.calls, root.budget.output_tokens) == (3, 10)
    with pytest.raises(bridge.BudgetExceeded):
        child.budget.reserve()
    with pytest.raises(ValueError):
        bridge.parse_action(
            '{"action":"craft","ingredients":{"m0_ore":true},'
            '"target_item":"m0_i1","output_count":2}'
        )
    with pytest.raises(ValueError):
        bridge.parse_action('{"action":"view_inventory","action":"finish"}')
    with pytest.raises(ValueError):
        bridge.parse_action('__import__("os").system("anything")')
    assert bridge.parse_action('{"action":"view_inventory"}') == {"action": "view_inventory"}
    for invalid in (
        '{"action":[]}',
        '{"action":"craft","ingredients":{"x":1},"target_item":[],"output_count":1}',
    ):
        with pytest.raises(ValueError):
            bridge.parse_action(invalid)


def test_public_prompt_never_uses_host_gold_or_depth_annotations():
    import textcraft_bridge as bridge

    task = {
        "goal": "Craft one thing",
        "id": "HOST_SECRET",
        "misc": {
            "initial_inventory": {"m0_ore": 2},
            "target_items": {"m0_i1": 1},
            "gold_trajectory": "SECRET_GOLD",
            "max_depth": "SECRET_DEPTH",
        },
    }
    prompt = bridge.initial_prompt(task, "flat")
    assert all(secret not in prompt for secret in ("HOST_SECRET", "SECRET_GOLD", "SECRET_DEPTH"))
    assert "m0_ore" in prompt and "m0_i1" in prompt
    recursive = bridge.initial_prompt(task, "recursive")
    assert '"max_agent_depth": 2' in recursive and '"max_agent_depth": 0' in prompt
