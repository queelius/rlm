import json
from types import SimpleNamespace

import pytest
import textcraft_bridge as bridge


def test_notebook_preserves_observed_recipes_but_not_stale_inventory_or_model_claims():
    import textcraft_recipe_memory as memory

    observed = dict(
        item="desk",
        can_craft=True,
        is_base=False,
        crafting_depth=1,
        in_inventory=0,
        recipes=[dict(ingredients={"wood": 2}, result_count=1)],
    )
    history = [
        dict(action=dict(action="get_info", items=["desk"]), feedback=[observed]),
        dict(response='{"item":"SECRET"}', feedback="Rejected action"),
        dict(action=dict(action="craft", target_item="other"), feedback="Error"),
    ]
    ledger = memory.recipe_ledger(history)
    assert list(ledger) == ["desk"]
    assert ledger["desk"]["recipes"] == [{"ingredients": {"wood": 2}, "result_count": 1}]
    assert "in_inventory" not in ledger["desk"]
    assert history[0]["feedback"][0]["in_inventory"] == 0


def test_recent_history_keeps_goal_current_inventory_and_old_observed_recipe():
    import textcraft_recipe_memory as memory

    frame = SimpleNamespace(
        targets={"desk": 1},
        initial_inventory={"wood": 2},
        inventory={"wood": 1},
        depth=0,
        max_depth=0,
        budget=bridge.Budget(),
    )
    history = [
        dict(
            action=dict(action="get_info", items=["desk"]),
            feedback=[dict(item="desk", recipes=[dict(ingredients={"wood": 2}, result_count=1)])],
        )
    ]
    history += [dict(action=dict(action="view_inventory"), feedback={"wood": 1}) for _ in range(6)]
    original = bridge.public_prompt
    with memory.installed("ledger_recent4"):
        prompt = bridge.public_prompt(frame, history, goal="Make a desk")
        payload = json.loads(prompt[len(bridge.INSTRUCTION) :])
        assert len(payload["history"]) == 4
        assert "desk" in payload["observed_recipe_notebook"]
        assert payload["current_inventory"] == {"wood": 1}
        assert payload["goal"] == "Make a desk"
    assert bridge.public_prompt is original
    with memory.installed("recent4"):
        payload = json.loads(bridge.public_prompt(frame, history)[len(bridge.INSTRUCTION) :])
        assert len(payload["history"]) == 4 and "observed_recipe_notebook" not in payload
    with pytest.raises(ValueError), memory.installed("unknown"):
        pass


def test_native_lookup_is_notebook_source_and_full_mode_retains_history():
    import textcraft_recipe_memory as memory

    world = bridge.load_world()
    frame = bridge.Frame(world, {}, {}, bridge.Budget(), 0)
    item = next(iter(world.recipes))
    action = dict(action="get_info", items=[item])
    reply = frame.apply(action)
    history = [dict(action=action, feedback=reply)]
    with memory.installed("ledger_full"):
        payload = json.loads(bridge.public_prompt(frame, history)[len(bridge.INSTRUCTION) :])
    assert payload["history"] == history
    assert payload["observed_recipe_notebook"][item]["recipes"] == reply[0]["recipes"]
    with memory.installed("history_full"):
        control = json.loads(bridge.public_prompt(frame, history)[len(bridge.INSTRUCTION) :])
    assert control["history"] == history
    assert "observed_recipe_notebook" not in control
    assert control["memory_description"] == payload["memory_description"]
