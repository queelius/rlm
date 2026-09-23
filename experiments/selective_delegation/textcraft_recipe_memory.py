"""Opt-in public-observation notebook; no environment or action changes."""

import copy
import json
from contextlib import contextmanager

import textcraft_bridge as bridge

MODES = ("history_full", "ledger_full", "ledger_recent4", "recent4")


def recipe_ledger(history):
    ledger = {}
    for row in history:
        action, feedback = row.get("action", {}), row.get("feedback")
        if action.get("action") != "get_info" or not isinstance(feedback, list):
            continue
        for info in feedback:
            if not isinstance(info, dict) or info.get("item") not in action.get("items", []):
                continue
            ledger[info["item"]] = {
                key: copy.deepcopy(info[key])
                for key in ("item", "can_craft", "is_base", "crafting_depth", "recipes")
                if key in info
            }
    return ledger


@contextmanager
def installed(mode):
    if mode not in MODES:
        raise ValueError("unknown public-memory condition")
    original = bridge.public_prompt

    def render(frame, history, context="", goal=None):
        full = original(frame, history, context=context, goal=goal)
        if not full.startswith(bridge.INSTRUCTION):
            raise ValueError("unexpected native public prompt")
        payload = json.loads(full[len(bridge.INSTRUCTION) :])
        if mode in ("ledger_recent4", "recent4"):
            payload["history"] = payload["history"][-4:]
        if mode in ("ledger_full", "ledger_recent4"):
            payload["observed_recipe_notebook"] = recipe_ledger(history)
        payload["memory_description"] = (
            "Any observed_recipe_notebook contains static facts from earlier get_info replies. "
            "Current quantities are in current_inventory, not earlier replies. "
            "can_craft means a recipe exists, not that ingredients are available."
        )
        return bridge.INSTRUCTION + json.dumps(payload, ensure_ascii=False)

    bridge.public_prompt = render
    try:
        yield
    finally:
        bridge.public_prompt = original
