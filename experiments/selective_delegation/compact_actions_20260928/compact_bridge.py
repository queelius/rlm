"""Experiment-local compact interface around the unchanged native TextCraft bridge."""

import copy
import json
import sys
from pathlib import Path

LIBRARY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LIBRARY))

import textcraft_bridge as native  # noqa: E402

INSTRUCTION = native.INSTRUCTION.replace(
    "fixed batch sizes: output_count must be a multiple of result_count, and ingredients "
    "must exactly equal the per-batch ingredients times that number of batches. Shared ",
    "fixed batch sizes: output_count must be a multiple of result_count. For craft, provide "
    "only target_item and output_count; ingredients are filled from a unique recipe already "
    "returned by get_info in this context. Unknown or ambiguous recipes and indivisible "
    "counts are rejected. No target, quantity, or inventory repair is performed. Shared ",
).replace(
    '{"action":"craft","ingredients":{"item":2},"target_item":"product","output_count":2}',
    '{"action":"craft","target_item":"product","output_count":2}',
)


def parse_action(text: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result

    action = json.loads(text, object_pairs_hook=unique)
    if not isinstance(action, dict) or action.get("action") != "craft":
        return native.parse_action(text)
    if set(action) - {"note"} != {"action", "target_item", "output_count"}:
        raise ValueError("compact craft requires exact fields without ingredients")
    # Reuse all bounded item/count/note checks without accepting an ingredient field.
    native.parse_action(json.dumps(dict(action, ingredients={"schema_placeholder": 1})))
    return action


def project_action(full_action: dict) -> dict:
    native.parse_action(json.dumps(full_action))
    return {k: copy.deepcopy(v) for k, v in full_action.items() if k != "ingredients"}


def bind_observed(action: dict, observed: dict) -> dict:
    action = parse_action(json.dumps(action))
    if action["action"] != "craft":
        return action
    recipes = observed.get(action["target_item"], {}).get("recipes")
    if not isinstance(recipes, list) or len(recipes) != 1:
        raise ValueError("craft requires one uniquely observed get_info recipe")
    recipe = recipes[0]
    count, ingredients = recipe.get("result_count"), recipe.get("ingredients")
    if type(count) is not int or count <= 0 or not native.quantities(ingredients):
        raise ValueError("observed recipe is not a positive integer recipe")
    if action["output_count"] % count:
        raise ValueError("output_count is not divisible by observed recipe result_count")
    batches = action["output_count"] // count
    executed = dict(action, ingredients={k: v * batches for k, v in ingredients.items()})
    native.parse_action(json.dumps(executed))
    return executed


class Frame(native.Frame):
    def __init__(self, world, inventory, targets, budget, max_depth, depth=0):
        if max_depth != 0 or depth != 0:
            raise ValueError("initial compact-action probe is flat only")
        super().__init__(world, inventory, targets, budget, max_depth, depth)
        self.observed_recipes = {}
        self.execution_assists = []

    def apply(self, action):
        requested = parse_action(json.dumps(action))
        executed = bind_observed(requested, self.observed_recipes)
        # Native code, not this adapter, checks inventory and performs consumption.
        reply = super().apply(executed)
        self.execution_assists.append(
            {
                "requested_action": copy.deepcopy(requested),
                "executed_action": copy.deepcopy(executed),
                "binding_source": "past native get_info only"
                if requested["action"] == "craft"
                else "unchanged non-craft",
            }
        )
        if requested["action"] == "get_info" and isinstance(reply, list):
            for info in reply:
                if info["item"] not in requested["items"]:
                    raise ValueError("native recipe reply includes an unqueried item")
                self.observed_recipes[info["item"]] = {
                    "recipes": copy.deepcopy(info["recipes"]),
                }
        return reply


def public_prompt(frame, history, context="", goal=None):
    rendered = native.public_prompt(frame, history, context=context, goal=goal)
    if not rendered.startswith(native.INSTRUCTION):
        raise ValueError("original public prompt contract changed")
    return INSTRUCTION + rendered[len(native.INSTRUCTION) :]


def initial_prompt(task, policy):
    if policy != "flat":
        raise ValueError("initial compact-action probe is flat only")
    rendered = native.initial_prompt(task, policy)
    return INSTRUCTION + rendered[len(native.INSTRUCTION) :]


def __getattr__(name):
    return getattr(native, name)
