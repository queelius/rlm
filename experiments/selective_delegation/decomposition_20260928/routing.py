"""Oracle-free, one-boundary decomposition admission policy."""

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

LIBRARY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LIBRARY))

import textcraft_bridge as native  # noqa: E402

MARKER = "\nPUBLIC_DECOMPOSITION_CONTROL\n"
CONTEXT = "Craft this prerequisite using public recipes; finish when its net target is met."


def _recipe(observed, item):
    recipes = observed.get(item, {}).get("recipes", [])
    if len(recipes) != 1:
        return None
    recipe = recipes[0]
    if type(recipe.get("result_count")) is not int or recipe["result_count"] <= 0:
        return None
    return recipe if native.quantities(recipe.get("ingredients")) else None


def choose(targets, initial, current, observed, calls_left, tokens_left, depth, delegated):
    result = {"query": None, "candidate": None, "adaptive": False, "reason": "no_candidate"}
    if depth > 0 or delegated:
        return {**result, "reason": "one_boundary_done"}
    demands = Counter()
    for item, quantity in sorted(targets.items()):
        shortage = max(0, initial.get(item, 0) + quantity - current.get(item, 0))
        if not shortage:
            continue
        if item not in observed:
            return {
                **result,
                "query": {"action": "get_info", "items": [item]},
                "reason": "common_public_discovery",
            }
        recipe = _recipe(observed, item)
        if recipe:
            batches = (shortage + recipe["result_count"] - 1) // recipe["result_count"]
            demands.update({k: v * batches for k, v in recipe["ingredients"].items()})
    missing = {k: v - current.get(k, 0) for k, v in demands.items() if v > current.get(k, 0)}
    for item in sorted(missing):
        if item not in observed:
            return {
                **result,
                "query": {"action": "get_info", "items": [item]},
                "reason": "common_public_discovery",
            }
    candidates = []
    for item, shortage in sorted(missing.items()):
        recipe = _recipe(observed, item)
        if not recipe:
            continue
        batches = (shortage + recipe["result_count"] - 1) // recipe["result_count"]
        needed = {k: v * batches for k, v in recipe["ingredients"].items()}
        unresolved = sorted(k for k, v in needed.items() if current.get(k, 0) < v)
        coupled = []
        for sibling, sibling_shortage in missing.items():
            if sibling == item:
                continue
            sibling_recipe = _recipe(observed, sibling)
            if not sibling_recipe:
                continue
            sibling_batches = (
                sibling_shortage + sibling_recipe["result_count"] - 1
            ) // sibling_recipe["result_count"]
            for ingredient in set(needed) & set(sibling_recipe["ingredients"]):
                total = (
                    needed[ingredient] + sibling_batches * sibling_recipe["ingredients"][ingredient]
                )
                if current.get(ingredient, 0) < total:
                    coupled.append(ingredient)

        def observed_depth(node, seen):
            if node in seen:
                return 0
            known = _recipe(observed, node)
            return (
                0
                if known is None
                else 1 + max(observed_depth(child, seen | {node}) for child in known["ingredients"])
            )

        candidates.append(
            {
                "targets": {item: shortage},
                "needed_inputs": needed,
                "unresolved_inputs": unresolved,
                "observed_shared_shortages": sorted(set(coupled)),
                "observed_depth_lower_bound": observed_depth(item, set()),
            }
        )
    if not candidates:
        return result
    candidate = candidates[0]  # Same lexical public candidate in every arm.
    result["candidate"] = candidate
    candidate["minimum_calls_remaining"] = (
        8 + 4 * candidate["observed_depth_lower_bound"] + 2 * len(candidate["unresolved_inputs"])
    )
    candidate["minimum_tokens_remaining"] = max(
        1024, 256 * (2 + candidate["observed_depth_lower_bound"])
    )
    if (
        calls_left < candidate["minimum_calls_remaining"]
        or tokens_left < candidate["minimum_tokens_remaining"]
    ):
        result["reason"] = "reserve_root_budget"
    elif not candidate["unresolved_inputs"]:
        result["reason"] = "branch_already_craftable"
    elif len(candidate["unresolved_inputs"]) < 2:
        result["reason"] = "insufficient_observed_branching"
    elif candidate["observed_shared_shortages"]:
        result["reason"] = "known_shared_stock_coupling"
    else:
        result.update(adaptive=True, reason="observed_branching_with_budget")
    return result


def make_bridge(mode):
    if mode not in ("flat", "fixed", "adaptive"):
        raise ValueError("undeclared decomposition policy")

    class Frame(native.Frame):
        def __init__(self, world, inventory, targets, budget, max_depth, depth=0):
            super().__init__(world, inventory, targets, budget, 0 if mode == "flat" else 1, depth)
            self.observed = {}
            self.delegated = False
            self.refusals = Counter()
            self.announced = None

        def routing(self):
            decision = choose(
                self.targets,
                self.initial_inventory,
                self.inventory,
                self.observed,
                self.budget.max_calls - self.budget.calls,
                self.budget.max_output_tokens - self.budget.output_tokens,
                self.depth,
                self.delegated,
            )
            required = decision["query"]
            if (
                required is None
                and decision["candidate"] is not None
                and (mode == "fixed" or mode == "adaptive" and decision["adaptive"])
            ):
                required = {
                    "action": "delegate",
                    "targets": decision["candidate"]["targets"],
                    "context": CONTEXT,
                }
            return {
                "policy": mode,
                "decision": decision,
                "required_next_action": required,
                "refusals": dict(self.refusals),
                "abort_requested": sum(self.refusals.values()) >= 2,
                "stock_reservation": "none; native shared inventory and per-node snapshots",
            }

        def apply(self, action):
            action = native.parse_action(json.dumps(action))
            # Native collector charges the response before apply. Enforce the decision
            # actually announced before that charge, not a changed budget threshold.
            required = (self.announced or self.routing())["required_next_action"]
            self.announced = None
            if required is not None:
                field = "targets" if required["action"] == "delegate" else "items"
                if action["action"] != required["action"] or action.get(field) != required[field]:
                    self.refusals[required["action"]] += 1
                    raise ValueError("admission instruction requires " + json.dumps(required))
            elif action["action"] == "delegate":
                raise ValueError("no delegation admitted at this public state/depth")
            reply = super().apply(action)
            if action["action"] == "get_info" and isinstance(reply, list):
                for info in reply:
                    # Intentionally omit native crafting_depth/can_craft/in_inventory metadata.
                    self.observed[info["item"]] = {
                        "recipes": [
                            {
                                "ingredients": copy.deepcopy(r["ingredients"]),
                                "result_count": r["result_count"],
                            }
                            for r in info["recipes"]
                        ]
                    }
            return reply

        def delegate(self, targets):
            if self.depth >= self.max_depth or self.delegated:
                raise ValueError("one public helper boundary only")
            self.delegated = True
            return Frame(
                self.world,
                self.inventory,
                targets,
                self.budget,
                self.max_depth,
                depth=self.depth + 1,
            )

    def public_prompt(frame, history, context="", goal=None):
        guidance = (
            "\nAdmission probe: required_next_action, when non-null, is mandatory. Emit its "
            "one JSON schema next; delegate context may be paraphrased but targets/counts must "
            "match. All actions and helpers consume the same global budget. When null, solve "
            "directly. Candidate facts use only previous public recipes and stock, not gold. "
            "Observed depth is a lower bound from partial knowledge, not native recipe depth."
        )
        control = frame.routing()
        if control["required_next_action"] is not None:
            rejections = sum(
                isinstance(step.get("feedback"), str)
                and step["feedback"].startswith("Rejected action:")
                for step in history
            )
            control["instruction_rejections"] = rejections
            control["abort_requested"] = control["abort_requested"] or rejections >= 2
        frame.announced = control
        return (
            native.public_prompt(frame, history, context=context, goal=goal)
            + guidance
            + MARKER
            + json.dumps(control)
        )

    namespace = {name: getattr(native, name) for name in dir(native) if not name.startswith("__")}
    namespace.update(Frame=Frame, public_prompt=public_prompt, __file__=__file__)
    return SimpleNamespace(**namespace)
