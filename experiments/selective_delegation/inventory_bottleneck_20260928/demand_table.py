"""Opt-in public arithmetic display, never a hidden-recipe or action-selection interface."""

from __future__ import annotations

import copy
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "inventory_public_math_20260928", Path(__file__).with_name("analyze.py")
)
math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(math)

MODES = ("demand", "masked")
DERIVED = ("remaining_demand", "stock_deficit", "new_batches", "frontier_deficit_kind")
DESCRIPTION = (
    "The public_quantity_table repeats only public inventory and earlier get_info facts. "
    "observed_recipes is null until queried; an empty list is an observed absence of recipes. "
    "Numeric remaining_demand, stock_deficit and new_batches, when supplied, are host-computed "
    "bookkeeping for the remaining root goal using ONLY already observed unique recipes. "
    "Shared ingredient demands are summed before rounding recipe batches. Demand includes "
    "available stock; deficit subtracts it. Unknown recipes stop the calculation, so values "
    "can change as more recipes are observed. Null computed values are unavailable, not zero. "
    "frontier_deficit_kind labels an unmet requirement without a unique observed recipe: "
    "known_base only follows a public is_base reply; other labels retain recipe uncertainty. "
    "These columns do not recommend an action. Rows are alphabetical, not an execution order. "
    "can_craft describes recipe existence, not present inventory sufficiency. "
    "matched_length_padding has no task meaning; ignore it."
)


def pair_payloads(payload):
    observed = {}
    for step in payload["history"]:
        action, feedback = step.get("action", {}), step.get("feedback")
        if action.get("action") != "get_info" or not isinstance(feedback, list):
            continue
        for reply in feedback:
            if reply["item"] not in action["items"]:
                raise ValueError("unrequested public recipe feedback")
            observed[reply["item"]] = copy.deepcopy(reply)
    targets, initial, stock = (
        payload["target_items"],
        payload["inventory_at_task_start"],
        payload["current_inventory"],
    )
    needs = math.public_needs(targets, initial, stock, observed)
    names = set(targets) | set(initial) | set(stock) | set(observed)
    for reply in observed.values():
        for recipe in reply.get("recipes", []):
            names.update(recipe["ingredients"])
    table = []
    for name in sorted(names):
        known, need = observed.get(name), needs["needed"].get(name)
        frontier_kind = None
        if name in needs["frontier_deficits"]:
            if known is not None and known.get("is_base") is True:
                frontier_kind = "known_base"
            elif known is None:
                frontier_kind = "unqueried_initial_stock" if name in initial else "unqueried_recipe"
            elif len(known.get("recipes", [])) > 1:
                frontier_kind = "ambiguous_recipes"
            else:
                frontier_kind = "observed_no_recipe"
        table.append(
            dict(
                item=name,
                stock=stock.get(name, 0),
                observed_is_base=known.get("is_base") if known is not None else None,
                observed_can_craft=known.get("can_craft") if known is not None else None,
                observed_recipes=known.get("recipes") if known is not None else None,
                remaining_demand=need["required"] if need is not None else None,
                stock_deficit=need["deficit"] if need is not None else None,
                new_batches=need["batches"] if need is not None else None,
                frontier_deficit_kind=frontier_kind,
            )
        )
    demand = copy.deepcopy(payload)
    demand.update(
        quantity_table_description=DESCRIPTION,
        public_quantity_table=table,
        matched_length_padding="",
    )
    masked = copy.deepcopy(demand)
    for row in masked["public_quantity_table"]:
        for key in DERIVED:
            row[key] = None
    return demand, masked


def input_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_pair(instruction, payload, tokenizer):
    values = dict(zip(MODES, pair_payloads(payload), strict=True))
    padding = dict.fromkeys(MODES, 0)
    for _ in range(16):
        prompts = {
            mode: instruction + json.dumps(value, ensure_ascii=False)
            for mode, value in values.items()
        }
        lengths = {mode: len(input_ids(tokenizer, prompt)) for mode, prompt in prompts.items()}
        target = max(lengths.values())
        if len(set(lengths.values())) == 1:
            return prompts, lengths
        for mode in MODES:
            padding[mode] += target - lengths[mode]
            values[mode]["matched_length_padding"] = " x" * padding[mode]
    raise ValueError("could not exactly match state-conditional prompt token lengths")


@contextmanager
def installed(bridge, mode, tokenizer):
    if mode not in MODES:
        raise ValueError("unknown quantity-table mode")
    original = bridge.public_prompt

    def render(frame, history, context="", goal=None):
        if frame.max_depth != 0 or frame.depth != 0:
            raise ValueError("this fixed pilot permits only flat root episodes")
        public = original(frame, history, context=context, goal=goal)
        if not public.startswith(bridge.INSTRUCTION):
            raise ValueError("native public instruction prefix differs")
        payload = json.loads(public[len(bridge.INSTRUCTION) :])
        prompts, _ = render_pair(bridge.INSTRUCTION, payload, tokenizer)
        return prompts[mode]

    bridge.public_prompt = render
    try:
        yield
    finally:
        bridge.public_prompt = original
