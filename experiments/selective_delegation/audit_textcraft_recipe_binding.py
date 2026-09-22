"""Retrospective public-only immediate recipe binding; not a policy rollout."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
AUDIT_SHA = "b0f97a7b98ce572e87d583bb738010417006f61d669af84b05724ad74d2d3c9d"
CATEGORIES = (
    "Insufficient ingredients",
    "Wrong amount of",
    "not divisible",
    "Missing required ingredient",
    "Extra ingredients not required",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    audit_path = ROOT / "analysis-textcraft-train-readiness-002.json"
    if sha(audit_path) != AUDIT_SHA:
        raise ValueError("fixed completed native audit changed")
    audit = json.loads(audit_path.read_text())
    output = ROOT / "textcraft-train-readiness-001"
    pins = {str(audit_path): AUDIT_SHA}
    counts, categories, episodes = Counter(), defaultdict(Counter), defaultdict(set)
    records, initial_states = [], {}
    for path in sorted((output / "nodes").glob("*.json")):
        digest = sha(path)
        if audit["native_audit"]["sha256"].get(str(path)) != digest:
            raise ValueError("node differs from independently replayed native receipt")
        pins[str(path)] = digest
        node = json.loads(path.read_text())
        if node["depth"] != 0 or len(node["public_history"]) != len(node["call_ids"]):
            raise ValueError("fixed flat complete public history required")
        inventory = dict(node["initial_inventory"])
        initial_states[path.name] = inventory.copy()
        recipes = {}
        for index, history in enumerate(node["public_history"]):
            action, feedback = history["action"], history["feedback"]
            call_id = node["call_ids"][index]
            if action["action"] == "get_info" and isinstance(feedback, list):
                for item in feedback:
                    recipes[item["item"]] = dict(
                        recipes=item["recipes"],
                        call_id=call_id,
                        history_index=index,
                        node=str(path),
                        node_sha256=digest,
                    )
            if action["action"] != "craft":
                continue
            counts["craft_calls"] += 1
            if isinstance(feedback, str) and feedback.startswith("Successfully"):
                # Update from ACTUAL successful action only; no hypothetical propagation.
                for item, quantity in action["ingredients"].items():
                    inventory[item] -= quantity
                    if inventory[item] < 0:
                        raise ValueError("public actual-success inventory became negative")
                target = action["target_item"]
                inventory[target] = inventory.get(target, 0) + action["output_count"]
                continue
            if not isinstance(feedback, str) or not feedback.startswith("Error:"):
                raise ValueError("unrecognized native craft feedback")
            labels = [label for label in CATEGORIES if label in feedback]
            if len(labels) != 1:
                raise ValueError("native error category is not uniquely recognized")
            label = labels[0]
            counts["errors"] += 1
            source = recipes.get(action["target_item"])
            seen = source["recipes"] if source else None
            required = None
            if not seen:
                reason = "recipe_not_previously_returned"
            elif len(seen) != 1:
                reason = "multiple_recipes_need_selection"
            elif any(not isinstance(v, int) for v in seen[0]["ingredients"].values()):
                reason = "tag_recipe_needs_choice"
            elif action["output_count"] % seen[0]["result_count"]:
                reason = "nondivisible_quantity_unchanged"
            else:
                counts["known_single_recipe_divisible"] += 1
                batches = action["output_count"] // seen[0]["result_count"]
                required = {k: v * batches for k, v in seen[0]["ingredients"].items()}
                reason = (
                    "immediate_craft_valid_at_actual_state"
                    if all(inventory.get(k, 0) >= v for k, v in required.items())
                    else "still_insufficient_inventory"
                )
            counts[reason] += 1
            categories[label][reason] += 1
            episodes[reason].add(path.name)
            records.append(
                dict(
                    call_id=call_id,
                    node=str(path),
                    history_index=index,
                    original_action=action,
                    original_feedback=feedback,
                    error_category=label,
                    public_recipe_source=source,
                    actual_inventory_before_call=inventory.copy(),
                    bound_ingredients=required,
                    reason=reason,
                )
            )
    expected = audit["native_audit"]["groups"]["train_public056"]
    if counts["errors"] != expected["invalids"]["native_action_error"]:
        raise ValueError("public error inventory differs from native audit")
    if counts["craft_calls"] != expected["actions"]["craft"]:
        raise ValueError("public craft inventory differs from native audit")
    return dict(
        schema="textcraft-public-recipe-binding-retrospective-v1",
        counts=dict(counts),
        by_original_error={k: dict(v) for k, v in categories.items()},
        episodes_by_reason={k: len(v) for k, v in episodes.items()},
        initial_states=initial_states,
        rejected_craft_calls=records,
        input_sha256=pins,
        script_sha256=sha(Path(__file__)),
        method="Traverse each authenticated flat node public_history in saved order, starting "
        "from its public initial inventory. Cache only prior successful get_info recipe returns. "
        "Apply only ACTUAL successful crafts to inventory. For each actual rejected craft retain "
        "model target/output quantity and test immediate observed-recipe binding at that actual "
        "state. No hypothetical success is propagated. No private world/gold recipe lookup, "
        "recipe selection, quantity rounding, recursive demand planning or model calls.",
        caveat="Local counterfactual checks are correlated and include already successful "
        "episodes; "
        "not measured policy gains, rescued episodes or an achievable success ceiling. Unknown "
        "recipes and invalid batch quantities remain rejected. Native audit authenticates inputs.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("immutable report exists")
    result = analyze()
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["counts"]))
