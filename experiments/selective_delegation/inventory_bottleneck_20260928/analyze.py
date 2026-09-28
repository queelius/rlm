"""Descriptive public-only inventory diagnostics on audited completed breadth traces.

No native world import, hidden recipe lookup, model inference, or model ancestry hash.
Recipe closure and demand arithmetic use only feedback available before each action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
OUTPUT = ROOT / "analysis-inventory-bottleneck-20260928"


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def unique_recipe(info):
    recipes = info.get("recipes", []) if info else []
    return recipes[0] if len(recipes) == 1 else None


def public_needs(targets: dict, initial: dict, inventory: dict, observed: dict) -> dict:
    """Aggregate shared-child demand BEFORE rounding batches on the observed DAG.

    This is a derived public-graph completion calculation, not a claim about the
    model's internal plan. Missing recipes remain unresolved, never oracle-filled.
    """
    order, active, visited, unknown = [], set(), set(), set()
    parents, recipes = defaultdict(set), {}

    def visit(item):
        require(item not in active, "cycle in observed dependency recipes")
        if item in visited:
            return
        active.add(item)
        recipe = unique_recipe(observed.get(item))
        if recipe:
            require(
                type(recipe["result_count"]) is int and recipe["result_count"] > 0,
                "noninteger recipe yield",
            )
            recipes[item] = recipe
            for child in recipe["ingredients"]:
                parents[child].add(item)
                visit(child)
        elif not (initial.get(item, 0) > 0 or observed.get(item, {}).get("is_base") is True):
            unknown.add(item)
        active.remove(item)
        visited.add(item)
        order.append(item)

    for item in targets:
        visit(item)
    demand = Counter({item: count + initial.get(item, 0) for item, count in targets.items()})
    needed, frontier, ready = {}, {}, []
    for item in reversed(order):
        required = demand[item]
        have = inventory.get(item, 0)
        deficit = max(0, required - have)
        record = dict(required=required, inventory=have, deficit=deficit, batches=None)
        recipe = recipes.get(item)
        if recipe:
            batches = (deficit + recipe["result_count"] - 1) // recipe["result_count"]
            record["batches"] = batches
            record["minimum_new_output"] = batches * recipe["result_count"]
            for child, amount in recipe["ingredients"].items():
                demand[child] += batches * amount
            if deficit and all(inventory.get(k, 0) >= n for k, n in recipe["ingredients"].items()):
                ready.append(item)
        elif deficit:
            frontier[item] = deficit
        needed[item] = record
    remaining_roots = [item for item in targets if needed[item]["deficit"]]
    completing_ready = bool(remaining_roots) and all(
        item in recipes
        and all(
            inventory.get(k, 0) >= n * needed[item]["batches"]
            for k, n in recipes[item]["ingredients"].items()
        )
        for item in remaining_roots
    )
    return dict(
        recipe_closure_complete=not unknown,
        unknown_recipe_names=sorted(unknown),
        completion_feasible=not frontier,
        frontier_deficits=frontier,
        publicly_known_base_deficits=sorted(
            k for k in frontier if observed.get(k, {}).get("is_base") is True
        ),
        needed=needed,
        ready_prerequisites=sorted(set(ready) - set(targets)),
        goal_completing_craft_ready=completing_ready,
        goal_met=not remaining_roots,
        shared_items=sorted(k for k, v in parents.items() if len(v) > 1),
        parents={k: sorted(v) for k, v in parents.items()},
    )


def static_info(info):
    return {k: info.get(k) for k in ("item", "is_base", "can_craft", "crafting_depth", "recipes")}


def analyze_node(node: dict) -> dict:
    history = node["public_history"]
    require(len(history) == len(node["call_ids"]), "history/call alignment differs")
    inventory, observed = dict(node["initial_inventory"]), {}
    assists = {a["call_id"]: a for a in node.get("execution_assists", [])}
    counts, seen_errors, traces = Counter(), set(), []
    previous_error_key, run_length, longest = None, 0, 0
    cross_consumptions, produced_at = defaultdict(list), {}
    first_closure, first_feasible = None, None
    for index, (row, cid) in enumerate(zip(history, node["call_ids"], strict=True)):
        before = public_needs(node["targets"], node["initial_inventory"], inventory, observed)
        action, feedback = row.get("action", {}), row.get("feedback")
        action_type = action.get("action", "invalid_schema")
        counts["calls"] += 1
        counts[action_type + "_calls"] += 1
        trace = dict(
            index=index,
            call_id=cid,
            action=action or None,
            feedback=feedback,
            stock_before=dict(inventory),
            public_before=before,
        )
        error_key = None
        if action_type == "get_info" and isinstance(feedback, list):
            repeats = 0
            for info in feedback:
                item = info["item"]
                require(item in action["items"], "unrequested public recipe reply")
                counts["queried_items"] += 1
                if item in observed and static_info(info) == static_info(observed[item]):
                    counts["requeried_unchanged_recipe_items"] += 1
                    repeats += 1
                observed[item] = info
                if "in_inventory" in info:
                    require(info["in_inventory"] == inventory.get(item, 0), "query stock mismatch")
            if repeats == len(feedback) and feedback:
                counts["query_calls_with_no_new_static_recipe"] += 1
                if before["recipe_closure_complete"]:
                    counts["redundant_query_calls_after_closure"] += 1
        elif action_type == "craft":
            executed = assists.get(cid, {}).get("executed_action", action)
            if cid in assists:
                require(executed == action, "assisted history execution mismatch")
                trace["requested_action"] = assists[cid]["requested_action"]
            trace["executed_action"] = executed
            insufficient = (
                isinstance(feedback, str) and "Insufficient ingredients in inventory" in feedback
            )
            if insufficient:
                counts["insufficient_stock_crafts"] += 1
                counts["insufficient_with_recipe_closure"] += int(before["recipe_closure_complete"])
                counts["insufficient_with_public_completion_feasible"] += int(
                    before["completion_feasible"]
                )
                counts["insufficient_with_ready_prerequisite"] += int(
                    bool(before["ready_prerequisites"])
                )
                physical = {k: v for k, v in executed.items() if k != "note"}
                error_key = canonical((physical, inventory))
                if error_key in seen_errors:
                    counts["repeat_insufficient_same_action_and_stock"] += 1
                    trace["repeat_same_action_stock"] = True
                seen_errors.add(error_key)
                if error_key == previous_error_key:
                    counts["consecutive_repeat_insufficient_same_action_and_stock"] += 1
                shortages = re.findall(r"([A-Za-z0-9_]+): need (\d+), have (\d+)", feedback)
                trace["reported_shortages"] = {
                    item: dict(need=int(need), have=int(have)) for item, need, have in shortages
                }
                witnesses = []
                for item, _, _ in shortages:
                    for prior in cross_consumptions[item]:
                        if (
                            executed["target_item"] in prior["other_known_parents"]
                            and produced_at.get(item, -1) < prior["index"]
                        ):
                            witnesses.append(dict(item=item, **prior))
                if witnesses:
                    counts["shortage_after_cross_branch_consumption_without_replenishment"] += 1
                    trace["cross_branch_consumption_witnesses"] = witnesses
            if isinstance(feedback, str) and feedback.startswith("Successfully crafted "):
                counts["successful_crafts"] += 1
                target = executed["target_item"]
                requested_need = before["needed"].get(target)
                if before["recipe_closure_complete"] and requested_need is not None:
                    minimum = requested_need.get("minimum_new_output")
                    if minimum is not None and executed["output_count"] > minimum:
                        counts["crafts_above_public_remaining_batch_demand"] += 1
                        trace["above_public_remaining_batch_demand"] = dict(
                            requested=executed["output_count"], minimum_new_output=minimum
                        )
                for item, amount in executed["ingredients"].items():
                    require(
                        inventory.get(item, 0) >= amount, "successful craft overdrew public stock"
                    )
                    siblings = set(before["parents"].get(item, [])) - {target}
                    if target in before["parents"].get(item, []) and siblings:
                        counts["successful_shared_ingredient_consumptions"] += 1
                        cross_consumptions[item].append(
                            dict(
                                index=index,
                                call_id=cid,
                                consuming_target=target,
                                consumed=amount,
                                other_known_parents=sorted(siblings),
                            )
                        )
                    inventory[item] -= amount
                    if inventory[item] == 0:
                        del inventory[item]
                inventory[target] = inventory.get(target, 0) + executed["output_count"]
                produced_at[target] = index
                after = public_needs(
                    node["targets"], node["initial_inventory"], inventory, observed
                )
                if before["completion_feasible"] and not after["completion_feasible"]:
                    counts["crafts_losing_public_graph_completion_feasibility"] += 1
                    trace["public_feasibility_lost"] = dict(
                        frontier_deficits=after["frontier_deficits"]
                    )
        if error_key is not None:
            run_length = run_length + 1 if error_key == previous_error_key else 1
            longest = max(longest, run_length)
        else:
            run_length = 0
        previous_error_key = error_key
        after = public_needs(node["targets"], node["initial_inventory"], inventory, observed)
        if first_closure is None and after["recipe_closure_complete"]:
            first_closure = index
        if first_feasible is None and after["completion_feasible"]:
            first_feasible = index
        trace["stock_after"] = dict(inventory)
        traces.append(trace)
    require(inventory == node["final_inventory"], "reconstructed public stock differs from audit")
    final = public_needs(node["targets"], node["initial_inventory"], inventory, observed)
    failed = node["native_score"] == 0
    flags = dict(
        any_insufficient=counts["insufficient_stock_crafts"] > 0,
        repeated_insufficient=counts["repeat_insufficient_same_action_and_stock"] > 0,
        repeated_query=counts["query_calls_with_no_new_static_recipe"] > 0,
        failed=failed,
        context_cap=node["status"] == "context_cap",
        failed_explicit_finish=failed and node["status"] == "finished",
        failed_with_complete_recipe_closure=failed and final["recipe_closure_complete"],
        failed_with_public_completion_feasible=failed and final["completion_feasible"],
        failed_with_ready_prerequisite=failed and bool(final["ready_prerequisites"]),
        failed_with_goal_completing_craft_ready=failed and final["goal_completing_craft_ready"],
        any_public_feasibility_loss=counts["crafts_losing_public_graph_completion_feasibility"] > 0,
        failed_with_public_feasibility_loss=failed
        and counts["crafts_losing_public_graph_completion_feasibility"] > 0,
        any_cross_branch_shortage=counts[
            "shortage_after_cross_branch_consumption_without_replenishment"
        ]
        > 0,
        failed_with_cross_branch_shortage=failed
        and counts["shortage_after_cross_branch_consumption_without_replenishment"] > 0,
        any_above_public_batch_demand=counts["crafts_above_public_remaining_batch_demand"] > 0,
    )
    return dict(
        native_score=node["native_score"],
        status=node["status"],
        counts=dict(counts),
        flags=flags,
        max_identical_insufficient_run=longest,
        first_recipe_closure_after_call=first_closure,
        first_public_feasibility_after_call=first_feasible,
        final_public=final,
        observed_recipe_items=len(observed),
        observed_recipes=observed,
        targets=node["targets"],
        final_inventory=inventory,
        initial_inventory=node["initial_inventory"],
        seconds=node["ended"] - node["started"],
        trace=traces,
    )


def aggregate(rows):
    if not rows:
        return {}
    return dict(
        episodes=len(rows),
        task_identities=len({r["task_id"] for r in rows}),
        successes=sum(r["native_score"] for r in rows),
        statuses=dict(Counter(r["status"] for r in rows)),
        flags=dict(
            sum((Counter({k: int(v) for k, v in r["flags"].items()}) for r in rows), Counter())
        ),
        counts=dict(sum((Counter(r["counts"]) for r in rows), Counter())),
        max_identical_insufficient_run=max(r["max_identical_insufficient_run"] for r in rows),
        median_identical_insufficient_run=sorted(r["max_identical_insufficient_run"] for r in rows)[
            len(rows) // 2
        ],
        episode_seconds_mean=mean(r["seconds"] for r in rows),
        episode_seconds_sum=sum(r["seconds"] for r in rows),
    )


def load(root: Path):
    rows, receipt_hashes = [], {}

    def read(path, expected=None):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(expected is None or expected == digest, "audited receipt changed: " + str(path))
        receipt_hashes[str(path)] = digest
        return json.loads(content)

    for panel in range(6):
        for world in (42, 50, 51, 52):
            for seed in ("original", "2291"):
                for mode in ("binder", "raw"):
                    directory = root / f"textcraft-breadth-p{panel:02d}-w{world}-s{seed}-{mode}-001"
                    audit = read(directory / "NATIVE-AUDIT.json")
                    plan = read(
                        directory / "PLAN.json", audit["sha256"][str(directory / "PLAN.json")]
                    )
                    require(
                        plan["teacher"] == "public" and plan["execution_mode"] == mode,
                        "discovery/execution mapping differs",
                    )
                    task_path = Path(plan["prepared"]) / "tasks.jsonl"
                    data = task_path.read_bytes()
                    require(
                        hashlib.sha256(data).hexdigest() == plan["tasks_sha256"], "tasks changed"
                    )
                    receipt_hashes[str(task_path)] = plan["tasks_sha256"]
                    tasks = {t["id"]: t for t in (json.loads(line) for line in data.splitlines())}
                    for job in plan["jobs"]:
                        task = tasks[job["task_id"]]
                        if task["misc"]["max_depth"] not in (4, 5):
                            continue
                        path = directory / "nodes" / f"{job['episode_id']}-n0.json"
                        node = read(path, audit["sha256"][str(path)])
                        episode_path = directory / "episodes" / f"{job['episode_id']}.json"
                        episode = read(episode_path, audit["sha256"][str(episode_path)])
                        require(
                            episode["observed"] and episode["native_score"] == node["native_score"],
                            "unknown or inconsistent native outcome",
                        )
                        result = analyze_node(node)
                        result.update(
                            panel=panel,
                            world=world,
                            training_seed=seed,
                            mode=mode,
                            task_id=job["task_id"],
                            repeat=job["repeat"],
                            episode_id=job["episode_id"],
                            node_path=str(path),
                            declared_depth=task["misc"]["max_depth"],
                            native_audit=str(directory / "NATIVE-AUDIT.json"),
                        )
                        rows.append(result)
    return rows, receipt_hashes


def main(args):
    rows, receipts = load(args.root)
    primary = [
        r
        for r in rows
        if r["panel"] == 0
        and r["world"] == 42
        and r["training_seed"] == "original"
        and r["mode"] == "binder"
    ]
    report = dict(
        schema="public-inventory-bottleneck-breadth-20260928-v1",
        primary_selection="Outcome-blind first panel00/world42/original training seed; "
        "all declared depth4/5 attempts, including successes.8 attempts/4 task identities.",
        broadening="Secondary: every discovery depth4/5 attempt in fixed primary breadth "
        "panels00–05, worlds42/50/51/52, both training seeds, raw and binder. Never panels06+.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primary=aggregate(primary),
        primary_episodes=[
            {k: v for k, v in r.items() if k not in ("trace", "observed_recipes")} for r in primary
        ],
        secondary={
            mode: aggregate([r for r in rows if r["mode"] == mode]) for mode in ("binder", "raw")
        },
        by_depth={
            f"{mode}_d{depth}": aggregate(
                [r for r in rows if r["mode"] == mode and r["declared_depth"] == depth]
            )
            for mode in ("binder", "raw")
            for depth in (4, 5)
        },
        by_world={
            f"{mode}_w{world}": aggregate(
                [r for r in rows if r["mode"] == mode and r["world"] == world]
            )
            for mode in ("binder", "raw")
            for world in (42, 50, 51, 52)
        },
        by_training_seed={
            f"{mode}_{seed}": aggregate(
                [r for r in rows if r["mode"] == mode and r["training_seed"] == seed]
            )
            for mode in ("binder", "raw")
            for seed in ("original", "2291")
        },
        observed_only=True,
        native_graph_oracle_used=False,
        definitions=dict(
            repeated_stock_error="Same canonical executed craft fields excluding note and "
            "same complete inventory after an earlier insufficient-stock failure; history, "
            "recipe knowledge and remaining budget may differ. Not identical policy state.",
            recipe_closure="Root dependency recipes publicly returned, recursively to names "
            "in initial inventory or explicitly identified public base items. Static closure "
            "is distinct from whether enough stock exists or the model understood the graph.",
            public_feasibility="Counterfactual arithmetic on only unique observed recipes: "
            "sum shared-child demand before rounding batches, subtract current stock once. "
            "All unresolved leaf deficits must be zero. No next action is selected.",
            prerequisite="A goal-reachable item with positive remaining derived demand and "
            "enough current stock for one observed-recipe batch. Root is counted separately.",
            cross_branch="A later native-reported item shortage follows consumption by a "
            "different already-known parent, with no intervening production of that item. "
            "A temporal witness, not proof the earlier craft was destructive.",
            feasibility_loss="Successful craft changes public-graph completion feasibility "
            "from true to false with the SAME observed recipes. This does not prove an "
            "oracle-world irrecoverable state if recipes remain unobserved.",
            overproduction="Successful output exceeds minimum batch-rounded outstanding "
            "demand in the current public DAG, restricted to complete static closure. "
            "Not necessarily harmful because excess resources may remain.",
        ),
        limitations="Descriptive, correlated repeated attempts on24 deep task identities; "
        "worlds and model seeds do not make independent goals. No causal effects or "
        "model-internal planning labels. Recipe inspection never uses hidden native graph.",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("SUMMARY.json", report),
        ("DETAILS.json", dict(rows=rows, sha256=receipts)),
    ):
        with (args.output / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(dict(primary=report["primary"], secondary=report["secondary"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args())
