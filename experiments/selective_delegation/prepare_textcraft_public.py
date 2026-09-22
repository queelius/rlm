"""CPU prototype: public root-first recipe discovery on the exact frozen32 TRAIN tasks.

One bounded task from PUBLIC-DISCOVERY-IMPLEMENTATION.md. Teacher inputs are only
targets, inventories and past public replies. The bridge/encoder remain unchanged.
Ruling: reject multiple-recipe alternatives rather than introduce a search framework;
the pinned synthetic world has one recipe per product. Failures are retained.
"""

import argparse
import hashlib
import heapq
import json
import time
from collections import Counter
from pathlib import Path

import prepare_textcraft_sft as reference
import textcraft_bridge as bridge

TASKS_SHA = "390dff9bb19d0fe71c7bec0505c97608013c66c65aea90a821839f01b615ab30"


def next_action(targets, initial_inventory, current_inventory, observed_recipes):
    """Choose from public information only; no task ID, world, score or gold argument."""
    if all(
        current_inventory.get(k, 0) - initial_inventory.get(k, 0) >= v for k, v in targets.items()
    ):
        return {"action": "finish", "message": "done"}
    # Build a product-before-ingredients topological order using queried edges only.
    edges, recipes, visiting, visited = {}, {}, set(), set()

    def visit(item):
        if item in visiting:
            raise ValueError("cycle in observed recipe graph")
        if item in visited:
            return
        visiting.add(item)
        info = observed_recipes.get(item)
        options = info["recipes"] if info is not None else []
        if len(options) > 1:
            raise ValueError("multiple recipe alternatives unsupported by bounded teacher")
        recipe = options[0] if options else None
        if recipe is not None:
            if (
                not bridge.quantities(recipe["ingredients"])
                or type(recipe["result_count"]) is not int
                or recipe["result_count"] <= 0
            ):
                raise ValueError("invalid public recipe quantities")
            recipes[item] = recipe
        edges[item] = sorted(recipe["ingredients"]) if recipe else []
        for child in edges[item]:
            visit(child)
        visiting.remove(item)
        visited.add(item)

    for item in sorted(targets, reverse=True):
        visit(item)
    incoming = Counter(child for children in edges.values() for child in children)
    ready = [item for item in edges if not incoming[item]]
    heapq.heapify(ready)
    order = []
    while ready:
        item = heapq.heappop(ready)
        order.append(item)
        for child in edges[item]:
            incoming[child] -= 1
            if not incoming[child]:
                heapq.heappush(ready, child)
    demand = Counter({k: initial_inventory.get(k, 0) + v for k, v in targets.items()})
    missing, batches, shortages = [], {}, {}
    for item in order:
        shortage = max(0, demand[item] - current_inventory.get(item, 0))
        shortages[item] = shortage
        if not shortage:
            continue
        if item not in observed_recipes:
            missing.append(item)
            continue
        if item not in recipes:
            raise ValueError("unavailable base resource or uncraftable item: " + item)
        recipe = recipes[item]
        count = (shortage + recipe["result_count"] - 1) // recipe["result_count"]
        batches[item] = count
        for ingredient, quantity in recipe["ingredients"].items():
            demand[ingredient] += count * quantity
    if missing:
        # Query before crafting; the next call recomputes aggregate demand from replies.
        return {"action": "get_info", "items": [missing[0]]}
    for item in sorted(batches):
        recipe = recipes[item]
        ingredients = {k: v * batches[item] for k, v in recipe["ingredients"].items()}
        # First satisfy the aggregated demand for shared ingredients, not one parent's
        # independent demand. This also avoids treating stock as reusable per branch.
        if all(
            shortages.get(k, 0) == 0 and current_inventory.get(k, 0) >= v
            for k, v in ingredients.items()
        ):
            return {
                "action": "craft",
                "ingredients": ingredients,
                "target_item": item,
                "output_count": batches[item] * recipe["result_count"],
            }
    raise ValueError("no feasible public craft under current inventory")


def trajectory(task, world, tokenizer, max_calls=96):
    # Only the driver sees a task object; teacher receives these explicit public copies.
    targets = dict(task["misc"]["target_items"])
    initial = dict(task["misc"]["initial_inventory"])
    frame = bridge.Frame(world, dict(initial), targets, bridge.Budget(max_calls, 8192), 0)
    known, query_steps, history, rows = {}, {}, [], []
    status, error = "unfinished", None
    try:
        while not frame.finished:
            try:
                cap = frame.budget.reserve()
            except bridge.BudgetExceeded:
                status = (
                    "global_call_cap" if frame.budget.calls >= max_calls else "global_token_cap"
                )
                break
            try:
                action = next_action(targets, initial, dict(frame.inventory), known)
            except ValueError as exc:
                status, error = "teacher_failure", str(exc)
                break
            prompt = bridge.public_prompt(frame, history, goal=task["goal"])
            encoded = reference.encode_row(prompt, action, tokenizer)
            if encoded["prompt_tokens"] + cap > 8192:
                status = "context_cap"
                break
            if encoded["target_tokens"] > cap:
                status = "target_output_cap"
                break
            bridge.parse_action(encoded["target"])
            provenance = {
                item: {
                    "query_step": query_steps[item],
                    "recipe_reply_sha256": hashlib.sha256(
                        json.dumps(known[item], sort_keys=True).encode()
                    ).hexdigest(),
                }
                for item in sorted(known)
            }
            frame.budget.charge(encoded["target_tokens"])
            feedback = frame.apply(action)
            rows.append(
                {
                    "task_id": task["id"],
                    "step": len(rows),
                    **encoded,
                    "feedback": feedback,
                    "public_recipe_provenance": provenance,
                }
            )
            history.append({"action": action, "feedback": feedback})
            if isinstance(feedback, str) and feedback.startswith("Error:"):
                status, error = "native_action_error", feedback
                break
            if action["action"] == "get_info":
                for info in feedback:
                    # Deliberately discard native crafting_depth/can_craft/in_inventory.
                    # This is the only route by which recipes enter the teacher.
                    known[info["item"]] = {
                        "is_base": info["is_base"],
                        "recipes": [
                            {
                                "ingredients": dict(r["ingredients"]),
                                "result_count": r["result_count"],
                            }
                            for r in info["recipes"]
                        ],
                    }
                    query_steps[info["item"]] = len(rows) - 1
        if frame.finished:
            status = "finished"
    except Exception as exc:
        status, error = "native_or_encoder_exception", f"{type(exc).__name__}: {exc}"
    # Qualification only: never consulted by next_action or during teacher decisions.
    score, details = frame.score()
    return dict(
        task_id=task["id"],
        status=status,
        error=error,
        native_score=score,
        native_details=details,
        eligible=status == "finished" and score == 1,
        rows=len(rows),
        calls_charged=frame.budget.calls,
        output_tokens=frame.budget.output_tokens,
        finish_included=frame.finished,
        public_recipe_count=len(known),
        initial_inventory=initial,
        final_inventory=dict(frame.inventory),
    ), rows


def prepare(root, output):
    if output.exists():
        raise FileExistsError("immutable public-discovery prototype already exists")
    source = root / "textcraft-train-inputs-001/tasks.jsonl"
    if reference.sha(source) != TASKS_SHA:
        raise ValueError("exact frozen047 TRAIN32 tasks required")
    tasks = [json.loads(line) for line in source.read_text().splitlines()]
    if len(tasks) != 32 or len({t["id"] for t in tasks}) != 32:
        raise ValueError("frozen32 identity mismatch")
    output.mkdir(parents=True)
    # Copy bytes, not a new selection; preserve before any teacher/native outcome.
    with (output / "tasks.jsonl").open("xb") as stream:
        stream.write(source.read_bytes())
    reference.save(
        output / "SELECTION.json",
        dict(
            task_ids=[t["id"] for t in tasks],
            tasks_sha256=TASKS_SHA,
            source=str(source),
            selected_before_replay=True,
            selection="same frozen047 tasks; no filter",
            training_accepted=False,
            teacher_inputs="targets, initial/current inventory, past queried recipes",
            limitations="Single observed recipe per product; "
            "cycles/unavailable resources explicit failures",
            preparer_sha256=reference.sha(Path(__file__)),
        ),
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        reference.BASE, local_files_only=True, trust_remote_code=False
    )
    world, started = bridge.load_world(), time.time()
    receipts, training_rows = [], []
    for task in tasks:
        receipt, rows = trajectory(task, world, tokenizer)
        reference.save(
            output / "traces" / (hashlib.sha256(task["id"].encode()).hexdigest()[:24] + ".json"),
            {"receipt": receipt, "rows": rows},
        )
        receipts.append(receipt)
        if receipt["eligible"]:
            training_rows.extend(rows)
    with (output / "rows.jsonl").open("x") as stream:
        for row in training_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = dict(
        schema="textcraft-public-discovery-prototype-v1",
        training_accepted=False,
        selected_task_count=32,
        eligible_task_count=sum(r["eligible"] for r in receipts),
        task_ids=[t["id"] for t in tasks],
        tasks=receipts,
        tasks_sha256=TASKS_SHA,
        failed_or_capped_tasks_retained=[r["task_id"] for r in receipts if not r["eligible"]],
        eligibility="Complete native-successful teacher traces only; "
        "every selected failure retained",
        rows=len(training_rows),
        rows_sha256=reference.sha(output / "rows.jsonl"),
        action_counts=dict(Counter(json.loads(r["target"])["action"] for r in training_rows)),
        prompt_tokens=sum(r["prompt_tokens"] for r in training_rows),
        supervised_tokens_including_eos=sum(r["target_tokens"] for r in training_rows),
        max_prompt_tokens=max((r["prompt_tokens"] for r in training_rows), default=0),
        max_target_tokens_including_eos=max((r["target_tokens"] for r in training_rows), default=0),
        max_prompt_plus_generation_cap=max(
            (r["prompt_tokens"] + 256 for r in training_rows), default=0
        ),
        future_effective16_updates=(len(training_rows) + 15) // 16,
        reference047={
            "rows": 366,
            "target_tokens": 8821,
            "updates": 23,
            "rows_sha256": reference.sha(root / "textcraft-train-inputs-001/rows.jsonl"),
        },
        dose_caveat="Same tasks/one prospective epoch, not matched rows/tokens/updates; "
        "no training accepted",
        model=str(reference.BASE),
        model_manifest_sha256=reference.sha(reference.BASE / "local-research-manifest.json"),
        eos_token_id=tokenizer.eos_token_id,
        started=started,
        ended=time.time(),
        source_sha256={
            str(p): reference.sha(p)
            for p in (
                Path(__file__),
                Path(reference.__file__),
                Path(bridge.__file__),
                Path(reference.original.__file__),
            )
        },
        trusted_source=bridge.trusted_provenance(),
        trace_sha256={
            str(p.relative_to(output)): reference.sha(p)
            for p in sorted((output / "traces").glob("*.json"))
        },
    )
    reference.save(output / "MANIFEST.json", manifest)
    return {
        k: manifest[k]
        for k in (
            "selected_task_count",
            "eligible_task_count",
            "rows",
            "action_counts",
            "supervised_tokens_including_eos",
            "max_prompt_plus_generation_cap",
            "failed_or_capped_tasks_retained",
            "future_effective16_updates",
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve(), args.output.resolve())))
