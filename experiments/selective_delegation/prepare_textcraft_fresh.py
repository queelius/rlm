"""Outcome-blind fresh VAL roots, frozen before native feasibility/token audits."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import prepare_textcraft as original
import prepare_textcraft_public as public
import prepare_textcraft_sft as training
import textcraft_bridge as bridge

SEED = 2026092222
STRATA = {2: 4, 3: 4, 4: 8}
sha, save = original.sha, original.save


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def select(candidates, exposed, train, strata=None):
    strata = STRATA if strata is None else strata
    excluded_ids = {r["id"] for r in exposed + train}
    excluded_roots = {i for r in exposed + train for i in r["misc"]["target_items"]}
    eligible = [
        r
        for r in candidates
        if r["id"] not in excluded_ids and not set(r["misc"]["target_items"]) & excluded_roots
    ]
    selected, used_roots = [], set()
    for depth, count in strata.items():
        ordered = sorted(
            [r for r in eligible if r["misc"]["max_depth"] == depth],
            key=lambda r: hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest(),
        )
        added = 0
        for row in ordered:
            roots = set(row["misc"]["target_items"])
            if roots & used_roots:
                continue
            selected.append(row)
            used_roots.update(roots)
            added += 1
            if added == count:
                break
        if added != count:
            raise ValueError("insufficient unique-root fixed stratum; no replacement")
    return selected, {
        "eligible_by_depth": dict(Counter(r["misc"]["max_depth"] for r in eligible)),
        "excluded_ids_or_roots": len(candidates) - len(eligible),
        "excluded_ids": sorted(excluded_ids),
        "excluded_roots": sorted(excluded_roots),
    }


def dependencies(targets, world):
    found = set()

    def visit(item):
        if item in found or item not in world.recipes:
            return
        found.add(item)
        for recipe in world.recipes[item]:
            for ingredient in recipe.ingredients:
                visit(ingredient)

    for target in targets:
        visit(target)
    return found


def freeze(root, output):
    if output.exists():
        raise FileExistsError("fresh immutable panel already exists")
    if sha(original.TASKS) != original.TASK_SHA or sha(training.TRAIN) != training.TRAIN_SHA:
        raise ValueError("official TRAIN/VAL changed")
    inventories = sorted(root.glob("textcraft*/tasks.jsonl"))
    plans = sorted(root.glob("textcraft*/PLAN.json"))
    exposed = [r for p in inventories for r in rows(p)]
    official = rows(original.TASKS)
    by_id = {r["id"]: r for r in official}
    unknown_plan_ids = set()
    for path in plans:
        plan = json.loads(path.read_text())
        for job in plan.get("jobs", []):
            identity = job.get("task_id")
            if identity in by_id:
                exposed.append(by_id[identity])
            elif identity:
                unknown_plan_ids.add(identity)
    train = rows(training.TRAIN)
    selected, inventory = select(official, exposed, train)
    output.mkdir(parents=True)
    with (output / "tasks.jsonl").open("x") as stream:
        for task in selected:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    save(
        output / "SELECTION.json",
        {
            "schema": "textcraft-fresh16-selection-v1",
            "selected_at": time.time(),
            "selected_before_replay": True,
            "selection_seed": SEED,
            "strata": STRATA,
            "selection": "SHA256(seed:id) within depth; distinct root goals; "
            "no outcomes/gold filter",
            "task_ids": [r["id"] for r in selected],
            "task_count": len(selected),
            "tasks_sha256": sha(output / "tasks.jsonl"),
            "inventory": inventory,
            "known_inventory_scope": "All current R/textcraft*/tasks.jsonl plus PLAN jobs; "
            "all official TRAIN root goals, not an unbounded historical guarantee",
            "unresolved_plan_task_ids": sorted(unknown_plan_ids),
            "input_sha256": {
                str(p): sha(p) for p in [original.TASKS, training.TRAIN, *inventories, *plans]
            },
            "source_sha256": {
                str(Path(m.__file__).resolve()): sha(Path(m.__file__))
                for m in (original, training, public, bridge)
            },
            "preparer_sha256": sha(Path(__file__)),
            "no_replacement": True,
            "upstream_commit": original.COMMIT,
            "license": "MIT",
            "recipe_seed": 42,
        },
    )
    return selected


def audit(root, output):
    from transformers import AutoTokenizer

    selection = json.loads((output / "SELECTION.json").read_text())
    if sha(output / "tasks.jsonl") != selection["tasks_sha256"]:
        raise ValueError("frozen task bytes differ")
    tasks, world = rows(output / "tasks.jsonl"), bridge.load_world()
    # Existing quantity-correct native gold actions; retain every failure.
    native = [bridge.replay_gold(task, world) for task in tasks]
    save(
        output / "REPLAY.json",
        {
            "tasks": native,
            "no_replacement": True,
            "selection_sha256": sha(output / "SELECTION.json"),
        },
    )
    tokens = original.token_audit(output)
    tokenizer = AutoTokenizer.from_pretrained(
        training.BASE, local_files_only=True, trust_remote_code=False
    )
    public_audit = []
    for task in tasks:
        result, _ = public.trajectory(task, world, tokenizer)
        public_audit.append(result)
    trained = rows(root / "textcraft-train-inputs-001/tasks.jsonl")
    train_dependencies = set().union(
        *(dependencies(t["misc"]["target_items"], world) for t in trained)
    )
    panel_dependencies = []
    for task in tasks:
        dep = dependencies(task["misc"]["target_items"], world)
        panel_dependencies.append(
            {
                "task_id": task["id"],
                "recipe_products": sorted(dep),
                "shared_with_actual_train32": sorted(dep & train_dependencies),
            }
        )
    save(
        output / "MANIFEST.json",
        {
            "schema": "textcraft-fresh16-qualified-v1",
            "status": "CPU_only_GPU_not_accepted",
            "task_count": 16,
            "task_ids": selection["task_ids"],
            "tasks_sha256": selection["tasks_sha256"],
            "selection_seed": SEED,
            "selection_sha256": sha(output / "SELECTION.json"),
            "replay_sha256": sha(output / "REPLAY.json"),
            "token_audit_sha256": sha(output / "TOKEN-AUDIT.json"),
            "world_sha256": tokens["world_sha256"],
            "trusted_source": bridge.trusted_provenance(),
            "model": tokens["model"],
            "model_manifest_sha256": tokens["model_manifest_sha256"],
            "all_native_gold_success": all(r["native_score"] == 1 for r in native),
            "ready": tokens["ready"] and all(r["native_score"] == 1 for r in native),
            "public_teacher_audit": public_audit,
            "dependency_overlap": panel_dependencies,
            "actual_train32_sha256": sha(root / "textcraft-train-inputs-001/tasks.jsonl"),
            "episode_seeds": [2026092204, 2026092205],
            "planned_episodes": 64,
            "planned_per_teacher": 32,
            "max_native_calls": 6144,
            "total_seconds_cap": 7200,
            "max_global_calls": 96,
            "max_global_output_tokens": 8192,
            "max_new_tokens": 256,
            "input_plus_output_limit": 8192,
            "no_truncation": True,
            "interpretation": "New root goals in the SAME recipe world, not independent worlds; "
            "depth-stratified exploratory panel; native/public-teacher feasibility "
            "is not model success.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("freeze", "audit"), required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.root.resolve(), args.output.resolve())
    else:
        audit(args.root.resolve(), args.output.resolve())
    print(json.dumps({"stage": args.stage, "output": str(args.output)}))
