"""Freeze structurally novel goals before native qualification or model readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx

import structure

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
SOURCE = ROOT / "source-textcraft-reserve-readout-001"
WORLD_SEED = 2026101401
SELECTION_SEED = 2026101402
STRATA = {3: 4, 4: 4}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def runtime():
    sys.path.insert(0, str(SOURCE))
    import prepare_textcraft_multiworld as worlds

    if Path(worlds.__file__).parent != SOURCE:
        raise ValueError("mixed native preparation runtime")
    worlds.prior.bridge.trusted_provenance()
    return worlds


def recipes(world) -> dict:
    if any(len(rows) != 1 for rows in world.recipes.values()):
        raise ValueError("single-recipe graph required")
    return {item: rows[0].ingredients for item, rows in world.recipes.items()}


def build_task(world, root: str, seed: int) -> dict:
    task = {
        "id": f"teaching_structure.w{seed}.{root}",
        "goal": f"Craft the following items: 2x {root}",
        "misc": {
            "target_items": {root: 2},
            "initial_inventory": {},
            "max_depth": world.get_crafting_depth(root),
        },
    }
    return dict(runtime().prior.construct(task, world), derived_world_seed=seed)


def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    worlds = runtime()
    plan_paths = sorted(ROOT.glob("textcraft*/**/PLAN.json"))
    prior_seeds, prepared, inputs = {42}, set(), {}
    for path in plan_paths:
        plan = json.loads(path.read_text())
        inputs[str(path)] = sha(path)
        if type(plan.get("world_seed")) is int:
            prior_seeds.add(plan["world_seed"])
        if plan.get("prepared") and (Path(plan["prepared"]) / "tasks.jsonl").exists():
            prepared.add(Path(plan["prepared"]) / "tasks.jsonl")
    if WORLD_SEED in prior_seeds:
        raise ValueError("candidate world already has a prior plan")
    prepared.update(ROOT.glob("textcraft*/tasks.jsonl"))
    exposed_roots = set()
    for path in sorted(prepared):
        inputs[str(path)] = sha(path)
        for line in path.read_text().splitlines():
            task = json.loads(line)
            exposed_roots.update(task.get("misc", {}).get("target_items", {}))

    index, inventory = structure.StructureIndex(), []
    for seed in sorted(prior_seeds):
        world = worlds.world(seed)
        mapping = recipes(world)
        count, large = 0, 0
        for root in sorted(mapping):
            graph = structure.dependency_graph(mapping, root)
            if graph is None:
                large += 1
            else:
                index.add(graph)
                count += 1
        inventory.append(
            {
                "world_seed": seed,
                "root_graphs_checked": count,
                "graphs_over_64_nodes": large,
                "world_sha256": worlds.prior.inventory.digest(
                    worlds.prior.inventory.snapshot(world)
                ),
            }
        )

    world = worlds.world(WORLD_SEED)
    mapping = recipes(world)
    candidates = []
    for root in sorted(mapping):
        depth = world.get_crafting_depth(root)
        if depth not in STRATA or root in exposed_roots:
            continue
        graph = structure.dependency_graph(mapping, root)
        # Prospective complexity bound, not native feasibility/model-outcome filtering.
        if graph is not None and sum(i in mapping for i in graph) <= 12:
            candidates.append((root, depth, graph))
    selected = structure.select(candidates, index, STRATA, SELECTION_SEED)
    output.mkdir(parents=True)
    record = {
        "schema": "teaching-fresh-structure-selection-v1",
        "frozen_epoch": time.time(),
        "world_seed": WORLD_SEED,
        "selection_seed": SELECTION_SEED,
        "strata": STRATA,
        "target_quantity": 2,
        "rule": "Exclude prior goal names; at most12 reachable recipe products and64 nodes; "
        "SHA256(seed:root) within depth. Exact directed rooted graph isomorphism, "
        "ignoring names/quantities, excludes old and within-panel duplicate structures.",
        "qualification_after_selection": True,
        "model_outcomes_used": False,
        "candidate_counts": dict(Counter(row[1] for row in candidates)),
        "prior_worlds": inventory,
        "exposed_goal_names": sorted(exposed_roots),
        "old_unique_topologies": sum(map(len, index.graphs.values())),
        "inventory_sha256": inputs,
        "source_sha256": {
            str(p): sha(p)
            for p in (
                Path(__file__).resolve(),
                Path(structure.__file__).resolve(),
                Path(worlds.__file__).resolve(),
                Path(worlds.prior.__file__).resolve(),
                Path(worlds.prior.inventory.__file__).resolve(),
                Path(worlds.prior.bridge.__file__).resolve(),
            )
        },
        "trusted_source": worlds.prior.bridge.trusted_provenance(),
        "environment": {"python": platform.python_version(), "networkx": nx.__version__},
        "world_sha256": worlds.prior.inventory.digest(worlds.prior.inventory.snapshot(world)),
        "tasks": [
            {
                "root": root,
                "depth": depth,
                "nodes": len(graph),
                "edges": sorted(map(list, graph.edges())),
                "node_kinds": dict(graph.nodes(data="kind")),
            }
            for root, depth, graph in selected
        ],
        "limitations": [
            "Same synthetic generator and vocabulary; novel rooted topology within recorded catalog, not new domain.",
            "Prior catalog covers native recipe worlds referenced by current TextCraft plans, not pretraining.",
            "Graphs over64 nodes cannot match selected graphs of64 nodes or fewer.",
            "Eight roots share one recipe world; no claim of eight independent domains.",
            "Depth/size design chosen after CPU-only novelty probe, before any new policy outcomes.",
        ],
    }
    save(output / "SELECTION.json", record)
    return record


def qualify(output: Path) -> dict:
    from transformers import AutoTokenizer

    selection = json.loads((output / "SELECTION.json").read_text())
    for filename, expected in selection["source_sha256"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("selected preparation source changed")
    worlds = runtime()
    world = worlds.world(selection["world_seed"])
    if (
        worlds.prior.inventory.digest(worlds.prior.inventory.snapshot(world))
        != selection["world_sha256"]
    ):
        raise ValueError("selected recipe world changed")
    tokenizer = AutoTokenizer.from_pretrained(
        worlds.prior.BASE, local_files_only=True, trust_remote_code=False
    )
    tasks, audits = [], []
    for row in selection["tasks"]:
        entry = {"root": row["root"]}
        try:
            task = build_task(world, row["root"], selection["world_seed"])
            tasks.append(task)
            entry.update(
                qualification=worlds.prior.qualify(task, world, tokenizer),
                statistics=worlds.prior.statistics(task, world),
            )
        except Exception as exc:
            entry["failure"] = f"{type(exc).__name__}: {exc}"
        audits.append(entry)
    with (output / "tasks.jsonl").open("x") as stream:
        for task in tasks:
            stream.write(json.dumps(task) + "\n")
    result = {
        "schema": "teaching-fresh-structure-qualified-v1",
        "selection_sha256": sha(output / "SELECTION.json"),
        "tasks_sha256": sha(output / "tasks.jsonl"),
        "world_seed": selection["world_seed"],
        "world_sha256": selection["world_sha256"],
        "task_count": len(tasks),
        "audits": audits,
        "ready": len(tasks) == 8
        and all(a.get("qualification", {}).get("native_score") == 1 for a in audits),
        "no_replacement": True,
        "inventory_rule": "Native conservative base-stock constructor, no distractors; not minimal inventory.",
        "interpretation": "Native teacher feasibility is not learned policy performance.",
    }
    save(output / "MANIFEST.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("freeze", "qualify"), required=True)
    args = parser.parse_args()
    result = freeze(args.output) if args.stage == "freeze" else qualify(args.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("ready", "task_count", "candidate_counts", "tasks", "audits")
            }
        )
    )
