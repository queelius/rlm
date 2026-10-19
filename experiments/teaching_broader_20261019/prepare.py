"""Freeze four structurally novel goals per additional recipe world (approved October 19 design).

Thin isolated adapter over the sealed October 14 fresh-structure preparation. It reuses that
module's runtime, task construction, qualification and novelty index unchanged, and differs only
in the world seed, the per-world strata (two depth-3 and two depth-4 roots) and the inclusion of
the October 14 pilot world in the exclusion catalog.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx

SEALED = Path("/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/teaching_transfer_20261014")
sys.path.insert(0, str(SEALED))
import prepare as base  # noqa: E402  (sealed October 14 module)
import structure  # noqa: E402

PILOT = Path("/project/alex_phd/runs/rlm-teaching-goal-20261014/fresh-structure-pilot-001")
FEASIBILITY = Path("/project/alex_phd/runs/rlm-teaching-goal-20261014/BROADER-STRUCTURE-NATIVE-FEASIBILITY.json")
STRATA = {3: 2, 4: 2}
SELECTION_SEEDS = {2026101501: 2026101511, 2026101502: 2026101512}
TASKS_PER_WORLD = 4

sha, save = base.sha, base.save


def declared_roots(world_seed: int) -> list:
    rows = json.loads(FEASIBILITY.read_text())["tasks"]
    return [row["root"] for row in rows if row["world_seed"] == world_seed]


def freeze(output: Path, world_seed: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if world_seed not in SELECTION_SEEDS:
        raise ValueError("unplanned world seed")
    worlds = base.runtime()
    plan_paths = sorted(base.ROOT.glob("textcraft*/**/PLAN.json"))
    prior_seeds, prepared, inputs = {42}, set(), {}
    for path in plan_paths:
        plan = json.loads(path.read_text())
        inputs[str(path)] = sha(path)
        if type(plan.get("world_seed")) is int:
            prior_seeds.add(plan["world_seed"])
        if plan.get("prepared") and (Path(plan["prepared"]) / "tasks.jsonl").exists():
            prepared.add(Path(plan["prepared"]) / "tasks.jsonl")
    pilot_selection = json.loads((PILOT / "SELECTION.json").read_text())
    prior_seeds.add(pilot_selection["world_seed"])
    inputs[str(PILOT / "SELECTION.json")] = sha(PILOT / "SELECTION.json")
    if world_seed in prior_seeds:
        raise ValueError("candidate world already has a prior plan")
    prepared.update(base.ROOT.glob("textcraft*/tasks.jsonl"))
    prepared.add(PILOT / "tasks.jsonl")
    exposed_roots = set()
    for path in sorted(prepared):
        inputs[str(path)] = sha(path)
        for line in path.read_text().splitlines():
            task = json.loads(line)
            exposed_roots.update(task.get("misc", {}).get("target_items", {}))

    index, inventory = structure.StructureIndex(), []
    for seed in sorted(prior_seeds):
        world = worlds.world(seed)
        mapping = base.recipes(world)
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

    world = worlds.world(world_seed)
    mapping = base.recipes(world)
    candidates = []
    for root in sorted(mapping):
        depth = world.get_crafting_depth(root)
        if depth not in STRATA or root in exposed_roots:
            continue
        graph = structure.dependency_graph(mapping, root)
        if graph is not None and sum(i in mapping for i in graph) <= 12:
            candidates.append((root, depth, graph))
    selection_seed = SELECTION_SEEDS[world_seed]
    hash_ranked = [root for root, _, _ in structure.select(candidates, index, STRATA, selection_seed)]
    declared = declared_roots(world_seed)
    # Use the roots declared in the pre-approval native feasibility check (outcome-blind,
    # recorded before any new policy outcome); verify each against the same novelty rules.
    by_root = {root: (root, depth, graph) for root, depth, graph in candidates}
    selected, used = [], structure.StructureIndex()
    for root in declared:
        if root not in by_root:
            raise ValueError("declared root is not an eligible candidate: " + root)
        row = by_root[root]
        if index.contains(row[2]):
            raise ValueError("declared root duplicates an old topology: " + root)
        if not used.add(row[2]):
            raise ValueError("declared root duplicates a within-panel topology: " + root)
        selected.append(row)
    if dict(Counter(row[1] for row in selected)) != STRATA:
        raise ValueError("declared roots do not satisfy the depth strata")
    output.mkdir(parents=True)
    record = {
        "schema": "teaching-broader-structure-selection-v1",
        "frozen_epoch": time.time(),
        "world_seed": world_seed,
        "selection_seed": selection_seed,
        "strata": STRATA,
        "target_quantity": 2,
        "rule": "Exclude prior goal names (including the October 14 pilot); at most12 reachable "
        "recipe products and64 nodes; SHA256(seed:root) within depth. Exact directed rooted "
        "graph isomorphism, ignoring names/quantities, excludes old (12 catalog worlds) and "
        "within-panel duplicate structures.",
        "qualification_after_selection": True,
        "model_outcomes_used": False,
        "candidate_counts": dict(Counter(row[1] for row in candidates)),
        "prior_worlds": inventory,
        "exposed_goal_names": sorted(exposed_roots),
        "old_unique_topologies": sum(map(len, index.graphs.values())),
        "inventory_sha256": inputs,
        "selection_rule": "declared_feasibility_roots_verified_against_novelty_index",
        "declared_feasibility_roots": declared,
        "declared_feasibility_sha256": sha(FEASIBILITY),
        "hash_ranked_roots": hash_ranked,
        "hash_ranked_roots_match_declared": hash_ranked == declared,
        "source_sha256": {
            str(p): sha(p)
            for p in (
                Path(__file__).resolve(),
                Path(base.__file__).resolve(),
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
            "Four roots share one recipe world; two worlds are not independent domains.",
            "Graphs over64 nodes cannot match selected graphs of64 nodes or fewer.",
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
    worlds = base.runtime()
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
            task = base.build_task(world, row["root"], selection["world_seed"])
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
        "schema": "teaching-broader-structure-qualified-v1",
        "selection_sha256": sha(output / "SELECTION.json"),
        "tasks_sha256": sha(output / "tasks.jsonl"),
        "world_seed": selection["world_seed"],
        "world_sha256": selection["world_sha256"],
        "task_count": len(tasks),
        "audits": audits,
        "ready": len(tasks) == TASKS_PER_WORLD
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
    parser.add_argument("--world-seed", type=int, required=True)
    parser.add_argument("--stage", choices=("freeze", "qualify"), required=True)
    args = parser.parse_args()
    result = (
        freeze(args.output, args.world_seed) if args.stage == "freeze" else qualify(args.output)
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "ready",
                    "task_count",
                    "candidate_counts",
                    "hash_ranked_roots_match_declared",
                    "declared_feasibility_roots",
                )
            }
            | {"roots": [t["root"] for t in result.get("tasks", [])] or
               [a["root"] for a in result.get("audits", [])]}
        )
    )
