"""CPU replay of identical-public-frame seed42/43 feasibility; no model calls."""

import argparse
import ast
import copy
import hashlib
import json
import sys
import time
import typing
from collections import Counter
from pathlib import Path

import prepare_textcraft_public as public

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
REFERENCE_SHA = "212410a12a6bb7e6fa772953664e3f6f8b94a9e785abf3a63ecd1eb27cb744d1"
bridge = public.bridge


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot(world):
    return {
        item: [
            {"ingredients": r.ingredients, "result_count": r.result_count, "depth": r.depth}
            for r in recipes
        ]
        for item, recipes in sorted(world.recipes.items())
    }


def replay(root, reference_path):
    from transformers import AutoTokenizer

    started = time.monotonic()
    if sha(reference_path) != REFERENCE_SHA:
        raise ValueError("immutable feasibility001 identity differs")
    expected = json.loads(reference_path.read_text())
    expected_sources = {Path(p).name: h for p, h in expected["source_sha256"].items()}
    source = bridge.PACKAGE / "synth_tasks.py"
    paths = [
        source,
        bridge.GENERATOR,
        bridge.ENV,
        Path(bridge.__file__),
        Path(public.__file__),
        Path(public.reference.__file__),
        root / "textcraft-inputs-001/tasks.jsonl",
    ]
    for path in paths:
        if sha(path) != expected_sources[path.name]:
            raise ValueError("source/input differs: " + str(path))
    a = bridge.load_world()
    generator = sys.modules["pinned_textcraft_synth_generator_d9c5857d"]
    b = generator.SynthRecipeDatabase()
    b.generate_all_recipes(seed=43, items_per_domain_tier=25)
    worlds = [a, b]
    world_hashes = {
        str(seed): digest(snapshot(world)) for seed, world in zip([42, 43], worlds, strict=True)
    }
    if world_hashes != expected["world_sha256"]:
        raise ValueError("regenerated recipe worlds differ")
    names = {"extract_base_materials_synth", "solve_crafting_task_synth"}
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    ast_hashes = {
        n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
        for n in nodes
    }
    if ast_hashes != expected["extracted_trusted_AST_sha256"]:
        raise ValueError("reviewed official function ASTs differ")
    namespace = {**vars(typing), "SynthRecipeDatabase": generator.SynthRecipeDatabase}
    # Only the two reviewed/hash-pinned official function bodies, never model-generated code.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    extract, solve = (
        namespace[name] for name in ("extract_base_materials_synth", "solve_crafting_task_synth")
    )
    tasks = [json.loads(line) for line in paths[-1].read_text().splitlines()]
    assert len(tasks) == 8
    tokenizer = AutoTokenizer.from_pretrained(
        public.reference.BASE, local_files_only=True, trust_remote_code=False
    )
    pairs, traces = [], {}
    for original in tasks:
        targets = original["misc"]["target_items"]
        requirements = []
        for world in worlds:
            solution = solve(world, targets, extract(world, targets, {}))
            assert solution is not None
            consumed = Counter()
            for step in solution[0]:
                for item, count in step["ingredients"].items():
                    if world.is_base_item(item):
                        consumed[item] += count
            requirements.append(dict(consumed))
        inventory = {
            item: max(req.get(item, 0) for req in requirements)
            for item in sorted(set().union(*requirements))
        }
        assert not set(targets) & set(inventory)
        prompts, outcomes = [], []
        for seed, world in zip([42, 43], worlds, strict=True):
            task = copy.deepcopy(original)
            task["misc"]["initial_inventory"] = inventory.copy()
            solution = solve(world, targets, inventory)
            assert solution is not None
            task["misc"]["gold_trajectory"] = solution[0]
            frame = bridge.Frame(world, inventory.copy(), targets, bridge.Budget(), 0)
            prompts.append(bridge.public_prompt(frame, [], goal=task["goal"]))
            qualified, rows = public.trajectory(task, world, tokenizer)
            legacy, legacy_rows = public.reference.trajectory(task, world, tokenizer)
            info = bridge.Frame(world, inventory.copy(), targets, bridge.Budget(), 0).apply(
                {"action": "get_info", "items": list(targets)}
            )
            traces[f"{task['id']}-world{seed}"] = {
                "public_rows_sha256": digest(rows),
                "legacy_rows_sha256": digest(legacy_rows),
            }
            outcomes.append(
                {
                    "seed": seed,
                    "public_teacher": qualified,
                    "public_first_action": json.loads(rows[0]["target"]) if rows else None,
                    "public_action_sha256": digest([json.loads(row["target"]) for row in rows]),
                    "legacy_teacher": legacy,
                    "legacy_first_action": json.loads(legacy_rows[0]["target"])
                    if legacy_rows
                    else None,
                    "legacy_first_craft": solution[0][0],
                    "root_query_reply": info,
                    "required_base_for_native_legacy_plan": requirements[len(outcomes)],
                    "public_max_prompt_plus_cap": max(
                        (row["prompt_tokens"] + 256 for row in rows), default=0
                    ),
                }
            )
        assert prompts[0] == prompts[1], original["id"]
        pairs.append(
            {
                "task_id": original["id"],
                "goal": original["goal"],
                "target_items": targets,
                "common_initial_inventory": inventory,
                "identical_initial_prompt": True,
                "initial_prompt_sha256": hashlib.sha256(prompts[0].encode()).hexdigest(),
                "legacy_first_actions_differ": (
                    outcomes[0]["legacy_first_action"] != outcomes[1]["legacy_first_action"]
                ),
                "root_query_replies_differ": (
                    outcomes[0]["root_query_reply"] != outcomes[1]["root_query_reply"]
                ),
                "worlds": outcomes,
            }
        )
    pairs = json.loads(json.dumps(pairs))  # Official target tuples serialize as JSON lists.
    if pairs != expected["pairs"]:
        raise ValueError("pair inventories/prompts/teacher outcomes differ from001")
    return {
        "schema": "textcraft-observational-ambiguity-replay-v1",
        "verified": True,
        "reproduced_report": str(reference_path),
        "reproduced_report_sha256": sha(reference_path),
        "summary": expected["summary"],
        "pairs": pairs,
        "world_sha256": world_hashes,
        "extracted_trusted_AST_sha256": ast_hashes,
        "reconstructed_trace_sha256": traces,
        "source_sha256": {str(p): sha(p) for p in [*paths, Path(__file__)]},
        "elapsed_cpu_seconds": time.monotonic() - started,
        "scope": "Deterministic reproduction of001; no selection, new task panel or GPU calls. "
        "Native inventories are sufficient, not proven minimal; differing first labels do not "
        "prove unique required actions or teacher-order causality.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=ROOT / "analysis-textcraft-observational-ambiguity-001.json",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise ValueError("immutable output exists")
    result = replay(args.root, args.verify_against)
    if args.report:
        with args.report.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"verified": result["verified"], "summary": result["summary"]}))
