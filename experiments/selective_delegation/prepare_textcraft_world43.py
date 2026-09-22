"""Derived seed43 tasks with the frozen eight roots; no model-outcome selection."""

import argparse
import ast
import json
import sys
import typing
from functools import lru_cache
from pathlib import Path

import inspect_textcraft_worlds as inventory
import prepare_textcraft as inputs
import textcraft_bridge as bridge

BASE = Path(
    "/project/alex_phd/research-cache/models/"
    "Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554"
)
TASK_SHA = "16a6663385759a9a16fa7ca44e6601b4d491234f8b04ccb9f2bf522c7ced8ab3"
WORLD_SHA = "9d71915420a2844d4fc94afc4fd4c9a5e19c6cc220e98a21c50dc738fdb003ff"
TASK_SOURCE_SHA = "5cc082f13d6e6671aae2dfcf043f68a0d099fb3ec05d75e9a50885cb35d033c0"


def worlds():
    old = bridge.load_world()
    world = sys.modules["pinned_textcraft_synth_generator_d9c5857d"].SynthRecipeDatabase()
    world.generate_all_recipes(seed=43, items_per_domain_tier=25)
    if inventory.digest(inventory.snapshot(world)) != WORLD_SHA:
        raise ValueError("qualified seed43 world changed")
    return old, world


def construct(task, world):
    path = bridge.PACKAGE / "synth_tasks.py"
    if inputs.sha(path) != TASK_SOURCE_SHA:
        raise ValueError("official task planner changed")
    names = {"extract_base_materials_synth", "solve_crafting_task_synth"}
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(nodes) != 2:
        raise ValueError("official helper inventory changed")
    namespace = {**vars(typing), "SynthRecipeDatabase": type(world)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    targets = dict(task["misc"]["target_items"])
    needed = namespace["extract_base_materials_synth"](world, targets, {})
    # Official extraction conservatively ignores batch yield division. Do not call
    # this a minimum-material inventory. Preserve old irrelevant base distractors.
    initial = {k: v for k, v in task["misc"]["initial_inventory"].items() if k not in needed}
    initial.update(needed)
    solution = namespace["solve_crafting_task_synth"](world, targets, initial)
    if solution is None:
        raise ValueError("official planner found no construction; no replacement")
    return dict(
        task,
        derived_world_seed=43,
        misc=dict(
            task["misc"],
            initial_inventory=initial,
            target_items=targets,
            gold_trajectory=solution[0],
            num_craft_steps=len(solution[0]),
        ),
    )


def statistics(task, world):
    visited = set()

    @lru_cache(None)
    def depth(item):
        if world.is_base_item(item):
            return 0
        visited.add(item)
        recipes = world.get_recipes_for_item(item)
        if len(recipes) != 1:
            raise ValueError("single recipe world required")
        return 1 + max(depth(i) for i in recipes[0].ingredients)

    targets, initial = task["misc"]["target_items"], task["misc"]["initial_inventory"]
    longest = max(depth(item) for item in targets)
    return dict(
        declared_depth=max(world.get_crafting_depth(i) for i in targets),
        dependency_chain=longest,
        reachable_products=len(visited),
        ingredient_edges=sum(len(world.get_recipes_for_item(i)[0].ingredients) for i in visited),
        maximum_fanin=max(len(world.get_recipes_for_item(i)[0].ingredients) for i in visited),
        craft_count=len(task["misc"]["gold_trajectory"]),
        batches=sum(step["target"][1] for step in task["misc"]["gold_trajectory"]),
        root_batch_sizes={i: world.get_recipes_for_item(i)[0].result_count for i in targets},
        initial_item_types=len(initial),
        initial_units=sum(initial.values()),
    )


def qualify(task, world, tokenizer):
    frame = bridge.Frame(
        world,
        dict(task["misc"]["initial_inventory"]),
        task["misc"]["target_items"],
        bridge.Budget(),
        max_depth=0,
    )
    history, lengths, actions = [], [], []
    for step in task["misc"]["gold_trajectory"]:
        actions.extend(
            [
                dict(action="get_info", items=[step["target"][0]]),
                dict(
                    action="craft",
                    target_item=step["target"][0],
                    ingredients=step["ingredients"],
                    output_count=step["result_count"],
                ),
            ]
        )
    actions.append(dict(action="finish", message="done"))
    for action in actions:
        prompt = bridge.public_prompt(frame, history, goal=task["goal"])
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        lengths.append(len(ids))
        tokens = (
            len(
                tokenizer.encode(
                    json.dumps(action, separators=(",", ":")), add_special_tokens=False
                )
            )
            + 1
        )
        if len(ids) + 256 > 8192 or tokens > 256:
            raise ValueError("constructive trace exceeds unchanged prompt/action cap")
        frame.budget.reserve()
        frame.budget.charge(tokens)
        reply = frame.apply(action)
        if isinstance(reply, str) and reply.startswith("Error:"):
            raise ValueError(reply)
        history.append(dict(action=action, feedback=reply))
    score, _ = frame.score()
    return dict(
        native_score=score,
        calls=len(actions),
        output_tokens=frame.budget.output_tokens,
        initial_prompt_tokens=lengths[0],
        max_prompt_plus_cap=max(lengths) + 256,
    )


def prepare(root, output):
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError("immutable panel exists")
    original = root / "textcraft-inputs-001/tasks.jsonl"
    if inputs.sha(original) != TASK_SHA:
        raise ValueError("frozen eight roots changed")
    tasks = list(map(json.loads, original.read_text().splitlines()))
    output.mkdir(parents=True)
    inputs.save(
        output / "SELECTION.json",
        dict(
            task_ids=[t["id"] for t in tasks],
            source_sha256=TASK_SHA,
            rule="All eight prior roots and quantities; no replacement/filter",
            before_native_replay=True,
            model_outcomes_used=False,
        ),
    )
    old, world = worlds()
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
    changed, audits = [], []
    for task in tasks:
        row = dict(task_id=task["id"], world42=statistics(task, old))
        try:
            new = construct(task, world)
            row.update(world43=statistics(new, world), qualification=qualify(new, world, tokenizer))
        except Exception as exc:
            new = dict(task, construction_failure=f"{type(exc).__name__}: {exc}")
            row["failure"] = new["construction_failure"]
        changed.append(new)
        audits.append(row)
    with (output / "tasks.jsonl").open("x") as stream:
        for task in changed:
            stream.write(json.dumps(task) + "\n")
    manifest = dict(
        schema="textcraft-derived-world43-panel-v1",
        world_seed=43,
        world_sha256=WORLD_SHA,
        tasks_sha256=inputs.sha(output / "tasks.jsonl"),
        original_tasks_sha256=TASK_SHA,
        task_count=8,
        audits=audits,
        ready=all(a.get("qualification", {}).get("native_score") == 1 for a in audits),
        inventory_rule="Official extract_base_materials_synth conservative required-base upper "
        "bound, preserving original irrelevant base distractors/quantities. Not minimum stock; "
        "different public inventories and difficulty, not matched alternate evidence.",
        derived_not_official_new_split=True,
        no_model_outcome_selection=True,
        trusted_source=bridge.trusted_provenance(),
        source_sha256={
            str(p): inputs.sha(p)
            for p in (
                Path(__file__).resolve(),
                Path(inventory.__file__).resolve(),
                Path(bridge.__file__).resolve(),
                Path(inputs.__file__).resolve(),
                bridge.PACKAGE / "synth_tasks.py",
            )
        },
    )
    inputs.save(output / "MANIFEST.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output)))
