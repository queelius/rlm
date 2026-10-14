"""Evaluate fixed teaching/dose actors on a prospectively frozen structural pilot."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import prepare as panel

SEEDS = (2026101403, 2026101404)
REFERENCES = {
    "known23": "textcraft-breadth-p00-w42-soriginal-corrected-raw-001",
    "repair23": "textcraft-teaching-order-20260928-001/eval-stable_visible-s2026092208-p00-w42-raw",
    "discovery23": "textcraft-breadth-p00-w42-soriginal-raw-001",
    "known69": "textcraft-teaching-dose-20260928-001/eval-known_recipe-cp69-p00-w42-raw",
}


def jobs(tasks: list, condition: str) -> list:
    if len(tasks) != 8 or len({task["id"] for task in tasks}) != 8:
        raise ValueError("eight unique tasks required")
    return [
        dict(
            episode_id=f"t{i:02d}-r{repeat}-flat-original",
            task_id=task["id"],
            repeat=repeat,
            seed=seed,
            policy="flat",
            prompt_profile="original",
            condition=condition,
        )
        for i, task in enumerate(tasks)
        for repeat, seed in enumerate(SEEDS)
    ]


def validate_binding(binding: dict) -> None:
    """Evaluation-only actor check; optimizer state is not loaded or rehashed."""
    path = Path(binding["path"])
    if panel.sha(path / "COMMIT.json") != binding["commit_sha256"]:
        raise ValueError("actor commit changed")
    commit = json.loads((path / "COMMIT.json").read_text())
    for name in ("adapter_model.safetensors", "adapter_config.json", "STATE.json"):
        if panel.sha(path / name) != commit["files"][name]:
            raise ValueError("actor file changed: " + name)
    if (
        commit["files"]["adapter_model.safetensors"] != binding["sha256"]
        or json.loads((path / "STATE.json").read_text()) != binding["state"]
        or panel.sha(path.parent / "PLAN.json") != binding["training_plan_sha256"]
    ):
        raise ValueError("actor binding changed")


def prepare(prepared: Path, output: Path, arm: str):
    worlds = panel.runtime()
    sys.path.insert(0, str(panel.SOURCE))
    import eval_textcraft as collector

    if Path(collector.__file__).parent != panel.SOURCE:
        raise ValueError("mixed collector runtime")
    selection = json.loads((prepared / "SELECTION.json").read_text())
    manifest = json.loads((prepared / "MANIFEST.json").read_text())
    if (
        not manifest["ready"]
        or manifest["selection_sha256"] != panel.sha(prepared / "SELECTION.json")
        or manifest["tasks_sha256"] != panel.sha(prepared / "tasks.jsonl")
    ):
        raise ValueError("fresh qualification differs")
    for filename, expected in selection["source_sha256"].items():
        if panel.sha(Path(filename)) != expected:
            raise ValueError("frozen preparation source changed")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    if [list(task["misc"]["target_items"])[0] for task in tasks] != [
        row["root"] for row in selection["tasks"]
    ]:
        raise ValueError("selected root order differs")
    reference = panel.ROOT / REFERENCES[arm] / "PLAN.json"
    audit = json.loads(reference.with_name("NATIVE-AUDIT.json").read_text())
    if panel.sha(reference) != audit["sha256"][str(reference)]:
        raise ValueError("previously audited actor reference changed")
    template = json.loads(reference.read_text())
    binding = template["adapter"]
    validate_binding(binding)
    condition = f"fresh_structure_{arm}"
    keys = (
        "model",
        "model_manifest_sha256",
        "trusted_source",
        "profile",
        "environment",
        "root_depth",
        "max_agent_depth",
        "max_global_calls",
        "max_global_output_tokens",
        "input_plus_output_limit",
        "max_new_tokens",
        "truncation",
        "sampling",
        "seed_rule",
        "child_return",
        "source_sha256",
    )
    plan = {key: copy.deepcopy(template[key]) for key in keys if key in template}
    plan.update(
        schema="teaching-fresh-structure-readout-v1",
        adapter=binding,
        fixed_adapter=binding["path"],
        adapters=None,
        optimizer=False,
        training_seed=2026092208,
        cumulative_training_updates=69 if arm == "known69" else 23,
        training_plan_sha256=binding["training_plan_sha256"],
        prepared=str(prepared.resolve()),
        manifest_sha256=panel.sha(prepared / "MANIFEST.json"),
        tasks_sha256=manifest["tasks_sha256"],
        selection_sha256=manifest["selection_sha256"],
        world_seed=manifest["world_seed"],
        world_sha256=manifest["world_sha256"],
        jobs=jobs(tasks, condition),
        conditions=[condition],
        parent_tasks=8,
        planned_episodes=16,
        planned_per_condition=16,
        planned_per_policy=16,
        budget_seconds=2700,
        max_native_calls=1536,
        seeds=list(SEEDS),
        execution_mode="raw",
        endpoint_selection=(
            "Fixed before fresh model outcomes; original23/repair23/discovery23/original69"
        ),
        reference_plan=str(reference),
        reference_plan_sha256=panel.sha(reference),
        caveat="Fresh graph-selected pilot, same vocabulary/generator and one fitted seed. "
        "Eight roots share prerequisites; not independent domains or new RL. "
        "Conservative base inventory and no distractors differ from earlier panels.",
    )
    for file in (
        Path(__file__).resolve(),
        Path(panel.__file__).resolve(),
        Path(panel.structure.__file__).resolve(),
    ):
        plan["source_sha256"][str(file)] = panel.sha(file)
    world = worlds.world(manifest["world_seed"])
    if (
        worlds.prior.inventory.digest(worlds.prior.inventory.snapshot(world))
        != plan["world_sha256"]
    ):
        raise ValueError("native world differs")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable evaluation plan differs")
    else:
        panel.save(path, plan)
    return collector, world, plan, tasks, binding


def run(args):
    collector, world, plan, tasks, binding = prepare(args.prepared, args.output, args.arm)
    if args.audit:
        import analyze_textcraft_profiles as profiles
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            collector.BASE, local_files_only=True, trust_remote_code=False
        )
        result = profiles.audit.analyze(args.output, tokenizer, world=world)
        result.pop("paired", None)
        result.pop("depth_strata", None)
        panel.save(args.output / "NATIVE-AUDIT.json", result)
    else:
        collector.run(
            argparse.Namespace(output=args.output, prepare_only=args.prepare_only),
            prepared_run=(plan, tasks),
            adapter=binding,
            world=world,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=REFERENCES, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    run(parser.parse_args())
