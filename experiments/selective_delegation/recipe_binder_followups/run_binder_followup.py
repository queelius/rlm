"""Prepare a fixed-seed/world recipe-binder follow-up without changing source003."""

import argparse
import copy
import json
from pathlib import Path

import textcraft_multiworld as multi


def output(world_seed, training_seed):
    return multi.c.ROOT / (
        f"textcraft-world{world_seed}-seed{multi.TRAINING_SEEDS[training_seed]}-"
        "public-recipe-binder-followup-001"
    )


def build(world_seed, training_seed):
    plan, tasks, adapter = multi.build(world_seed, training_seed, "public")
    plan = copy.deepcopy(plan)
    condition = f"world{world_seed}_seed{training_seed}_public_recipe_binder"
    plan.update(
        schema="textcraft-public-observed-recipe-binding-followup-v1",
        conditions=[condition],
        budget_seconds=1800,
        execution_assist=(
            "For craft only, preserve model target_item/output_count and replace ingredients "
            "only from one prior public get_info recipe if divisible; otherwise unchanged."
        ),
        caveat=(
            "Automatic execution assist, not a learned strategy or planner. This is a paired "
            "replication on an exposed eight-goal/two-seed reserve protocol."
        ),
    )
    plan["jobs"] = [dict(job, condition=condition) for job in plan["jobs"]]
    here = Path(__file__).resolve()
    plan["source_sha256"][str(here)] = multi.c.inputs.sha(here)
    source003 = Path(multi.__file__).resolve().parent / "SOURCE.json"
    plan["source_sha256"][str(source003)] = multi.c.inputs.sha(source003)
    return plan, tasks, adapter


def prepare(world_seed, training_seed):
    plan, tasks, adapter = build(world_seed, training_seed)
    path = output(world_seed, training_seed) / "PLAN.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("immutable binder follow-up PLAN differs")
    if not path.exists():
        multi.c.save(path, plan)
    return plan, tasks, adapter


def run(prepare_only, world_seed, training_seed):
    plan, tasks, adapter = prepare(world_seed, training_seed)
    multi.c.run(
        argparse.Namespace(output=output(world_seed, training_seed), prepare_only=prepare_only),
        prepared_run=(plan, tasks),
        adapter=adapter,
        world=None if prepare_only else multi.checked_world(plan),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, choices=(47, 48), required=True)
    parser.add_argument("--training-seed", choices=("original", "2291"), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.prepare_only, args.world, args.training_seed)
