"""Read fixed visible-order checkpoint23 on an existing paired breadth panel."""

import argparse
import copy
import json
import sys
from pathlib import Path

import prepare


def load_runtime(execution: str):
    directory = prepare.ROOT / (
        "source-textcraft-reserve-readout-001"
        if execution == "raw"
        else "source-textcraft-recipe-binder-003"
    )
    sys.path.insert(0, str(directory))
    import eval_textcraft as collector
    import eval_textcraft_trained as shared
    import prepare_textcraft_multiworld as worlds

    prepare.require(Path(collector.__file__).parent == directory, "mixed raw/binder runtime")
    return collector, shared, worlds


def inputs(args: argparse.Namespace) -> tuple:
    collector, shared, worlds = load_runtime(args.execution)
    seed_label = "original" if args.seed == 2026092208 else "2291"
    template_directory = prepare.ROOT / (
        f"textcraft-breadth-p{args.panel:02d}-w{args.world}-s{seed_label}-{args.execution}-001"
    )
    template_path = template_directory / "PLAN.json"
    audit = json.loads((template_directory / "NATIVE-AUDIT.json").read_text())
    prepare.require(
        prepare.sha(template_path) == audit["sha256"][str(template_path)],
        "paired reference PLAN changed",
    )
    template = json.loads(template_path.read_text())
    prepared = Path(template["prepared"])
    for name, key in (("tasks.jsonl", "tasks_sha256"), ("MANIFEST.json", "manifest_sha256")):
        prepare.require(prepare.sha(prepared / name) == template[key], "frozen panel changed")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    training = args.root / f"train-{args.mode}-seed{args.seed}"
    train_plan = json.loads((training / "PLAN.json").read_text())
    prepare.require(train_plan["seed"] == args.seed, "wrong training seed")
    manifest = json.loads((args.root / args.mode / "MANIFEST.json").read_text())
    binding = shared.endpoint(
        training / "checkpoint-0023",
        training_plan_sha256=prepare.sha(training / "PLAN.json"),
        rows_sha256=manifest["rows_sha256"],
    )
    plan = copy.deepcopy(template)
    condition = f"order_{args.mode}_s{args.seed}_p{args.panel:02d}_w{args.world}_{args.execution}"
    plan.update(
        schema="textcraft-visible-order-readout-v1",
        teacher=args.mode,
        adapter=binding,
        fixed_adapter=binding["path"],
        training_plan_sha256=binding["training_plan_sha256"],
        conditions=[condition],
        jobs=[dict(job, condition=condition) for job in template["jobs"]],
        teaching_order_contract_sha256=prepare.sha(training / "ORDER-CONTROL-CONTRACT.json"),
        reference_plan=str(template_path),
        reference_plan_sha256=prepare.sha(template_path),
        endpoint_selection="Fixed checkpoint23 after one matched-action epoch; no selection",
        caveat=template["caveat"]
        + " Reused exploratory panel; new teacher conditioning histories, "
        "matched actions/labels/minibatch target order. Offline oracle schedule, visible names.",
    )
    for path in (Path(__file__).resolve(), Path(prepare.__file__).resolve()):
        plan["source_sha256"][str(path)] = prepare.sha(path)
    world = worlds.world(args.world)
    digest = worlds.prior.inventory.digest(worlds.prior.inventory.snapshot(world))
    prepare.require(digest == plan["world_sha256"], "runtime world differs from frozen panel")
    output = args.root / (
        f"eval-{args.mode}-s{args.seed}-p{args.panel:02d}-w{args.world}-{args.execution}"
    )
    return collector, world, output, plan, tasks, binding


def run(args: argparse.Namespace) -> None:
    collector, world, output, plan, tasks, binding = inputs(args)
    path = output / "PLAN.json"
    if path.exists():
        prepare.require(json.loads(path.read_text()) == plan, "immutable evaluation plan differs")
    else:
        prepare.save(path, plan)
    if args.audit:
        import analyze_textcraft_profiles as profiles
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            collector.BASE, local_files_only=True, trust_remote_code=False
        )
        result = profiles.audit.analyze(output, tokenizer, world=world)
        result.pop("paired", None)
        result.pop("depth_strata", None)
        prepare.save(output / "NATIVE-AUDIT.json", result)
        return
    collector.run(
        argparse.Namespace(output=output, prepare_only=args.prepare_only),
        prepared_run=(plan, tasks),
        adapter=binding,
        world=world,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=prepare.OUTPUT)
    parser.add_argument("--mode", choices=prepare.MODES, required=True)
    parser.add_argument("--seed", type=int, choices=(2026092208, 2026092291), default=2026092208)
    parser.add_argument("--panel", type=int, choices=range(6), default=0)
    parser.add_argument("--world", type=int, choices=(42, 50, 51, 52), default=42)
    parser.add_argument("--execution", choices=("raw", "binder"), default="raw")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    run(parser.parse_args())
