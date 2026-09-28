"""Fixed cumulative SFT23/46/69 readouts on the existing paired panel00 worlds."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import dose_common as d


def load_runtime(execution: str):
    directory = d.ROOT / (
        "source-textcraft-reserve-readout-001"
        if execution == "raw"
        else "source-textcraft-recipe-binder-003"
    )
    sys.path.insert(0, str(directory))
    import eval_textcraft as collector
    import eval_textcraft_trained as shared
    import prepare_textcraft_multiworld as worlds

    d.require(Path(collector.__file__).parent == directory, "mixed raw/binder runtime")
    return collector, shared, worlds


def endpoint(root: Path, teacher: str, step: int, shared) -> dict:
    original, prior, _ = d.original(teacher)
    if step == 23:
        return shared.endpoint(
            original / "checkpoint-0023",
            training_plan_sha256=d.sha(original / "PLAN.json"),
            rows_sha256=prior["rows_sha256"],
        )
    d.require(step in (46, 69), "fixed prospective endpoints only")
    import psutil

    training = root / f"train-{teacher}"
    plan = d.read(training / "PLAN.json")
    d.require(
        plan["teacher"] == teacher
        and plan["seed"] == d.SEED
        and plan["schema"] == "textcraft-matched-teacher-continuation-20260928-v1"
        and plan["fixed_readout_updates"] == [46, 69]
        and plan["rows_sha256"] == prior["rows_sha256"],
        "teacher-dose plan differs",
    )
    for path, expected in plan["source_sha256"].items():
        d.require(d.sha(Path(path)) == expected, "continuation source changed")
    checkpoint = training / f"checkpoint-{step:04d}"
    state, commit = d.verify_checkpoint(checkpoint)
    d.require(
        state["continuation_plan_sha256"] == d.sha(training / "PLAN.json"),
        "checkpoint belongs to another continuation",
    )
    step_paths = [training / "steps" / f"{i:04d}.json" for i in range(24, step + 1)]
    d.validate_dose(state, [d.read(p) for p in step_paths], step)
    owners = sorted(training.glob("OWNER-*.json"))
    d.require(bool(owners), "authenticated training owner required")
    terminals = {}
    for path in owners:
        owner = d.read(path)
        d.require(owner["source_sha256"] == plan["source_sha256"], "owner source differs")
        terminal_path = path.with_name(path.name.replace("OWNER-", "TERMINAL-"))
        terminal = d.read(terminal_path)
        terminals[str(terminal_path)] = dict(sha256=d.sha(terminal_path), status=terminal)
        try:
            process = psutil.Process(owner["pid"])
            d.require(
                abs(process.create_time() - owner["create_time"]) >= 0.01
                or process.status() == psutil.STATUS_ZOMBIE,
                "training owner still live",
            )
        except psutil.NoSuchProcess:
            pass
    d.require(
        max(t["status"]["step"] for t in terminals.values()) >= step,
        "owner has not reached fixed endpoint",
    )
    config = d.read(checkpoint / "adapter_config.json")
    d.require(
        config["r"] == 8
        and config["lora_alpha"] == 16
        and config["base_model_name_or_path"] == prior["model"],
        "adapter config differs",
    )
    return dict(
        path=str(checkpoint),
        sha256=commit["files"]["adapter_model.safetensors"],
        commit_sha256=d.sha(checkpoint / "COMMIT.json"),
        state=state,
        training_plan_sha256=d.sha(training / "PLAN.json"),
        training_rows_sha256=plan["rows_sha256"],
        original_training_plan_sha256=d.sha(original / "PLAN.json"),
        step_receipts_sha256={str(p): d.sha(p) for p in step_paths},
        terminals=terminals,
        qualification="Fixed committed boundary; later owner failure, if any, retained "
        "in terminals and never used to select a different checkpoint",
    )


def replace_endpoint(
    template: dict, binding: dict, *, teacher: str, step: int, world: int, execution: str
) -> dict:
    plan = copy.deepcopy(template)
    condition = f"dose_{teacher}_cp{step}_p00_w{world}_{execution}"
    plan.update(
        schema="textcraft-fixed-teacher-dose-readout-20260928-v1",
        teacher=teacher,
        cumulative_training_updates=step,
        cumulative_training_epochs=step // 23,
        adapter=binding,
        fixed_adapter=binding["path"],
        conditions=[condition],
        jobs=[dict(job, condition=condition) for job in template["jobs"]],
    )
    return plan


def inputs(args):
    collector, shared, worlds = load_runtime(args.execution)
    reference = d.ROOT / f"textcraft-breadth-p00-w{args.world}-soriginal-{args.execution}-001"
    template_path = reference / "PLAN.json"
    audit = d.read(reference / "NATIVE-AUDIT.json")
    d.require(
        d.sha(template_path) == audit["sha256"][str(template_path)], "paired reference PLAN changed"
    )
    template = d.read(template_path)
    prepared = Path(template["prepared"])
    for name, key in (("tasks.jsonl", "tasks_sha256"), ("MANIFEST.json", "manifest_sha256")):
        d.require(d.sha(prepared / name) == template[key], "frozen panel changed")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    binding = endpoint(args.root, args.teacher, args.step, shared)
    plan = replace_endpoint(
        template,
        binding,
        teacher=args.teacher,
        step=args.step,
        world=args.world,
        execution=args.execution,
    )
    plan.update(
        training_plan_sha256=binding["training_plan_sha256"],
        reference_plan=str(template_path),
        reference_plan_sha256=d.sha(template_path),
        endpoint_selection="Fixed original23, cumulative46 or69; no outcome selection",
        caveat=template["caveat"] + " Reused exploratory panel. Dose control preserves "
        "each original teacher dataset/history and row order; only continued optimization. "
        "Known-recipe denotes quantity-corrected privileged teacher, not discovery.",
    )
    for path in (
        Path(__file__).resolve(),
        Path(d.__file__).resolve(),
        Path(collector.__file__).resolve(),
        Path(worlds.__file__).resolve(),
    ):
        plan["source_sha256"][str(path)] = d.sha(path)
    world = worlds.world(args.world)
    digest = worlds.prior.inventory.digest(worlds.prior.inventory.snapshot(world))
    d.require(digest == plan["world_sha256"], "runtime recipe world differs")
    output = args.root / f"eval-{args.teacher}-cp{args.step}-p00-w{args.world}-{args.execution}"
    return collector, world, output, plan, tasks, binding


def run(args):
    collector, world, output, plan, tasks, binding = inputs(args)
    path = output / "PLAN.json"
    if path.exists():
        d.require(d.read(path) == plan, "immutable evaluation plan differs")
    else:
        d.save(path, plan)
    if args.audit:
        import analyze_textcraft_profiles as profiles
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            collector.BASE, local_files_only=True, trust_remote_code=False
        )
        result = profiles.audit.analyze(output, tokenizer, world=world)
        result.pop("paired", None)
        result.pop("depth_strata", None)
        d.save(output / "NATIVE-AUDIT.json", result)
    else:
        collector.run(
            argparse.Namespace(output=output, prepare_only=args.prepare_only),
            prepared_run=(plan, tasks),
            adapter=binding,
            world=world,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=d.OUTPUT)
    parser.add_argument("--teacher", choices=d.ORIGINALS, required=True)
    parser.add_argument("--step", type=int, choices=(23, 46, 69), required=True)
    parser.add_argument("--world", type=int, choices=(42, 50), default=42)
    parser.add_argument("--execution", choices=("raw", "binder"), default="raw")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    run(parser.parse_args())
