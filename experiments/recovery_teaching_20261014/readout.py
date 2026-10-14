"""Full-start A evaluation of warm, one-step recovery and ordinary teaching."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def fit_module():
    path = Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("recovery_fit", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def endpoint(directory):
    fit = fit_module()
    summary = fit.read(directory / "SUMMARY.json")
    if not summary.get("complete") or not summary.get("endpoint_usable"):
        raise ValueError("complete usable teaching endpoint required")
    cp = directory / "boundaries/sample-0001/checkpoint-0001"
    if summary["endpoint"] != str(cp) or summary["actual_optimizer_steps"] != [1]:
        raise ValueError("fixed one-step endpoint required")
    commit = fit.read(cp / "COMMIT.json")
    for name in ("adapter_model.safetensors", "adapter_config.json", "STATE.json"):
        if fit.sha(cp / name) != commit["files"][name]:
            raise ValueError("committed endpoint changed")
    state = fit.read(cp / "STATE.json")
    if state["step"] != 1 or state["plan_sha256"] != fit.sha(directory / "PLAN.json"):
        raise ValueError("endpoint training identity changed")
    return dict(
        path=str(cp),
        sha256=commit["files"]["adapter_model.safetensors"],
        commit_sha256=fit.sha(cp / "COMMIT.json"),
    )


def prepare(args, p=None):
    fit = fit_module()
    p = p or fit.native()
    if args.arm not in ("warm", "recovery", "ordinary"):
        raise ValueError("fixed three readout arms required")
    if fit.sha(fit.DATA / "MANIFEST.json") != fit.MANIFEST_SHA:
        raise ValueError("frozen teaching provenance changed")
    data = fit.read(fit.DATA / "MANIFEST.json")
    source = Path(data["source_collection"]) / "PLAN.json"
    if fit.sha(source) != data["source_collection_plan_sha256"]:
        raise ValueError("original A reference changed")
    plan = fit.read(source)
    directory = Path(plan["prepared"])
    if (
        fit.sha(directory / "MANIFEST.json") != data["source_dataset_manifest_sha256"]
        or fit.sha(directory / "tasks.jsonl") != data["source_tasks_sha256"]
    ):
        raise ValueError("original normal-start A tasks changed")
    tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text().splitlines()]
    if [t["id"] for t in tasks] != data["arms"]["recovery"]["task_ids"]:
        raise ValueError("exact ordered eight A identities required")
    if args.arm == "warm":
        binding = p.adapter_binding(Path(data["future_training_start"]["path"]))
        if binding != data["future_training_start"]:
            raise ValueError("fixed warm actor differs")
    else:
        binding = endpoint(fit.OUTPUT / f"train-{args.arm}")
        training = fit.read(fit.OUTPUT / f"train-{args.arm}" / "PLAN.json")
        if training["arm"] != args.arm or training["data_manifest_sha256"] != fit.MANIFEST_SHA:
            raise ValueError("teaching arm identity changed")
    plan.update(
        schema="recovery-teaching-normal-start-readout-20261014-v1",
        phase="readout",
        update=0 if args.arm == "warm" else 1,
        arm=args.arm,
        adapter=binding,
        planned_episodes=16,
        budget_seconds=5400,
        jobs=[
            dict(
                episode_id=f"t{i:02d}-r{repeat}-flat",
                task_id=t["id"],
                repeat=repeat,
                seed=202609290210 + repeat,
                policy="flat",
                prompt_profile="original",
                condition=f"recovery_teaching_{args.arm}",
            )
            for repeat in range(2)
            for i, t in enumerate(tasks)
        ],
        source_sha256=p.source_pins(Path(__file__), "raw"),
        teaching_manifest_sha256=fit.MANIFEST_SHA,
        endpoint_rule="Fixed warm and one-step endpoints, ordinary starts; no B selection",
        caveat="Exploratory TRAIN A efficacy, not untouched generalization. Two stochastic "
        "draws per root; teacher histories and8 labels differ. Not new RL.",
    )
    for path in (Path(fit.__file__), fit.LIBRARY / "collect.py"):
        plan["source_sha256"][str(path)] = fit.sha(path)
    output = args.output / "PLAN.json"
    if output.exists():
        if fit.read(output) != plan:
            raise ValueError("immutable readout plan differs")
    else:
        p.c.save(output, plan)
    return plan, tasks


def run(args):
    fit = fit_module()
    if args.output.resolve().parent != fit.OUTPUT:
        raise ValueError("scoped recovery readout output required")
    path = fit.LIBRARY / "collect.py"
    sys.path.insert(0, str(fit.LIBRARY))
    spec = importlib.util.spec_from_file_location("recovery_native_collect", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.p.__file__).resolve() != fit.LIBRARY / "probe_common.py":
        raise ValueError("mixed native collector import")
    p = fit.native()
    module.p = SimpleNamespace(**vars(p))
    module.p.prepare_collection = lambda native_args, entrypoint: prepare(native_args, p)
    args.mode, args.phase, args.update, args.hours = "raw", "readout", 1, 1.5
    module.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("warm", "recovery", "ordinary"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
