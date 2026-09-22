"""One fixed teacher on sixteen frozen fresh VAL roots in the original recipe world."""

import argparse
import json
from pathlib import Path

import eval_textcraft_public as public
import eval_textcraft_trained as shared

c = shared.collector
MANIFEST_SHA = "a145cb33aa6d2d62566c75bfcf432763369bf648f766bde6c99cda4bf6369a33"
TASKS_SHA = "da9f7498ffc136d24cc348523fe7be63586fc09bb2220ec05c35e86474382e8c"
WORLD_SHA = "f76ce3978c038be9624b3c7387aa30033970508c6afecb1f8f08315aa0693808"


def endpoint(teacher):
    if teacher not in ("privileged", "public"):
        raise ValueError("unknown teacher")
    adapter = (
        c.ROOT
        / (
            "textcraft-action-sft-001"
            if teacher == "privileged"
            else "textcraft-public-discovery-sft-001"
        )
        / "checkpoint-0023"
    )
    if teacher == "privileged":
        return shared.endpoint(adapter)
    contract_path = adapter.parent / "TEACHER-CONTRACT.json"
    if c.inputs.sha(contract_path) != public.CONTRACT_SHA:
        raise ValueError("fixed public teacher contract changed")
    public.validate_teacher(
        json.loads((adapter.parent / "PLAN.json").read_text()),
        json.loads(contract_path.read_text()),
    )
    return shared.endpoint(
        adapter, training_plan_sha256=public.PLAN_SHA, rows_sha256=public.ROWS_SHA
    )


def prepare(args):
    if not 0 < args.hours <= 1:
        raise ValueError("each fixed teacher arm is capped at60minutes")
    if (
        c.inputs.sha(args.prepared / "MANIFEST.json") != MANIFEST_SHA
        or c.inputs.sha(args.prepared / "tasks.jsonl") != TASKS_SHA
    ):
        raise ValueError("exact outcome-blind fresh16 input bytes required")
    manifest = json.loads((args.prepared / "MANIFEST.json").read_text())
    if not manifest["ready"] or manifest["world_sha256"] != WORLD_SHA:
        raise ValueError("qualified original world42 required")
    for name, key in (
        ("SELECTION.json", "selection_sha256"),
        ("REPLAY.json", "replay_sha256"),
        ("TOKEN-AUDIT.json", "token_audit_sha256"),
    ):
        if c.inputs.sha(args.prepared / name) != manifest[key]:
            raise ValueError("fresh input qualification changed")
    tasks = list(map(json.loads, (args.prepared / "tasks.jsonl").read_text().splitlines()))
    if len(tasks) != 16 or [t["id"] for t in tasks] != manifest["task_ids"]:
        raise ValueError("all16 frozen tasks required")
    binding = endpoint(args.teacher)
    base_args = argparse.Namespace(**vars(args))
    base_args.prepared = c.ROOT / "textcraft-inputs-001"
    base_args.adapter = Path(binding["path"])
    plan, _, _ = shared.prepare(base_args, require_endpoint=False)
    condition = "fresh_" + args.teacher
    jobs = [
        dict(
            episode_id=f"t{i:02d}-r{repeat}-flat-original",
            task_id=task["id"],
            repeat=repeat,
            seed=seed,
            policy="flat",
            condition=condition,
            prompt_profile="original",
        )
        for i, task in enumerate(tasks)
        for repeat, seed in enumerate(c.SEEDS)
    ]
    audit = json.loads((args.prepared / "TOKEN-AUDIT.json").read_text())
    lengths = [r["initial_prompt_tokens"]["flat"] for r in audit["tasks"]]
    plan.update(
        schema="textcraft-fresh16-teacher-readout-v1",
        profile="original",
        teacher=args.teacher,
        prepared=str(args.prepared.resolve()),
        tasks_sha256=TASKS_SHA,
        manifest_sha256=MANIFEST_SHA,
        world_seed=42,
        world_sha256=WORLD_SHA,
        jobs=jobs,
        planned_episodes=32,
        planned_per_condition=32,
        parent_tasks=16,
        conditions=[condition],
        max_native_calls=3072,
        adapter=binding,
        fixed_adapter=binding["path"],
        training_plan_sha256=binding["training_plan_sha256"],
        max_agent_depth={"flat": 0},
        budget_seconds=args.hours * 3600,
        initial_token_audit={"prompt_tokens": lengths, "max_prompt_plus_cap": max(lengths) + 256},
        prompt_difference="Original flat interface unchanged between fixed048/056 adapters. "
        "New root goals in shared world42, not independent recipe worlds.",
        caveat="16distinctroots/two correlated seeds;14share actualTRAIN32prerequisites. "
        "Freshness covers frozen currentR inventory, not every historical run.",
    )
    for key in (
        "instruction_reminder",
        "initial_token_audit_by_profile",
        "base_original_plan_sha256",
        "base_reminder_plan_sha256",
        "baseline_output",
        "baseline_plan_sha256",
        "baseline_analysis_sha256",
    ):
        plan.pop(key, None)
    for module in (public, shared):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = c.inputs.sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = c.inputs.sha(Path(__file__))
    path = args.output / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable fresh readout PLAN changed")
    else:
        c.save(path, plan)
    return plan, tasks, binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", choices=("privileged", "public"), required=True)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan, tasks, binding = prepare(args)
    c.run(args, prepared_run=(plan, tasks), adapter=binding)
