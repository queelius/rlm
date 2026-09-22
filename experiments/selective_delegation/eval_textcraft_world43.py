"""One fixed teacher arm in the qualified changed recipe world; sixteen flat slots."""

import argparse
import json
from pathlib import Path

import eval_textcraft_public as public
import eval_textcraft_trained as shared
import prepare_textcraft_world43 as panel

c = shared.collector


def prepare(args):
    if not 0 < args.hours <= 0.75:
        raise ValueError("each of two arms is capped at45minutes")
    manifest = json.loads((args.prepared / "MANIFEST.json").read_text())
    if (
        not manifest["ready"]
        or manifest["world_sha256"] != panel.WORLD_SHA
        or c.inputs.sha(args.prepared / "tasks.jsonl") != manifest["tasks_sha256"]
    ):
        raise ValueError("qualified immutable world43 tasks required")
    adapter = (
        c.ROOT
        / (
            "textcraft-action-sft-001"
            if args.teacher == "privileged"
            else "textcraft-public-discovery-sft-001"
        )
        / "checkpoint-0023"
    )
    base_args = argparse.Namespace(**vars(args))
    base_args.prepared, base_args.adapter = c.ROOT / "textcraft-inputs-001", adapter
    plan, original, _ = shared.prepare(base_args, require_endpoint=False)
    tasks = list(map(json.loads, (args.prepared / "tasks.jsonl").read_text().splitlines()))
    if [(t["id"], t["goal"], t["misc"]["target_items"]) for t in tasks] != [
        (t["id"], t["goal"], t["misc"]["target_items"]) for t in original
    ]:
        raise ValueError("all eight original root goals/quantities must be preserved")
    if args.teacher == "privileged":
        binding = shared.endpoint(adapter)
    else:
        contract = json.loads((adapter.parent / "TEACHER-CONTRACT.json").read_text())
        if c.inputs.sha(adapter.parent / "TEACHER-CONTRACT.json") != public.CONTRACT_SHA:
            raise ValueError("fixed public teacher contract changed")
        public.validate_teacher(json.loads((adapter.parent / "PLAN.json").read_text()), contract)
        binding = shared.endpoint(
            adapter, training_plan_sha256=public.PLAN_SHA, rows_sha256=public.ROWS_SHA
        )
    condition = "world43_" + args.teacher
    jobs = [dict(j, condition=condition) for j in plan["jobs"] if j["prompt_profile"] == "original"]
    plan.update(
        schema="textcraft-world43-transfer-v1",
        profile="original",
        teacher=args.teacher,
        prepared=str(args.prepared.resolve()),
        tasks_sha256=manifest["tasks_sha256"],
        manifest_sha256=c.inputs.sha(args.prepared / "MANIFEST.json"),
        world_seed=43,
        world_sha256=panel.WORLD_SHA,
        jobs=jobs,
        planned_episodes=16,
        conditions=[condition],
        max_native_calls=1536,
        adapter=binding,
        fixed_adapter=str(adapter),
        training_plan_sha256=binding["training_plan_sha256"],
        initial_token_audit={
            "prompt_tokens": [
                a["qualification"]["initial_prompt_tokens"] for a in manifest["audits"]
            ]
        },
        prompt_difference="Same original public interface in changed world43 and reconstructed "
        "sufficient inventory; no world/recipe labels in prompts. Compare fixed teacher adapters "
        "within43; world42 versus43 difficulty/inventory is NOT matched.",
    )
    plan.pop("instruction_reminder", None)
    plan.pop("initial_token_audit_by_profile", None)
    for module in (panel, public, panel.inventory):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = c.inputs.sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = c.inputs.sha(Path(__file__))
    path = args.output / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable world43 arm PLAN changed")
    else:
        c.save(path, plan)
    return plan, tasks, binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", choices=("privileged", "public"), required=True)
    parser.add_argument("--hours", type=float, default=0.75)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan, tasks, binding = prepare(args)
    c.run(
        args,
        prepared_run=(plan, tasks),
        adapter=binding,
        world=None if args.prepare_only else panel.worlds()[1],
    )
