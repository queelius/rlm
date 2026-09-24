"""Quantity-corrected teacher endpoints on the unchanged frozen breadth panel."""

import argparse
import json
from pathlib import Path

import textcraft_breadth as b

ROWS_SHA = "24ea72cb1242f2e0d819d8fb115737864de03fb750e064f145f9ec48a245e6d6"


def output(panel, world, seed, mode):
    return b.ROOT / f"textcraft-breadth-p{panel:02d}-w{world}-s{seed}-corrected-{mode}-001"


def build(panel, world, seed, mode):
    plan, tasks, _ = b.build(panel, world, seed, mode)
    directory = b.ROOT / f"textcraft-quantity-matched-seed{b.m.TRAINING_SEEDS[seed]}-001"
    adapter = b.m.original.shared.endpoint(
        directory / "checkpoint-0023",
        training_plan_sha256=b.m.c.inputs.sha(directory / "PLAN.json"),
        rows_sha256=ROWS_SHA,
    )
    condition = f"breadth_p{panel}_w{world}_{seed}_corrected_{mode}"
    plan.update(
        teacher="quantity_corrected_original",
        adapter=adapter,
        fixed_adapter=adapter["path"],
        training_plan_sha256=adapter["training_plan_sha256"],
        conditions=[condition],
    )
    plan["jobs"] = [dict(j, condition=condition) for j in plan["jobs"]]
    plan["source_sha256"][str(Path(__file__).resolve())] = b.m.c.inputs.sha(Path(__file__))
    return plan, tasks, adapter


def compare(args):
    import analyze_textcraft_profiles as profiles

    arms, rows, plans = [], [], []
    for mode in ("raw", "binder"):
        directory = output(args.panel, args.world, args.seed, mode)
        report = directory / "NATIVE-AUDIT.json"
        arm = json.loads(report.read_text())
        for path in [directory / "PLAN.json", *(directory / "episodes").glob("*.json")]:
            if arm["sha256"].get(str(path)) != b.m.c.inputs.sha(path):
                raise ValueError("native-audited corrected receipt changed")
        plans.append(json.loads((directory / "PLAN.json").read_text()))
        arms.append(
            dict(native_audit=str(report), sha256=b.m.c.inputs.sha(report), groups=arm["groups"])
        )
        rows.append(
            {
                (r["task_id"], r["repeat"]): r
                for r in (
                    json.loads(p.read_text()) for p in (directory / "episodes").glob("*.json")
                )
            }
        )
    if b.m.normalize(plans[0]["jobs"]) != b.m.normalize(plans[1]["jobs"]):
        raise ValueError("paired slots differ")
    for key in ("budget_seconds", "tasks_sha256", "world_sha256", "fixed_adapter"):
        if plans[0][key] != plans[1][key]:
            raise ValueError("paired contract differs")
    b.m.c.save(
        args.compare,
        dict(
            arms=arms,
            binder_minus_raw=profiles.compare(plans[0]["jobs"], *rows),
            caveat=plans[0]["caveat"],
            teacher="quantity_corrected_original",
            planned_per_arm=16,
        ),
    )


def run(args):
    plan, tasks, adapter = build(args.panel, args.world, args.seed, args.mode)
    directory = output(args.panel, args.world, args.seed, args.mode)
    if (directory / "PLAN.json").exists():
        if json.loads((directory / "PLAN.json").read_text()) != plan:
            raise ValueError("immutable corrected PLAN differs")
    else:
        b.m.c.save(directory / "PLAN.json", plan)
    if args.report:
        import analyze_textcraft_profiles as profiles
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(b.m.c.BASE, local_files_only=True)
        result = profiles.audit.analyze(directory, tokenizer, world=b.m.checked_world(plan))
        result.pop("paired", None)
        result.pop("depth_strata", None)
        b.m.c.save(args.report, result)
        return
    b.m.c.run(
        argparse.Namespace(output=directory, prepare_only=args.prepare_only),
        prepared_run=(plan, tasks),
        adapter=adapter,
        world=None if args.prepare_only else b.m.checked_world(plan),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=int, default=0)
    parser.add_argument("--world", type=int, default=42)
    parser.add_argument("--seed", choices=("original", "2291"), default="original")
    parser.add_argument("--mode", choices=("raw", "binder"), default="raw")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    compare(args) if args.compare else run(args)
