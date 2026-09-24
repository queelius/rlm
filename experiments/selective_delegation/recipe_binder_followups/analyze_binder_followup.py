"""Native replay of a binder follow-up and receipt-pinned unchanged-public comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import analyze_textcraft_profiles as profiles
import run_binder_followup as followup
import textcraft_multiworld as multi


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_arm(report, output):
    for arm in report["arms"]:
        hashes = arm.get("sha256", {})
        if hashes.get(str(output / "PLAN.json")) == sha(output / "PLAN.json"):
            return arm
    raise ValueError("authenticated public baseline arm not found")


def run(world_seed, training_seed, baseline, baseline_audit, report_path):
    if report_path.exists() or report_path.with_suffix(".md").exists():
        raise FileExistsError("immutable analysis exists")
    plan, _, _ = followup.build(world_seed, training_seed)
    output = followup.output(world_seed, training_seed)
    if json.loads((output / "PLAN.json").read_text()) != plan:
        raise ValueError("saved binder PLAN differs")
    baseline_report = json.loads(baseline_audit.read_text())
    arm = public_arm(baseline_report, baseline)
    for path in [baseline / "PLAN.json", *(baseline / "episodes").glob("*.json")]:
        if arm["sha256"].get(str(path)) != sha(path):
            raise ValueError("authenticated baseline receipt differs")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True)
    result = profiles.audit.analyze(output, tokenizer, world=multi.checked_world(plan))
    result.pop("paired", None)
    result.pop("depth_strata", None)
    pilot_rows = {(r["task_id"], r["repeat"]): r for r in (
        json.loads(p.read_text()) for p in (output / "episodes").glob("*.json")
    )}
    baseline_rows = {(r["task_id"], r["repeat"]): r for r in (
        json.loads(p.read_text()) for p in (baseline / "episodes").glob("*.json")
    )}
    result["binder_minus_unchanged_public"] = profiles.compare(
        plan["jobs"], baseline_rows, pilot_rows
    )
    result["baseline_audit"] = str(baseline_audit)
    result["baseline_audit_sha256"] = sha(baseline_audit)
    result["baseline_groups"] = arm["groups"]
    result["caveat"] = plan["caveat"]
    multi.c.save(report_path, result)
    report_path.with_suffix(".md").write_text(
        "# Recipe-binding follow-up\n\n" + plan["caveat"] + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, choices=(47, 48), required=True)
    parser.add_argument("--training-seed", choices=("original", "2291"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.world, args.training_seed, args.baseline, args.baseline_audit, args.report)
