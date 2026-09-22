"""Native-audited procedural prompt versus fixed052 original, all sixteen slots."""

import argparse
import json
from pathlib import Path

import analyze_textcraft_profiles as profiles
import eval_textcraft_procedure as control

audit = profiles.audit


def pair(jobs, old_rows, new_rows):
    left = {
        (r["task_id"], r["repeat"]): r
        for r in old_rows.values()
        if r["condition"] == "trained_original"
    }
    right = {(r["task_id"], r["repeat"]): r for r in new_rows.values()}
    return profiles.compare(jobs, left, right)


def analyze(output, baseline_report):
    from transformers import AutoTokenizer

    c = control.collector
    if c.inputs.sha(baseline_report) != control.REFERENCE_REPORT_SHA:
        raise ValueError("fixed independently audited052 report required")
    baseline = audit.read(baseline_report)
    old_output = c.ROOT / "textcraft-trained-readout-001"
    old_plan = audit.read(old_output / "PLAN.json")
    if c.inputs.sha(old_output / "PLAN.json") != control.REFERENCE_PLAN_SHA:
        raise ValueError("fixed052 PLAN changed")
    plan = audit.read(output / "PLAN.json")
    if (
        plan["schema"] != "textcraft-procedure-control-v1"
        or plan["procedural_instruction"] != c.PROCEDURAL_INSTRUCTION
        or plan["reference_report_sha256"] != control.REFERENCE_REPORT_SHA
    ):
        raise ValueError("fixed procedure profile differs")
    old_jobs = [j for j in old_plan["jobs"] if j["prompt_profile"] == "original"]
    if plan["jobs"] != control.jobs(old_jobs):
        raise ValueError("unmatched task/seed slots")
    for key in (
        "tasks_sha256",
        "manifest_sha256",
        "seeds",
        "sampling",
        "model",
        "model_manifest_sha256",
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
        "seed_rule",
        "truncation",
    ):
        audit.require(plan[key] == old_plan[key], "unmatched contract: " + key)
    for key in ("path", "sha256", "commit_sha256"):
        audit.require(plan["adapter"][key] == old_plan["adapter"][key], "different adapter")
    for job in old_jobs:
        path = old_output / "episodes" / (job["episode_id"] + ".json")
        audit.require(
            c.inputs.sha(path) == baseline["sha256"][str(path)], "baseline episode changed"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    result = audit.analyze(output, tokenizer)
    rows = {p.stem: audit.read(p) for p in (output / "episodes").glob("*.json")}
    result.pop("paired", None)
    result.pop("depth_strata", None)
    result.update(
        procedure_minus_original=pair(plan["jobs"], baseline["episode_rows"], rows),
        baseline_group=baseline["groups"]["trained_original"],
        episode_rows=rows,
        baseline_report=str(baseline_report),
        baseline_report_sha256=control.REFERENCE_REPORT_SHA,
        caveat="Same fixed048 checkpoint23 and16 exposed task/seed slots. Additional procedural "
        "instruction is a package baseline, not isolated teacher causality. Eight correlated "
        "parents. Owner cap45min versus05290min; missing outcomes unknown, not failed. "
        "Physical new costs exclude reused052 receipts.",
    )
    result["sha256"][str(Path(__file__))] = c.inputs.sha(Path(__file__))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise ValueError("immutable analysis exists")
    result = analyze(args.output, args.baseline_report)
    control.collector.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write("# TextCraft procedural prompt control\n\n" + result["caveat"] + "\n\n")
        stream.write(
            "Paired result: "
            + json.dumps(
                {k: v for k, v in result["procedure_minus_original"].items() if k != "rows"}
            )
            + "\n\n"
        )
        stream.write("New outcomes and costs: " + json.dumps(result["groups"]) + "\n\n")
        stream.write("Reused original052: " + json.dumps(result["baseline_group"]) + "\n")
    print(json.dumps(result["procedure_minus_original"]))
