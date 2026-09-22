"""Independent flat TextCraft prompt/weight comparisons; incomplete slots stay unknown."""

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import analyze_textcraft as audit


def compare(jobs, left, right):
    rows = []
    for job in jobs:
        key = job["task_id"], job["repeat"]
        a, b = left.get(key), right.get(key)
        known = bool(a and b and a["observed"] and b["observed"])
        rows.append(
            dict(
                task_id=key[0],
                repeat=key[1],
                known=known,
                left=a["native_score"] if a and a["observed"] else None,
                right=b["native_score"] if b and b["observed"] else None,
                delta=b["native_score"] - a["native_score"] if known else None,
            )
        )
    known = [r for r in rows if r["known"]]
    parents = sorted({j["task_id"] for j in jobs})
    difference = interval = None
    if len(known) == len(rows):
        values = [mean(r["delta"] for r in rows if r["task_id"] == p) for p in parents]
        difference = mean(values)
        rng = random.Random(2026092206)
        draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(20000))
        interval = [draws[int(19999 * q)] for q in (0.025, 0.975)]
    return dict(
        planned_pairs=len(jobs),
        known_pairs=len(known),
        unknown_pairs=len(rows) - len(known),
        wins=sum(r["delta"] > 0 for r in known),
        losses=sum(r["delta"] < 0 for r in known),
        ties=sum(r["delta"] == 0 for r in known),
        complete_panel_difference=difference,
        ci95=interval,
        rows=rows,
    )


def analyze(output, baseline_report, expected_collector_sha, reminder_report=None):
    from transformers import AutoTokenizer

    plan = audit.read(output / "PLAN.json")
    audit.require(
        plan["schema"]
        in ("textcraft-native-instruction-control-v1", "textcraft-fixed-action-sft23-readout-v1"),
        "unsupported flat readout",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    result = audit.analyze(output, tokenizer, expected_collector_sha256=expected_collector_sha)
    base = audit.read(baseline_report)
    # Consume the already independent044 native audit; verify small receipt hashes only.
    base_output = audit.collector.ROOT / "textcraft-pilot-001"
    base_plan = audit.read(base_output / "PLAN.json")
    audit.require(
        base["sha256"][str(base_output / "PLAN.json")]
        == audit.collector.inputs.sha(base_output / "PLAN.json"),
        "baseline PLAN changed",
    )
    for field in (
        "tasks_sha256",
        "manifest_sha256",
        "seeds",
        "model_manifest_sha256",
        "sampling",
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
        "truncation",
    ):
        audit.require(base_plan[field] == plan[field], "base/readout mismatch: " + field)
    rows = {p.stem: audit.read(p) for p in (output / "episodes").glob("*.json")}
    base_rows = {}
    for job in base_plan["jobs"]:
        if job["policy"] != "flat":
            continue
        path = base_output / "episodes" / (job["episode_id"] + ".json")
        if path.exists():
            audit.require(
                base["sha256"].get(str(path)) == audit.collector.inputs.sha(path),
                "baseline episode differs from independent audit",
            )
            base_rows[(job["task_id"], job["repeat"])] = audit.read(path)
    jobs = [j for j in base_plan["jobs"] if j["policy"] == "flat"]
    comparisons = {}
    if plan["schema"] == "textcraft-native-instruction-control-v1":
        right = {(r["task_id"], r["repeat"]): r for r in rows.values()}
        comparisons["base_reminder_minus_base_original"] = compare(jobs, base_rows, right)
    else:
        # Trained readout pairs prompt profiles internally. Base049 must have its own
        # completed independent report before its matched training contrast is formed.
        original = {
            (r["task_id"], r["repeat"]): r
            for r in rows.values()
            if r["condition"] == "trained_original"
        }
        reminder = {
            (r["task_id"], r["repeat"]): r
            for r in rows.values()
            if r["condition"] == "trained_instruction_control"
        }
        comparisons["trained_original_minus_base_original"] = compare(jobs, base_rows, original)
        comparisons["trained_reminder_minus_trained_original"] = compare(jobs, original, reminder)
        audit.require(reminder_report is not None, "trained readout requires audited049 report")
        reference = audit.read(reminder_report)
        ref_output = audit.collector.ROOT / "textcraft-instruction-control-001"
        audit.require(
            reference["sha256"][str(ref_output / "PLAN.json")]
            == plan["base_reminder_plan_sha256"]
            == audit.collector.inputs.sha(ref_output / "PLAN.json"),
            "base049 PLAN identity changed",
        )
        ref_rows = reference["episode_rows"]
        for eid in ref_rows:
            path = ref_output / "episodes" / (eid + ".json")
            audit.require(
                reference["sha256"][str(path)] == audit.collector.inputs.sha(path),
                "audited049 episode changed",
            )
        ref_by_slot = {(r["task_id"], r["repeat"]): r for r in ref_rows.values()}
        comparisons["trained_reminder_minus_base_reminder"] = compare(jobs, ref_by_slot, reminder)
        result["reminder_report_sha256"] = audit.collector.inputs.sha(reminder_report)
        result["base_reminder_groups"] = reference["groups"]
    result["profile_comparisons"] = comparisons
    result["baseline_report"] = str(baseline_report)
    result["baseline_report_sha256"] = audit.collector.inputs.sha(baseline_report)
    result["baseline_groups"] = base["groups"]
    result["planned_jobs"] = plan["jobs"]
    result["episode_rows"] = rows
    result["sha256"][str(Path(__file__))] = audit.collector.inputs.sha(Path(__file__))
    result["method"]["profile_caveat"] = (
        "Exposed eight-task panel and two correlated seeds. Post-hoc prompt qualification. "
        "Native copied-state finish opportunities are not model successes. "
        "No complete-panel effect/CI when any paired outcome is unknown."
    )
    # Remove the old flat-vs-recursive summaries: those conditions are not this readout.
    result.pop("paired", None)
    result.pop("depth_strata", None)
    return result


def markdown(result):
    lines = [
        "# TextCraft flat prompt/weight comparison",
        "",
        result["method"]["profile_caveat"],
        "",
        "| Condition | Successes /16 | Failures | Missing / unavailable | Calls | Tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, g in result["groups"].items():
        c = g["cost"]
        lines.append(
            f"| {name} | {g['won']} | {g['observed_failures']} | "
            f"{g['missing']} / {g['unavailable']} | {c['calls']} | "
            f"{c['prompt_tokens'] + c['completion_tokens']} |"
        )
    for name, contrast in result["profile_comparisons"].items():
        lines += ["", name + ": " + json.dumps({k: v for k, v in contrast.items() if k != "rows"})]
    lines += [
        "",
        "Native replay verifies original prompt bytes/token IDs, sampling, seeds, "
        "model/adapter receipts, inventories, feedback and native score. Unknowns are not zeros.",
        "",
        "Groups/status/error/cost detail: " + json.dumps(result["groups"]),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--expected-collector-sha256", required=True)
    parser.add_argument("--reminder-report", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise ValueError("immutable report exists")
    result = analyze(
        args.output,
        args.baseline_report,
        args.expected_collector_sha256,
        reminder_report=args.reminder_report,
    )
    audit.collector.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(result))
    print(json.dumps(result["profile_comparisons"]))
