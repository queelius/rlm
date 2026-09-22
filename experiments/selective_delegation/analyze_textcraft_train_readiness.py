"""Native terminal audit and all-group readiness classification for fixed TRAIN8x4."""

import argparse
import json
from collections import Counter
from pathlib import Path

import analyze_textcraft as audit
import eval_textcraft_train_readiness as reader

PLAN_SHA = "e96a7ac7e9ded7695217378cd1668607c68cc5c0bbe244d40a88842b4c691444"
COLLECTOR_SHA = "ddc7d49f5749258e620fc651de1cc84d96a581e44ee958965ad4f0e03fad1223"


def group_readiness(jobs, rows, calls, audits):
    groups = {}
    for task_id in dict.fromkeys(j["task_id"] for j in jobs):
        selected = [j for j in jobs if j["task_id"] == task_id]
        ids = {j["episode_id"] for j in selected}
        recorded = {eid: rows[eid] for eid in ids if eid in rows}
        known = {eid: row for eid, row in recorded.items() if row["observed"]}
        for eid in known:
            audit.require(audits[eid]["replayed"], "known reward requires native replay")
            audit.require(audits[eid]["native_score"] in (0, 1), "binary native reward required")
        successes = sum(audits[eid]["native_score"] for eid in known)
        planned, observed = len(selected), len(known)
        classification = (
            "incomplete"
            if observed < planned
            else "all_zero"
            if successes == 0
            else "all_success"
            if successes == planned
            else "mixed"
        )
        groups[task_id] = dict(
            planned=planned,
            recorded=len(recorded),
            observed=observed,
            missing=planned - len(recorded),
            unavailable=len(recorded) - observed,
            successes=successes,
            observed_failures=observed - successes,
            classification=classification,
            success_rate_bounds=[successes / planned, (successes + planned - observed) / planned],
            rewards_by_repeat={
                str(j["repeat"]): audits[j["episode_id"]]["native_score"]
                if j["episode_id"] in known
                else None
                for j in selected
            },
            errors=dict(sum((Counter(r["errors"]) for r in recorded.values()), Counter())),
            error_episode_counts={
                kind: sum(r["errors"].get(kind, 0) > 0 for r in recorded.values())
                for kind in ("invalid_schema", "native_action_error", "rejected_action")
            },
            cost=reader.c.cost([call for call in calls if call["episode_id"] in ids]),
        )
    return groups


def analyze(output):
    from transformers import AutoTokenizer

    c = reader.c
    audit.require(c.inputs.sha(output / "PLAN.json") == PLAN_SHA, "fixed063 PLAN required")
    plan = audit.read(output / "PLAN.json")
    tasks = list(map(json.loads, (Path(plan["prepared"]) / "tasks.jsonl").read_text().splitlines()))
    audit.require(
        plan["jobs"] == reader.jobs(tasks) and len(tasks) == 8,
        "all8TRAIN groups/four fixed samples required",
    )
    audit.require(
        plan["split"] == "train"
        and plan["profile"] == "original"
        and plan["adapter"]["training_plan_sha256"] == reader.public.PLAN_SHA,
        "fixed public056 TRAIN readout required",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    native = audit.analyze(output, tokenizer, draws=1, expected_collector_sha256=COLLECTOR_SHA)
    rows = {p.stem: audit.read(p) for p in (output / "episodes").glob("*.json")}
    calls = [audit.read(p) for p in (output / "calls").glob("*.json")]
    groups = group_readiness(plan["jobs"], rows, calls, native["audits"])
    # The generic paired-recursion/bootstrap descriptions do not apply to this one-arm TRAIN run.
    for key in ("paired", "depth_strata", "method"):
        native.pop(key, None)
    return dict(
        schema="textcraft-train-readiness-analysis-v1",
        split="train",
        planned_groups=8,
        planned_episodes=32,
        groups=groups,
        classification_counts=dict(Counter(g["classification"] for g in groups.values())),
        native_audit=native,
        adapter=plan["adapter"],
        source_sha256={str(Path(__file__).resolve()): c.inputs.sha(Path(__file__))},
        method="Eight exact SFT TRAIN tasks, four correlated samples each; no bootstrap or "
        "held-out inference. All planned groups retained. Incomplete means any missing/unavailable "
        "sample, not all-zero. Native terminal rewards include paths with schema/action errors; "
        "error counts are diagnostic, never a reward filter. Partial calls without episodes "
        "still count toward their planned task's physical cost. No automatic RL admission.",
    )


def markdown(report):
    lines = [
        "# TextCraft TRAIN rollout readiness",
        "",
        report["method"],
        "",
        "| TRAIN task | Success / observed / planned | Missing / unavailable | Group | "
        "Schema / action / rejected errors | Calls | Input / output tokens | Native seconds |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for task_id, group in report["groups"].items():
        cost, errors = group["cost"], group["errors"]
        lines.append(
            f"| {task_id} | {group['successes']} / {group['observed']} / {group['planned']} "
            f"| {group['missing']} / {group['unavailable']} | {group['classification']} "
            f"| {errors.get('invalid_schema', 0)} / {errors.get('native_action_error', 0)} "
            f"/ {errors.get('rejected_action', 0)} | {cost['calls']} "
            f"| {cost['prompt_tokens']} / {cost['completion_tokens']} "
            f"| {cost['native_service_seconds']:.2f} |"
        )
    lines += [
        "",
        "Group counts: " + json.dumps(report["classification_counts"]),
        "",
        "All-call costs: " + json.dumps(report["native_audit"]["physical_cost"]),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise ValueError("immutable report exists")
    result = analyze(args.output)
    reader.c.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(result))
    print(json.dumps(result["classification_counts"]))
