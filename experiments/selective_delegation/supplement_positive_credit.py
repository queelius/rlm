"""Explicit four-unavailable-slot coverage recovery; never rerun completed outcomes."""

import argparse
import copy
import json
from pathlib import Path

import analyze_textcraft_positive_partial as audit
import textcraft_positive_replay as experiment

c, read, sha = experiment.c, audit.read, experiment.sha
ORIGINAL = experiment.READOUT
OUTPUT = c.ROOT / "textcraft-positive-credit-supplement-002"
AUDIT = c.ROOT / "analysis-textcraft-positive-credit-recovered-001.json"
IDS = [f"t{task:02d}-r{repeat}-flat-original" for task in (14, 15) for repeat in (0, 1)]


def prepare():
    receipt = read(AUDIT)
    original = read(ORIGINAL / "PLAN.json")
    if receipt["arms"][1]["sha256"][str(ORIGINAL / "PLAN.json")] != sha(ORIGINAL / "PLAN.json"):
        raise ValueError("original audit PLAN changed")
    jobs = receipt["unavailable_jobs"]
    if [j["episode_id"] for j in jobs] != IDS:
        raise ValueError("exact four unavailable slots only")
    for job in jobs:
        if job not in original["jobs"]:
            raise ValueError("foreign slot")
        path = ORIGINAL / "episodes" / (job["episode_id"] + ".json")
        if path.exists() and (
            read(path)["observed"] or receipt["arms"][1]["sha256"].get(str(path)) != sha(path)
        ):
            raise ValueError("completed/changed episode cannot be retried")
    binding = experiment.endpoint_binding()
    if binding != original["adapter"]:
        raise ValueError("fixed endpoint changed")
    plan = copy.deepcopy(original)
    plan.update(
        schema="positive-credit-four-slot-supplement-v1",
        jobs=jobs,
        planned_episodes=4,
        planned_per_condition=4,
        parent_tasks=2,
        budget_seconds=1800,
        max_native_calls=384,
        supplemental_of=str(ORIGINAL),
        original_plan_sha256=sha(ORIGINAL / "PLAN.json"),
        original_audit_sha256=sha(AUDIT),
        recovery_scope="Rerun interrupted14r0 from initial state; run3never-observed slots. "
        "Same original seeds,prompts,endpoint,per-episode caps. "
        "Original cost/unknown attempt retained. "
        "Supplemental operational coverage, not an unchanged60minute throughput comparison.",
    )
    for module in (
        c,
        c.bridge,
        c.inputs,
        c.probe,
        c.probe.runtime,
        c.probe.campaign,
        audit,
        experiment,
    ):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__))
    path = OUTPUT / "PLAN.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("immutable supplemental PLAN changed")
    else:
        c.save(path, plan)
    tasks = list(map(json.loads, (Path(plan["prepared"]) / "tasks.jsonl").read_text().splitlines()))
    return plan, tasks, binding


def analyze(report):
    from transformers import AutoTokenizer

    plan, _, _ = prepare()
    original = read(AUDIT)
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    supplement = audit.analyze(
        OUTPUT, tokenizer, expected_task_count=16, expected_jobs=plan["jobs"]
    )
    supplement.pop("paired", None)
    supplement.pop("depth_strata", None)
    # Consume already independently audited original results; verify consumed small receipts.
    directories = (experiment.REFERENCE_READOUT, ORIGINAL)
    rows = []
    for arm, directory in zip(original["arms"], directories, strict=True):
        current = {}
        for path in (directory / "episodes").glob("*.json"):
            if arm["sha256"].get(str(path)) != sha(path):
                raise ValueError("original audited episode changed")
            row = read(path)
            current[(row["task_id"], row["repeat"])] = row
        rows.append(current)
    for path in (OUTPUT / "episodes").glob("*.json"):
        row = read(path)
        key = row["task_id"], row["repeat"]
        if row["episode_id"] not in IDS or rows[1].get(key, {}).get("observed"):
            raise ValueError("duplicate observed outcome replacement forbidden")
        rows[1][key] = row
    jobs = read(ORIGINAL / "PLAN.json")["jobs"]
    effect = experiment.reference.comparison.profiles.compare(jobs, *rows)
    known = [r for r in rows[1].values() if r["observed"]]
    success = sum(r["native_score"] for r in known)
    costs = [original["arms"][1]["physical_cost"], supplement["physical_cost"]]
    summed = {key: sum(cost[key] for cost in costs) for key in costs[0]}
    result = dict(
        schema="positive-credit-supplemented-native-comparison-v1",
        planned=32,
        observed=len(known),
        unknown=32 - len(known),
        won=success,
        success_rate_bounds=[success / 32, (success + 32 - len(known)) / 32],
        positive_minus_signed=effect,
        supplemental_native_audit=supplement,
        original_audit_sha256=sha(AUDIT),
        original_plus_supplement_cost=summed,
        all_cost_components=costs,
        original_one_hour_result={
            "groups": original["arms"][1]["groups"],
            "owner_wall_seconds": original["arms"][1]["owner_wall_seconds"],
            "collection_cap_seconds": 3600,
            "positive_minus_signed": original["positive_minus_signed"],
            "difference_identification_bounds": original["difference_identification_bounds"],
        },
        supplemented_coverage_contract="Same per-episode limits, "
        "not matched one-hour total compute. "
        "Additional30minute collection/coldstart and repeated interrupted prefix are charged. "
        "Selection used unavailable coverage only, never success/failure of completed slots.",
        wrapper_sha256=sha(Path(__file__)),
        interpretation=plan["recovery_scope"],
    )
    c.save(report, result)
    brief = {k: v for k, v in result.items() if k != "supplemental_native_audit"}
    brief["positive_minus_signed"] = {k: v for k, v in effect.items() if k != "rows"}
    with report.with_suffix(".md").open("x") as f:
        f.write(
            "# Positive-credit supplemental coverage\n\n"
            + plan["recovery_scope"]
            + "\n\n```json\n"
            + json.dumps(brief, indent=2)
            + "\n```\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report:
        analyze(args.report)
    else:
        plan, tasks, binding = prepare()
        c.run(
            argparse.Namespace(output=OUTPUT, prepare_only=args.prepare_only),
            prepared_run=(plan, tasks),
            adapter=binding,
        )
