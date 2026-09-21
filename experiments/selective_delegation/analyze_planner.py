"""Receipt-based paired analysis; condition names and inventory come from PLAN."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

import analysis
import analyze_execution
import eval_planner
import probe

SEED = 2026092110
DRAWS = 20000


def measured(records):
    result = eval_planner.cost(records)
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def analyze(output: Path, cases_path: Path, *, draws=DRAWS, seed=SEED, comparison_output=None):
    output, cases_path = Path(output).resolve(), Path(cases_path).resolve()
    hashes = {}

    def track(path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"input/checkpoint/source hash changed: {path}")
        hashes[str(path)] = digest
        return data

    def read(path):
        return json.loads(track(path))

    plan = read(output / "PLAN.json")
    sources = [(output, plan)]
    if comparison_output is not None:
        comparison_output = Path(comparison_output).resolve()
        other = read(comparison_output / "PLAN.json")
        if (
            not plan["conditions"]
            or not other["conditions"]
            or set(plan["conditions"]) & set(other["conditions"])
        ):
            raise ValueError("comparison requires disjoint nonempty condition sets")
        for field in ("cases_sha256", "case_ids", "repeats", "seed", "execution"):
            if field not in plan or field not in other or plan[field] != other[field]:
                raise ValueError("comparison contract differs: " + field)
        for field in (
            "source_sha256",
            "model",
            "model_manifest_sha256",
            "split",
            "mode",
            "temperature",
            "top_p",
            "top_k",
            "caps",
            "max_context",
            "helper_budget_policy",
            "helper_contract",
            "architecture",
        ):
            if plan.get(field) != other.get(field):
                raise ValueError("comparison contract differs: " + field)
        sources.append((comparison_output, other))
    raw = track(cases_path)
    if hashes[str(cases_path)] != plan["cases_sha256"]:
        raise ValueError("cases differ from immutable evaluation PLAN")
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    by_id = {row["id"]: row for row in cases}
    parents, repeats = plan["case_ids"], plan["repeats"]
    conditions = [
        condition for _, source_plan in sources for condition in source_plan["conditions"]
    ]
    if (
        not parents
        or repeats < 1
        or draws < 1
        or not conditions
        or len(set(parents)) != len(parents)
        or len(set(conditions)) != len(conditions)
        or len(by_id) != len(cases)
        or any(p not in by_id for p in parents)
    ):
        raise ValueError("invalid planned inventory")
    for source_output, source_plan in sources:
        for path, expected in source_plan.get("dependencies", {}).items():
            track(path, expected)
        helper = source_plan.get("helper_contract", {})
        if helper.get("adapter"):
            helper_adapter = Path(helper["adapter"])
            for name, expected in helper.get("adapter_binding", {}).items():
                track(helper_adapter / name, expected)
            track(helper_adapter.parent / "PLAN.json", helper["training_plan_sha256"])
        if helper.get("reminder_source"):
            track(helper["reminder_source"], helper["reminder_source_sha256"])
        if source_plan.get("adapter"):
            adapter = Path(source_plan["adapter"])
            for name, expected in source_plan.get("adapter_files_sha256", {}).items():
                track(adapter / name, expected)
            if source_plan.get("training_plan_sha256"):
                track(adapter.parent / "PLAN.json", source_plan["training_plan_sha256"])
        if source_plan.get("model_manifest_sha256"):
            track(
                Path(source_plan["model"]) / "local-research-manifest.json",
                source_plan["model_manifest_sha256"],
            )
        for path in sorted(source_output.glob("OWNER-*.json")):
            owner = read(path)
            if owner.get("source"):
                track(owner["source"], source_plan.get("source_sha256"))

    def receipt_paths(directory):
        for source_output, source_plan in sources:
            for path in sorted((source_output / directory).glob("*.json")):
                yield path, source_plan["conditions"]

    calls = {}
    for path, source_conditions in receipt_paths("calls"):
        row = read(path)
        cid = row["call_id"]
        if cid in calls or path.stem != cid or row["condition"] not in source_conditions:
            raise ValueError("duplicate or unexpected call identity")
        for field, ids in (
            ("prompt_tokens", "input_token_ids"),
            ("completion_tokens", "output_token_ids"),
        ):
            if ids in row and row.get("usage", {}).get(field) != len(row[ids]):
                raise ValueError("native token IDs disagree with usage receipt")
        calls[cid] = row
    episodes = {}
    inventory = {(p, r, c) for p in parents for r in range(repeats) for c in conditions}
    for path, source_conditions in receipt_paths("episodes"):
        row = read(path)
        key = row["case_id"], row["repeat"], row["condition"]
        if key not in inventory or key in episodes or row["condition"] not in source_conditions:
            raise ValueError("unexpected or duplicate episode")
        episodes[key] = row
    values, linked = {}, set()
    groups = {}
    denominator = len(parents) * repeats
    for condition in conditions:
        rows, records, plans, lengths = [], [], [], []
        per_parent_plans = {p: set() for p in parents}
        for parent in parents:
            for repeat in range(repeats):
                row = episodes.get((parent, repeat, condition))
                ids = row["call_ids"] if row else []
                if len(set(ids)) != len(ids) or linked.intersection(ids):
                    raise ValueError("unexpected reuse of physical calls")
                if any(cid not in calls or calls[cid]["condition"] != condition for cid in ids):
                    raise ValueError("episode references missing/mismatched call")
                linked.update(ids)
                current = [calls[cid] for cid in ids]
                records.extend(current)
                finals = [c for c in current if c["role"] == "final"]
                roots = [c for c in current if c["role"] == "root"]
                if len(finals) > 1 or len(roots) > 1:
                    raise ValueError("multiple root/final calls in episode")
                final = finals[0] if finals else None
                grade = (
                    probe.grade(final["text"], by_id[parent])
                    if final and final["available"]
                    else probe.grade("", by_id[parent])
                )
                status = row["status"] if row else "missing_episode"
                if final and final["available"]:
                    status = "scored" if grade["valid"] else "invalid_final"
                elif status == "scored":
                    status = "missing_final"
                scored = {
                    "case_id": parent,
                    "repeat": repeat,
                    "status": status,
                    "em": float(grade["correct"]),
                    "f1": grade["f1"],
                    "valid_final": grade["valid"],
                    "present": row is not None,
                }
                rows.append(scored)
                values[parent, repeat, condition] = scored
                if roots and roots[0]["available"]:
                    try:
                        root_plan = eval_planner.parse_plan(roots[0]["text"])
                    except (ValueError, TypeError):
                        continue
                    canonical = json.dumps(root_plan, sort_keys=True, ensure_ascii=False)
                    plans.append(canonical)
                    per_parent_plans[parent].add(canonical)
                    lengths.append(len(root_plan["subquestions"]))
        deployment = measured(records)
        groups[condition] = {
            "planned_episodes": denominator,
            "recorded_episodes": sum(r["present"] for r in rows),
            "missing_episodes": sum(not r["present"] for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "valid_finals": sum(r["valid_final"] for r in rows),
            "correct": sum(r["em"] for r in rows),
            "em": mean(r["em"] for r in rows),
            "f1": mean(r["f1"] for r in rows),
            "valid_generated_plans": len(plans),
            "distinct_plans": len(set(plans)),
            "plan_lengths": dict(Counter(str(n) for n in lengths)),
            "distinct_plans_per_parent": {p: len(plans) for p, plans in per_parent_plans.items()},
            "physical_cost": measured([r for r in calls.values() if r["condition"] == condition]),
            "deployed_cost": deployment,
            "per_planned_attempt": {
                key: deployment[key] / denominator
                for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
            },
        }
    rng = random.Random(seed)
    samples = [[rng.randrange(len(parents)) for _ in parents] for _ in range(draws)]
    comparisons = {}
    for left, right in combinations(conditions, 2):
        comparison = {"left": left, "right": right}
        for metric in ("em", "f1"):
            differences = [
                mean(
                    values[p, r, right][metric] - values[p, r, left][metric] for r in range(repeats)
                )
                for p in parents
            ]
            comparison[metric] = analyze_execution.interval(differences, samples)
        for label, direction in (("wins", 1), ("losses", -1)):
            changes = []
            for parent in parents:
                for repeat in range(repeats):
                    a, b = values[parent, repeat, left], values[parent, repeat, right]
                    if b["em"] - a["em"] != direction:
                        continue
                    missing = (
                        not a["present"]
                        or not b["present"]
                        or "missing_final" in (a["status"], b["status"])
                    )
                    category = (
                        "missing_involved"
                        if missing
                        else "both_scored"
                        if a["status"] == b["status"] == "scored"
                        else "protocol_involved"
                    )
                    changes.append(
                        {
                            "case_id": parent,
                            "repeat": repeat,
                            "category": category,
                            "left_status": a["status"],
                            "right_status": b["status"],
                        }
                    )
            comparison[label] = {
                "episodes": len(changes),
                "parents": sorted({r["case_id"] for r in changes}),
                "rows": changes,
                **{
                    category: sum(r["category"] == category for r in changes)
                    for category in ("both_scored", "protocol_involved", "missing_involved")
                },
            }
        comparisons[right + "_minus_" + left] = comparison
    starts = [read(path) for path, _ in receipt_paths("starts")]
    for path in (
        Path(__file__),
        Path(analysis.__file__),
        Path(analyze_execution.__file__),
        Path(eval_planner.__file__),
        Path(eval_planner.planner.__file__),
        Path(probe.__file__),
        probe.MUSIQUE / "metrics/answer.py",
    ):
        track(path)
    return {
        "output": str(output),
        "comparison_output": str(comparison_output) if comparison_output is not None else None,
        "cases": str(cases_path),
        "source_plans": [
            {"output": str(source_output), **source_plan} for source_output, source_plan in sources
        ],
        "input_source_checkpoint_sha256": hashes,
        "plan_identity": {
            key: plan.get(key)
            for key in (
                "model",
                "adapter",
                "adapter_files_sha256",
                "source_sha256",
                "split",
                "mode",
                "execution",
                "architecture",
                "policy",
                "helper_contract",
                "conditions",
            )
        },
        "method": {
            "independent_parent_count": len(parents),
            "episodes_per_condition": denominator,
            "repeats": repeats,
            "draws": draws,
            "seed": seed,
            "parent_order": parents,
            "bootstrap": "Python random.Random.randrange, paired parent means; linearly "
            "interpolated percentile 95% CI; all planned attempts, missing/invalid zero",
            "metric": "official MuSiQue alias-max EM/F1 via probe.grade",
        },
        "groups": groups,
        "comparisons": comparisons,
        "cluster_overlap": analysis.component_overlap([by_id[p] for p in parents]),
        "physical_cost": measured(list(calls.values())),
        "unlinked_call_ids": sorted(set(calls) - linked),
        "unresolved_starts": sum(s["call_id"] not in calls for s in starts),
        "extra_start_attempts": len(starts) - len({s["call_id"] for s in starts}),
        "cautions": [
            "Repeats share a parent and are not independent observations; parent-level "
            "bootstrap intervals are exploratory, unadjusted, and not fully independent "
            "where atomic components overlap.",
            "Missing episodes score zero on planned denominators: incomplete runs give "
            "lower bounds, not completed-run estimates. Unknown usage is not free.",
            "Protocol-involved changes include malformed outputs and generation failures; "
            "a protocol recovery is not by itself evidence of better semantic planning.",
            "Distinct plans count exact serialized question lists, not semantic diversity. "
            "Costs include actual native calls, with no hypothetical unused call budget.",
            "Condition labels come from PLAN; identify the actual adapter checkpoint "
            "before calling a comparison an SFT or RL effect.",
        ],
    }


def markdown(report):
    method = report["method"]
    lines = [
        "# Planner evaluation",
        "",
        f"Source: `{report['output']}`",
        "",
        f"{method['independent_parent_count']} parents, not {method['episodes_per_condition']} "
        "independent episodes per condition. All planned attempts remain in denominators.",
        "",
        "| Condition | EM | F1 | Correct / planned | Calls | Tokens / planned attempt |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, row in report["groups"].items():
        lines.append(
            f"| {condition} | {row['em']:.2%} | {row['f1']:.2%} | "
            f"{row['correct']:g}/{row['planned_episodes']} | {row['physical_cost']['calls']} | "
            f"{row['per_planned_attempt']['total_tokens']:.1f} |"
        )
    if report.get("comparison_output"):
        lines += ["", f"Comparison source: `{report['comparison_output']}`", ""]
    lines += [
        "",
        "Paired differences (percentage points):"
        if report["comparisons"]
        else "Unpaired single-condition report; no effect estimate.",
        "",
    ]
    for name, comparison in report["comparisons"].items():
        intervals = []
        for metric in ("em", "f1"):
            row = comparison[metric]
            intervals.append(
                f"{metric.upper()} {100 * row['estimate']:+.2f} "
                f"[{100 * row['ci95'][0]:+.2f}, {100 * row['ci95'][1]:+.2f}]"
            )
        lines.append(f"- {name}: " + "; ".join(intervals) + ".")
        for label in ("wins", "losses"):
            row = comparison[label]
            lines.append(
                f"  - {label}: {row['episodes']} episodes / {len(row['parents'])} parents; "
                f"{row['both_scored']} both scored, {row['protocol_involved']} protocol "
                f"involved, {row['missing_involved']} missing involved."
            )
    lines += ["", "Status and plan-shape diagnostics:", ""]
    for condition, row in report["groups"].items():
        lines.append(
            f"- {condition}: statuses `{json.dumps(row['status_counts'], sort_keys=True)}`; "
            f"plan lengths `{json.dumps(row['plan_lengths'], sort_keys=True)}`; "
            f"{row['distinct_plans']} distinct valid question lists."
        )
    lines += [
        "",
        f"Bootstrap: {method['draws']} draws, seed {method['seed']}; "
        "paired parent means, percentile 95% intervals.",
        "",
        f"Connected atomic-component clusters: "
        f"{report['cluster_overlap']['connected_parent_clusters']}.",
        "",
        *["- " + caution for caution in report["cautions"]],
        "",
    ]
    return "\n".join(lines)


def write_report(report, path):
    path = Path(path)
    sibling = path.with_suffix(".md")
    if path == sibling or path.exists() or sibling.exists():
        raise FileExistsError("choose unused distinct JSON and markdown paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    with sibling.open("x") as handle:
        handle.write(markdown(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "cases", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        parser.error("report already exists; choose a new immutable report path")
    report = analyze(args.output, args.cases, comparison_output=args.comparison_output)
    write_report(report, args.report)
    print(markdown(report))


if __name__ == "__main__":
    main()
