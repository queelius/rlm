"""Read-only, receipt-based analysis of the frozen 2x2 execution diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import analysis
import probe

ARMS = ("model_bundled", "reference_bundled", "model_isolated", "reference_isolated")
SEED = 2026092191
DRAWS = 20000
CONTRASTS = {
    "isolation_model": (-1, 0, 1, 0),
    "isolation_reference": (0, -1, 0, 1),
    "reference_gain_bundled": (-1, 1, 0, 0),
    "reference_gain_isolated": (0, 0, -1, 1),
    "interaction": (1, -1, -1, 1),
}
CAUTIONS = [
    "All planned parents/repeats retained; missing/invalid finals score zero. Missing rows make "
    "estimates lower bounds, not completed-run estimates.",
    "Bootstrap resamples paired parent means, conditional on one fixed checkpoint per parent; "
    "repeats are not independent end-to-end runs. Intervals are exploratory and unadjusted.",
    "Reference questions are privileged annotations. Isolation changes context visibility, "
    "dependency binding, sequential calls and response format together, not a pure mechanism.",
    "Last-helper scoring against composed-question gold is diagnostic: the model's final "
    "subquestion may have a different target; EM mismatch need not mean semantic error.",
    "Original finals retain the old checkpoint/provisional answer. New planner evaluation "
    "without a provisional answer is an architecture change.",
    "No aggregate isolation advantage was established; isolated SFT execution is a diagnostic "
    "choice, not a winning-arm claim. Finals rescue more matches than they destroy.",
]


def cost(records):
    result = {
        "calls": len(records),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "unknown_usage_calls": 0,
    }
    for record in records:
        usage = record.get("usage") or {}
        known = record.get("available", False)
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage.get(key)
            if type(value) is int and value >= 0:
                result[key] += value
            else:
                known = False
        result["unknown_usage_calls"] += not known
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def interval(values, samples):
    estimates = sorted(mean(values[i] for i in draw) for draw in samples)

    def percentile(q):
        location = (len(estimates) - 1) * q
        low = int(location)
        high = min(low + 1, len(estimates) - 1)
        return estimates[low] + (estimates[high] - estimates[low]) * (location - low)

    return {"estimate": mean(values), "ci95": [percentile(0.025), percentile(0.975)]}


def analyze(output: Path, cases_path: Path, *, draws=DRAWS, seed=SEED):
    output, cases_path = output.resolve(), cases_path.resolve()
    hashes = {}

    def read(path, *, jsonl=False):
        path = Path(path).resolve()
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        return (
            [json.loads(line) for line in content.splitlines() if line.strip()]
            if jsonl
            else json.loads(content)
        )

    plan = read(output / "PLAN.json")
    cases = read(cases_path, jsonl=True)
    if hashes[str(cases_path)] != plan["cases_sha256"]:
        raise ValueError("cases hash does not match execution PLAN")
    by_id = {case["id"]: case for case in cases}
    parents, repeats = plan["case_ids"], plan["repeats"]
    if len(by_id) != len(cases) or len(set(parents)) != len(parents):
        raise ValueError("duplicate cases/parents")
    if not parents or repeats < 1 or draws < 1 or tuple(plan["arms"]) != ARMS:
        raise ValueError("invalid planned dimensions")
    source = Path(plan["source_output"])
    read(source / "PLAN.json")
    if hashes[str((source / "PLAN.json").resolve())] != plan["source_plan_sha256"]:
        raise ValueError("source PLAN changed")
    roots, states = {}, {}
    for parent in parents:
        for path, expected in plan["source_checkpoints"][parent]["source_hashes"].items():
            read(path)
            if hashes[str(Path(path).resolve())] != expected:
                raise ValueError(f"source changed: {path}")
        record = read(source / "calls" / f"{parent}-checkpoint.json")
        roots[parent] = record
        states[parent] = probe.checkpoint(record["text"])
    calls = {}
    for path in sorted((output / "calls").glob("*.json")):
        record = read(path)
        if record["call_id"] in calls or path.stem != record["call_id"]:
            raise ValueError("duplicate/misnamed call")
        calls[record["call_id"]] = record
    starts = [read(path) for path in sorted((output / "starts").glob("*.json"))]
    episodes = {}
    expected_keys = {(p, r, a) for p in parents for r in range(repeats) for a in ARMS}
    for path in sorted((output / "episodes").glob("*.json")):
        row = read(path)
        key = (row["case_id"], row["repeat"], row["arm"])
        if key not in expected_keys or key in episodes:
            raise ValueError("unexpected/duplicate episode")
        episodes[key] = row
    scores, groups, helper_results, failures = {}, {}, {}, []
    referenced = set()
    for arm in ARMS:
        rows, records, deployed, transitions = [], [], [], []
        for parent in parents:
            for repeat in range(repeats):
                key = (parent, repeat, arm)
                row = episodes.get(key)
                identity = f"{parent}-r{repeat}-{arm}"
                ids = row["new_call_ids"] if row else []
                if row and row["reused_checkpoint_call_id"] != roots[parent]["call_id"]:
                    raise ValueError("reused checkpoint mismatch")
                if any(cid not in calls or not cid.startswith(identity + "-") for cid in ids):
                    raise ValueError("missing or mismatched episode call")
                if referenced.intersection(ids):
                    raise ValueError("new call reused across episodes")
                referenced.update(ids)
                current = [calls[cid] for cid in ids]
                records.extend(current)
                deployed.extend([roots[parent], *current])
                final = calls.get(identity + "-final") if row else None
                if final and final["call_id"] not in ids:
                    raise ValueError("final not linked by episode")
                graded = (
                    probe.grade(final["text"], by_id[parent])
                    if final and final.get("available")
                    else probe.grade("", by_id[parent])
                )
                scores[key] = {"em": float(graded["correct"]), "f1": graded["f1"]}
                rows.append(
                    {
                        "present": row is not None,
                        "available": bool(final and final.get("available")),
                        "valid": graded["valid"],
                        **scores[key],
                    }
                )
                if row and row.get("helper_parse_failure"):
                    failures.append(
                        {
                            "case_id": parent,
                            "repeat": repeat,
                            "arm": arm,
                            "episode_id": identity,
                            "error": row.get("error"),
                            "call_ids": ids,
                        }
                    )
                if arm.endswith("isolated"):
                    helper = calls.get(identity + "-helper2") if row else None
                    if helper and helper["call_id"] not in ids:
                        raise ValueError("helper not linked by episode")
                    hgrade = (
                        probe.grade(helper["text"], by_id[parent])
                        if helper and helper.get("available")
                        else probe.grade("", by_id[parent])
                    )
                    transitions.append(
                        {
                            "case_id": parent,
                            "repeat": repeat,
                            "helper_valid": hgrade["valid"],
                            "helper_correct": hgrade["correct"],
                            "helper_f1": hgrade["f1"],
                            "final_correct": graded["correct"],
                        }
                    )
        groups[arm] = {
            "planned": len(rows),
            "episodes": sum(r["present"] for r in rows),
            "available": sum(r["available"] for r in rows),
            "valid": sum(r["valid"] for r in rows),
            "correct": sum(r["em"] for r in rows),
            "em": mean(r["em"] for r in rows),
            "f1": mean(r["f1"] for r in rows),
            "new_physical_cost": cost(records),
            "hypothetical_deployed_cost": cost(deployed),
        }
        if transitions:
            summary = {
                "planned": len(transitions),
                "valid": sum(t["helper_valid"] for t in transitions),
                "em": mean(t["helper_correct"] for t in transitions),
                "f1": mean(t["helper_f1"] for t in transitions),
            }
            for label, h, f in (
                ("correct_to_wrong", True, False),
                ("wrong_to_correct", False, True),
            ):
                subset = [
                    t for t in transitions if t["helper_correct"] == h and t["final_correct"] == f
                ]
                summary[label] = {
                    "episodes": len(subset),
                    "parents": sorted({t["case_id"] for t in subset}),
                    "rows": subset,
                }
            helper_results[arm] = summary
    rng = random.Random(seed)
    samples = [[rng.randrange(len(parents)) for _ in parents] for _ in range(draws)]
    contrasts = {}
    for name, weights in CONTRASTS.items():
        contrasts[name] = {}
        for metric in ("em", "f1"):
            values = [
                sum(
                    weight * mean(scores[p, r, arm][metric] for r in range(repeats))
                    for arm, weight in zip(ARMS, weights, strict=True)
                )
                for p in parents
            ]
            contrasts[name][metric] = interval(values, samples)
    for path in (Path(__file__), Path(probe.__file__), Path(analysis.__file__)):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    official = probe.MUSIQUE / "metrics" / "answer.py"
    hashes[str(official)] = hashlib.sha256(official.read_bytes()).hexdigest()
    return {
        "output": str(output),
        "cases": str(cases_path),
        "input_and_code_sha256": hashes,
        "method": {
            "draws": draws,
            "seed": seed,
            "parents_in_PLAN_order": parents,
            "repeats": repeats,
            "resampling": "Python random.Random.randrange; paired "
            "parent means; linearly interpolated percentile 95% CI; failures zero",
        },
        "groups": groups,
        "contrasts": contrasts,
        "isolated_last_helper": helper_results,
        "parse_failures": failures,
        "missing_episodes": len(expected_keys) - len(episodes),
        "source_exclusions": plan.get("source_exclusions"),
        "cluster_overlap": analysis.component_overlap([by_id[p] for p in parents]),
        "second_question_literal_hash1": {
            "model": sum("#1" in states[p]["subquestions"][1] for p in parents),
            "reference": sum(
                "#1" in by_id[p]["metadata"]["question_decomposition"][1]["question"]
                for p in parents
            ),
        },
        "costs": {
            "new_physical": cost(list(calls.values())),
            "historical_reused_once": cost(list(roots.values())),
            "including_historical_once": cost([*calls.values(), *roots.values()]),
            "start_receipts": len(starts),
            "unresolved_starts": sum(s["call_id"] not in calls for s in starts),
            "extra_start_attempts": len(starts) - len({s["call_id"] for s in starts}),
            "transport_failure_calls": sum(not c.get("available") for c in calls.values()),
            "unlinked_new_calls": sorted(set(calls) - referenced),
        },
        "cautions": CAUTIONS,
    }


def markdown(report):
    lines = [
        "# Execution diagnostic",
        "",
        f"Source: `{report['output']}`",
        "",
        "| Arm | EM | F1 | Correct / planned | New calls | New tokens | Deployed tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in report["groups"].items():
        lines.append(
            f"| {arm} | {row['em']:.2%} | {row['f1']:.2%} | "
            f"{row['correct']:g}/{row['planned']} | {row['new_physical_cost']['calls']} | "
            f"{row['new_physical_cost']['total_tokens']} | "
            f"{row['hypothetical_deployed_cost']['total_tokens']} |"
        )
    lines += [
        "",
        "Paired contrasts in percentage points:",
        "",
        "| Contrast | EM [95% CI] | F1 [95% CI] |",
        "|---|---:|---:|",
    ]
    for name, metrics in report["contrasts"].items():
        cells = [
            f"{v['estimate'] * 100:+.2f} [{v['ci95'][0] * 100:+.2f}, {v['ci95'][1] * 100:+.2f}]"
            for v in metrics.values()
        ]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += [
        "",
        f"Bootstrap: {report['method']['draws']} draws; "
        f"seed {report['method']['seed']}. All planned repeats included, failures zero.",
        "",
        "Costs (new acquisition, historical once, and their sum are separate):",
        "",
        "```json",
        json.dumps(report["costs"], indent=2),
        "```",
        "",
        "Last-helper diagnostics:",
        "",
        "| Arm | Helper EM | Helper F1 | Correct→wrong episodes / parents | "
        "Wrong→correct episodes / parents |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, row in report["isolated_last_helper"].items():
        harm, rescue = row["correct_to_wrong"], row["wrong_to_correct"]
        lines.append(
            f"| {arm} | {row['em']:.2%} | {row['f1']:.2%} | "
            f"{harm['episodes']} / {len(harm['parents'])} | "
            f"{rescue['episodes']} / {len(rescue['parents'])} |"
        )
    lines += [
        "",
        f"Helper parse failures: {len(report['parse_failures'])}; "
        f"missing episodes: {report['missing_episodes']}.",
        "",
        *["- " + caution for caution in report["cautions"]],
        "",
    ]
    return "\n".join(lines)


def write_report(report, path):
    path = Path(path)
    sibling = path.with_suffix(".md")
    if path == sibling or path.exists() or sibling.exists():
        raise FileExistsError("report and markdown must be distinct unused paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with sibling.open("x") as handle:
        handle.write(markdown(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        parser.error("report already exists; choose a new immutable report path")
    report = analyze(args.output, args.cases)
    write_report(report, args.report)
    print(json.dumps({"report": str(args.report), "groups": report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
