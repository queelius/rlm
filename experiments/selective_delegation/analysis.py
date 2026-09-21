"""Parent-clustered exploratory analysis of common-state intervention repeats.

Leave-one-repeat-out selection uses outcomes from the SAME parent. It diagnoses
repeatable heterogeneity; it is neither a deployable router nor an oracle bound.
Only the standard library is needed; this script never loads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean

ARMS = ("finish", "reconsider", "targeted", "decompose")
METRICS = ("em", "f1")
BOOTSTRAP_SEED = 2026092107


def average(values):
    return mean(values) if values else None


def interval(values, samples, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap over whole parent contributions, never episodes."""
    if not values:
        return {"estimate": None, "ci95": None, "parents": 0}
    generator = random.Random(seed)
    draws = sorted(mean(generator.choices(values, k=len(values))) for _ in range(samples))
    return {
        "estimate": mean(values),
        "ci95": [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]],
        "parents": len(values),
    }


def call_cost(records):
    """Known token totals are lower bounds whenever a usage field is missing."""
    result = dict(
        calls=len(records),
        prompt_tokens=0,
        completion_tokens=0,
        unknown_usage_calls=0,
        unavailable_calls=0,
    )
    for record in records:
        usage = record.get("usage") or {}
        unknown = False
        for field in ("prompt_tokens", "completion_tokens"):
            value = usage.get(field)
            if type(value) is int and value >= 0:
                result[field] += value
            else:
                unknown = True
        result["unknown_usage_calls"] += unknown
        result["unavailable_calls"] += not record.get("available", False)
    result["known_total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    result["token_totals_are_lower_bounds"] = bool(result["unknown_usage_calls"])
    return result


def add_cost(costs, denominator):
    fields = (
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "unknown_usage_calls",
        "unavailable_calls",
        "known_total_tokens",
    )
    result = {key: sum(value[key] for value in costs) for key in fields}
    result["token_totals_are_lower_bounds"] = bool(result["unknown_usage_calls"])
    result["mean_calls_per_planned_deployment"] = (
        result["calls"] / denominator if denominator else None
    )
    result["mean_known_tokens_per_planned_deployment"] = (
        result["known_total_tokens"] / denominator if denominator else None
    )
    return result


def choose_arm(table, parents, training):
    # Fixed tie order prefers finish; F1 breaks EM ties before arm order.
    scores = {}
    for arm in ARMS:
        scores[arm] = tuple(
            sum(table[(parent, arm, repeat)][metric] for parent in parents for repeat in training)
            for metric in METRICS
        )
    return max(ARMS, key=scores.__getitem__)


def cross_validate(table, parents, repeats, samples):
    # Every included parent has all continuations or a recorded failed checkpoint.
    if repeats != 3:
        return {"status": "requires_exactly_three_repeats", "parents": 0}
    contributions = {
        key: {metric: [] for metric in METRICS} for key in ("selected", "fixed", "finish")
    }
    costs = {key: [] for key in contributions}
    decisions = []
    selected_counts = Counter()
    for _parent in parents:
        for key in contributions:
            for metric in METRICS:
                contributions[key][metric].append([])
    for held in range(repeats):
        training = [repeat for repeat in range(repeats) if repeat != held]

        fixed = choose_arm(table, parents, training) if parents else "finish"
        for index, parent in enumerate(parents):
            selected = choose_arm(table, [parent], training)
            selected_counts[selected] += 1
            decisions.append(
                {
                    "case_id": parent,
                    "held_repeat": held,
                    "selected_arm": selected,
                    "fixed_arm": fixed,
                }
            )
            for key, arm in (("selected", selected), ("fixed", fixed), ("finish", "finish")):
                row = table[(parent, arm, held)]
                costs[key].append(row["deployed_cost"])
                for metric in METRICS:
                    contributions[key][metric][index].append(row[metric])
    values = {
        key: {metric: [mean(v) for v in rows] for metric, rows in metrics.items()}
        for key, metrics in contributions.items()
    }
    result = {
        "status": "computed",
        "parents": len(parents),
        "parent_ids": parents,
        "decisions": decisions,
        "selected_arm_counts": dict(selected_counts),
        "selection_rule": "maximize other-two-repeat EM, then F1, then fixed ARMS order",
        "interpretation": "Optimistic same-parent information diagnostic; not a deployable "
        "router, held-out-parent generalization estimate, or max-of-noisy-means oracle.",
        "fixed_comparator": "One global arm selected on other two repeats across the same parents.",
        "ci_scope": "Parent-cluster bootstrap of held-repeat contributions, conditional on "
        "fitted fold selections; excludes selection-refitting uncertainty.",
    }
    for key in values:
        result[key] = {metric: average(rows) for metric, rows in values[key].items()}
        result[key]["deployed_cost"] = add_cost(costs[key], len(parents) * repeats)
    for baseline in ("fixed", "finish"):
        result["selected_minus_" + baseline] = {
            metric: interval(
                [
                    a - b
                    for a, b in zip(
                        values["selected"][metric], values[baseline][metric], strict=True
                    )
                ],
                samples,
            )
            for metric in METRICS
        }
    return result


def component_overlap(cases):
    roots = {case["id"]: case["id"] for case in cases}
    owners = {}
    missing = []

    def root(parent):
        while roots[parent] != parent:
            parent = roots[parent]
        return parent

    for case in cases:
        components = case.get("metadata", {}).get("component_ids", [])
        if not components:
            missing.append(case["id"])
        for component in components:
            if component in owners:
                roots[root(case["id"])] = root(owners[component])
            owners[component] = case["id"]
    sizes = Counter(root(case["id"]) for case in cases)
    return {
        "connected_parent_clusters": len(sizes),
        "cluster_sizes": sorted(sizes.values(), reverse=True),
        "largest_cluster": max(sizes.values(), default=0),
        "parents_sharing_components": sum(size for size in sizes.values() if size > 1),
        "parents_missing_component_ids": missing,
        "interpretation": "Parent-bootstrap intervals condition on this parent sample; shared "
        "atomic components violate full parent independence. These are exploratory intervals, "
        "not component-cluster generalization guarantees.",
    }


def native_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def format_overlay(output, followup, cases, read, hashes, episodes, calls, paths, samples):
    """Verify and overlay observed pairs in memory; never manufacture collector files."""
    plan = read(followup / "PLAN.json")
    if Path(plan["source_output"]).resolve() != output:
        raise ValueError("format source output identity differs")
    if plan["source_plan_sha256"] != hashes[str(output / "PLAN.json")]:
        raise ValueError("format source plan hash differs")
    if plan["cases_sha256"] != hashes[str(cases)]:
        raise ValueError("format cases hash differs")
    for path, expected in plan["source_hashes"].items():
        if hashes.get(path) != expected:
            raise ValueError("format source hash differs: " + path)
    original = {row["episode_id"]: (key, row) for key, row in episodes.items()}
    if set(plan["episodes"]) - original.keys():
        raise ValueError("format plan names unknown original episodes")
    new_calls = {}
    for path in sorted((followup / "calls").glob("*.json")):
        row = read(path)
        if row["call_id"] in calls or row["call_id"] in new_calls:
            raise ValueError("duplicate format call identity")
        new_calls[row["call_id"]] = row
    overlay, pairs = {}, []
    for path in sorted((followup / "pairs").glob("*.json")):
        pair = read(path)
        identity = pair["episode_id"]
        if identity not in plan["episodes"]:
            raise ValueError("unexpected format pair")
        key, source = original[identity]
        contract = plan["episodes"][identity]
        if key in overlay or (pair["case_id"], pair["arm"], pair["repeat"]) != key:
            raise ValueError("duplicate or mismatched format coordinate")
        if not source["available"] or pair["seed"] != source["seed"]:
            raise ValueError("format pair source availability or seed differs")
        required = {str(paths[key])} | {
            str(output / "calls" / (call_id + ".json")) for call_id in source["call_ids"]
        }
        if set(pair["source_hashes"]) != required:
            raise ValueError("format pair source hash closure differs")
        for source_path, expected in pair["source_hashes"].items():
            if (
                hashes.get(source_path) != expected
                or plan["source_hashes"].get(source_path) != expected
            ):
                raise ValueError("format pair source hash differs: " + source_path)
        requests = {call_id: calls[call_id]["request_digest"] for call_id in source["call_ids"]}
        if (
            pair["source_request_digests"] != requests
            or contract["source_request_digests"] != requests
        ):
            raise ValueError("format source request binding differs")
        for call_id in source["call_ids"]:
            if native_digest(calls[call_id]["request"]) != requests[call_id]:
                raise ValueError("format source request digest differs")
        if pair["reused_call_ids"] != source["call_ids"][:-1]:
            raise ValueError("format reused checkpoint/helper differ")
        new_id = pair["new_call_id"]
        if new_id != identity + "-short-final" or new_id not in new_calls:
            raise ValueError("format final call receipt missing or identity differs")
        new = new_calls[new_id]
        if new["available"] != pair["available"]:
            raise ValueError("format final availability differs")
        if (
            native_digest(new["request"]) != new["request_digest"]
            or native_digest(new["prompt"]) != contract["prompt_digest"]
        ):
            raise ValueError("format final prompt/request digest differs")
        sampling = new["request"]["sampling_params"]
        if any(
            sampling[field] != contract[field] for field in ("seed", "temperature", "max_tokens")
        ):
            raise ValueError("format final sampling differs")
        if contract["seed"] != source["seed"] or contract["max_tokens"] != 128:
            raise ValueError("format sampling source binding differs")
        if any(pair["old"][field] != source[field] for field in ("valid", "correct", "f1")):
            raise ValueError("format original grade differs")
        overlay[key] = {
            **source,
            **pair["new"],
            "available": pair["available"],
            "call_ids": pair["reused_call_ids"] + [new_id],
        }
        pairs.append(pair)
    comparisons = {}
    for arm in ARMS:
        selected = [pair for pair in pairs if pair["arm"] == arm]
        per_parent = {}
        for pair in selected:
            parent = per_parent.setdefault(pair["case_id"], {metric: [] for metric in METRICS})
            for metric, field in (("em", "correct"), ("f1", "f1")):
                old = float(pair["old"][field]) if pair["old"]["valid"] else 0.0
                new = (
                    float(pair["new"][field]) if pair["available"] and pair["new"]["valid"] else 0.0
                )
                parent[metric].append(new - old)
        comparisons[arm] = {
            "observed_pairs": len(selected),
            "old": {
                metric: average(
                    [
                        float(pair["old"][field]) if pair["old"]["valid"] else 0.0
                        for pair in selected
                    ]
                )
                for metric, field in (("em", "correct"), ("f1", "f1"))
            },
            "new": {
                metric: average(
                    [
                        float(pair["new"][field])
                        if pair["available"] and pair["new"]["valid"]
                        else 0.0
                        for pair in selected
                    ]
                )
                for metric, field in (("em", "correct"), ("f1", "f1"))
            },
            "new_minus_old": {
                metric: interval([mean(row[metric]) for row in per_parent.values()], samples)
                for metric in METRICS
            },
        }
    info = {
        "output": str(followup),
        "phrase_instruction": plan["phrase_instruction"],
        "planned_new_finals": len(plan["episodes"]),
        "observed_pairs": len(pairs),
        "source_exclusions": plan["source_exclusions"],
        "arms": comparisons,
        "comparison_denominator": "Observed old/new pairs only; unavailable new finals score zero. "
        "Primary arm/CV analysis retains all original planned parents and missing-pair accounting.",
        "deployment_note": "Reuse original checkpoint/helper and substitute NEW final. "
        "The old final is acquisition cost only, never hypothetical deployment cost.",
    }
    starts = [read(path) for path in sorted((followup / "starts").glob("*.json"))]
    return overlay, new_calls, starts, info


def analyze(output: Path, cases: Path, *, bootstrap_samples=2000, format_output=None):
    if bootstrap_samples < 20:
        raise ValueError("at least 20 bootstrap samples required")
    output, cases = Path(output).resolve(), Path(cases).resolve()
    hashes = {}

    def read(path, jsonl=False):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return (
            [json.loads(line) for line in raw.splitlines() if line.strip()]
            if jsonl
            else json.loads(raw)
        )

    plan = read(output / "PLAN.json")
    all_cases = read(cases, jsonl=True)
    by_id = {row["id"]: row for row in all_cases}
    parents, repeats = plan["case_ids"], plan["repeats"]
    if len(by_id) != len(all_cases) or len(set(parents)) != len(parents):
        raise ValueError("duplicate case identities")
    if not parents or set(parents) - by_id.keys() or repeats < 1 or tuple(plan["arms"]) != ARMS:
        raise ValueError("invalid planned case/arm/repeat inventory")
    if plan.get("cases_sha256") not in (None, hashes[str(cases)]):
        raise ValueError("cases differ from collector plan")
    checkpoints = {}
    for path in sorted((output / "checkpoints").glob("*.json")):
        row = read(path)
        key = row["case_id"]
        if key not in parents or key in checkpoints:
            raise ValueError("unexpected or duplicate checkpoint")
        checkpoints[key] = row
    calls = {}
    for path in sorted((output / "calls").glob("*.json")):
        row = read(path)
        if row["call_id"] in calls:
            raise ValueError("duplicate physical call receipt")
        calls[row["call_id"]] = row
    episodes, episode_paths = {}, {}
    for path in sorted((output / "episodes").glob("*.json")):
        row = read(path)
        key = row["case_id"], row["arm"], row["repeat"]
        if (
            key in episodes
            or key[0] not in parents
            or key[1] not in ARMS
            or key[2] not in range(repeats)
        ):
            raise ValueError("unexpected or duplicate episode coordinate")
        episodes[key] = row
        episode_paths[key] = path
    starts = [read(path) for path in sorted((output / "starts").glob("*.json"))]
    original_cost = call_cost(list(calls.values()))
    format_info = None
    new_cost = call_cost([])
    if format_output is not None:
        episodes, new_calls, new_starts, format_info = format_overlay(
            output,
            Path(format_output).resolve(),
            cases,
            read,
            hashes,
            episodes,
            calls,
            episode_paths,
            bootstrap_samples,
        )
        new_cost = call_cost(list(new_calls.values()))
        calls.update(new_calls)
        starts.extend(new_starts)
    unresolved_starts = sum(row["call_id"] not in calls for row in starts)
    table = {}
    for parent in parents:
        checkpoint = checkpoints.get(parent)
        checkpoint_failed = checkpoint is not None and (
            checkpoint.get("error") is not None or checkpoint.get("state") is None
        )
        for arm in ARMS:
            for repeat in range(repeats):
                key = parent, arm, repeat
                episode = episodes.get(key)
                if episode is not None and (checkpoint is None or checkpoint_failed):
                    raise ValueError("continuation without valid checkpoint")
                if checkpoint_failed:
                    status = checkpoint.get("error") or "checkpoint_invalid"
                elif episode is None:
                    status = "missing_episode"
                elif not episode.get("available", False):
                    status = "transport_failure"
                elif not episode.get("valid", False):
                    status = "invalid_final"
                else:
                    status = "scored"
                ids = list(dict.fromkeys(episode.get("call_ids", []))) if episode else []
                initial_id = checkpoint.get("call_id") if checkpoint else None
                if initial_id and initial_id not in ids:
                    ids.insert(0, initial_id)
                receipts = [
                    calls.get(identity, {"available": False, "usage": {}}) for identity in ids
                ]
                em = float(bool(episode.get("correct"))) if status == "scored" else 0.0
                f1 = float(episode["f1"]) if status == "scored" else 0.0
                if not math.isfinite(f1) or not 0 <= f1 <= 1:
                    raise ValueError("invalid stored F1")
                table[key] = {
                    "case_id": parent,
                    "arm": arm,
                    "repeat": repeat,
                    "status": status,
                    "em": em,
                    "f1": f1,
                    "checkpoint_failure": checkpoint_failed,
                    "deployed_cost": call_cost(receipts),
                    "missing_call_receipts": sum(identity not in calls for identity in ids),
                }
    arm_results, paired, parent_metrics = {}, {}, {}
    for arm in ARMS:
        rows = [table[(parent, arm, repeat)] for parent in parents for repeat in range(repeats)]
        metrics = {
            metric: [
                mean(table[(parent, arm, repeat)][metric] for repeat in range(repeats))
                for parent in parents
            ]
            for metric in METRICS
        }
        parent_metrics[arm] = metrics
        counts = Counter(row["status"] for row in rows)
        arm_results[arm] = {
            "planned_deployments": len(rows),
            "recorded_episodes": sum(
                (parent, arm, repeat) in episodes for parent in parents for repeat in range(repeats)
            ),
            "checkpoint_failures": sum(row["checkpoint_failure"] for row in rows),
            "missing_episodes": counts["missing_episode"],
            "status_counts": dict(counts),
            "deployed_cost": add_cost([row["deployed_cost"] for row in rows], len(rows)),
            **{metric: mean(values) for metric, values in metrics.items()},
            "parent_cluster_ci": {
                metric: interval(values, bootstrap_samples) for metric, values in metrics.items()
            },
        }
        if arm != "finish":
            paired[arm] = {
                metric: interval(
                    [
                        value - baseline
                        for value, baseline in zip(
                            metrics[metric], parent_metrics["finish"][metric], strict=True
                        )
                    ],
                    bootstrap_samples,
                )
                for metric in METRICS
            }
    complete = [
        parent
        for parent in parents
        if all(
            table[(parent, arm, repeat)]["status"] != "missing_episode"
            for arm in ARMS
            for repeat in range(repeats)
        )
    ]
    cv = cross_validate(table, complete, repeats, bootstrap_samples)
    collection_complete = len(complete) == len(parents)
    physical = call_cost(list(calls.values()))
    failures = {
        "checkpoint_failures": sum(
            row.get("error") is not None or row.get("state") is None for row in checkpoints.values()
        ),
        "checkpoint_transport_failures": sum(
            row.get("error") == "checkpoint_transport_failure" for row in checkpoints.values()
        ),
        "transport_episodes": sum(row["status"] == "transport_failure" for row in table.values()),
        "invalid_final_episodes": sum(row["status"] == "invalid_final" for row in table.values()),
        "missing_episodes": sum(row["status"] == "missing_episode" for row in table.values()),
        "missing_call_receipt_references": sum(
            row["missing_call_receipts"] for row in table.values()
        ),
        "parents_excluded_from_cv": len(parents) - len(complete),
    }
    gains = [
        cv.get("selected_minus_" + baseline, {}).get("em", {}) for baseline in ("fixed", "finish")
    ]
    signal = (
        collection_complete
        and len(complete) >= 16
        and failures["transport_episodes"] == 0
        and failures["checkpoint_transport_failures"] == 0
        and failures["missing_call_receipt_references"] == 0
        and physical["unavailable_calls"] == 0
        and unresolved_starts == 0
        and all(
            gain.get("estimate") is not None and gain["estimate"] >= 0.03 and gain["ci95"][0] > 0
            for gain in gains
        )
    )
    return {
        "schema": "selective-delegation-analysis-v1",
        "output": str(output),
        "case_input": str(cases),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashes,
        "parents": len(parents),
        "complete_parents": len(complete),
        "splits": dict(Counter(by_id[parent]["split"] for parent in parents)),
        "repeats": repeats,
        "denominator_policy": "All planned parents x repeats per arm. Recorded checkpoint failures "
        "score zero in every arm. Missing continuations score zero only as an explicit incomplete "
        "collection lower bound and are excluded by whole parent from cross-validation.",
        "grading": "Stored official MuSiQue alias-max EM/F1 from collector; "
        "unavailable/invalid outcomes zero.",
        "bootstrap": {
            "unit": "parent",
            "samples": bootstrap_samples,
            "seed": BOOTSTRAP_SEED,
            "method": "percentile",
            "ci": 0.95,
            "interpretation": "exploratory and conditional "
            "on shared checkpoints and parent inventory; no multiplicity correction",
        },
        "component_overlap": component_overlap([by_id[parent] for parent in parents]),
        "repeat_scope": "Three continuation seeds conditional on one fixed initial checkpoint per "
        "parent, not independent end-to-end repetitions.",
        "arms": arm_results,
        "paired_vs_finish": paired,
        "cross_validation": cv,
        "physical_cost": physical,
        "physical_cost_split": {"original_acquisition": original_cost, "new_final_only": new_cost},
        "format_comparison": format_info,
        "failures": failures,
        "unresolved_started_attempts": unresolved_starts,
        "physical_cost_note": "One receipt per physical call, shared checkpoints counted once. "
        "Unresolved starts may add unobserved work; unknown tokens are not treated as known zero.",
        "deployed_cost_note": "Checkpoint once per deployment plus its actual continuation "
        "calls. Router/training costs absent: no router trained or executed in this probe.",
        "parent_arm_metrics": [
            {
                "case_id": parent,
                "arm": arm,
                **{metric: parent_metrics[arm][metric][index] for metric in METRICS},
            }
            for index, parent in enumerate(parents)
            for arm in ARMS
        ],
        "decision": {
            "format_checked": format_info is not None,
            "training_candidate": signal and format_info is not None,
            "collection_complete": collection_complete,
            "repeatable_headroom_signal": signal,
            "rule": "Exploratory strong-signal screen, not an absolute training gate: "
            "complete collection, no transport contamination, >=16 parents, selected-minus-fixed "
            "and selected-minus-finish EM >=0.03 and both conditional bootstrap lower bounds >0.",
            "next_step": "Train a bounded observation-only controller and evaluate untouched "
            "parents; this signal alone does not establish learnability."
            if signal and format_info is not None
            else "Check the answer-format follow-up before treating this as a training candidate."
            if signal
            else "No strong repeatability flag at this sample size; this is not evidence of "
            "no headroom. Assess pilot plus validation uncertainty and learning value before "
            "choosing training or another targeted probe.",
        },
    }


def markdown(report):
    lines = [
        f"Selective delegation: {report['complete_parents']}/{report['parents']} complete parents, "
        f"{report['repeats']} repeats.",
        "",
        "| Arm | EM | F1 | Missing | Checkpoint failures | Known tokens/deployment |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, row in report["arms"].items():
        lines.append(
            f"| {arm} | {row['em']:.3f} | {row['f1']:.3f} | {row['missing_episodes']} | "
            f"{row['checkpoint_failures']} | "
            f"{row['deployed_cost']['mean_known_tokens_per_planned_deployment']:.1f} |"
        )
    lines += ["", report["denominator_policy"], ""]
    for arm, comparisons in report["paired_vs_finish"].items():
        value = comparisons["em"]
        lines.append(
            f"{arm} minus finish EM: {value['estimate']:+.3f}, parent-bootstrap95% "
            f"[{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]."
        )
    cv = report["cross_validation"]
    if cv.get("parents", 0):
        lines += [
            "",
            cv["interpretation"],
            f"Other-two-repeat selection / fixed / finish EM: {cv['selected']['em']:.3f} / "
            f"{cv['fixed']['em']:.3f} / {cv['finish']['em']:.3f}.",
        ]
        for baseline in ("fixed", "finish"):
            value = cv["selected_minus_" + baseline]["em"]
            lines.append(
                f"Selection minus {baseline}: {value['estimate']:+.3f},95% "
                f"[{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]."
            )
        lines.append(cv["ci_scope"])
    physical = report["physical_cost"]
    lines += [
        "",
        f"Physical cost: {physical['calls']} calls, {physical['known_total_tokens']} known "
        f"tokens, {physical['unknown_usage_calls']} calls with unknown usage; "
        f"{report['unresolved_started_attempts']} unresolved started attempts.",
        report["deployed_cost_note"],
        "",
        report["repeat_scope"],
        f"Atomic-component overlap: {report['component_overlap']['connected_parent_clusters']} "
        f"connected clusters; largest has {report['component_overlap']['largest_cluster']} "
        "parents.",
        report["component_overlap"]["interpretation"],
        "",
        report["decision"]["next_step"],
        "All intervals are exploratory and unadjusted for multiple comparisons.",
        "",
    ]
    if report["format_comparison"] is not None:
        info = report["format_comparison"]
        costs = report["physical_cost_split"]
        lines += [
            f"Answer format follow-up: {info['observed_pairs']}/{info['planned_new_finals']} "
            "new final pairs.",
            info["phrase_instruction"],
            info["deployment_note"],
            f"Physical split: {costs['original_acquisition']['calls']} original acquisition "
            f"calls + {costs['new_final_only']['calls']} new final-only calls.",
            "Matched old → new EM by arm: "
            + "; ".join(
                f"{arm} {row['old']['em']:.3f} → {row['new']['em']:.3f} (n={row['observed_pairs']})"
                for arm, row in info["arms"].items()
                if row["observed_pairs"]
            ),
            "",
        ]
    return "\n".join(lines)


def write_report(report, path):
    path = Path(path)
    sibling = path.with_suffix(".md")
    if path == sibling or path.exists() or sibling.exists():
        raise FileExistsError("report JSON or markdown already exists (or paths coincide)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with sibling.open("x") as stream:
        stream.write(markdown(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--format-output", type=Path)
    args = parser.parse_args()
    value = analyze(
        args.output,
        args.cases,
        bootstrap_samples=args.bootstrap_samples,
        format_output=args.format_output,
    )
    write_report(value, args.report)
    print(json.dumps(value["decision"], sort_keys=True))
