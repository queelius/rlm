"""Read-only, planned-denominator analysis of matched frozen-root helper evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

import eval_helper
import eval_planner
import probe

SEED = 2026092113


def measured(records):
    result = eval_planner.cost(records)
    durations = [r["ended"] - r["started"] for r in records if "ended" in r and "started" in r]
    if any(d < 0 for d in durations):
        raise ValueError("negative native latency")
    result.update(
        total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        known_latency_seconds=sum(durations),
        mean_returned_call_latency_seconds=mean(durations) if durations else None,
        unknown_latency_calls=len(records) - len(durations),
    )
    return result


def component_clusters(cases):
    roots, owners = {c["id"]: c["id"] for c in cases}, {}

    def root(p):
        while roots[p] != p:
            p = roots[p]
        return p

    for case in cases:
        for component in case.get("metadata", {}).get("component_ids", []):
            if component in owners:
                roots[root(case["id"])] = root(owners[component])
            owners[component] = case["id"]
    groups = {}
    for case in cases:
        groups.setdefault(root(case["id"]), []).append(case["id"])
    return list(groups.values())


def clustered_interval(differences, clusters, draws, seed):
    """Resample whole clusters; retain parent weighting when cluster sizes differ."""
    rng, estimates = random.Random(seed), []
    for _ in range(draws):
        sampled = [p for _ in clusters for p in clusters[rng.randrange(len(clusters))]]
        estimates.append(mean(differences[p] for p in sampled))
    estimates.sort()

    def percentile(q):
        location = (len(estimates) - 1) * q
        lower = int(location)
        upper = min(lower + 1, len(estimates) - 1)
        return estimates[lower] + (estimates[upper] - estimates[lower]) * (location - lower)

    return {"estimate": mean(differences.values()), "ci95": [percentile(0.025), percentile(0.975)]}


def analyze(output, cases_path, *, draws=20000, seed=SEED):
    output, cases_path = Path(output).resolve(), Path(cases_path).resolve()
    hashes, cache = {}, {}

    def track(path, expected=None):
        path = str(Path(path).resolve())
        if path not in cache:
            cache[path] = Path(path).read_bytes()
            hashes[path] = hashlib.sha256(cache[path]).hexdigest()
        if expected is not None and hashes[path] != expected:
            raise ValueError("source/receipt hash differs: " + path)
        return cache[path]

    def read(path):
        return json.loads(track(path))

    plan = read(output / "PLAN.json")
    if hashlib.sha256(track(cases_path)).hexdigest() != plan["cases_sha256"]:
        raise ValueError("cases differ from immutable PLAN")
    cases = [json.loads(line) for line in track(cases_path).splitlines() if line.strip()]
    by_id = {c["id"]: c for c in cases}
    parents, repeats, conditions = plan["case_ids"], plan["repeats"], plan["conditions"]
    if (
        plan.get("schema") != "matched-frozen-root-helper-evaluation-v1"
        or conditions != list(eval_helper.CONDITIONS)
        or not parents
        or repeats < 1
        or draws < 1
        or len(set(parents)) != len(parents)
        or len(by_id) != len(cases)
        or any(p not in by_id for p in parents)
    ):
        raise ValueError("unexpected helper evaluation inventory")
    # Hash the analysis inputs once. Record checkpoint bindings, but do not repeatedly
    # reread multi-GB model/checkpoint ancestry already sealed by the collector.
    for path, digest in plan.get("source_hashes", {}).items():
        if Path(path).suffix == ".json":
            track(path, digest)
    calls, starts, episodes = {}, {}, {}
    for directory, target in (("calls", calls), ("starts", starts)):
        for path in sorted((output / directory).glob("*.json")):
            row = read(path)
            cid = row["call_id"]
            if cid != path.stem or cid in target or row["condition"] not in conditions:
                raise ValueError("unexpected native call identity")
            if row["role"] not in ("helper", "final"):
                raise ValueError("historical roots must not appear as new physical calls")
            if "request" in row and probe.runtime.digest(row["request"]) != row["request_digest"]:
                raise ValueError("native request digest mismatch")
            for field, ids in (
                ("prompt_tokens", "input_token_ids"),
                ("completion_tokens", "output_token_ids"),
            ):
                if (
                    directory == "calls"
                    and ids in row
                    and field in row.get("usage", {})
                    and row["usage"][field] != len(row[ids])
                ):
                    raise ValueError("native token IDs differ from usage")
            target[cid] = row
    inventory = {(p, r, c) for p in parents for r in range(repeats) for c in conditions}
    for path in sorted((output / "episodes").glob("*.json")):
        row = read(path)
        key = row["case_id"], row["repeat"], row["condition"]
        if key not in inventory or key in episodes or path.stem != row["episode_id"]:
            raise ValueError("unexpected episode identity")
        episodes[key] = row
    matched_fields = (
        "plan",
        "plan_valid",
        "seed",
        "source_episode_id",
        "source_hashes",
        "reused_root_call_id",
        "root_request_digest",
    )
    for p in parents:
        for r in range(repeats):
            counterparts = [episodes[p, r, c] for c in conditions if (p, r, c) in episodes]
            for row in counterparts[1:]:
                if any(row[k] != counterparts[0][k] for k in matched_fields):
                    raise ValueError("matched root/plan/seed contract differs")
            contracts = {}
            for row in counterparts:
                for cid in row["call_ids"]:
                    if cid not in calls:
                        continue  # Linked-call validation below gives the precise error.
                    call = calls[cid]
                    key = cid.removeprefix(row["episode_id"])
                    request = call["request"]
                    contract = {k: request.get(k) for k in ("seed", "sampling", "model")}
                    if key in contracts and contracts[key] != contract:
                        raise ValueError("matched downstream sampling contract differs")
                    contracts[key] = contract
            if counterparts and plan.get("source_output"):
                row = counterparts[0]
                root = read(
                    Path(plan["source_output"]) / "calls" / (row["reused_root_call_id"] + ".json")
                )
                root_plan = eval_helper.validate_root(
                    root,
                    by_id[p],
                    plan["model"],
                    plan["root_adapter_binding"]["adapter_model.safetensors"],
                )
                if root_plan != row["plan"] or root["request_digest"] != row["root_request_digest"]:
                    raise ValueError("matched source root differs")
    values, linked, groups = {}, set(), {}
    for condition in conditions:
        rows = []
        for p in parents:
            for repeat in range(repeats):
                row = episodes.get((p, repeat, condition))
                current = []
                for cid in row["call_ids"] if row else []:
                    if cid in linked or cid not in calls or calls[cid]["condition"] != condition:
                        raise ValueError("missing/reused/mismatched physical call")
                    linked.add(cid)
                    current.append(calls[cid])
                trace, steps, status = [], [], "missing_episode" if row is None else row["status"]
                if row and row["plan_valid"]:
                    for i, question in enumerate(row["plan"]["subquestions"]):
                        cid = row["episode_id"] + f"-helper-{i + 1}"
                        call = calls.get(cid) if cid in row["call_ids"] else None
                        step = {
                            "step": i + 1,
                            "question": question,
                            "valid": False,
                            "observed": bool(call and call["available"]),
                        }
                        if call:
                            resolved = eval_planner.bind_question(
                                question, [s["answer"] for s in trace]
                            )
                            step["resolved_question"] = resolved
                            expected = eval_helper.helper_prompt(by_id[p], resolved, condition)
                            if call["request"]["prompt"] != expected:
                                raise ValueError(
                                    "helper request differs from public reconstruction"
                                )
                            if call["available"]:
                                try:
                                    answer = eval_planner.parse_helper_answer(call["text"])
                                except (ValueError, TypeError):
                                    status = "invalid_helper"
                                else:
                                    step.update(valid=True, answer=answer)
                                    trace.append(
                                        {
                                            "step": i + 1,
                                            "question": question,
                                            "resolved_question": resolved,
                                            "answer": answer,
                                        }
                                    )
                            else:
                                status = "helper_unavailable"
                        steps.append(step)
                        if not step["valid"]:
                            break
                    if trace != row["helper_trace"]:
                        raise ValueError("saved helper trace differs from native answers")
                finals = [c for c in current if c["role"] == "final"]
                if len(finals) > 1:
                    raise ValueError("multiple final calls")
                final = finals[0] if finals else None
                grade = probe.grade("", by_id[p])
                if final:
                    expected = eval_planner.final_prompt(
                        by_id[p], row["plan"], {"execution": "isolated", "steps": trace}
                    )
                    if final["request"]["prompt"] != expected:
                        raise ValueError("final request differs from frozen plan/helper trace")
                    if final["available"]:
                        grade = probe.grade(final["text"], by_id[p])
                        status = "scored" if grade["valid"] else "invalid_final"
                    else:
                        status = "final_unavailable"
                elif status == "scored":
                    status = "missing_final"
                missing = (
                    row is None
                    or "unavailable" in status
                    or "failure" in status
                    or status.startswith("missing")
                )
                scored = {
                    "case_id": p,
                    "repeat": repeat,
                    "condition": condition,
                    "present": row is not None,
                    "status": status,
                    "unobserved": missing,
                    "valid_final": grade["valid"],
                    "em": float(grade["correct"]),
                    "f1": grade["f1"],
                    "answer": grade["parsed"],
                    "steps": steps,
                }
                values[p, repeat, condition] = scored
                rows.append(scored)
        records = [r for r in calls.values() if r["condition"] == condition]
        unresolved = [
            r for cid, r in starts.items() if cid not in calls and r["condition"] == condition
        ]
        groups[condition] = {
            "planned_episodes": len(rows),
            "recorded_episodes": sum(r["present"] for r in rows),
            "missing_episodes": sum(not r["present"] for r in rows),
            "unobserved_outcomes": sum(r["unobserved"] for r in rows),
            "returned_protocol_invalid_outcomes": sum(
                not r["unobserved"] and not r["valid_final"] for r in rows
            ),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "correct": sum(r["em"] for r in rows),
            "em": mean(r["em"] for r in rows),
            "f1": mean(r["f1"] for r in rows),
            "valid_finals": sum(r["valid_final"] for r in rows),
            "helper_steps_observed": sum(s["observed"] for r in rows for s in r["steps"]),
            "helper_steps_valid": sum(s["valid"] for r in rows for s in r["steps"]),
            "new_physical_cost": measured(records + unresolved),
            "new_cost_by_role": {
                role: measured([r for r in records + unresolved if r["role"] == role])
                for role in ("helper", "final")
            },
        }
    clusters = component_clusters([by_id[p] for p in parents])
    comparisons = {}
    for left, right in combinations(conditions, 2):
        comparison = {"left": left, "right": right}
        for metric in ("em", "f1"):
            delta = {
                p: mean(
                    values[p, r, right][metric] - values[p, r, left][metric] for r in range(repeats)
                )
                for p in parents
            }
            comparison[metric] = clustered_interval(delta, clusters, draws, seed)
            comparison[metric]["parent_bootstrap_ci95"] = clustered_interval(
                delta, [[p] for p in parents], draws, seed
            )["ci95"]
        changes = {"wins": [], "losses": []}
        agreement = Counter(
            both_valid_steps=0,
            same_answer_steps=0,
            changed_answer_steps=0,
            changed_resolved_question_steps=0,
            both_valid_same_question_steps=0,
            changed_answer_same_question_steps=0,
            unavailable_or_invalid_steps=0,
        )
        agreement_rows = []
        for p in parents:
            for r in range(repeats):
                a, b = values[p, r, left], values[p, r, right]
                delta = b["em"] - a["em"]
                if delta:
                    category = (
                        "missing_involved"
                        if a["unobserved"] or b["unobserved"]
                        else "both_valid"
                        if a["valid_final"] and b["valid_final"]
                        else "protocol_involved"
                    )
                    changes["wins" if delta > 0 else "losses"].append(
                        {
                            "case_id": p,
                            "repeat": r,
                            "category": category,
                            "left_status": a["status"],
                            "right_status": b["status"],
                        }
                    )
                for i in range(max(len(a["steps"]), len(b["steps"]))):
                    x = a["steps"][i] if i < len(a["steps"]) else {}
                    y = b["steps"][i] if i < len(b["steps"]) else {}
                    if not x.get("valid") or not y.get("valid"):
                        agreement["unavailable_or_invalid_steps"] += 1
                        continue
                    changed = x["answer"] != y["answer"]
                    same_q = x["resolved_question"] == y["resolved_question"]
                    agreement["both_valid_steps"] += 1
                    agreement["changed_answer_steps" if changed else "same_answer_steps"] += 1
                    agreement["changed_resolved_question_steps"] += not same_q
                    agreement["both_valid_same_question_steps"] += same_q
                    agreement["changed_answer_same_question_steps"] += changed and same_q
                    agreement_rows.append(
                        {
                            "case_id": p,
                            "repeat": r,
                            "step": i + 1,
                            "left_answer": x["answer"],
                            "right_answer": y["answer"],
                            "same_resolved_question": same_q,
                            "answer_changed": changed,
                        }
                    )
        for label, rows in changes.items():
            comparison[label] = {
                "episodes": len(rows),
                "parents": sorted({r["case_id"] for r in rows}),
                "rows": rows,
                **{
                    c: sum(r["category"] == c for r in rows)
                    for c in ("both_valid", "protocol_involved", "missing_involved")
                },
            }
        comparison["helper_agreement"] = {**agreement, "rows": agreement_rows}
        comparisons[right + "_minus_" + left] = comparison
    unresolved = [r for cid, r in starts.items() if cid not in calls]
    for path in (
        Path(__file__),
        Path(eval_helper.__file__),
        Path(eval_planner.__file__),
        Path(probe.__file__),
        probe.MUSIQUE / "metrics/answer.py",
    ):
        track(path)
    return {
        "output": str(output),
        "cases": str(cases_path),
        "source_plan": plan,
        "input_receipt_analysis_sha256": hashes,
        "checkpoint_verification": "Collector-sealed model/adapter bindings retained from PLAN; "
        "analysis checks consumed JSON source receipts, not multi-GB weight ancestry again.",
        "method": {
            "parents": len(parents),
            "episodes_per_condition": len(parents) * repeats,
            "repeats": repeats,
            "draws": draws,
            "seed": seed,
            "component_clusters": clusters,
            "parents_missing_component_ids": [
                p for p in parents if not by_id[p].get("metadata", {}).get("component_ids")
            ],
            "metric": "official MuSiQue alias-max EM/F1 via probe.grade",
        },
        "primary_comparison": "trained_helper_minus_base_helper",
        "groups": groups,
        "comparisons": comparisons,
        "new_physical_cost": measured(list(calls.values()) + unresolved),
        "unlinked_call_ids": sorted(set(calls) - linked),
        "unresolved_start_ids": sorted(r["call_id"] for r in unresolved),
        "outcomes": list(values.values()),
        "cautions": [
            "All planned outcomes remain in denominators. Missing/unavailable outcomes are "
            "unobserved, not demonstrated scientific failures; incomplete scores are lower bounds.",
            "Primary intervals resample connected atomic-component clusters with parent weighting; "
            "missing component IDs fall back to singleton parents. Parent-only intervals are also "
            "reported. Few clusters can make bootstrap intervals misleadingly narrow.",
            "Helper agreement is exact string agreement at matching generated plan steps, not "
            "annotated-step accuracy. Changed upstream predictions may change downstream "
            "questions.",
            "Protocol recoveries do not establish better semantic reasoning. Both-valid gains "
            "are descriptive; full-source finals can bypass or repair helper answers.",
            "New costs exclude every historical root. Unknown token/latency totals are not zero "
            "cost; latency sums are service time, not end-to-end wall time.",
            "All three comparisons are exploratory, paired, and unadjusted; trained minus base "
            "is primary. The reminder arm controls for the extra JSON instruction, not all "
            "formatting effects.",
        ],
    }


def markdown(report):
    method = report["method"]
    lines = [
        "# Frozen-root helper evaluation",
        "",
        f"Source: `{report['output']}`",
        "",
        f"{method['parents']} parents; {method['episodes_per_condition']} planned "
        "episodes per arm. "
        f"{len(method['component_clusters'])} connected atomic-component clusters.",
        "",
        "| Arm | EM | F1 | Correct / planned | Unobserved | Invalid protocol | "
        "New calls | New tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in report["groups"].items():
        cost = row["new_physical_cost"]
        lines.append(
            f"| {condition} | {row['em']:.2%} | {row['f1']:.2%} | "
            f"{row['correct']:g}/{row['planned_episodes']} | {row['unobserved_outcomes']} | "
            f"{row['returned_protocol_invalid_outcomes']} | {cost['calls']} | "
            f"{cost['total_tokens']} |"
        )
    lines += [
        "",
        "Primary comparison: trained helper minus base helper. Differences below are "
        "percentage points with exploratory component-cluster 95% intervals.",
        "",
    ]
    for name, pair in report["comparisons"].items():
        metrics = [
            f"{m.upper()} {100 * pair[m]['estimate']:+.2f} "
            f"[{100 * pair[m]['ci95'][0]:+.2f}, {100 * pair[m]['ci95'][1]:+.2f}]"
            for m in ("em", "f1")
        ]
        lines.append(f"- {name}: " + "; ".join(metrics) + ".")
        for label in ("wins", "losses"):
            row = pair[label]
            lines.append(
                f"  - {label}: {row['episodes']} episodes, {len(row['parents'])} parents; "
                f"{row['both_valid']} both-valid, {row['protocol_involved']} protocol-involved, "
                f"{row['missing_involved']} unobserved-involved."
            )
        row = pair["helper_agreement"]
        lines.append(
            f"  - Helper answers changed at {row['changed_answer_steps']}/"
            f"{row['both_valid_steps']} both-valid steps; {row['changed_resolved_question_steps']} "
            "of these step pairs had different resolved questions."
        )
    lines += ["", "Per-arm protocol and actual cost:", ""]
    for condition, row in report["groups"].items():
        cost = row["new_physical_cost"]
        lines.append(
            f"- {condition}: statuses {json.dumps(row['status_counts'], sort_keys=True)}; "
            f"helper JSON valid {row['helper_steps_valid']}/{row['helper_steps_observed']} "
            "observed; "
            f"known service time {cost['known_latency_seconds']:.1f}s; "
            f"{cost['unknown_usage_calls']} unknown-usage and "
            f"{cost['unknown_latency_calls']} unknown-latency calls."
        )
    plan = report["source_plan"]
    lines += [
        "",
        f"Root checkpoint: `{plan.get('root_adapter')}`; binding "
        f"`{json.dumps(plan.get('root_adapter_binding'), sort_keys=True)}`.",
        "",
        f"Helper checkpoint: `{plan.get('helper_adapter')}`; binding "
        f"`{json.dumps(plan.get('helper_adapter_binding'), sort_keys=True)}`.",
        "",
        f"Bootstrap: {method['draws']} draws, seed {method['seed']}.",
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
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with sibling.open("x") as handle:
        handle.write(markdown(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "cases", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.output, args.cases)
    write_report(report, args.report)
    print(markdown(report))


if __name__ == "__main__":
    main()
