"""Matched MuSiQue end-to-end policies with explicitly different architectures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import eval_planner
import probe

SEED = 2026092115


def validate_panel(plans):
    first = plans[0]
    fields = (
        "cases_sha256",
        "case_ids",
        "repeats",
        "seed",
        "temperature",
        "top_p",
        "top_k",
        "model",
        "model_manifest_sha256",
        "split",
    )
    for plan in plans:
        for field in fields:
            if field not in plan or plan[field] != first[field]:
                raise ValueError("matched policy panel differs: " + field)
        if plan["caps"]["final"] != first["caps"]["final"]:
            raise ValueError("matched final cap differs")
        if plan.get("mode", "planner") not in ("planner", "direct"):
            raise ValueError("only explicit planner/direct MuSiQue policies supported")


def compare(planner_report, direct_report, *, draws=20000, seed=SEED):
    hashes, cache = {}, {}

    def track(path, expected=None):
        path = str(Path(path).resolve())
        if path not in cache:
            cache[path] = Path(path).read_bytes()
            hashes[path] = hashlib.sha256(cache[path]).hexdigest()
        if expected is not None and hashes[path] != expected:
            raise ValueError("input hash changed: " + path)
        return cache[path]

    def read(path):
        return json.loads(track(path))

    original = [read(planner_report), read(direct_report)]
    if original[0]["cases"] != original[1]["cases"]:
        raise ValueError("matched cases path differs")
    sources = []
    for report in original:
        for reported in report["source_plans"]:
            plan = read(Path(reported["output"]) / "PLAN.json")
            if reported != {"output": reported["output"], **plan}:
                raise ValueError("source PLAN differs from original report")
            sources.append((Path(reported["output"]), plan, report))
    validate_panel([p for _, p, _ in sources])
    first = sources[0][1]
    cases_path = Path(original[0]["cases"])
    cases = [
        json.loads(line)
        for line in track(cases_path, first["cases_sha256"]).splitlines()
        if line.strip()
    ]
    if any(
        c.get("dataset") == "hotpotqa" or "supporting_facts" in c.get("metadata", {}) for c in cases
    ):
        raise ValueError("this comparison uses MuSiQue metrics only")
    by_id = {c["id"]: c for c in cases}
    parents, repeats = first["case_ids"], first["repeats"]
    if not parents or repeats < 1 or len(set(parents)) != len(parents):
        raise ValueError("invalid planned panel")
    groups, values, observed_sampling = {}, {}, {}
    for output, plan, original_report in sources:
        owners = list(output.glob("OWNER-*.json"))
        if not owners or len(owners) != len(list(output.glob("TERMINAL-*.json"))):
            raise ValueError("source owner still active")
        for path in owners:
            owner = read(path)
            terminal = read(path.with_name(path.name.replace("OWNER-", "TERMINAL-")))
            if terminal["failure"] or terminal["stopped"]:
                raise ValueError("source owner did not finish")
            track(owner["source"], plan["source_sha256"])
        calls = {}
        for path in sorted((output / "calls").glob("*.json")):
            row = read(path)
            if row["call_id"] != path.stem or row["call_id"] in calls:
                raise ValueError("duplicate/mismatched native receipt")
            if probe.runtime.digest(row["request"]) != row["request_digest"]:
                raise ValueError("native request digest differs")
            calls[row["call_id"]] = row
        episodes = {}
        for path in sorted((output / "episodes").glob("*.json")):
            row = read(path)
            key = row["case_id"], row["repeat"], row["condition"]
            if key in episodes:
                raise ValueError("duplicate episode")
            episodes[key] = row
        inventory = {(p, r, c) for p in parents for r in range(repeats) for c in plan["conditions"]}
        if set(episodes) != inventory:
            raise ValueError("incomplete or unexpected planned episode inventory")
        starts = [read(p) for p in sorted((output / "starts").glob("*.json"))]
        if any(s["call_id"] not in calls for s in starts):
            raise ValueError("unresolved native attempts: complete policy comparison required")
        for condition in plan["conditions"]:
            mode = plan.get("mode", "planner")
            policy = "direct_base" if mode == "direct" else "planner_" + condition
            if policy in groups:
                raise ValueError("duplicate policy identity")
            rows, linked = [], set()
            for p in parents:
                for repeat in range(repeats):
                    episode = episodes[p, repeat, condition]
                    expected_seed = (
                        plan["seed"] + int(probe.runtime.digest(p)[:6], 16) + repeat * 100
                    )
                    if episode["seed"] != expected_seed:
                        raise ValueError("matched episode seed differs")
                    records = [calls[cid] for cid in episode["call_ids"]]
                    if linked.intersection(episode["call_ids"]):
                        raise ValueError("reused physical call")
                    linked.update(episode["call_ids"])
                    finals = [c for c in records if c["role"] == "final"]
                    if len(finals) > 1:
                        raise ValueError("multiple final receipts")
                    final = finals[0] if finals else None
                    status = episode["status"]
                    if final:
                        request = final["request"]
                        if request["seed"] != expected_seed + 2 or request["adapter_enabled"]:
                            raise ValueError("matched base-final model/seed differs")
                        sampling = request["sampling"]
                        if (
                            sampling["max_new_tokens"] != plan["caps"]["final"]
                            or sampling["temperature"] != plan["temperature"]
                        ):
                            raise ValueError("matched final sampling differs from PLAN")
                        contract = {
                            "seed": request["seed"],
                            "model": request["model"],
                            "sampling": sampling,
                        }
                        if (p, repeat) in observed_sampling and observed_sampling[
                            p, repeat
                        ] != contract:
                            raise ValueError(
                                "matched native final sampling differs between policies"
                            )
                        observed_sampling[p, repeat] = contract
                        if mode == "direct" and request["prompt"] != eval_planner.direct_prompt(
                            by_id[p]
                        ):
                            raise ValueError("direct public prompt reconstruction differs")
                    grade = probe.grade(
                        final["text"] if final and final["available"] else "", by_id[p]
                    )
                    if final and final["available"]:
                        status = "scored" if grade["valid"] else "invalid_final"
                    elif final:
                        status = "final_unavailable"
                    unobserved = (
                        "failure" in status
                        or "unavailable" in status
                        or status.startswith("missing")
                    )
                    value = {
                        "case_id": p,
                        "repeat": repeat,
                        "status": status,
                        "valid": grade["valid"],
                        "unobserved": unobserved,
                        "em": float(grade["correct"]),
                        "f1": grade["f1"],
                    }
                    values[p, repeat, policy] = value
                    rows.append(value)
            native = [c for c in calls.values() if c["condition"] == condition]
            cost = analyze_helper.measured(native)
            group = {
                "source": str(output),
                "condition": condition,
                "mode": mode,
                "architecture": plan["architecture"],
                "source_sha256": plan["source_sha256"],
                "adapter": plan.get("adapter")
                if mode == "planner" and condition != "base"
                else None,
                "adapter_binding": plan.get("adapter_files_sha256")
                if mode == "planner" and condition != "base"
                else None,
                "planned": len(rows),
                "correct": sum(r["em"] for r in rows),
                "em": mean(r["em"] for r in rows),
                "f1": mean(r["f1"] for r in rows),
                "valid_finals": sum(r["valid"] for r in rows),
                "status_counts": dict(Counter(r["status"] for r in rows)),
                "native_cost": cost,
                "tokens_per_attempt": cost["total_tokens"] / len(rows),
                "calls_per_attempt": cost["calls"] / len(rows),
                "role_counts": dict(Counter(c["role"] for c in native)),
                "unlinked_call_ids": sorted(set(c["call_id"] for c in native) - linked),
            }
            for metric in ("correct", "em", "f1"):
                if abs(group[metric] - original_report["groups"][condition][metric]) > 1e-12:
                    raise ValueError("native regrading disagrees with source report")
            groups[policy] = group
    if set(groups) != {"direct_base", "planner_base", "planner_sft", "planner_rl"}:
        raise ValueError("expected direct/base/SFT/RL policies")
    clusters = analyze_helper.component_clusters([by_id[p] for p in parents])
    contrasts = {}
    for left, right in (
        ("planner_sft", "direct_base"),
        ("planner_sft", "planner_rl"),
        ("planner_base", "direct_base"),
        ("planner_rl", "direct_base"),
    ):
        contrast = {"left": left, "right": right}
        for metric in ("em", "f1"):
            delta = {
                p: mean(
                    values[p, r, right][metric] - values[p, r, left][metric] for r in range(repeats)
                )
                for p in parents
            }
            contrast[metric] = analyze_helper.clustered_interval(delta, clusters, draws, seed)
        for label, sign in (("wins", 1), ("losses", -1)):
            changes = []
            for p in parents:
                for repeat in range(repeats):
                    a, b = values[p, repeat, left], values[p, repeat, right]
                    if b["em"] - a["em"] != sign:
                        continue
                    category = (
                        "unobserved_involved"
                        if a["unobserved"] or b["unobserved"]
                        else ("both_valid" if a["valid"] and b["valid"] else "protocol_involved")
                    )
                    changes.append(
                        {
                            "case_id": p,
                            "repeat": repeat,
                            "category": category,
                            "left_status": a["status"],
                            "right_status": b["status"],
                        }
                    )
            contrast[label] = {
                "episodes": len(changes),
                "parents": sorted({r["case_id"] for r in changes}),
                "rows": changes,
                "categories": dict(Counter(r["category"] for r in changes)),
            }
        contrast["right_over_left_token_ratio"] = (
            groups[right]["native_cost"]["total_tokens"]
            / groups[left]["native_cost"]["total_tokens"]
        )
        contrasts[right + "_minus_" + left] = contrast
    for path in (
        Path(__file__),
        Path(analyze_helper.__file__),
        Path(eval_planner.__file__),
        Path(probe.__file__),
        probe.MUSIQUE / "metrics/answer.py",
    ):
        track(path)
    return {
        "source_reports": [str(Path(p).resolve()) for p in (planner_report, direct_report)],
        "cases": str(cases_path),
        "input_native_source_sha256": hashes,
        "source_plans": [{"output": str(o), **p} for o, p, _ in sources],
        "method": {
            "parents": len(parents),
            "repeats": repeats,
            "draws": draws,
            "seed": seed,
            "component_clusters": clusters,
            "verified_native_final_contracts": len(observed_sampling),
            "matched_fields": "case hash/order, repeats, episode/final seeds, model identity, "
            "final token cap, temperature/top-p/top-k and complete native sampling object",
            "metric": "official MuSiQue alias-max EM/F1 via probe.grade",
        },
        "groups": groups,
        "contrasts": contrasts,
        "primary_contrasts": ["direct_base_minus_planner_sft", "planner_rl_minus_planner_sft"],
        "cautions": [
            "These are distinct end-to-end policies, not matched architecture treatments. "
            "Direct finals see question/documents; planner finals additionally see generated "
            "plans/helper traces, and planners incur upstream protocol failures and extra calls.",
            "Matched sampling seeds do not mean matched random trajectories for different "
            "prompts. Root/helper budgets and input token counts differ; costs are actual, "
            "not compute-matched or a causal estimate of decomposition's effect.",
            "All planned episodes are retained. Protocol gains are separated from both-valid "
            "changes; neither is automatically a semantic reasoning improvement.",
            "Intervals resample whole connected atomic-component clusters with parent "
            "weighting. This is a secondary, exploratory, unadjusted panel comparison.",
            "The model/adapter binding is inherited from sealed source plans; source code "
            "and consumed receipts are hashed, without rehashing multi-GB weights.",
        ],
    }


def markdown(report):
    token_ratio = report["contrasts"]["direct_base_minus_planner_sft"][
        "right_over_left_token_ratio"
    ]
    lines = [
        "# MuSiQue: end-to-end policy comparison",
        "",
        "Different prompts and execution architectures are compared explicitly; this is "
        "not a causal decomposition ablation.",
        "",
        "| Policy | Correct / planned | EM | F1 | Valid finals | Calls | Tokens / attempt |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("direct_base", "planner_base", "planner_sft", "planner_rl"):
        row = report["groups"][policy]
        lines.append(
            f"| {policy} | {row['correct']:g}/{row['planned']} | {row['em']:.2%} | "
            f"{row['f1']:.2%} | {row['valid_finals']} | {row['native_cost']['calls']} | "
            f"{row['tokens_per_attempt']:.1f} |"
        )
    lines += [
        "",
        "Paired differences in percentage points, with component-cluster 95% intervals:",
        "",
    ]
    for name, row in report["contrasts"].items():
        metrics = [
            f"{m.upper()} {row[m]['estimate'] * 100:+.2f} "
            f"[{row[m]['ci95'][0] * 100:+.2f}, {row[m]['ci95'][1] * 100:+.2f}]"
            for m in ("em", "f1")
        ]
        lines.append(f"- {name}: " + "; ".join(metrics) + ".")
        for label in ("wins", "losses"):
            change = row[label]
            lines.append(
                f"  - {label}: {change['episodes']} episodes / {len(change['parents'])} "
                f"parents; {change['categories']}."
            )
    lines += [
        "",
        "Primary contrasts are direct minus SFT and RL minus SFT. Direct uses "
        f"{token_ratio:.1%} "
        "of the SFT policy's actual tokens on this panel.",
        "",
        f"Panel: {report['method']['parents']} parents × {report['method']['repeats']} repeats; "
        f"{len(report['method']['component_clusters'])} connected components; "
        f"{report['method']['draws']} bootstrap draws, seed {report['method']['seed']}.",
        "",
        "Original reports: " + "; ".join(f"`{p}`" for p in report["source_reports"]),
        "",
        *["- " + caution for caution in report["cautions"]],
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("planner-report", "direct-report", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    sibling = args.report.with_suffix(".md")
    if args.report.exists() or sibling.exists() or args.report == sibling:
        parser.error("choose unused distinct JSON/Markdown paths")
    report = compare(args.planner_report, args.direct_report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with sibling.open("x") as handle:
        handle.write(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
