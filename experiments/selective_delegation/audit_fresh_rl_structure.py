"""Exploratory fresh003 structure audit using only the completed original policy readout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_fresh_strata
import analyze_helper
import probe

SEED = 2026092123
POLICIES = ("planner_sft", "planner_rl", "direct_base")


def audit(root, *, draws=10000):
    hashes = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        if expected is not None and hashes[str(path)] != expected:
            raise ValueError("consumed source differs from completed audit: " + str(path))
        return json.loads(raw)

    original = read(root / "analysis-fresh-contract-policy-001.json")
    audited = original["input_native_source_sha256"]
    exposure = read(root / "analysis-document-exposure-001.json")
    cases_path = root / "fresh-dev-inputs-003/cases.jsonl"
    raw = cases_path.read_bytes()
    cases_hash = hashlib.sha256(raw).hexdigest()
    if (
        cases_hash != analyze_fresh_strata.CASES_SHA
        or cases_hash != (exposure["sources_sha256"]["fresh-dev-inputs-003/cases.jsonl"])
    ):
        raise ValueError("fixed fresh003/exposure identity differs")
    hashes[str(cases_path.resolve())] = cases_hash
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    partitions = analyze_fresh_strata.strata(cases, exposure["panels"]["freshdev64"])
    parents, by_id = partitions["all64"], {c["id"]: c for c in cases}
    exposed = set(partitions["exact_train_document_overlap"])
    clusters = analyze_helper.component_clusters(cases)
    records = {}
    for policy in POLICIES:
        source_group = original["groups"][policy]
        output = Path(source_group["source"])
        plan = read(output / "PLAN.json", audited[str(output / "PLAN.json")])
        if plan["case_ids"] != parents or plan["repeats"] != 2:
            raise ValueError("not full64 x2 readout")
        owners = list(output.glob("OWNER-*.json"))
        if not owners or {p.stem[6:] for p in owners} != {
            p.stem[9:] for p in output.glob("TERMINAL-*.json")
        }:
            raise ValueError("unfinished source")
        for owner in owners:
            terminal = read(owner.with_name(owner.name.replace("OWNER-", "TERMINAL-")))
            if terminal["failure"] or terminal["stopped"]:
                raise ValueError("failed or stopped source")
        condition, mode = source_group["condition"], source_group["mode"]
        for parent in parents:
            for repeat in range(2):
                identity = f"{parent}-r{repeat}-{condition}-" + (
                    "direct" if mode == "direct" else "isolated"
                )
                path = output / "episodes" / (identity + ".json")
                episode = read(path, audited[str(path)])
                finals = [cid for cid in episode["call_ids"] if cid.endswith("-final")]
                if len(finals) > 1:
                    raise ValueError("multiple finals")
                final = None
                if finals:
                    path = output / "calls" / (finals[0] + ".json")
                    final = read(path, audited[str(path)])
                    if not final["available"]:
                        raise ValueError("unexpected unobserved final")
                elif episode["status"] not in (
                    "invalid_plan",
                    "invalid_dependency",
                    "invalid_helper",
                ):
                    raise ValueError("unobserved episode cannot be called an incorrect answer")
                grade = probe.grade(final["text"] if final else "", by_id[parent])
                trace = episode.get("helper_trace", [])
                generated = episode.get("plan", {}).get("subquestions", [])
                last = trace[-1]["answer"] if trace else None
                last_grade = (
                    probe.grade(json.dumps({"answer": last}), by_id[parent]) if last else None
                )
                records[parent, repeat, policy] = dict(
                    case_id=parent,
                    repeat=repeat,
                    hops=by_id[parent]["metadata"]["hops"],
                    exact_train_document_overlap=parent in exposed,
                    em=float(grade["correct"]),
                    f1=grade["f1"],
                    valid=grade["valid"],
                    status=episode["status"],
                    plan=generated,
                    planned_steps=len(generated),
                    helper_calls=len(trace),
                    final_answer=grade["parsed"],
                    last_helper_answer=last,
                    last_helper_em=float(last_grade["correct"]) if last_grade else None,
                    last_helper_f1=last_grade["f1"] if last_grade else None,
                )
        current = [records[p, r, policy] for p in parents for r in range(2)]
        for metric in ("em", "f1"):
            if abs(mean(row[metric] for row in current) - source_group[metric]) > 1e-12:
                raise ValueError("regrade differs from official completed policy audit")
    pairs = []
    for p in parents:
        for repeat in range(2):
            a, b = [records[p, repeat, policy] for policy in POLICIES[:2]]
            step_delta = b["planned_steps"] - a["planned_steps"]
            relation = "more" if step_delta > 0 else "fewer" if step_delta < 0 else "same_length"
            pairs.append(
                dict(
                    case_id=p,
                    repeat=repeat,
                    hops=a["hops"],
                    exact_train_document_overlap=p in exposed,
                    relation=relation,
                    step_delta=step_delta,
                    exact_same_plan=a["plan"] == b["plan"],
                    em_delta=b["em"] - a["em"],
                    f1_delta=b["f1"] - a["f1"],
                    sft_steps=a["planned_steps"],
                    rl_steps=b["planned_steps"],
                )
            )
    for hops in (2, 3):
        for overlap in (True, False):
            partitions[f"hop{hops}_overlap{int(overlap)}"] = [
                p
                for p in parents
                if by_id[p]["metadata"]["hops"] == hops and (p in exposed) == overlap
            ]
    summaries = {}
    for name, subset in partitions.items():
        groups = {}
        for policy in POLICIES:
            rows = [records[p, r, policy] for p in subset for r in range(2)]
            groups[policy] = dict(
                planned=len(rows),
                correct=sum(r["em"] for r in rows),
                em=mean(r["em"] for r in rows),
                f1=mean(r["f1"] for r in rows),
                planned_steps=sum(r["planned_steps"] for r in rows),
                mean_planned_steps=mean(r["planned_steps"] for r in rows),
                step_histogram=dict(Counter(r["planned_steps"] for r in rows)),
                actual_helper_calls=sum(r["helper_calls"] for r in rows),
                final_correct_last_helper_not_exact=sum(
                    r["em"] == 1 and r["last_helper_em"] == 0 for r in rows
                ),
                final_wrong_last_helper_exact=sum(
                    r["em"] == 0 and r["last_helper_em"] == 1 for r in rows
                ),
            )
        paired = [r for r in pairs if r["case_id"] in subset]
        changes = {}
        for relation in ("more", "same_length", "fewer"):
            rs = [r for r in paired if r["relation"] == relation]
            changes[relation] = dict(
                attempts=len(rs),
                wins=sum(r["em_delta"] == 1 for r in rs),
                losses=sum(r["em_delta"] == -1 for r in rs),
                net_steps=sum(r["step_delta"] for r in rs),
            )
        restricted = analyze_fresh_strata.restrict_clusters(clusters, subset)
        intervals = {}
        if not name.startswith("hop"):
            for metric in ("em", "f1", "planned_steps"):
                delta = {
                    p: mean(
                        records[p, r, "planner_rl"][metric] - records[p, r, "planner_sft"][metric]
                        for r in range(2)
                    )
                    for p in subset
                }
                intervals[metric] = analyze_helper.clustered_interval(
                    delta, restricted, draws, SEED
                )
        summaries[name] = dict(
            parents=len(subset),
            component_clusters=restricted,
            groups=groups,
            paired_length_changes=changes,
            exact_same_plan_attempts=sum(r["exact_same_plan"] for r in paired),
            different_same_length_attempts=sum(
                r["relation"] == "same_length" and not r["exact_same_plan"] for r in paired
            ),
            rl_minus_sft=intervals,
        )
    generated_length_bins = {}
    for policy in POLICIES[:2]:
        groups = {}
        for length in sorted({r["planned_steps"] for k, r in records.items() if k[2] == policy}):
            rows = [
                r for k, r in records.items() if k[2] == policy and r["planned_steps"] == length
            ]
            groups[length] = dict(
                attempts=len(rows),
                correct=sum(r["em"] for r in rows),
                em=mean(r["em"] for r in rows),
                f1=mean(r["f1"] for r in rows),
            )
        generated_length_bins[policy] = groups
    for path in (
        Path(__file__),
        Path(analyze_helper.__file__),
        Path(analyze_fresh_strata.__file__),
        Path(probe.__file__),
        probe.MUSIQUE / "metrics/answer.py",
    ):
        hashes[str(path.resolve())] = probe.campaign.sha(path)
    return dict(
        method=dict(
            primary="all64 balanced32/32; not natural whole-development mixture",
            metric="official MuSiQue alias-max EM/F1",
            draws=draws,
            seed=SEED,
            bootstrap="Paired parent means, full-panel components restricted within stratum",
        ),
        summaries=summaries,
        generated_length_bins=generated_length_bins,
        paired_attempts=pairs,
        per_attempt=[dict(policy=k[2], **r) for k, r in records.items()],
        source_sha256=hashes,
        caveats=[
            "Exposure strata are associated with hop/difficulty, not a causal exposure effect.",
            "Generated length is policy-dependent/post-treatment: its bins are descriptive.",
            "Different same-length strings are not proof of semantic replanning.",
            "Last-helper/final disagreement is mechanical; last steps are not aligned to "
            "annotations or necessarily answer the original question; not automatically a rescue.",
            "Exploratory subgroup intervals are unadjusted, not a tested interaction.",
        ],
    )


def markdown(report):
    lines = [
        "# Fresh RL structure: complete-panel exploratory audit",
        "",
        report["method"]["primary"],
        "",
    ]
    for name, result in report["summaries"].items():
        lines += [
            f"## {name}: {result['parents']} parents",
            "",
            "| Policy | Correct/attempts | F1 | Planned steps | Mean steps | "
            "Step histogram | Actual helpers |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
        for policy, g in result["groups"].items():
            lines.append(
                f"| {policy} | {g['correct']:g}/{g['planned']} | {g['f1']:.4f} | "
                f"{g['planned_steps']} | {g['mean_planned_steps']:.4f} | "
                f"{g['step_histogram']} | {g['actual_helper_calls']} |"
            )
        lines += [
            "",
            "Paired count changes: " + json.dumps(result["paired_length_changes"], sort_keys=True),
            f"Exact same plans: {result['exact_same_plan_attempts']}; different same-length "
            f"plans: {result['different_same_length_attempts']}.",
            "",
        ]
        for metric, ci in result["rl_minus_sft"].items():
            lines.append(f"- RL−SFT {metric}: {ci['estimate']:+.5f}, CI95 {ci['ci95']}.")
        lines += [""]
    lines += ["## Limits", "", *["- " + note for note in report["caveats"]], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    md = args.report.with_suffix(".md")
    if args.report == md or args.report.exists() or md.exists():
        parser.error("choose new immutable JSON and Markdown paths")
    report = audit(args.root.resolve())
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with md.open("x") as stream:
        stream.write(markdown(report))
    print(markdown(report))
