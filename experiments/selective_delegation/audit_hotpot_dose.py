"""Paired official-Hotpot comparison of fixed SFT16 and SFT48 root checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import probe
import score_hotpot

SEED = 2026092116


def outcome(final, gold, status):
    value = {
        "em": 0.0,
        "f1": 0.0,
        "valid": False,
        "unobserved": False,
        "answer": None,
        "status": status,
    }
    if final and final["available"]:
        try:
            answer = score_hotpot.parse_answer(final["text"])
        except (ValueError, TypeError):
            value["status"] = "invalid_final"
        else:
            em, f1 = score_hotpot.official_score(answer, gold)
            value.update(em=em, f1=f1, valid=True, answer=answer, status="scored")
    elif final:
        value.update(unobserved=True, status="final_unavailable")
    else:
        value["unobserved"] = (
            "failure" in status or "unavailable" in status or status.startswith("missing")
        )
    return value


def audit(early, full, cases_path, early_report, full_report, *, draws=20000):
    hashes, cache = {}, {}

    def track(path, expected=None):
        path = str(Path(path).resolve())
        if path not in cache:
            cache[path] = Path(path).read_bytes()
            hashes[path] = hashlib.sha256(cache[path]).hexdigest()
        if expected is not None and hashes[path] != expected:
            raise ValueError("source hash mismatch: " + path)
        return cache[path]

    def read(path):
        return json.loads(track(path))

    plans = [read(Path(source) / "PLAN.json") for source in (early, full)]
    fields = (
        "cases_sha256",
        "case_ids",
        "repeats",
        "seed",
        "caps",
        "temperature",
        "top_p",
        "top_k",
        "execution",
        "split",
        "source_sha256",
        "model",
        "model_manifest_sha256",
    )
    for field in fields:
        if field not in plans[0] or plans[0][field] != plans[1].get(field):
            raise ValueError("paired source contract differs: " + field)
    cases = score_hotpot._load_cases(Path(cases_path))
    track(cases_path, plans[0]["cases_sha256"])
    parents, repeats = plans[0]["case_ids"], plans[0]["repeats"]
    if len(parents) != 32 or repeats != 2:
        raise ValueError("expected the fixed 32-parent two-repeat Hotpot dose panel")
    values, groups, contracts = {}, {}, {}
    for name, source, plan, expected_step, original_path in zip(
        ("sft16", "sft48"),
        (Path(early), Path(full)),
        plans,
        (16, 48),
        (early_report, full_report),
        strict=True,
    ):
        original = read(original_path)
        if original["sources"]["cases_sha256"] != plan["cases_sha256"]:
            raise ValueError("official source report cases differ")
        owners = list(source.glob("OWNER-*.json"))
        if not owners or len(owners) != len(list(source.glob("TERMINAL-*.json"))):
            raise ValueError("incomplete source owner")
        for owner_path in owners:
            owner = read(owner_path)
            terminal = read(owner_path.with_name(owner_path.name.replace("OWNER-", "TERMINAL-")))
            if terminal["failure"] or terminal["stopped"]:
                raise ValueError("source owner failed/stopped")
            track(owner["source"], plan["source_sha256"])
        state_path = Path(plan["adapter"]) / "STATE.json"
        state = json.loads(track(state_path, plan["adapter_files_sha256"]["STATE.json"]))
        if state["step"] != expected_step:
            raise ValueError("wrong fixed checkpoint step")
        calls = {}
        for path in sorted((source / "calls").glob("*.json")):
            call = read(path)
            if call["condition"] != "sft":
                continue
            if call["call_id"] != path.stem or call["call_id"] in calls:
                raise ValueError("duplicate/mismatched call")
            if probe.runtime.digest(call["request"]) != call["request_digest"]:
                raise ValueError("native request digest differs")
            calls[call["call_id"]] = call
        episodes = {}
        for path in sorted((source / "episodes").glob("*.json")):
            row = read(path)
            if row["condition"] == "sft":
                key = row["case_id"], row["repeat"]
                if key in episodes:
                    raise ValueError("duplicate episode")
                episodes[key] = row
        inventory = {(p, r) for p in parents for r in range(repeats)}
        if set(episodes) != inventory:
            raise ValueError("planned dose inventory incomplete or mismatched")
        rows, linked = [], set()
        for p, repeat in sorted(inventory):
            episode = episodes[p, repeat]
            if linked.intersection(episode["call_ids"]):
                raise ValueError("physical call reused")
            linked.update(episode["call_ids"])
            records = [calls[cid] for cid in episode["call_ids"]]
            for call in records:
                suffix = call["call_id"].removeprefix(episode["episode_id"])
                request = call["request"]
                contract = {k: request[k] for k in ("seed", "sampling", "model", "role")}
                # Root adapter differs intentionally. All downstream calls remain base.
                if call["role"] != "root" and request["adapter_enabled"]:
                    raise ValueError("downstream model not frozen base")
                if call["role"] in ("root", "final"):
                    key = p, repeat, suffix
                    if key in contracts and contracts[key] != contract:
                        raise ValueError("paired native sampling differs")
                    contracts[key] = contract
            finals = [c for c in records if c["role"] == "final"]
            if len(finals) > 1:
                raise ValueError("multiple final calls")
            result = outcome(finals[0] if finals else None, cases[p]["answer"], episode["status"])
            result.update(case_id=p, repeat=repeat)
            values[name, p, repeat] = result
            rows.append(result)
        for path in sorted((source / "starts").glob("*.json")):
            start = read(path)
            if start["condition"] == "sft" and start["call_id"] not in calls:
                raise ValueError("unresolved SFT call")
        group = {
            "source": str(source.resolve()),
            "checkpoint": plan["adapter"],
            "adapter_binding": plan["adapter_files_sha256"],
            "planned": len(rows),
            "correct": sum(r["em"] for r in rows),
            "em": mean(r["em"] for r in rows),
            "f1": mean(r["f1"] for r in rows),
            "valid_finals": sum(r["valid"] for r in rows),
            "unobserved": sum(r["unobserved"] for r in rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "native_cost": analyze_helper.measured(list(calls.values())),
            "unlinked_call_ids": sorted(set(calls) - linked),
        }
        for metric in ("em", "f1"):
            if abs(group[metric] - original["conditions"]["sft"][metric]) > 1e-12:
                raise ValueError("native official regrade differs from prior official report")
        groups[name] = group
    contrast = {}
    for metric in ("em", "f1"):
        differences = {
            p: mean(
                values["sft48", p, r][metric] - values["sft16", p, r][metric]
                for r in range(repeats)
            )
            for p in parents
        }
        contrast[metric] = analyze_helper.clustered_interval(
            differences, [[p] for p in parents], draws, SEED
        )
    for label, sign in (("wins", 1), ("losses", -1)):
        rows = []
        for p in parents:
            for repeat in range(repeats):
                a, b = values["sft16", p, repeat], values["sft48", p, repeat]
                if b["em"] - a["em"] != sign:
                    continue
                category = (
                    "unobserved"
                    if a["unobserved"] or b["unobserved"]
                    else ("both_valid" if a["valid"] and b["valid"] else "protocol_involved")
                )
                rows.append(
                    {
                        "case_id": p,
                        "repeat": repeat,
                        "category": category,
                        "early_status": a["status"],
                        "full_status": b["status"],
                        "early_answer": a["answer"],
                        "full_answer": b["answer"],
                    }
                )
        contrast[label] = {
            "episodes": len(rows),
            "parents": sorted({r["case_id"] for r in rows}),
            "categories": dict(Counter(r["category"] for r in rows)),
            "rows": rows,
        }
    for path in (
        Path(__file__),
        Path(score_hotpot.__file__),
        score_hotpot.EVALUATOR,
        Path(analyze_helper.__file__),
        Path(probe.__file__),
    ):
        track(path)
    return {
        "metric": "Official HotpotQA answer EM/F1; no MuSiQue receipt metrics used",
        "groups": groups,
        "sft48_minus_sft16": contrast,
        "source_report_paths": [str(Path(p).resolve()) for p in (early_report, full_report)],
        "input_source_receipt_sha256": hashes,
        "method": {
            "parents": 32,
            "repeats": 2,
            "draws": draws,
            "seed": SEED,
            "bootstrap": "Paired parent means; percentile95%; no atomic component metadata",
            "matched_plan_fields": list(fields),
        },
        "limitations": [
            "This compares two fixed checkpoints on an already inspected 32-question explorer "
            "sample, not a general training-dose curve or the full Hotpot development set.",
            "More updates change planner language/structure and downstream execution, not only "
            "protocol reliability. Both-valid changes are descriptive, not a complete mechanism.",
            "The early checkpoint does not recover the base model's higher score. These results "
            "do not support a simple claim that additional SFT updates caused the transfer loss; "
            "they also do not establish absence of overfitting elsewhere.",
            "No new primary checkpoint is selected. The fixed SFT48 warm start remains unchanged. "
            "Intervals are exploratory, paired by parent, and unadjusted.",
        ],
    }


def markdown(report):
    lines = [
        "# Hotpot fixed SFT-dose audit",
        "",
        "Same frozen 32 parents × two repeats, source009 and sampling contract; official "
        "Hotpot answer metrics regraded from native finals.",
        "",
        "| Checkpoint | Correct / planned | F1 | Valid finals | Status counts |",
        "|---|---:|---:|---:|---|",
    ]
    for name, row in report["groups"].items():
        lines.append(
            f"| {name} | {row['correct']:g}/{row['planned']} | {row['f1']:.2%} | "
            f"{row['valid_finals']} | {row['statuses']} |"
        )
    lines += ["", "Full-dose SFT48 minus early SFT16:", ""]
    for metric in ("em", "f1"):
        row = report["sft48_minus_sft16"][metric]
        lines.append(
            f"- {metric.upper()}: {100 * row['estimate']:+.2f} points; paired-parent "
            f"95% interval [{100 * row['ci95'][0]:+.2f}, {100 * row['ci95'][1]:+.2f}]."
        )
    for label in ("wins", "losses"):
        row = report["sft48_minus_sft16"][label]
        lines.append(
            f"- {label}: {row['episodes']} episodes / {len(row['parents'])} parents; "
            f"{row['categories']}."
        )
    lines += [
        "",
        f"Bootstrap: {report['method']['draws']} draws, seed {SEED}.",
        "",
        *["- " + limitation for limitation in report["limitations"]],
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("early", "full", "cases", "early-report", "full-report", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    sibling = args.report.with_suffix(".md")
    if args.report.exists() or sibling.exists() or args.report == sibling:
        parser.error("choose unused distinct JSON/Markdown outputs")
    report = audit(args.early, args.full, args.cases, args.early_report, args.full_report)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with sibling.open("x") as stream:
        stream.write(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
