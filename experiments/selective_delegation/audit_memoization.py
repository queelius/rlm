"""Retrospective exact-request memoization opportunity, separately within each RL update."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import analyze_helper
import probe


def duplicate_groups(records):
    grouped = {}
    for row in records:
        request = row["request"]
        if row["role"] not in ("helper", "final"):
            continue
        if request["adapter_enabled"] or request["adapter_sha256"] is not None:
            raise ValueError("audit restricted to frozen base-model downstream requests")
        # Full request is deliberately stricter than a hand-selected key: it includes
        # prompt, native input IDs, role, model, adapter flags, seed and every sampler field.
        grouped.setdefault(probe.runtime.digest(request), []).append(row)
    groups, avoidable = [], []
    fields = ("text", "output_token_ids", "available", "finish_reason", "terminated", "error")
    for digest, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda r: (r["started"], r["call_id"]))
        disagreements = [
            f for f in fields if any(r.get(f) != members[0].get(f) for r in members[1:])
        ]
        replayable = not disagreements and all(r["available"] for r in members)
        if replayable:
            avoidable.extend(members[1:])
        groups.append(
            {
                "request_digest": digest,
                "role": members[0]["role"],
                "retained_call_id": members[0]["call_id"],
                "call_ids": [r["call_id"] for r in members],
                "redundant_calls": len(members) - 1,
                "disagreement_fields": disagreements,
                "safely_replayable_observed": replayable,
                "all_redundant_recorded_cost": analyze_helper.measured(members[1:]),
            }
        )
    return groups, avoidable


def audit(output):
    output = Path(output).resolve()
    hashes = {}

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    plan = read(output / "PLAN.json")
    owners = list(output.glob("OWNER-*.json"))
    if not owners or len(owners) != len(list(output.glob("TERMINAL-*.json"))):
        raise ValueError("only a completed RL run may be audited")
    for path in owners:
        read(path)
        terminal = read(path.with_name(path.name.replace("OWNER-", "TERMINAL-")))
        if terminal.get("failure"):
            raise ValueError("source owner failed")
    batches, all_records, downstream, all_avoidable = {}, [], [], []
    for batch in sorted(output.glob("batch-*")):
        manifest = read(batch / "BATCH.json")
        if manifest["episodes"] != 64:
            raise ValueError("incomplete RL batch")
        records, ids = [], set()
        for path in sorted((batch / "calls").glob("*.json")):
            row = read(path)
            if path.stem != row["call_id"] or row["call_id"] in ids:
                raise ValueError("duplicate/mismatched receipt identity")
            if probe.runtime.digest(row["request"]) != row["request_digest"]:
                raise ValueError("native request digest mismatch")
            if row["request"]["model"] != plan["model"] or row["role"] != row["request"]["role"]:
                raise ValueError("request role/model differs from source")
            if row["request"]["input_token_ids"] != row["input_token_ids"]:
                raise ValueError("native prompt tokens differ from request")
            for field, token_ids in (
                ("prompt_tokens", "input_token_ids"),
                ("completion_tokens", "output_token_ids"),
            ):
                if token_ids in row and row["usage"].get(field) != len(row[token_ids]):
                    raise ValueError("native token usage differs")
            ids.add(row["call_id"])
            records.append(row)
        for path in sorted((batch / "starts").glob("*.json")):
            if read(path)["call_id"] not in ids:
                raise ValueError("unresolved native start; exact physical inventory unknown")
        groups, avoidable = duplicate_groups(records)
        eligible = [r for r in records if r["role"] in ("helper", "final")]
        batches[batch.name] = {
            "all_native_cost": analyze_helper.measured(records),
            "downstream_cost": analyze_helper.measured(eligible),
            "duplicate_request_groups": len(groups),
            "duplicate_physical_calls": sum(g["redundant_calls"] for g in groups),
            "groups_with_disagreements": sum(bool(g["disagreement_fields"]) for g in groups),
            "observed_identical_avoidable_cost": analyze_helper.measured(avoidable),
            "avoidable_by_role": {
                role: analyze_helper.measured([r for r in avoidable if r["role"] == role])
                for role in ("helper", "final")
            },
            "duplicate_groups": groups,
        }
        all_records.extend(records)
        downstream.extend(eligible)
        all_avoidable.extend(avoidable)
    if not batches:
        raise ValueError("no completed batches")
    for path in (Path(__file__), Path(analyze_helper.__file__), Path(probe.__file__)):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    total, eligible, avoidable = [
        analyze_helper.measured(rows) for rows in (all_records, downstream, all_avoidable)
    ]
    return {
        "source": str(output),
        "source_plan": plan,
        "input_source_receipt_sha256": hashes,
        "batches": batches,
        "all_native_cost": total,
        "downstream_cost": eligible,
        "observed_identical_avoidable_cost": avoidable,
        "duplicate_request_groups": sum(b["duplicate_request_groups"] for b in batches.values()),
        "groups_with_disagreements": sum(b["groups_with_disagreements"] for b in batches.values()),
        "fractions": {
            "downstream_calls": avoidable["calls"] / eligible["calls"],
            "all_native_calls": avoidable["calls"] / total["calls"],
            "all_native_tokens": avoidable["total_tokens"] / total["total_tokens"],
            "recorded_call_service_seconds": avoidable["known_latency_seconds"]
            / total["known_latency_seconds"],
        },
        "method": "Group the entire scientific request object by its existing native digest, "
        "separately inside each completed update. Root calls are excluded. Retain earliest "
        "physical call by start time; count later calls only when all returned outputs/statuses "
        "agree exactly and all are available. No cross-update deduplication.",
        "limitations": [
            "This is a retrospective exact-request reuse ceiling on the recorded workload, "
            "not measured speedup, general determinism, improved rewards, or algorithmic "
            "sample efficiency.",
            "Agreement is checked on text, native output IDs, availability, finish reason, "
            "termination and errors. Matching seeds alone are not evidence of equivalent requests.",
            "Avoidable latency is the sum of recorded duplicate-call service durations, not "
            "wall-clock savings. Optimizer/replay/checkpoint time is outside this call audit; "
            "GPU memory occupancy is not converted into compute savings.",
            "Fresh downstream seeds differ by update, so repeats across updates are not pooled. "
            "No root policy sampling is removed or changed, and no cache or search algorithm "
            "was implemented.",
        ],
    }


def markdown(report):
    total, eligible, avoid = [
        report[k]
        for k in ("all_native_cost", "downstream_cost", "observed_identical_avoidable_cost")
    ]
    lines = [
        "# Within-update frozen-request reuse audit",
        "",
        f"Source: `{report['source']}`",
        "",
        f"{report['duplicate_request_groups']} repeated exact-request groups; "
        f"{report['groups_with_disagreements']} groups with output/status disagreements.",
        "",
        f"Ideal within-update reuse could avoid {avoid['calls']}/{eligible['calls']} downstream "
        f"calls ({report['fractions']['downstream_calls']:.1%}), or "
        f"{avoid['calls']}/{total['calls']} "
        "of all native calls, on this recorded workload.",
        "",
        f"Those duplicate receipts consumed {avoid['prompt_tokens']:,} input and "
        f"{avoid['completion_tokens']:,} output tokens "
        f"({report['fractions']['all_native_tokens']:.1%} of all native tokens), and "
        f"{avoid['known_latency_seconds']:.1f} recorded service seconds. This is not a "
        "measured wall-clock speedup.",
        "",
        "| Update | Downstream calls | Duplicate groups | Identical avoidable calls | "
        "Avoidable tokens | Recorded service seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["batches"].items():
        cost = row["observed_identical_avoidable_cost"]
        lines.append(
            f"| {name} | {row['downstream_cost']['calls']} | "
            f"{row['duplicate_request_groups']} | {cost['calls']} | "
            f"{cost['total_tokens']} | {cost['known_latency_seconds']:.1f} |"
        )
    lines += ["", report["method"], "", *["- " + text for text in report["limitations"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sibling = args.report.with_suffix(".md")
    if args.report.exists() or sibling.exists() or args.report == sibling:
        parser.error("choose unused distinct JSON/Markdown report paths")
    report = audit(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with sibling.open("x") as handle:
        handle.write(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
