"""Strict matched RL checkpoint16 versus24 readout; never relabel source receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import analyze_planner
import probe

SEED = 2026092175
CASES_SHA256 = "6251b27db8acc4fcc195f614b661acc49e5dcdb60c9c61f01826d3c4caf86153"


def validate_contract(first, second):
    for plan in (first, second):
        if plan["conditions"] != ["rl"] or plan["execution"] != "isolated":
            raise ValueError("only single-condition isolated RL evaluation supported")
        if plan["helper_contract"]["mode"] != "trained_helper":
            raise ValueError("fixed trained helper required")
    required = (
        "cases_sha256",
        "case_ids",
        "repeats",
        "seed",
        "execution",
        "source_sha256",
        "model",
        "model_manifest_sha256",
        "split",
        "temperature",
        "top_p",
        "top_k",
        "caps",
        "helper_contract",
    )
    for field in required:
        if field not in first or field not in second or first[field] != second[field]:
            raise ValueError("matched RL dose contract differs: " + field)
    for field in ("mode", "max_context", "helper_budget_policy", "architecture", "environment"):
        if first.get(field) != second.get(field):
            raise ValueError("matched RL dose contract differs: " + field)


def score_episode(episode, records, case):
    finals = [r for r in records if r["role"] == "final"]
    if len(finals) > 1:
        raise ValueError("multiple final receipts")
    final = finals[0] if finals else None
    status = episode["status"] if episode else "missing_episode"
    observed = bool(episode) and bool(records) and all(r["available"] for r in records)
    if final and final["available"]:
        grade = probe.grade(final["text"], case)
        status = "scored" if grade["valid"] else "invalid_final"
        observed = True
    else:
        grade = {"correct": False, "f1": 0.0, "valid": False}
        if status not in {"invalid_plan", "invalid_dependency", "invalid_helper"}:
            observed = False
        if status == "scored":
            status = "missing_final"
    return {
        "em": float(grade["correct"]),
        "f1": grade["f1"],
        "valid": grade["valid"],
        "observed": observed,
        "status": status,
    }


def paired(old, new, cases, repeats, *, draws=20000, seed=SEED):
    parents = [c["id"] for c in cases]
    clusters = analyze_helper.component_clusters(cases)
    result = {"left": "rl16", "right": "rl24", "component_clusters": clusters}
    for metric in ("em", "f1"):
        delta = {
            p: mean(new[p, r][metric] - old[p, r][metric] for r in range(repeats)) for p in parents
        }
        result[metric] = analyze_helper.clustered_interval(delta, clusters, draws, seed)
    for label, sign in (("wins", 1), ("losses", -1)):
        rows = []
        for p in parents:
            for r in range(repeats):
                a, b = old[p, r], new[p, r]
                if b["em"] - a["em"] != sign:
                    continue
                category = (
                    "unobserved_involved"
                    if not a["observed"] or not b["observed"]
                    else "both_valid"
                    if a["valid"] and b["valid"]
                    else "protocol_involved"
                )
                rows.append({"case_id": p, "repeat": r, "category": category, "rl16": a, "rl24": b})
        result[label] = {
            "episodes": len(rows),
            "rows": rows,
            "categories": dict(Counter(r["category"] for r in rows)),
        }
    result["lower_bound_difference_not_effect_estimate"] = any(
        not r["observed"] for table in (old, new) for r in table.values()
    )
    return result


def compare(old_output, new_output, cases_path, *, draws=20000, seed=SEED):
    outputs = [Path(p).resolve() for p in (old_output, new_output)]
    cases_path = Path(cases_path).resolve()
    hashes = {}

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in hashes and hashes[str(path)] != digest:
            raise ValueError("consumed receipt changed: " + str(path))
        hashes[str(path)] = digest
        return json.loads(raw)

    plans = [read(p / "PLAN.json") for p in outputs]
    validate_contract(*plans)
    first = plans[0]
    if (
        first["cases_sha256"] != CASES_SHA256
        or len(first["case_ids"]) != 64
        or first["repeats"] != 2
        or first["split"] != "development"
    ):
        raise ValueError("requires frozen fresh003 64-parent two-repeat panel")
    cases_raw = cases_path.read_bytes()
    hashes[str(cases_path)] = hashlib.sha256(cases_raw).hexdigest()
    if hashes[str(cases_path)] != CASES_SHA256:
        raise ValueError("frozen cases changed")
    by_id = {c["id"]: c for c in map(json.loads, cases_raw.splitlines())}
    cases = [by_id[p] for p in first["case_ids"]]
    helper_state = read(Path(first["helper_contract"]["adapter"]) / "STATE.json")
    if helper_state["step"] != 36:
        raise ValueError("requires frozen helper checkpoint36")
    reports, tables = {}, {}
    for label, output, plan, step in zip(("rl16", "rl24"), outputs, plans, (16, 24), strict=True):
        owners = list(output.glob("OWNER-*.json"))
        if not owners or {p.name[6:] for p in owners} != {
            p.name[9:] for p in output.glob("TERMINAL-*.json")
        }:
            raise ValueError("source owner unfinished or absent: " + str(output))
        terminals = [read(p.with_name(p.name.replace("OWNER-", "TERMINAL-"))) for p in owners]
        adapter = Path(plan["adapter"])
        state = read(adapter / "STATE.json")
        if state["step"] != step:
            raise ValueError("wrong RL checkpoint step")
        if step == 24:
            training = read(adapter.parent / "PLAN.json")
            if (
                Path(training["continuation"]["ancestor_checkpoint"]).resolve()
                != Path(first["adapter"]).resolve()
            ):
                raise ValueError("RL24 does not continue compared RL16 checkpoint")
        audit = analyze_planner.analyze(output, cases_path, draws=1, seed=seed)
        for path, digest in audit["input_source_checkpoint_sha256"].items():
            if path in hashes and hashes[path] != digest:
                raise ValueError("native audit disagrees with consumed hash: " + path)
            hashes[path] = digest
        calls = {p.stem: read(p) for p in sorted((output / "calls").glob("*.json"))}
        episodes = {}
        for path in sorted((output / "episodes").glob("*.json")):
            row = read(path)
            episodes[row["case_id"], row["repeat"]] = row
        table = {}
        for case in cases:
            p = case["id"]
            for repeat in range(2):
                episode = episodes.get((p, repeat))
                records = [calls[c] for c in episode["call_ids"]] if episode else []
                expected_seed = first["seed"] + int(probe.runtime.digest(p)[:6], 16) + repeat * 100
                if episode and episode["seed"] != expected_seed:
                    raise ValueError("episode seed differs")
                for record in records:
                    request = record["request"]
                    if probe.runtime.digest(request) != record["request_digest"]:
                        raise ValueError("native request digest differs")
                    if request["model"] != plan["model"]:
                        raise ValueError("native model differs")
                    if request["helper_contract"] != plan["helper_contract"]:
                        raise ValueError("native helper contract differs")
                    sampling = request["sampling"]
                    for key in ("temperature", "top_p", "top_k"):
                        if sampling[key] != plan[key]:
                            raise ValueError("native sampling differs: " + key)
                    role = record["role"]
                    if role == "final":
                        if (
                            request["adapter_enabled"]
                            or request["adapter_sha256"] is not None
                            or request["seed"] != expected_seed + 2
                            or sampling["max_new_tokens"] != plan["caps"]["final"]
                        ):
                            raise ValueError("base final seed/adapter/cap differs")
                    elif role == "root" and (
                        not request["adapter_enabled"]
                        or request["seed"] != expected_seed
                        or sampling["max_new_tokens"] != plan["caps"]["root"]
                        or request["adapter_sha256"]
                        != plan["adapter_files_sha256"]["adapter_model.safetensors"]
                    ):
                        raise ValueError("root checkpoint/seed/cap differs")
                    elif role == "helper":
                        index = int(record["call_id"].rsplit("-", 1)[1])
                        if (
                            not request["adapter_enabled"]
                            or request["model_instance"] != "helper"
                            or request["seed"] != expected_seed + index
                            or sampling["max_new_tokens"] != episode["helper_per_call_cap"]
                            or request["adapter_sha256"]
                            != plan["helper_contract"]["adapter_binding"][
                                "adapter_model.safetensors"
                            ]
                        ):
                            raise ValueError("frozen helper checkpoint/seed/cap differs")
                table[p, repeat] = score_episode(episode, records, case)
        group = dict(audit["groups"]["rl"])
        for metric in ("em", "f1"):
            if abs(mean(r[metric] for r in table.values()) - group[metric]) > 1e-12:
                raise ValueError("official native regrading disagrees")
        group.update(
            observed=sum(r["observed"] for r in table.values()),
            unobserved=sum(not r["observed"] for r in table.values()),
            protocol_failures=sum(r["observed"] and not r["valid"] for r in table.values()),
            unresolved_starts=audit["unresolved_starts"],
            extra_start_attempts=audit["extra_start_attempts"],
            unlinked_call_ids=audit["unlinked_call_ids"],
            terminals=terminals,
            physical_cost=analyze_helper.measured(list(calls.values())),
        )
        reports[label], tables[label] = group, table
    for source in (Path(__file__), Path(analyze_helper.__file__)):
        hashes[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "source_plans": [{"output": str(o), **p} for o, p in zip(outputs, plans, strict=True)],
        "cases": str(cases_path),
        "input_source_checkpoint_sha256": hashes,
        "groups": reports,
        "comparison": paired(tables["rl16"], tables["rl24"], cases, 2, draws=draws, seed=seed),
        "rows": [
            {"case_id": p, "repeat": r, "rl16": tables["rl16"][p, r], "rl24": tables["rl24"][p, r]}
            for p, r in tables["rl16"]
        ],
        "method": {
            "parents": 64,
            "repeats": 2,
            "planned_per_policy": 128,
            "draws": draws,
            "seed": seed,
            "metric": "official MuSiQue alias-max EM/F1",
        },
        "cautions": [
            "Exploratory reuse of the exposed fresh003 panel, deliberately balanced 32 two-hop "
            "and 32 three-hop parents; not the natural whole-development mix.",
            "RL24 minus RL16 changes root training dose and second-pass data exposure together. "
            "Paired component-cluster bootstrap intervals are unadjusted, not confirmation.",
            "Missing/unavailable outcomes contribute zero only to planned-denominator lower "
            "bounds, not observed scientific failures. If either run is incomplete, the "
            "difference of these bounds is not a treatment-effect bound.",
            "Returned invalid plans/helpers/finals are observed protocol zeros. Costs count "
            "all actual native calls; unknown usage/latency is not free. No SFT/direct controls.",
        ],
    }


def markdown(report):
    lines = [
        "# MuSiQue RL dose: checkpoint24 versus16",
        "",
        "64 balanced fresh003 parents × 2 repeats; same frozen helper36 and base final.",
        "",
        "| Policy | Correct/planned | F1 | Observed | Protocol failures | Calls | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, g in report["groups"].items():
        cost = g["physical_cost"]
        lines.append(
            f"| {name} | {g['correct']:g}/{g['planned_episodes']} | {g['f1']:.4f} | "
            f"{g['observed']} | {g['protocol_failures']} | {cost['calls']} | "
            f"{cost['total_tokens']} |"
        )
    lines.append("")
    for name, g in report["groups"].items():
        cost = g["physical_cost"]
        lines.append(
            f"{name}: {g['unobserved']} unobserved outcomes; "
            f"{cost['failed_calls']} failed native calls; {g['unresolved_starts']} unresolved "
            f"starts; {cost['known_latency_seconds']:.1f}s summed known native call time "
            f"({cost['unknown_latency_calls']} unknown latencies)."
        )
    pair = report["comparison"]
    lines += ["", f"{len(pair['component_clusters'])} atomic-component bootstrap clusters.", ""]
    for metric in ("em", "f1"):
        row = pair[metric]
        lines.append(
            f"RL24−RL16 {metric.upper()}: {row['estimate'] * 100:+.2f} points; "
            f"95% CI [{row['ci95'][0] * 100:+.2f}, {row['ci95'][1] * 100:+.2f}]."
        )
    for label in ("wins", "losses"):
        lines += ["", f"{label.title()}: {pair[label]['episodes']}; {pair[label]['categories']}."]
    if pair["lower_bound_difference_not_effect_estimate"]:
        lines += ["", "INCOMPLETE: differences above compare lower bounds, not effect estimates."]
    lines += ["", *report["cautions"], "", "Sources:", ""]
    lines += [f"- {p['output']} — {p['adapter']}" for p in report["source_plans"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("rl16-output", "rl24-output", "cases", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    sibling = args.report.with_suffix(".md")
    if args.report == sibling or args.report.exists() or sibling.exists():
        raise FileExistsError("choose unused JSON and Markdown paths")
    if args.draws < 1:
        raise ValueError("positive bootstrap draws required")
    report = compare(args.rl16_output, args.rl24_output, args.cases, draws=args.draws)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with sibling.open("x") as stream:
        stream.write(markdown(report))


if __name__ == "__main__":
    main()
