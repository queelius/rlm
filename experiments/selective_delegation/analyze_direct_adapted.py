"""Immutable paired analysis of direct helper-adapter receipts, with optional saved controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

import analyze_helper
import eval_direct_adapted as runner
import eval_planner as evaluation
import probe

SEED = 2026092116


def validate_call(call, case, repeat, condition, plan):
    request = call["request"]
    enabled = condition == "helper_sft_direct"
    adapter = plan["helper_adapter_binding"]["adapter_model.safetensors"] if enabled else None
    expected = {
        "prompt": evaluation.direct_prompt(case),
        "condition": condition,
        "role": "direct_answer",
        "model": plan["model"],
        "adapter_enabled": enabled,
        "adapter_name": "helper_sft" if enabled else None,
        "adapter_sha256": adapter,
        "seed": plan.get("seed", evaluation.SEED)
        + int(probe.runtime.digest(case["id"])[:6], 16)
        + repeat * 100
        + 2,
        "sampling": runner.SAMPLING,
    }
    if probe.runtime.digest(request) != call["request_digest"] or any(
        request.get(k) != value for k, value in expected.items()
    ):
        raise ValueError("direct native request identity differs")
    for key in ("condition", "role", "model", "adapter_enabled", "adapter_name", "adapter_sha256"):
        if call.get(key) != expected[key]:
            raise ValueError("direct role/model/adapter receipt differs")
    if call["input_token_ids"] != request["input_token_ids"]:
        raise ValueError("request/native input tokens differ")
    for field, ids in (
        ("prompt_tokens", "input_token_ids"),
        ("completion_tokens", "output_token_ids"),
    ):
        actual = call.get("usage", {}).get(field)
        if actual is not None and (ids not in call or actual != len(call[ids])):
            raise ValueError("native token usage differs")
    if call["available"] and (
        not call.get("output_token_ids")
        or any(
            field not in call.get("usage", {}) for field in ("prompt_tokens", "completion_tokens")
        )
    ):
        raise ValueError("available direct call lacks native usage")


def compare(values, parents, repeats, clusters, left, right, *, draws, seed):
    result = {"left": left, "right": right}
    for metric in ("em", "f1"):
        delta = {
            p: mean(
                values[p, r, right][metric] - values[p, r, left][metric] for r in range(repeats)
            )
            for p in parents
        }
        result[metric] = analyze_helper.clustered_interval(delta, clusters, draws, seed)
    for sign, name in ((1, "wins"), (-1, "losses")):
        rows = []
        for parent in parents:
            for repeat in range(repeats):
                a, b = values[parent, repeat, left], values[parent, repeat, right]
                if (b["em"] - a["em"]) * sign > 0:
                    category = (
                        "missing_involved"
                        if not a["present"] or not b["present"]
                        else ("both_valid" if a["valid"] and b["valid"] else "protocol_involved")
                    )
                    rows.append({"case_id": parent, "repeat": repeat, "category": category})
        result[name] = {
            "episodes": len(rows),
            "parents": sorted({r["case_id"] for r in rows}),
            **{
                c: sum(r["category"] == c for r in rows)
                for c in ("both_valid", "protocol_involved", "missing_involved")
            },
            "rows": rows,
        }
    return result


def analyze(
    output, cases_path, *, baseline_output=None, planner_outputs=(), draws=20000, seed=SEED
):
    output, cases_path = Path(output).resolve(), Path(cases_path).resolve()
    hashes = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("input/source/receipt hash differs: " + str(path))
        hashes[str(path)] = digest
        return json.loads(data)

    def track(path, expected=None):
        path = Path(path).resolve()
        digest = probe.campaign.sha(path)
        if expected is not None and digest != expected:
            raise ValueError("input/source/checkpoint hash differs: " + str(path))
        hashes[str(path)] = digest

    plan = read(output / "PLAN.json")
    panel = plan.get("panel", "fresh003")
    spec = runner.panel_spec(panel)
    if panel != "fresh003" and (baseline_output or planner_outputs):
        raise ValueError(
            "Hotpot optional historical comparisons unsupported: incompatible seeds/schema"
        )
    track(cases_path, plan["cases_sha256"])
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    runner.validate_panel(cases, panel)
    parents, repeats = plan["case_ids"], plan["repeats"]
    if (
        plan["schema"] != "paired-direct-helper-adapter-v1"
        or plan["conditions"] != list(runner.CONDITIONS)
        or parents != [c["id"] for c in cases]
        or repeats != 2
        or plan["cases_sha256"] != spec["cases_sha256"]
        or plan["seed"] != spec["seed"]
        or plan["metric"] != spec["metric"]
        or plan["split"] != spec["split"]
        or plan["parents"] != spec["parents"]
        or plan["maximum_calls"] != spec["parents"] * 4
        or (panel != "fresh003" and plan.get("dataset") != spec["dataset"])
        or plan["sampling"] != runner.SAMPLING
        or draws < 1
    ):
        raise ValueError("unexpected direct-adapted planned contract")
    by_id = {c["id"]: c for c in cases}
    for path, digest in plan["dependencies"].items():
        track(path, digest)
    adapter = Path(plan["helper_adapter"])
    for name, digest in plan["helper_adapter_binding"].items():
        track(adapter / name, digest)
    track(adapter.parent / "PLAN.json", plan["helper_training_plan_sha256"])
    for name, key in (
        ("local-research-manifest.json", "model_manifest_sha256"),
        ("generation_config.json", "generation_config_sha256"),
    ):
        track(Path(plan["model"]) / name, plan[key])
    calls, episodes, starts = {}, {}, {}
    for directory, target, identity_field in (
        ("calls", calls, "call_id"),
        ("starts", starts, "call_id"),
        ("episodes", episodes, "episode_id"),
    ):
        for path in sorted((output / directory).glob("*.json")):
            row = read(path)
            identity = row[identity_field]
            if path.stem != identity or identity in target:
                raise ValueError("duplicate or malformed physical identity")
            target[identity] = row
    expected_calls = {
        f"{p}-r{r}-{c}-answer": (p, r, c)
        for p in parents
        for r in range(2)
        for c in runner.CONDITIONS
    }
    if set(calls) - set(expected_calls) or set(starts) - set(expected_calls):
        raise ValueError("call outside planned direct inventory")
    for cid, call in calls.items():
        p, r, c = expected_calls[cid]
        validate_call(call, by_id[p], r, c, plan)
        if cid not in starts or starts[cid]["request_digest"] != call["request_digest"]:
            raise ValueError("call lacks matching native start")
    for cid, start in starts.items():
        p, r, c = expected_calls[cid]
        validate_call(start, by_id[p], r, c, plan)
    values, linked, indexed = {}, set(), {}
    for eid, row in episodes.items():
        key = row["case_id"], row["repeat"], row["condition"]
        if eid + "-answer" not in expected_calls or key != expected_calls[eid + "-answer"]:
            raise ValueError("episode outside planned inventory")
        if key in indexed or row["call_ids"] != [eid + "-answer"]:
            raise ValueError("direct episode must reference exactly its single answer call")
        indexed[key] = row
    for cid, (p, r, c) in expected_calls.items():
        row, call = indexed.get((p, r, c)), calls.get(cid)
        if row is not None and call is None:
            raise ValueError("direct episode references missing call")
        grade = runner.eval_helper.grade_final(
            call["text"] if call and call["available"] else "", by_id[p]
        )
        status = (
            "missing_episode"
            if row is None
            else "generation_failure"
            if not call["available"]
            else ("scored" if grade["valid"] else "invalid_answer")
        )
        if row is not None:
            linked.add(cid)
            if (
                any(row[k] != grade[k] for k in ("valid", "correct", "f1", "parsed", "metric"))
                or row["status"] != status
            ):
                raise ValueError("saved direct grade differs from official regrade")
        values[p, r, c] = {
            "em": float(grade["correct"]) if row else 0.0,
            "f1": grade["f1"] if row else 0.0,
            "valid": bool(row and grade["valid"]),
            "present": row is not None,
            "status": status,
        }
    # Fresh-dev should form47 connected component clusters; calculate, do not assume.
    clusters = analyze_helper.component_clusters(cases)
    groups = {}

    def add_group(condition, records):
        subset = [values[p, r, condition] for p in parents for r in range(2)]
        cost = analyze_helper.measured(records)
        groups[condition] = {
            "planned": len(subset),
            "recorded": sum(v["present"] for v in subset),
            "valid": sum(v["valid"] for v in subset),
            "correct": sum(v["em"] for v in subset),
            "em": mean(v["em"] for v in subset),
            "f1": mean(v["f1"] for v in subset),
            "status_counts": dict(Counter(v["status"] for v in subset)),
            "physical_cost": cost,
            "per_planned_attempt": {k: cost[k] / len(subset) for k in ("calls", "total_tokens")},
        }

    unresolved = [r for cid, r in starts.items() if cid not in calls]
    for c in runner.CONDITIONS:
        add_group(c, [r for r in list(calls.values()) + unresolved if r["condition"] == c])

    baseline = (
        Path(baseline_output).resolve()
        if baseline_output
        else output.parent / "fresh-contract-direct-001"
    )
    baseline_agreement = {
        "output": str(baseline) if panel == "fresh003" else None,
        "present": panel == "fresh003" and (baseline / "PLAN.json").exists(),
    }
    if baseline_agreement["present"]:
        old_plan = read(baseline / "PLAN.json")
        fields = ("cases_sha256", "case_ids", "repeats", "seed", "model", "model_manifest_sha256")
        contract_matches = (
            all(old_plan.get(k) == plan.get(k) for k in fields) and old_plan.get("mode") == "direct"
        )
        baseline_agreement["panel_model_contract_matches"] = contract_matches
        matched = Counter()
        for p in parents:
            for r in range(2):
                path = baseline / "calls" / f"{p}-r{r}-base-direct-final.json"
                current = calls.get(f"{p}-r{r}-base_direct-answer")
                if not path.exists() or current is None:
                    matched["missing_pair"] += 1
                    continue
                old = read(path)
                request_ok = probe.runtime.digest(old["request"]) == old["request_digest"]
                same = (
                    contract_matches
                    and request_ok
                    and all(
                        old["request"].get(k) == current["request"].get(k)
                        for k in (
                            "prompt",
                            "input_token_ids",
                            "model",
                            "adapter_enabled",
                            "adapter_sha256",
                            "seed",
                            "sampling",
                        )
                    )
                )
                matched["same_scientific_request" if same else "different_scientific_request"] += 1
                if old["available"] and current["available"]:
                    matched["both_available"] += 1
                    matched["same_output_tokens"] += old.get("output_token_ids") == current.get(
                        "output_token_ids"
                    )
                    matched["same_output_text"] += old["text"] == current["text"]
        baseline_agreement["pairs"] = dict(matched)
        baseline_agreement["note"] = (
            "Identity and actual outputs checked separately; no receipts reused and no "
            "bitwise equality assumed."
        )

    comparison_sources = []
    for path in planner_outputs:
        source = Path(path).resolve()
        other = read(source / "PLAN.json")
        if any(
            other.get(k) != plan.get(k)
            for k in (
                "cases_sha256",
                "case_ids",
                "repeats",
                "seed",
                "model",
                "model_manifest_sha256",
            )
        ):
            raise ValueError("planner panel/model/seed mapping differs")
        helper = other.get("helper_contract", {})
        if (
            other.get("execution") != "isolated"
            or other.get("mode", "planner") != "planner"
            or other.get("temperature") != 0.5
            or helper.get("mode") != "trained_helper"
            or helper.get("adapter_binding") != plan["helper_adapter_binding"]
        ):
            raise ValueError("planner fixed helper/execution contract differs")
        for path, digest in other.get("dependencies", {}).items():
            track(path, digest)
        for name, digest in other.get("adapter_files_sha256", {}).items():
            track(Path(other["adapter"]) / name, digest)
        pcalls = {p.stem: read(p) for p in (source / "calls").glob("*.json")}
        prows = {}
        for path in (source / "episodes").glob("*.json"):
            row = read(path)
            key = row["case_id"], row["repeat"], row["condition"]
            if key in prows:
                raise ValueError("duplicate planner episode")
            prows[key] = row
        for condition in other["conditions"]:
            if condition not in ("sft", "rl"):
                continue
            label = "planner_" + condition
            if label in groups:
                raise ValueError("duplicate planner comparison condition")
            for p in parents:
                for r in range(2):
                    row = prows.get((p, r, condition))
                    seed_root = evaluation.SEED + int(probe.runtime.digest(p)[:6], 16) + r * 100
                    if row and row["seed"] != seed_root:
                        raise ValueError("planner episode seed mapping differs")
                    records = [pcalls[cid] for cid in row["call_ids"]] if row else []
                    for call in records:
                        req = call["request"]
                        role = call["role"]
                        expected_adapter = (
                            other["adapter_files_sha256"]["adapter_model.safetensors"]
                            if role == "root"
                            else plan["helper_adapter_binding"]["adapter_model.safetensors"]
                            if role == "helper"
                            else None
                        )
                        if (
                            role not in ("root", "helper", "final")
                            or call["condition"] != condition
                            or probe.runtime.digest(req) != call["request_digest"]
                            or req["model"] != plan["model"]
                            or req["adapter_enabled"] != (role != "final")
                            or req["adapter_sha256"] != expected_adapter
                        ):
                            raise ValueError("planner saved role/model/adapter identity differs")
                        for field, ids in (
                            ("prompt_tokens", "input_token_ids"),
                            ("completion_tokens", "output_token_ids"),
                        ):
                            if ids in call and call.get("usage", {}).get(field) != len(call[ids]):
                                raise ValueError("planner native token usage differs")
                    finals = [call for call in records if call["role"] == "final"]
                    if len(finals) > 1:
                        raise ValueError("multiple planner final calls")
                    final = finals[0] if finals else None
                    if final:
                        req = final["request"]
                        if (
                            probe.runtime.digest(req) != final["request_digest"]
                            or req["seed"] != seed_root + 2
                            or req["adapter_enabled"]
                            or req["adapter_sha256"] is not None
                            or req["sampling"] != runner.SAMPLING
                        ):
                            raise ValueError("planner final seed/sampling/base identity differs")
                    grade = probe.grade(
                        final["text"] if final and final["available"] else "", by_id[p]
                    )
                    values[p, r, label] = {
                        "em": float(grade["correct"]),
                        "f1": grade["f1"],
                        "valid": grade["valid"],
                        "present": row is not None,
                        "status": ("scored" if grade["valid"] else "invalid_final")
                        if final and final["available"]
                        else row["status"]
                        if row
                        else "missing_episode",
                    }
            add_group(label, [c for c in pcalls.values() if c["condition"] == condition])
        comparison_sources.append({"output": str(source), "plan": other})
    pairs = {
        right + "_minus_" + left: compare(
            values, parents, repeats, clusters, left, right, draws=draws, seed=seed
        )
        for left, right in combinations(groups, 2)
    }
    for path in (
        Path(__file__),
        Path(analyze_helper.__file__),
        Path(runner.__file__),
        Path(probe.__file__),
        *runner.eval_helper.metric_sources(spec["dataset"]),
    ):
        track(path)
    return {
        "output": str(output),
        "source_plan": plan,
        "method": {
            "parents": len(parents),
            "episodes_per_arm": len(parents) * repeats,
            "repeats": repeats,
            "clusters": clusters,
            "draws": draws,
            "seed": seed,
            "metric": spec["metric"],
            "panel": panel,
            "parents_missing_component_ids": sum(
                not c.get("metadata", {}).get("component_ids") for c in cases
            ),
            "bootstrap": "Connected component clusters, parent-weighted paired repeat means; "
            "Python Random.randrange and interpolated percentile95; "
            "missing/invalid planned outcomes zero; missing component IDs use singleton-parent "
            "fallback, not verified atomic independence.",
        },
        "groups": groups,
        "comparisons": pairs,
        "baseline_agreement": baseline_agreement,
        "comparison_sources": comparison_sources,
        "input_source_receipt_sha256": hashes,
        "unlinked_call_ids": sorted(set(calls) - linked),
        "unresolved_start_ids": [r["call_id"] for r in unresolved],
        "all_direct_episodes_present": len(episodes) == len(parents) * repeats * 2,
        "cautions": [
            "Missing outcomes are incomplete lower bounds, not observed errors; "
            "unknown cost is not zero.",
            f"{len(parents)} parents, not{len(parents) * repeats} independent repeats; "
            "clusters calculated from cases. Missing component IDs use singleton-parent "
            "fallback, not verified atomic independence.",
            "Both-valid gains are not proof of a decomposition mechanism. Helper adapter "
            "is transferred from subquestion training to an original-question prompt.",
            "Optional planner comparisons are saved adaptive-development architecture "
            "contrasts, not unseen-held generalization claims.",
        ],
    }


def markdown(report):
    method = report["method"]
    lines = [
        "# Direct helper-adapter control",
        "",
        f"Source: `{report['output']}`",
        "",
        f"{method['parents']} parents, {method['episodes_per_arm']} planned attempts per arm; "
        f"{len(method['clusters'])} bootstrap clusters. Metric: {method['metric']}. "
        f"{method.get('parents_missing_component_ids', 0)} parents lack component IDs "
        "and use singleton-parent fallback, not verified atomic independence.",
        "",
        "| Arm | EM | F1 | Correct/planned | Calls | Tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c, g in report["groups"].items():
        lines.append(
            f"| {c} | {g['em']:.2%} | {g['f1']:.2%} | {g['correct']:g}/{g['planned']} | "
            f"{g['physical_cost']['calls']} | {g['physical_cost']['total_tokens']} |"
        )
    lines += ["", f"Paired cluster bootstrap: {method['draws']} draws, seed {method['seed']}.", ""]
    for name, pair in report["comparisons"].items():
        lines.append(
            f"- {name}: "
            + "; ".join(
                f"{m.upper()} {100 * pair[m]['estimate']:+.2f}pp "
                f"[{100 * pair[m]['ci95'][0]:+.2f}, {100 * pair[m]['ci95'][1]:+.2f}]"
                for m in ("em", "f1")
            )
        )
        for change in ("wins", "losses"):
            row = pair[change]
            lines.append(
                f"  - {change}: {row['episodes']} episodes; {row['both_valid']} both-valid, "
                f"{row['protocol_involved']} protocol/generation, "
                f"{row['missing_involved']} missing-involved."
            )
    lines += [
        "",
        "Statuses: "
        + json.dumps({c: g["status_counts"] for c, g in report["groups"].items()}, sort_keys=True),
        "",
        "Historical direct-base agreement: "
        + json.dumps(report["baseline_agreement"], sort_keys=True),
        "",
        *report["cautions"],
    ]
    return "\n".join(lines) + "\n"


def write_report(report, path):
    path = Path(path)
    sibling = path.with_suffix(".md")
    if path.exists() or sibling.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with sibling.open("x") as stream:
        stream.write(markdown(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--planner-output", type=Path, action="append", default=[])
    args = parser.parse_args()
    write_report(
        analyze(
            args.output,
            args.cases,
            baseline_output=args.baseline_output,
            planner_outputs=args.planner_output,
        ),
        args.report,
    )
