"""Secondary component-cluster and dependency audit of one completed planner transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import eval_planner
import probe

SEED = 2026092114


def reference_defects(questions):
    """Scan every parsed question, including steps unreachable in the saved execution."""
    defects = []
    for step, question in enumerate(questions, 1):
        for match in re.finditer(r"#(\d+)\b", question):
            reference = int(match.group(1))
            if not 1 <= reference < step:
                defects.append(
                    {
                        "step": step,
                        "reference": reference,
                        "kind": "nonpositive"
                        if reference < 1
                        else "self"
                        if reference == step
                        else "forward",
                    }
                )
    return defects


def audit(original_report, examples_path, *, draws=20000, seed=SEED):
    hashes, cache = {}, {}

    def track(path, expected=None):
        path = str(Path(path).resolve())
        if path not in cache:
            cache[path] = Path(path).read_bytes()
            hashes[path] = hashlib.sha256(cache[path]).hexdigest()
        if expected is not None and expected != hashes[path]:
            raise ValueError("input hash mismatch: " + path)
        return cache[path]

    def read(path):
        return json.loads(track(path))

    original = read(original_report)
    output, cases_path = Path(original["output"]), Path(original["cases"])
    plan = read(output / "PLAN.json")
    owners = sorted(output.glob("OWNER-*.json"))
    if not owners or len(owners) != len(list(output.glob("TERMINAL-*.json"))):
        raise ValueError("source run must be completed")
    for path in owners:
        owner = read(path)
        terminal = read(path.with_name(path.name.replace("OWNER-", "TERMINAL-")))
        if terminal["failure"] or terminal["stopped"]:
            raise ValueError("source owner did not complete successfully")
        track(owner["source"], plan["source_sha256"])
    for path, expected in plan["dependencies"].items():
        track(path, expected)
    cases = [
        json.loads(line)
        for line in track(cases_path, plan["cases_sha256"]).splitlines()
        if line.strip()
    ]
    by_id = {c["id"]: c for c in cases}
    parents, repeats, conditions = plan["case_ids"], plan["repeats"], plan["conditions"]
    if conditions != ["base", "sft"] or len(parents) != 64 or repeats != 2:
        raise ValueError("expected the completed 64-parent paired four-hop transfer")
    calls, episodes = {}, {}
    for path in sorted((output / "calls").glob("*.json")):
        call = read(path)
        if call["call_id"] != path.stem or call["call_id"] in calls:
            raise ValueError("duplicate/mismatched native call")
        if probe.runtime.digest(call["request"]) != call["request_digest"]:
            raise ValueError("native request digest mismatch")
        calls[call["call_id"]] = call
    for path in sorted((output / "episodes").glob("*.json")):
        row = read(path)
        key = row["case_id"], row["repeat"], row["condition"]
        if key in episodes:
            raise ValueError("duplicate episode")
        episodes[key] = row
    inventory = {(p, r, c) for p in parents for r in range(repeats) for c in conditions}
    if set(episodes) != inventory:
        raise ValueError("completed planned inventory differs")
    values, groups, diagnostics, linked = {}, {}, [], set()
    for condition in conditions:
        stops, statuses, lengths, invalid_stops, defects_by_kind = [Counter() for _ in range(5)]
        condition_rows, defective, invalid_dependency_parents = [], [], set()
        for p in parents:
            for repeat in range(repeats):
                row = episodes[p, repeat, condition]
                records = [calls[cid] for cid in row["call_ids"]]
                if linked.intersection(row["call_ids"]):
                    raise ValueError("native call unexpectedly reused")
                linked.update(row["call_ids"])
                roots = [c for c in records if c["role"] == "root"]
                finals = [c for c in records if c["role"] == "final"]
                if len(roots) != 1 or len(finals) > 1:
                    raise ValueError("unexpected root/final count")
                root = roots[0]
                if root["request"]["prompt"] != eval_planner.planner_prompt(by_id[p]):
                    raise ValueError("root prompt reconstruction differs")
                stops[root.get("finish_reason", "unknown")] += 1
                statuses[row["status"]] += 1
                if row["status"] == "invalid_dependency":
                    invalid_dependency_parents.add(p)
                try:
                    generated = eval_planner.parse_plan(root["text"]) if root["available"] else None
                except (ValueError, TypeError):
                    generated = None
                if bool(generated) != row["plan_valid"] or (generated and generated != row["plan"]):
                    raise ValueError("root parse disagrees with recorded plan")
                issues = reference_defects(generated["subquestions"]) if generated else []
                if generated:
                    lengths[str(len(generated["subquestions"]))] += 1
                else:
                    invalid_stops[root.get("finish_reason", "unknown")] += 1
                if issues:
                    first_bad = min(i["step"] for i in issues)
                    detail = {
                        "case_id": p,
                        "repeat": repeat,
                        "condition": condition,
                        "status": row["status"],
                        "defects": issues,
                        "subquestions": generated["subquestions"],
                        "successful_helper_steps": len(row.get("helper_trace", [])),
                        "first_defective_step": first_bad,
                        "not_reached_as_runtime_dependency_error": row["status"]
                        != "invalid_dependency",
                        "defects_after_first_bad_step": sum(i["step"] > first_bad for i in issues),
                    }
                    defective.append(detail)
                    diagnostics.append(detail)
                    defects_by_kind.update(i["kind"] for i in issues)
                final = finals[0] if finals else None
                grade = probe.grade(final["text"] if final and final["available"] else "", by_id[p])
                value = {"em": float(grade["correct"]), "f1": grade["f1"]}
                values[p, repeat, condition] = value
                condition_rows.append(value)
        groups[condition] = {
            "planned": len(condition_rows),
            "correct": sum(v["em"] for v in condition_rows),
            "em": mean(v["em"] for v in condition_rows),
            "f1": mean(v["f1"] for v in condition_rows),
            "root_stop_counts": dict(stops),
            "invalid_plan_stop_counts": dict(invalid_stops),
            "status_counts": dict(statuses),
            "valid_plan_lengths": dict(lengths),
            "static_defective_plans": len(defective),
            "static_defective_parents": len({d["case_id"] for d in defective}),
            "static_defect_occurrences": dict(defects_by_kind),
            "runtime_invalid_dependency_parents": sorted(invalid_dependency_parents),
            "static_defects_not_reached_at_runtime": sum(
                d["not_reached_as_runtime_dependency_error"] for d in defective
            ),
            "native_role_counts": dict(
                Counter(c["role"] for c in calls.values() if c["condition"] == condition)
            ),
        }
        for metric in ("em", "f1", "correct"):
            if abs(groups[condition][metric] - original["groups"][condition][metric]) > 1e-12:
                raise ValueError("native regrading disagrees with original report")
    if groups["base"]["correct"] != 27 or groups["sft"]["correct"] != 24:
        raise ValueError("unexpected four-hop point estimates")
    clusters = analyze_helper.component_clusters([by_id[p] for p in parents])
    intervals = {}
    for metric in ("em", "f1"):
        delta = {
            p: mean(
                values[p, r, "sft"][metric] - values[p, r, "base"][metric] for r in range(repeats)
            )
            for p in parents
        }
        intervals[metric] = analyze_helper.clustered_interval(delta, clusters, draws, seed)
        intervals[metric]["original_parent_only_ci95"] = original["comparisons"]["sft_minus_base"][
            metric
        ]["ci95"]
    training_plan_path = Path(plan["adapter"]).parent / "PLAN.json"
    training_plan = json.loads(track(training_plan_path, plan["training_plan_sha256"]))
    examples = [
        json.loads(line)
        for line in track(examples_path, training_plan["examples_sha256"]).splitlines()
        if line.strip()
    ]
    train_issues = [
        {
            "id": e["id"],
            "defects": reference_defects(eval_planner.parse_plan(e["target"])["subquestions"]),
        }
        for e in examples
    ]
    for path in (
        Path(__file__),
        Path(analyze_helper.__file__),
        Path(eval_planner.__file__),
        Path(eval_planner.planner.__file__),
        Path(probe.__file__),
        probe.MUSIQUE / "metrics/answer.py",
    ):
        track(path)
    return {
        "original_report": str(Path(original_report).resolve()),
        "source_output": str(output),
        "cases": str(cases_path),
        "training_examples": str(Path(examples_path).resolve()),
        "input_source_receipt_sha256": hashes,
        "model": plan["model"],
        "adapter": plan["adapter"],
        "adapter_binding": plan["adapter_files_sha256"],
        "method": {
            "draws": draws,
            "seed": seed,
            "parents": len(parents),
            "repeats": repeats,
            "component_clusters": clusters,
            "cluster_count": len(clusters),
            "bootstrap": "Paired whole connected-component clusters, parent-weighted means; "
            "percentile intervals. Secondary exploratory sensitivity analysis.",
        },
        "groups": groups,
        "sft_minus_base": intervals,
        "native_cost": analyze_helper.measured(list(calls.values())),
        "native_receipts": len(calls),
        "unlinked_call_ids": sorted(set(calls) - linked),
        "training_target_audit": {
            "targets": len(examples),
            "targets_with_static_defects": sum(bool(r["defects"]) for r in train_issues),
            "defective_targets": [r for r in train_issues if r["defects"]],
        },
        "static_defect_episodes": diagnostics,
        "limitations": [
            "Static checking covers strict-JSON parsed plans only; truncated/unparseable roots "
            "are not repaired or interpreted as dependency-valid.",
            "Forward/self references are executor-contract defects, not an assessment of semantic "
            "plan quality. A plan with fewer than four questions is not necessarily wrong.",
            "Scanning all parsed questions reveals defects masked by earlier helper failures; "
            "it does not show how those hypothetical later helpers would have answered.",
            "No causal attribution to chain length follows from these descriptive counts. The "
            "128-token root cap can truncate plans; no longer-budget counterfactual was run.",
            "Only 24 connected clusters underpin this exploratory unadjusted interval. Original "
            "point estimates are unchanged; this is not a new model evaluation or training run.",
        ],
    }


def markdown(report):
    lines = [
        "# Four-hop transfer: secondary structural audit",
        "",
        f"Source: `{report['source_output']}`",
        "",
        "Native finals reproduce base 27/128 versus SFT 24/128. The point estimates "
        "are unchanged; uncertainty now resamples 24 connected atomic-component clusters.",
        "",
    ]
    for metric, row in report["sft_minus_base"].items():
        lines.append(
            f"- SFT minus base {metric.upper()}: {100 * row['estimate']:+.2f} points; "
            f"component-cluster 95% interval [{100 * row['ci95'][0]:+.2f}, "
            f"{100 * row['ci95'][1]:+.2f}]. Original parent-only interval: "
            f"[{100 * row['original_parent_only_ci95'][0]:+.2f}, "
            f"{100 * row['original_parent_only_ci95'][1]:+.2f}]."
        )
    lines += ["", "Root formatting and dependency structure:", ""]
    for condition, row in report["groups"].items():
        lines.append(
            f"- {condition}: root stops {row['root_stop_counts']}; "
            f"invalid-plan stops {row['invalid_plan_stop_counts']}; "
            f"{row['static_defective_plans']} parsed plans on {row['static_defective_parents']} "
            f"parents have static dependency defects ({row['static_defect_occurrences']}). "
            f"{row['static_defects_not_reached_at_runtime']} defective plans were masked "
            "before a runtime dependency error."
        )
        lines.append(
            f"  - Runtime statuses: {row['status_counts']}; "
            f"valid plan lengths: {row['valid_plan_lengths']}; "
            f"native role counts: {row['native_role_counts']}."
        )
    training = report["training_target_audit"]
    lines += [
        "",
        f"Training targets: {training['targets_with_static_defects']}/"
        f"{training['targets']} contain static dependency defects.",
        "",
        f"Native call receipts: {report['native_receipts']}; all original receipts remain "
        "unchanged. Source/report/cases/native receipt hashes are in the JSON audit.",
        "",
        f"Bootstrap: {report['method']['draws']} draws, seed {report['method']['seed']}.",
        "",
        *["- " + s for s in report["limitations"]],
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("original-report", "training-examples", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    sibling = args.report.with_suffix(".md")
    if args.report.exists() or sibling.exists() or args.report == sibling:
        parser.error("choose unused distinct JSON/Markdown audit paths")
    report = audit(args.original_report, args.training_examples)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with sibling.open("x") as handle:
        handle.write(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
