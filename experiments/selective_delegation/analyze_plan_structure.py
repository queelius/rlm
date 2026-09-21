"""Structure-only diversity of the completed 256-parent, four-candidate TRAIN pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import eval_planner
import rl_planner


def structure(plan):
    """Ordered graph, not semantic equivalence; mirror and check the actual one-pass binder."""
    questions = plan["subquestions"]
    edges, invalid = [], []
    occurrences = 0
    for step, question in enumerate(questions, 1):
        refs = [int(match.group(1)) for match in re.finditer(r"#(\d+)\b", question)]
        occurrences += len(refs)
        bad = [index for index in refs if not 1 <= index < step]
        try:
            eval_planner.bind_question(question, ["predicted answer"] * (step - 1))
            binder_valid = True
        except ValueError:
            binder_valid = False
        if binder_valid != (not bad):
            raise ValueError("reference classifier disagrees with runtime binder")
        edges.append(sorted({index for index in refs if 1 <= index < step}))
        for index in bad:
            invalid.append(
                {
                    "step": step,
                    "reference": index,
                    "kind": "invalid_index"
                    if index < 1
                    else "self"
                    if index == step
                    else "forward",
                    "outside_plan": not 1 <= index <= len(questions),
                }
            )
    return {
        "canonical": [len(questions), edges],
        "valid": not invalid,
        "invalid_references": invalid,
        "reference_occurrences": occurrences,
    }


def summarize_group(rows):
    if sorted(r["candidate"] for r in rows) != list(range(4)):
        raise ValueError("exactly four unique candidates required")
    rows = sorted(rows, key=lambda row: row["candidate"])
    rewards = [row["reward"] for row in rows]
    advantages = rl_planner.rloo(rewards)
    structures = [structure(row["plan"]) if row["plan_valid"] else None for row in rows]
    valid = [s for s in structures if s is not None and s["valid"]]
    signatures = {json.dumps(s["canonical"]) for s in valid}
    by_length = defaultdict(set)
    for entry in valid:
        by_length[entry["canonical"][0]].add(json.dumps(entry["canonical"][1]))
    return {
        "rewards": rewards,
        "mixed_rewards": len(set(rewards)) > 1,
        "advantages": advantages,
        "structures": structures,
        "distinct_valid_text_lists": len(
            {json.dumps(r["plan"]["subquestions"]) for r in rows if r["plan_valid"]}
        ),
        "distinct_valid_structures": len(signatures),
        "invalid_json_plans": sum(s is None for s in structures),
        "invalid_structure_plans": sum(s is not None and not s["valid"] for s in structures),
        "category": "multiple_valid_structures"
        if len(signatures) > 1
        else "same_valid_structure"
        if signatures
        else "no_valid_structure",
        "step_count_varies": len(by_length) > 1,
        "same_length_dependency_varies": any(len(edges) > 1 for edges in by_length.values()),
        "nonzero_advantages": sum(a != 0 for a in advantages),
        "absolute_advantage_mass": sum(abs(a) for a in advantages),
        "absolute_advantage_token_mass": sum(
            abs(a) * r["root_tokens"] for a, r in zip(advantages, rows, strict=True)
        ),
        "fully_scored_reward_variation": len({r["reward"] for r in rows if r["status"] == "scored"})
        > 1,
    }


def analyze(audit_path):
    audit_path = Path(audit_path).resolve()
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    source = Path(audit["source"])
    if (
        audit["completed_prefix_updates"] != 16
        or audit["excluded_incomplete_batches"]
        or len(audit["batches"]) != 16
        or audit["source_plan"]["parent_schedule"] != "consecutive"
    ):
        raise ValueError("requires the complete16-update consecutive-parent audit")
    audited_hashes = audit["input_source_receipt_sha256"]
    consumed = {str(audit_path): hashlib.sha256(audit_bytes).hexdigest()}

    def read_verified(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if audited_hashes.get(str(path)) != digest:
            raise ValueError("consumed receipt differs from completed audit: " + str(path))
        consumed[str(path)] = digest
        return json.loads(data)

    plan = read_verified(source / "PLAN.json")
    if plan != audit["source_plan"]:
        raise ValueError("source PLAN differs from completed audit")
    parents = [p for batch in plan["case_ids_by_update"] for p in batch]
    if len(parents) != len(set(parents)) or len(parents) != 256:
        raise ValueError("expected256 unique TRAIN parents")
    groups, structure_counts, lengths, invalid_kinds = [], Counter(), Counter(), Counter()
    invalid_examples = []
    for batch in audit["batches"]:
        update = batch["update"]
        if batch["parent_ids"] != plan["case_ids_by_update"][update - 1]:
            raise ValueError("audited parent order differs")
        previous = {g["case_id"]: g for g in batch["groups"]}
        for parent in batch["parent_ids"]:
            rows = []
            for candidate in range(4):
                identity = f"u{update:02d}-{parent}-c{candidate}"
                base = source / f"batch-{update:04d}"
                row = read_verified(base / "episodes" / (identity + ".json"))
                root = read_verified(base / "calls" / (identity + "-root.json"))
                if (
                    row["case_id"] != parent
                    or row["candidate"] != candidate
                    or row["update"] != update
                    or not root["available"]
                ):
                    raise ValueError("candidate/root receipt identity differs")
                try:
                    parsed = eval_planner.parse_plan(root["text"])
                except (ValueError, TypeError):
                    parsed = None
                if bool(parsed) != row["plan_valid"] or (parsed and parsed != row["plan"]):
                    raise ValueError("saved plan differs from actual strict-parsed root")
                row["root_tokens"] = len(root["output_token_ids"])
                rows.append(row)
            group = summarize_group(rows)
            if group["rewards"] != previous[parent]["rewards"] or any(
                not math.isclose(a, b, abs_tol=1e-12)
                for a, b in zip(group["advantages"], previous[parent]["advantages"], strict=True)
            ):
                raise ValueError("reward/advantage differs from completed audit")
            group.update(case_id=parent, update=update)
            groups.append(group)
            for candidate, result in enumerate(group["structures"]):
                if result is None:
                    continue
                lengths[str(result["canonical"][0])] += 1
                if result["valid"]:
                    structure_counts[json.dumps(result["canonical"])] += 1
                else:
                    invalid_kinds.update(r["kind"] for r in result["invalid_references"])
                    invalid_examples.append(
                        {"update": update, "case_id": parent, "candidate": candidate, **result}
                    )
    if len(groups) != 256 or {g["case_id"] for g in groups} != set(parents):
        raise ValueError("completed audit groups must cover all256 parents exactly once")
    buckets = {}
    for category in ("multiple_valid_structures", "same_valid_structure", "no_valid_structure"):
        selected = [g for g in groups if g["category"] == category]
        mixed = [g for g in selected if g["mixed_rewards"]]
        buckets[category] = {
            "all_groups": len(selected),
            "mixed_reward_groups": len(mixed),
            "mixed_fully_scored_groups": sum(g["fully_scored_reward_variation"] for g in mixed),
            "mixed_with_invalid_structure_or_json": sum(
                bool(g["invalid_structure_plans"] or g["invalid_json_plans"]) for g in mixed
            ),
            "nonzero_advantages": sum(g["nonzero_advantages"] for g in mixed),
            "absolute_advantage_mass": sum(g["absolute_advantage_mass"] for g in mixed),
            "absolute_advantage_token_mass": sum(g["absolute_advantage_token_mass"] for g in mixed),
        }
    for path in (Path(__file__), Path(eval_planner.__file__), Path(rl_planner.__file__)):
        consumed[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source": str(source),
        "completed_audit": str(audit_path),
        "parents": 256,
        "root_candidates": 1024,
        "groups": groups,
        "categories": buckets,
        "mixed_reward_groups": sum(g["mixed_rewards"] for g in groups),
        "groups_step_count_varies": sum(g["step_count_varies"] for g in groups),
        "groups_same_length_dependency_varies": sum(
            g["same_length_dependency_varies"] for g in groups
        ),
        "groups_both_variations": sum(
            g["step_count_varies"] and g["same_length_dependency_varies"] for g in groups
        ),
        "invalid_json_plans": sum(g["invalid_json_plans"] for g in groups),
        "invalid_dependency_plans": len(invalid_examples),
        "invalid_reference_occurrences": {
            kind: invalid_kinds[kind] for kind in ("invalid_index", "self", "forward")
        },
        "invalid_examples": invalid_examples,
        "plan_length_counts": dict(sorted(lengths.items())),
        "valid_structure_counts": dict(structure_counts.most_common()),
        "valid_dependency_plans": sum(structure_counts.values()),
        "distinct_valid_structures": len(structure_counts),
        "nonzero_advantage_trajectories": sum(g["nonzero_advantages"] for g in groups),
        "total_absolute_advantage_mass": sum(g["absolute_advantage_mass"] for g in groups),
        "method": {
            "canonical": "[ordered step count, "
            "sorted unique earlier-step reference indices per step]",
            "reference_syntax": r"#(\d+)\b, checked against eval_planner.bind_question",
            "invalid": "Invalid JSON separate. #0 invalid index; #current self; #later forward. "
            "Invalid reference occurrences count individually; duplicate valid edges collapse. "
            "Other # text is literal according to the runtime, not a rejected reference.",
            "group_category": "Count distinct valid dependency structures only; invalid candidates "
            "retained "
            "in all4 rewards/advantages, and mixed invalid contamination separately counted.",
            "credit": "Actual rl_planner.rloo over all4 binary terminal rewards; report nonzero "
            "trajectory count, sum absolute scalar advantages, and sum absolute advantage times "
            "emitted root-token count. "
            "Neither is a gradient norm or evidence of causal structural advantage.",
            "provenance": "Reuse completed audit; verify hashes of consumed JSON receipts/code. "
            "No model/adapter/optimizer weights rehashed or loaded; no fresh readout accessed.",
        },
        "input_source_receipt_sha256": consumed,
        "caution": "Same graph does not imply same semantics; different graph does not prove "
        "better decomposition. Full-source finals can bypass plans. Pre-update rewards across "
        "different parent blocks are not a learning curve or generalization result. "
        "This is descriptive TRAIN credit analysis.",
    }


def markdown(report):
    lines = [
        "# Completed TRAIN plan structure",
        "",
        f"Source: `{report['source']}`",
        "",
        "256 parents ×4 candidates =1024 saved roots;16 pre-update training batches.",
        "",
        "| Valid structures in group | All groups | Mixed reward | Nonzero advantages "
        "| Absolute advantage mass |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, values in report["categories"].items():
        lines.append(
            f"| {category} | {values['all_groups']} | {values['mixed_reward_groups']} | "
            f"{values['nonzero_advantages']} | {values['absolute_advantage_mass']:.6f} |"
        )
    lines += [
        "",
        f"Step-count variation: {report['groups_step_count_varies']}/256 groups. "
        f"Same-length dependency variation: {report['groups_same_length_dependency_varies']}/256; "
        f"both: {report['groups_both_variations']}.",
        "",
        f"Invalid root JSON: {report['invalid_json_plans']}; invalid dependency plans: "
        f"{report['invalid_dependency_plans']}; invalid-reference occurrences: "
        f"`{json.dumps(report['invalid_reference_occurrences'], sort_keys=True)}`.",
        "",
        report["method"]["canonical"],
        "",
        report["method"]["credit"],
        "",
        report["caution"],
        "",
        f"Completed provenance audit: `{report['completed_audit']}`.",
    ]
    return "\n".join(lines) + "\n"


def write_report(report, path):
    path = Path(path)
    if path.exists() or path.with_suffix(".md").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    with path.with_suffix(".md").open("x") as stream:
        stream.write(markdown(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    write_report(analyze(args.audit), args.report)
