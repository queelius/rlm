"""Reproduce completed044/049/052 query behavior using only saved public feedback."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import textcraft_bridge as bridge

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
BRIDGE_SHA = "1553db71f5c4dd4738e50d087c0a33d99d9ea7f97a084e136e4360837f2a7d40"
CONFIGS = (
    ("base_original", "textcraft-pilot-001", "analysis-textcraft-pilot-001.json", None),
    (
        "base_reminder",
        "textcraft-instruction-control-001",
        "analysis-textcraft-instruction-control-001.json",
        None,
    ),
    (
        "trained_original",
        "textcraft-trained-readout-001",
        "analysis-textcraft-trained-readout-001.json",
        "original",
    ),
    (
        "trained_reminder",
        "textcraft-trained-readout-001",
        "analysis-textcraft-trained-readout-001.json",
        "instruction_control",
    ),
)
TEACHER_CONFIGS = (
    (
        "privileged_original",
        "textcraft-trained-readout-001",
        "analysis-textcraft-trained-readout-001.json",
        "original",
    ),
    (
        "privileged_reminder",
        "textcraft-trained-readout-001",
        "analysis-textcraft-trained-readout-001.json",
        "instruction_control",
    ),
    (
        "public_original",
        "textcraft-public-discovery-readout-001",
        "analysis-textcraft-public-teacher-001.json",
        "original",
    ),
    (
        "public_reminder",
        "textcraft-public-discovery-readout-001",
        "analysis-textcraft-public-teacher-001.json",
        "instruction_control",
    ),
)
TEACHER_PAIRS = (
    ("privileged_original", "public_original"),
    ("privileged_reminder", "public_reminder"),
)
BOOLS = ("first_physical_action_root_query", "first_valid_query_root", "ever_root_query")
COUNTS = (
    "calls",
    "invalid_schema",
    "valid_query_calls",
    "root_query_calls",
    "nonexistent_query_calls",
    "nonexistent_item_mentions",
    "repeated_nonexistent_mentions",
)
ROW_KEYS = (
    "episode_id",
    "task_id",
    "repeat",
    "observed",
    "reason",
    *BOOLS,
    "valid_query_calls",
    "nonexistent_query_calls",
    "repeated_nonexistent_mentions",
)
TEACHER_EXTRA_COUNTS = (
    "repeat_returned_static_recipe_calls",
    "repeat_returned_static_recipe_mentions",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def known_static_repeats(queries, history):
    """Count only queries whose item recipe appeared in strictly earlier public feedback."""
    known = set()
    repeated_calls, repeated_mentions = 0, 0
    for query in queries:
        for prior in history[: query["index"]]:
            feedback = prior.get("feedback")
            if isinstance(feedback, list):
                known.update(item["item"] for item in feedback if item.get("recipes"))
        repeats = set(query["items"]).intersection(known)
        repeated_calls += bool(repeats)
        repeated_mentions += len(repeats)
    return repeated_calls, repeated_mentions


def summarize(rows, extra_counts=()):
    observed = [row for row in rows if row["observed"]]
    result = dict(planned=len(rows), observed=len(observed), unknown=len(rows) - len(observed))
    result.update(
        {key: sum(row[key] for row in observed) for key in (*BOOLS, *COUNTS, *extra_counts)}
    )
    result.update(
        never_root_query_observed=sum(not row["ever_root_query"] for row in observed),
        episodes_with_nonexistent_query=sum(row["nonexistent_query_calls"] > 0 for row in observed),
        nonexistent_fraction_valid_queries=(
            result["nonexistent_query_calls"] / result["valid_query_calls"]
            if result["valid_query_calls"]
            else None
        ),
    )
    return result


def analyze(root, configs=CONFIGS, matched_pairs=None, known_recipe_diagnostic=False):
    if sha(Path(bridge.__file__)) != BRIDGE_SHA:
        raise ValueError("strict parser must match source052")
    rows, inventory, provenance = {}, [], {}
    for label, directory, report_name, profile in configs:
        directory, report_path = root / directory, root / report_name
        report, plan = read(report_path), read(directory / "PLAN.json")
        provenance[str(report_path)] = sha(report_path)
        provenance[str(directory / "PLAN.json")] = sha(directory / "PLAN.json")
        assert report["sha256"][str(directory / "PLAN.json")] == sha(directory / "PLAN.json")
        jobs = [
            job
            for job in plan["jobs"]
            if job["policy"] == "flat" and (profile is None or job["prompt_profile"] == profile)
        ]
        assert len(jobs) == 16

        def bound(path, report=report, label=label):
            digest = sha(path)
            assert report["sha256"][str(path)] == digest, str(path)
            inventory.append([label, str(path), digest])
            return read(path)

        out = []
        for job in jobs:
            path = directory / "episodes" / (job["episode_id"] + ".json")
            row = {key: job[key] for key in ("episode_id", "task_id", "repeat", "seed")}
            if not path.exists():
                out.append({**row, "observed": False, "reason": "missing"})
                continue
            episode = bound(path)
            if not episode["observed"]:
                out.append({**row, "observed": False, "reason": "unavailable"})
                continue
            node = bound(directory / "nodes" / (job["episode_id"] + "-n0.json"))
            calls = [bound(directory / "calls" / (cid + ".json")) for cid in episode["call_ids"]]
            assert all(call["available"] and call["node_id"] == "n0" for call in calls)
            assert node["call_ids"] == episode["call_ids"]
            assert len(node["public_history"]) == len(calls)
            frame = next(
                json.loads(line)
                for line in calls[0]["request"]["prompt"].splitlines()
                if line.startswith('{"goal":')
            )
            roots = set(frame["target_items"])
            assert len(roots) == 1
            queries, invalid = [], 0
            for index, (call, history) in enumerate(
                zip(calls, node["public_history"], strict=True)
            ):
                try:
                    action = bridge.parse_action(call["text"])
                except ValueError:
                    assert history["response"] == call["text"]
                    invalid += 1
                    continue
                assert action == history["action"]
                if action["action"] != "get_info":
                    continue
                feedback = history["feedback"]
                assert isinstance(feedback, list)
                assert [item["item"] for item in feedback] == action["items"]
                missing = [
                    item["item"]
                    for item in feedback
                    if not item["can_craft"]
                    and not item["is_base"]
                    and item["in_inventory"] == 0
                    and item["recipes"] == []
                ]
                queries.append(
                    dict(
                        index=index,
                        call_id=call["call_id"],
                        items=action["items"],
                        root=bool(roots.intersection(action["items"])),
                        nonexistent=missing,
                    )
                )
            names = [item for query in queries for item in query["nonexistent"]]
            counter = Counter(names)
            row = {
                **row,
                "observed": True,
                "calls": len(calls),
                "invalid_schema": invalid,
                "valid_query_calls": len(queries),
                "first_physical_action_root_query": bool(
                    queries and queries[0]["index"] == 0 and queries[0]["root"]
                ),
                "first_valid_query_root": bool(queries and queries[0]["root"]),
                "ever_root_query": any(query["root"] for query in queries),
                "root_query_calls": sum(query["root"] for query in queries),
                "nonexistent_query_calls": sum(bool(query["nonexistent"]) for query in queries),
                "nonexistent_item_mentions": len(names),
                "repeated_nonexistent_mentions": sum(count - 1 for count in counter.values()),
            }
            if known_recipe_diagnostic:
                repeated_calls, repeated_mentions = known_static_repeats(
                    queries, node["public_history"]
                )
                row["repeat_returned_static_recipe_calls"] = repeated_calls
                row["repeat_returned_static_recipe_mentions"] = repeated_mentions
            out.append(row)
        rows[label] = out
    matched = {}
    if matched_pairs is None:
        matched_pairs = (
            ("base_original", "trained_original"),
            ("base_reminder", "trained_reminder"),
        )
    for left_name, right_name in matched_pairs:
        left = {(row["task_id"], row["repeat"]): row for row in rows[left_name]}
        right = {(row["task_id"], row["repeat"]): row for row in rows[right_name]}
        assert set(left) == set(right)
        keys = [key for key in left if left[key]["observed"] and right[key]["observed"]]
        assert all(left[key]["seed"] == right[key]["seed"] for key in keys)
        matched[right_name + "_vs_" + left_name] = {
            "planned_pairs": 16,
            "observed_pairs": len(keys),
            "unknown_pairs": 16 - len(keys),
            "base": summarize(
                [left[key] for key in keys],
                TEACHER_EXTRA_COUNTS if known_recipe_diagnostic else (),
            ),
            "trained": summarize(
                [right[key] for key in keys],
                TEACHER_EXTRA_COUNTS if known_recipe_diagnostic else (),
            ),
            "boolean_transitions": {
                field: {
                    "base_yes_trained_no": sum(
                        left[k][field] and not right[k][field] for k in keys
                    ),
                    "base_no_trained_yes": sum(
                        not left[k][field] and right[k][field] for k in keys
                    ),
                }
                for field in BOOLS
            },
            "unknown_slots": [dict(task_id=k[0], repeat=k[1]) for k in left if k not in keys],
        }
        if configs == TEACHER_CONFIGS:
            matched[right_name + "_vs_" + left_name]["semantic_mapping"] = {
                "base": "privileged teacher",
                "trained": "public-information teacher",
            }
    return {
        "schema": "textcraft-query-transfer-reproducible-audit-v1",
        "groups": {
            name: summarize(group, TEACHER_EXTRA_COUNTS if known_recipe_diagnostic else ())
            for name, group in rows.items()
        },
        "matched_comparisons": matched,
        "episode_rows": {
            name: [
                {
                    k: row[k]
                    for k in (*ROW_KEYS, *(TEACHER_EXTRA_COUNTS if known_recipe_diagnostic else ()))
                    if k in row
                }
                for row in group
            ]
            for name, group in rows.items()
        },
        "native_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_and_report_sha256": {
            **provenance,
            str(Path(__file__)): sha(__file__),
            str(Path(bridge.__file__)): sha(bridge.__file__),
        },
        "method": "All completed matched slots; root from first public goal frame only. "
        "Strict query schema, including queries for nonexistent names. Nonexistent means native "
        "feedback can_craft=false,is_base=false,inventory0,recipes[]. Missing/unavailable remain "
        "unknown. Verify every used native receipt against prior independent audit. "
        "No model calls, gold recipe inspection, teacher-order causal claim or outcome selection.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify-against", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--comparison", choices=("transfer", "teacher"), default="transfer")
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise ValueError("immutable report exists")
    configs = TEACHER_CONFIGS if args.comparison == "teacher" else CONFIGS
    pairs = TEACHER_PAIRS if args.comparison == "teacher" else None
    result = analyze(
        args.root,
        configs,
        pairs,
        known_recipe_diagnostic=args.comparison == "teacher",
    )
    if args.verify_against:
        old = read(args.verify_against)
        for key in ("groups", "matched_comparisons", "episode_rows", "native_inventory_sha256"):
            if result[key] != old[key]:
                raise ValueError("scientific replay mismatch: " + key)
        result["reproduced_report"] = str(args.verify_against)
        result["reproduced_report_sha256"] = sha(args.verify_against)
    if args.report:
        with args.report.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"verified": bool(args.verify_against), "groups": result["groups"]}))
