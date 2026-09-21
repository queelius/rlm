#!/usr/bin/env python3
"""Observed plan-length helper-gating diagnostic; never selects new outcomes."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    values.sort()
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap(rows, clusters, seed: int) -> list[float]:
    by_id: dict[str, list[float]] = {}
    for row in rows:
        by_id.setdefault(row["case_id"], []).append(row["rule"] - row["plan"])
    parent_values = {case: sum(values) / len(values) for case, values in by_id.items()}
    cluster_values = [[parent_values[case] for case in cluster] for cluster in clusters]
    rng = random.Random(seed)
    draws = []
    for _ in range(20000):
        sampled = [cluster_values[rng.randrange(len(cluster_values))] for _ in cluster_values]
        values = [value for cluster in sampled for value in cluster]
        draws.append(sum(values) / len(values))
    return [
        sum(parent_values.values()) / len(parent_values),
        percentile(draws, 0.025),
        percentile(draws, 0.975),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092180)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan_root = args.root / "plan-only-001"
    analysis_path = plan_root / "ANALYSIS.json"
    analysis = json.loads(analysis_path.read_text())
    episodes = []
    for name in glob.glob(str(plan_root / "episodes" / "*.json")):
        row = json.loads(Path(name).read_text())
        if row.get("mode") != "plan_only":
            continue
        policy = row["policy"]
        root_call = args.root / f"fresh-contract-{policy}-001" / "calls" / (
            row["source_root_call_id"] + ".json"
        )
        source_episode = args.root / f"fresh-contract-{policy}-001" / "episodes" / (
            row["source_episode_id"] + ".json"
        )
        plan = json.loads(json.loads(root_call.read_text())["text"])["subquestions"]
        row["n_subquestions"] = len(plan)
        row["factual"] = json.loads(source_episode.read_text())
        episodes.append(row)
    result = {
        "schema": "helper-gating-plan-length-observational-v1",
        "rule": "execute iff n_subquestions > 2",
        "seed": args.seed,
        "source_sha256": {str(analysis_path): sha(analysis_path)},
    }
    result["policies"] = {}
    for policy in ("sft", "rl"):
        rows = [row for row in episodes if row["policy"] == policy]
        if len(rows) != 128:
            raise ValueError((policy, len(rows)))
        for row in rows:
            row["plan"] = int(row["new"]["correct"])
            row["execute"] = int(row["old"]["correct"])
            row["rule"] = row["execute"] if row["n_subquestions"] > 2 else row["plan"]
            row["oracle"] = max(row["plan"], row["execute"])
        costs = {}
        for name, choose in {"plan_only": False, "execute": True, "rule": None}.items():
            total = Counter()
            for row in rows:
                execute = choose if choose is not None else row["n_subquestions"] > 2
                cost = (
                    row["factual"]["deployed_cost"]
                    if execute
                    else row["hypothetical_deployed_cost"]
                )
                total.update(cost)
            costs[name] = {
                key: total[key] for key in ("calls", "prompt_tokens", "completion_tokens")
            }
            costs[name]["total_tokens"] = (
                costs[name]["prompt_tokens"] + costs[name]["completion_tokens"]
            )
        clusters = analysis["groups"][policy]["component_clusters"]
        result["policies"][policy] = {
            "planned": len(rows),
            "plan_length_counts": dict(Counter(row["n_subquestions"] for row in rows)),
            "correct": {
                name: sum(row[name] for row in rows)
                for name in ("plan", "execute", "rule", "oracle")
            },
            "plan_execute_outcomes": {
                f"plan_{plan}_execute_{execute}": count
                for (plan, execute), count in Counter(
                    (row["plan"], row["execute"]) for row in rows
                ).items()
            },
            "rule_vs_plan_when_execute": {
                f"plan_{plan}_execute_{execute}": count
                for (plan, execute), count in Counter(
                    (row["plan"], row["execute"])
                    for row in rows
                    if row["n_subquestions"] > 2
                ).items()
            },
            "rule_minus_plan_component_bootstrap": bootstrap(rows, clusters, args.seed),
            "costs_root_once": costs,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
