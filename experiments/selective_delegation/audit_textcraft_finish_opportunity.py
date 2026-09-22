"""Public-state finish opportunity on observed044 traces; not model performance."""

import argparse
import json
from pathlib import Path

import eval_textcraft as collector


def audit(output, report):
    if report.exists():
        raise ValueError("immutable report exists")
    plan = json.loads((output / "PLAN.json").read_text())
    tasks = {
        t["id"]: t
        for t in map(json.loads, (Path(plan["prepared"]) / "tasks.jsonl").read_text().splitlines())
    }
    world, rows, hashes = collector.bridge.load_world(), [], {}
    for path in sorted((output / "episodes").glob("*.json")):
        episode = json.loads(path.read_text())
        hashes[str(path)] = collector.inputs.sha(path)
        if not episode["observed"]:
            continue
        if episode["node_count"] != 1:
            raise ValueError("this small audit supports root-only traces")
        task = tasks[episode["task_id"]]
        frame = collector.bridge.Frame(
            world,
            dict(task["misc"]["initial_inventory"]),
            task["misc"]["target_items"],
            collector.bridge.Budget(),
            max_depth=0,
        )
        node_path = output / "nodes" / (episode["episode_id"] + "-n0.json")
        node = json.loads(node_path.read_text())
        hashes[str(node_path)] = collector.inputs.sha(node_path)
        first = None
        for index, entry in enumerate(node["public_history"]):
            met = all(
                frame.inventory.get(k, 0) - frame.initial_inventory.get(k, 0) >= v
                for k, v in frame.targets.items()
            )
            # Counterfactual native finish on a copy, never modify the saved trajectory.
            candidate = collector.bridge.Frame(
                world, dict(frame.inventory), frame.targets, collector.bridge.Budget(), max_depth=0
            )
            candidate.initial_inventory = dict(frame.initial_inventory)
            candidate.apply({"action": "finish", "message": "done"})
            native_score, _ = candidate.score()
            if (native_score == 1) != met:
                raise ValueError("public net-goal criterion disagrees with native finish")
            if met and first is None:
                first = index
            if "action" in entry:
                try:
                    feedback = frame.apply(entry["action"])
                except ValueError:
                    continue
                if feedback != entry["feedback"]:
                    raise ValueError("saved public transition differs")
        rows.append(
            dict(
                episode_id=episode["episode_id"],
                policy=episode["policy"],
                observed_native_score=episode["native_score"],
                calls=episode["global_calls"],
                earliest_finish_call_index=first,
                avoidable_calls_if_finish_chosen_at_first_opportunity=(
                    episode["global_calls"] - first - 1 if first is not None else 0
                ),
            )
        )
    result = dict(
        definition="Public net-inventory criterion, independently checked by native finish on "
        "copied states. No model call, changed score, or success prediction. Observed044 only; "
        "hypothetical one finish call replaces continuation. No inference-time early-stop rule.",
        observed_episodes=len(rows),
        planned_episodes=len(plan["jobs"]),
        rows=rows,
        opportunities=sum(r["earliest_finish_call_index"] is not None for r in rows),
        observed_failures_with_opportunity=sum(
            r["earliest_finish_call_index"] is not None and r["observed_native_score"] == 0
            for r in rows
        ),
        avoidable_calls=sum(
            r["avoidable_calls_if_finish_chosen_at_first_opportunity"] for r in rows
        ),
        sha256={
            **hashes,
            str(output / "PLAN.json"): collector.inputs.sha(output / "PLAN.json"),
            str(Path(__file__)): collector.inputs.sha(Path(__file__)),
            str(Path(collector.bridge.__file__)): collector.inputs.sha(
                Path(collector.bridge.__file__)
            ),
        },
    )
    collector.save(report, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output, args.report)
