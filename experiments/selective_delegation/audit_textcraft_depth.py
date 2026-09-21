"""Additive host-only constructive depth audit; no new task selection or model calls."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def depth_row(task, panel):
    misc = task["misc"]
    # Resource units retain causal craft depth. Initial units have depth zero.
    # Consume shallowest available units first; no hidden recipe graph is needed.
    pools = defaultdict(Counter)
    for item, count in misc["initial_inventory"].items():
        pools[item][0] = count

    def consume(item, count, mutate=True):
        used = []
        for depth, available in sorted(pools[item].items()):
            n = min(count, available)
            if n:
                used.append(depth)
                if mutate:
                    pools[item][depth] -= n
                count -= n
            if not count:
                return max(used, default=0)
        raise ValueError("gold resource inventory insufficient: " + item)

    step_depths = []
    for step in misc["gold_trajectory"]:
        depth = 1 + max(consume(item, count) for item, count in step["ingredients"].items())
        pools[step["target"][0]][depth] += step["result_count"]
        step_depths.append(depth)
    target_depths = {
        item: consume(item, count + misc["initial_inventory"].get(item, 0), False)
        for item, count in misc["target_items"].items()
    }
    return {
        "panel": panel,
        "task_id": task["id"],
        "declared_depth": misc["max_depth"],
        "gold_crafts": len(step_depths),
        "query_craft_finish_rows": 2 * len(step_depths) + 1,
        "target_chain_depth": max(target_depths.values()),
        "all_gold_steps_max_chain": max(step_depths, default=0),
        "target_chain_depths": target_depths,
    }


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, output):
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError("immutable audit exists")
    rows, sources = [], {}
    for label, name in (
        ("TRAIN", "textcraft-train-inputs-001"),
        ("pilot_VAL", "textcraft-inputs-001"),
    ):
        path = root / name / "tasks.jsonl"
        sources[str(path)] = sha(path)
        selection = root / name / "SELECTION.json"
        sources[str(selection)] = sha(selection)
        for line in path.read_text().splitlines():
            rows.append(depth_row(json.loads(line), label))
    groups = {}
    for label in ("TRAIN", "pilot_VAL"):
        part = [r for r in rows if r["panel"] == label]
        groups[label] = {
            "tasks": len(part),
            "table": [
                {"declared_depth": d, "gold_crafts": n, "target_chain_depth": k, "tasks": count}
                for (d, n, k), count in sorted(
                    Counter(
                        (r["declared_depth"], r["gold_crafts"], r["target_chain_depth"])
                        for r in part
                    ).items()
                )
            ],
            "declared_depth_differs_from_constructive_chain": sum(
                r["declared_depth"] != r["target_chain_depth"] for r in part
            ),
        }
    report = {
        "schema": "textcraft-constructive-depth-audit-v1",
        "rows": rows,
        "groups": groups,
        "source_sha256": {**sources, str(Path(__file__).resolve()): sha(Path(__file__))},
        "method": "Track fungible ingredient units along the fixed gold craft sequence. "
        "Initial units depth0; each craft output depth1+maximum consumed depth; "
        "consume lowest-depth units first. Final root counts include initial count "
        "plus required net additions. Query/finish do not add craft depth.",
        "limitations": "Constructive trajectory depth under a stated allocation, not "
        "a globally minimal required chain or a new task label. Alternative recipes, "
        "skip-level edges, initial stock and gold redundancy can change effective work. "
        "No model outcomes, selection changes or rewritten inputs.",
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    lines = [
        "# TextCraft declared versus constructive crafting depth",
        "",
        report["method"],
        "",
        report["limitations"],
        "",
        "| Panel | Task | Declared depth | Gold crafts | "
        "Constructive root chain | Query/craft/finish rows |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['panel']} | {r['task_id']} | {r['declared_depth']} | {r['gold_crafts']} | "
        f"{r['target_chain_depth']} | {r['query_craft_finish_rows']} |"
        for r in rows
    ]
    with output.with_suffix(".md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root.resolve(), args.output.resolve())))
