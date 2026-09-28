"""Independent replay of saved reordered rows, without calling the scheduling policy."""

import argparse
import json
import sys
from collections import defaultdict

import prepare


def audit(root) -> dict:
    reference, bridge = prepare.load_source()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        reference.BASE, local_files_only=True, trust_remote_code=False
    )
    world = bridge.load_world()
    original = [
        json.loads(line) for line in (prepare.CORRECTED / "rows.jsonl").read_text().splitlines()
    ]
    tasks = {
        t["id"]: t
        for t in map(json.loads, (prepare.CORRECTED / "tasks.jsonl").read_text().splitlines())
    }
    result = {}
    for mode in prepare.MODES:
        directory = root / mode
        manifest = json.loads((directory / "MANIFEST.json").read_text())
        prepare.require(prepare.sha(directory / "rows.jsonl") == manifest["rows_sha256"], "rows")
        rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines()]
        by_task = defaultdict(list)
        for expected, row in zip(original, rows, strict=True):
            prepare.require(row["task_id"] == expected["task_id"], "task presentation order")
            prepare.require(row["target"] == expected["target"], "target presentation order")
            prepare.require(
                [t for t in row["labels"] if t != -100]
                == [t for t in expected["labels"] if t != -100],
                "labels",
            )
            by_task[row["task_id"]].append(row)
        queries = 0
        for task_id, trace in by_task.items():
            task = tasks[task_id]
            frame = bridge.Frame(
                world,
                dict(task["misc"]["initial_inventory"]),
                task["misc"]["target_items"],
                bridge.Budget(96, 8192),
                0,
            )
            history = []
            visible = set(frame.targets) | set(frame.inventory)
            for row in sorted(trace, key=lambda r: r["step"]):
                action = json.loads(row["target"])
                prompt = bridge.public_prompt(frame, history, goal=task["goal"])
                encoded = reference.encode_row(prompt, action, tokenizer)
                prepare.require(all(row[k] == value for k, value in encoded.items()), "prompt/mask")
                if action["action"] == "get_info":
                    prepare.require(set(action["items"]).issubset(visible), "hidden query name")
                    queries += 1
                frame.budget.charge(encoded["target_tokens"])
                feedback = frame.apply(action)
                prepare.require(feedback == row["feedback"], "native feedback differs")
                history.append({"action": action, "feedback": feedback})
                if action["action"] == "get_info":
                    for info in feedback:
                        visible.add(info["item"])
                        for recipe in info["recipes"]:
                            visible.update(recipe["ingredients"])
                visible.update(frame.inventory)
            prepare.require(frame.finished and frame.score()[0] == 1, "native completion")
        result[mode] = {
            "native_successful_tasks": len(by_task),
            "reconstructed_prompt_mask_rows": len(rows),
            "visible_queries": queries,
            "all_target_and_label_tokens_paired_by_row": True,
            "rows_sha256": manifest["rows_sha256"],
        }
    sys.path.insert(0, str(prepare.ROOT / "source-048-textcraft-action-sft"))
    from train_planner import epoch_order

    result["paired_batch_target_tokens"] = {
        str(seed): [
            sum(original[i]["target_tokens"] for i in epoch_order(366, seed, 0)[start : start + 16])
            for start in range(0, 366, 16)
        ]
        for seed in (2026092208, 2026092291)
    }
    result["method"] = (
        "Saved-row native replay and prompt/mask reconstruction; scheduler not called."
    )
    prepare.save(root / "CPU-AUDIT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=prepare.Path, default=prepare.OUTPUT)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
