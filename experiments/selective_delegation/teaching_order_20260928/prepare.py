"""Replay matched-action teachers with public-name visibility and fixed craft order."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
OUTPUT = ROOT / "textcraft-teaching-order-20260928-001"
SOURCE = ROOT / "source-047-textcraft-action-inputs"
CORRECTED = ROOT / "textcraft-quantity-matched-inputs-004"
DISCOVERY = ROOT / "textcraft-public-discovery-prototype-001"
ROWS_SHA = "24ea72cb1242f2e0d819d8fb115737864de03fb750e064f145f9ec48a245e6d6"
DISCOVERY_SHA = "dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a"
TASKS_SHA = "ed21650d6a7399201c671996c0c80c6381368d634a604525a0036de6157575a7"
ORDER_SEED = 2026092801
MODES = ("stable_visible", "random_visible")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def choose_index(
    actions: list[dict],
    pending: list[int],
    visible: set[str],
    inventory: dict,
    queried: set[str],
    rng: random.Random | None = None,
) -> int:
    """Preserve craft order; select only queries grounded in public names."""
    first_craft = next((i for i in pending if actions[i]["action"] == "craft"), None)
    if first_craft is not None:
        action = actions[first_craft]
        if action["target_item"] in queried and all(
            inventory.get(k, 0) >= n for k, n in action["ingredients"].items()
        ):
            return first_craft
    queries = [
        i
        for i in pending
        if actions[i]["action"] == "get_info" and set(actions[i]["items"]).issubset(visible)
    ]
    if queries:
        return min(queries) if rng is None else rng.choice(queries)
    if len(pending) == 1 and actions[pending[0]]["action"] == "finish":
        return pending[0]
    raise ValueError("no visible query or feasible next original craft; preserve failed attempt")


def load_source():
    sys.path.insert(0, str(SOURCE))
    import prepare_textcraft_sft as reference

    return reference, reference.bridge


def replay(task: dict, old: list[dict], public: list[dict], mode: str, world, tokenizer) -> tuple:
    reference, bridge = load_source()
    actions = [json.loads(row["target"]) for row in old]
    pending, emitted, rows, history = list(range(len(actions))), [], [], []
    targets, initial = task["misc"]["target_items"], task["misc"]["initial_inventory"]
    frame = bridge.Frame(world, dict(initial), targets, bridge.Budget(96, 8192), max_depth=0)
    visible, queried = set(targets) | set(initial), set()
    seed = ORDER_SEED + int(hashlib.sha256(task["id"].encode()).hexdigest()[:8], 16)
    rng = random.Random(seed) if mode == "random_visible" else None
    while pending:
        index = choose_index(actions, pending, visible, frame.inventory, queried, rng)
        action = actions[index]
        if action["action"] == "get_info":
            require(set(action["items"]).issubset(visible), "unobserved query name")
        prompt = bridge.public_prompt(frame, history, goal=task["goal"])
        encoded = reference.encode_row(prompt, action, tokenizer)
        require(encoded["target"] == old[index]["target"], "literal target changed")
        require(
            [t for t in encoded["labels"] if t != -100]
            == [t for t in old[index]["labels"] if t != -100],
            "supervised label token sequence changed",
        )
        require(encoded["prompt_tokens"] + frame.budget.reserve() <= 8192, "context cap")
        require(encoded["target_tokens"] <= frame.budget.reserve(), "target cap")
        frame.budget.charge(encoded["target_tokens"])
        feedback = frame.apply(action)
        require(
            not (isinstance(feedback, str) and feedback.startswith("Error:")),
            f"native action error: {feedback}",
        )
        rows.append(
            {
                "task_id": task["id"],
                "step": len(rows),
                "source_action_index": index,
                **encoded,
                "feedback": feedback,
            }
        )
        history.append({"action": action, "feedback": feedback})
        if action["action"] == "get_info":
            for info in feedback:
                queried.add(info["item"])
                visible.add(info["item"])
                for recipe in info["recipes"]:
                    visible.update(recipe["ingredients"])
        visible.update(frame.inventory)
        pending.remove(index)
        emitted.append(index)
    score, details = frame.score()
    require(score == 1 and frame.finished, "native teacher replay did not succeed")
    require(Counter(r["target"] for r in rows) == Counter(r["target"] for r in old), "multiset")
    require(
        [actions[i] for i in emitted if actions[i]["action"] == "craft"]
        == [a for a in actions if a["action"] == "craft"],
        "original craft order changed",
    )
    public_targets = [r["target"] for r in public]
    receipt = {
        "task_id": task["id"],
        "native_score": score,
        "native_details": details,
        "eligible": True,
        "finish_included": frame.finished,
        "rows": len(rows),
        "original_action_indices_in_native_order": emitted,
        "queries": sum(a["action"] == "get_info" for a in actions),
        "all_query_names_public": True,
        "same_action_and_label_multiset": True,
        "original_craft_order_preserved": True,
        "same_native_action_order_as_discovery": [r["target"] for r in rows] == public_targets,
        "same_native_action_order_as_known_recipe": emitted == list(range(len(actions))),
        "target_tokens": sum(r["target_tokens"] for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "initial_inventory": initial,
        "final_inventory": dict(frame.inventory),
        "random_seed": seed if rng else None,
    }
    # Training row index remains aligned with the known teacher. Native chronology
    # remains in step and the receipt. Source048's shuffle then pairs target sequences
    # and supervised-token denominators in every corresponding optimizer batch.
    return receipt, sorted(rows, key=lambda r: r["source_action_index"])


def prepare(output: Path) -> dict:
    require(not output.exists(), "immutable preparation directory already exists")
    for path, expected in (
        (CORRECTED / "rows.jsonl", ROWS_SHA),
        (CORRECTED / "tasks.jsonl", TASKS_SHA),
        (DISCOVERY / "rows.jsonl", DISCOVERY_SHA),
    ):
        require(sha(path) == expected, f"frozen teacher input changed: {path}")
    tasks = [json.loads(line) for line in (CORRECTED / "tasks.jsonl").read_text().splitlines()]
    old_rows = [json.loads(line) for line in (CORRECTED / "rows.jsonl").read_text().splitlines()]
    public_rows = [json.loads(line) for line in (DISCOVERY / "rows.jsonl").read_text().splitlines()]
    old, public = defaultdict(list), defaultdict(list)
    for row in old_rows:
        old[row["task_id"]].append(row)
    for row in public_rows:
        public[row["task_id"]].append(row)
    reference, bridge = load_source()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        reference.BASE, local_files_only=True, trust_remote_code=False
    )
    world = bridge.load_world()
    output.mkdir(parents=True)
    save(
        output / "PREPARE-PLAN.json",
        {
            "question": "Does public-name observability in matched-action supervision "
            "improve transfer?",
            "modes": MODES,
            "order_seed": ORDER_SEED,
            "task_count": 32,
            "rows": 366,
            "corrected_rows_sha256": ROWS_SHA,
            "tasks_sha256": TASKS_SHA,
            "discovery_rows_sha256": DISCOVERY_SHA,
            "scope": "Offline oracle action multiset; public query-name visibility repaired; "
            "not a claim of an entirely public-information teacher decision policy.",
            "training_row_order": "Original known-teacher action index; "
            "paired target labels per batch",
            "preparer_sha256": sha(Path(__file__)),
        },
    )
    manifests = {}
    for mode in MODES:
        directory = output / mode
        directory.mkdir()
        with (directory / "tasks.jsonl").open("xb") as stream:
            stream.write((CORRECTED / "tasks.jsonl").read_bytes())
        receipts, rows = [], []
        try:
            for task in tasks:
                receipt, task_rows = replay(
                    task, old[task["id"]], public[task["id"]], mode, world, tokenizer
                )
                save(directory / "tasks" / f"{task['id']}.json", receipt)
                receipts.append(receipt)
                rows.extend(task_rows)
        except Exception as exc:
            save(
                directory / "FAILED.json",
                {"error": f"{type(exc).__name__}: {exc}", "completed_tasks": len(receipts)},
            )
            raise
        require(len(rows) == 366 and len(receipts) == 32, "incomplete fixed teacher dataset")
        require(
            [r["target"] for r in rows] == [r["target"] for r in old_rows],
            "training presentation target order differs from known teacher",
        )
        require(sum(r["target_tokens"] for r in rows) == 8820, "supervised dose differs")
        with (directory / "rows.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        manifest = {
            "schema": "textcraft-visible-query-order-v1",
            "mode": mode,
            "rows": len(rows),
            "eligible_task_count": len(receipts),
            "selected_task_count": len(tasks),
            "tasks_sha256": TASKS_SHA,
            "rows_sha256": sha(directory / "rows.jsonl"),
            "model": str(reference.BASE),
            "model_manifest_sha256": sha(reference.BASE / "local-research-manifest.json"),
            "same_action_and_label_multiset_tasks": 32,
            "all_query_names_public": True,
            "original_craft_order_preserved": True,
            "training_target_order_preserved": True,
            "native_action_order_matches_discovery_tasks": sum(
                r["same_native_action_order_as_discovery"] for r in receipts
            ),
            "native_action_order_matches_known_tasks": sum(
                r["same_native_action_order_as_known_recipe"] for r in receipts
            ),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "supervised_tokens_including_eos": 8820,
            "max_prompt_plus_generation_cap": max(r["prompt_tokens"] + 256 for r in rows),
            "order_seed": ORDER_SEED,
            "source_sha256": {
                str(Path(__file__)): sha(Path(__file__)),
                str(Path(reference.__file__)): sha(Path(reference.__file__)),
                str(Path(bridge.__file__)): sha(Path(bridge.__file__)),
            },
            "trusted_source": bridge.trusted_provenance(),
            "tasks": receipts,
        }
        require(
            manifest["native_action_order_matches_discovery_tasks"] < 32,
            "entire control duplicates discovery; retain evidence and design a distinct order",
        )
        save(directory / "MANIFEST.json", manifest)
        manifests[mode] = {
            k: v for k, v in manifest.items() if k not in ("tasks", "trusted_source")
        }
    save(output / "SUMMARY.json", manifests)
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
