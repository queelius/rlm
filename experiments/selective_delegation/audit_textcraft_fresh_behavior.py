"""Receipt-bound behavior audit for completed fresh007 TextCraft arms."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import audit_textcraft_query_transfer as query

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
REPORT = ROOT / "analysis-textcraft-fresh-001.json"
INPUTS = ROOT / "textcraft-fresh-inputs-001"
ARMS = (
    ("privileged_old", ROOT / "textcraft-fresh-privileged-001", 0),
    ("public_teacher", ROOT / "textcraft-fresh-public-001", 1),
)
# Established in TEXTCRAFT-FRESH-SCOPE.md before this paired report was opened.
EXPOSED_ROOT_TASKS = {
    "textcraft_synth.val.151",
    "textcraft_synth.val.24",
    "textcraft_synth.val.462",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def exposure_summary(rows: list[dict], exposed_tasks: set[str]) -> dict:
    """Keep the complete frozen panel primary; report exact-ID subset separately."""
    def count(subset):
        return {
            "slots": len(subset),
            "old_successes": sum(row["old"] for row in subset),
            "public_successes": sum(row["public"] for row in subset),
        }

    return {
        "full_primary": count(rows),
        "identifier_absent_supplement": count(
            [row for row in rows if row["task_id"] not in exposed_tasks]
        ),
    }


def craft_error_category(feedback: str) -> str:
    """Map only the native environment's explicit craft-error phrases."""
    if "No recipe found" in feedback:
        return "no_recipe"
    if "Missing required ingredient" in feedback:
        return "missing_required"
    if "Extra ingredients not required" in feedback:
        return "extra_ingredient"
    if "Insufficient ingredients in inventory" in feedback:
        return "insufficient_inventory"
    if "Wrong amount" in feedback:
        return "wrong_amount"
    return "other_craft_error"


def task_depths(plan: dict) -> dict:
    tasks_path = INPUTS / "tasks.jsonl"
    if sha(tasks_path) != plan["tasks_sha256"]:
        raise ValueError("frozen fresh task bytes differ from PLAN")
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    return {task["id"]: task["misc"]["max_depth"] for task in tasks}


def collect(
    label: str, directory: Path, arm: dict, depths: dict
) -> tuple[list[dict], list[list[str]]]:
    plan_path = directory / "PLAN.json"
    plan = read(plan_path)
    if arm["sha256"].get(str(plan_path)) != sha(plan_path):
        raise ValueError("completed report does not bind native PLAN")
    jobs = plan["jobs"]
    if len(jobs) != 32 or {(job["task_id"], job["repeat"]) for job in jobs}.__len__() != 32:
        raise ValueError("exact 16-parent x two-repeat fresh panel required")
    rows, inventory = [], []

    def bound(path: Path) -> dict:
        digest = sha(path)
        if arm["sha256"].get(str(path)) != digest:
            raise ValueError(f"native receipt not bound by completed analysis: {path}")
        inventory.append([label, str(path), digest])
        return read(path)

    for job in jobs:
        episode = bound(directory / "episodes" / f"{job['episode_id']}.json")
        node = bound(directory / "nodes" / f"{job['episode_id']}-n0.json")
        calls = [bound(directory / "calls" / f"{call_id}.json") for call_id in episode["call_ids"]]
        if not episode["observed"] or not all(call["available"] for call in calls):
            raise ValueError("completed fresh panel unexpectedly unavailable")
        if node["call_ids"] != episode["call_ids"] or len(node["public_history"]) != len(calls):
            raise ValueError("node/call history differs")
        frame = next(
            json.loads(line)
            for line in calls[0]["request"]["prompt"].splitlines()
            if line.startswith('{"goal":')
        )
        roots = set(frame["target_items"])
        queries, invalid = [], 0
        for index, (call, history) in enumerate(zip(calls, node["public_history"], strict=True)):
            try:
                action = query.bridge.parse_action(call["text"])
            except ValueError:
                invalid += 1
                continue
            if action != history["action"]:
                raise ValueError("saved action differs from public history")
            if action["action"] != "get_info":
                continue
            feedback = history["feedback"]
            if not isinstance(feedback, list):
                raise ValueError("get_info lacks public feedback")
            missing = [
                item["item"]
                for item in feedback
                if not item["can_craft"]
                and not item["is_base"]
                and item["in_inventory"] == 0
                and item["recipes"] == []
            ]
            queries.append(
                {
                    "index": index,
                    "items": action["items"],
                    "root": bool(roots.intersection(action["items"])),
                    "nonexistent": missing,
                }
            )
        names = [name for item in queries for name in item["nonexistent"]]
        repeats = query.known_static_repeats(queries, node["public_history"])
        craft_errors = Counter(
            craft_error_category(str(history["feedback"]))
            for history in node["public_history"]
            if history.get("action", {}).get("action") == "craft"
            and not str(history.get("feedback", "")).startswith("Successfully crafted")
        )
        errors = arm["audits"][job["episode_id"]]["errors"]
        rows.append(
            {
                "episode_id": job["episode_id"],
                "task_id": job["task_id"],
                "repeat": job["repeat"],
                "seed": job["seed"],
                "depth": depths[job["task_id"]],
                "success": episode["native_score"],
                "calls": len(calls),
                "invalid_schema": invalid,
                "first_root_query": bool(queries and queries[0]["root"]),
                "ever_root_query": any(item["root"] for item in queries),
                "nonexistent_query_calls": sum(bool(item["nonexistent"]) for item in queries),
                "repeated_nonexistent_mentions": sum(
                    count - 1 for count in Counter(names).values()
                ),
                "repeat_returned_static_recipe_calls": repeats[0],
                "repeat_returned_static_recipe_mentions": repeats[1],
                "native_action_error": errors.get("native_action_error", 0),
                "rejected_action": errors.get("rejected_action", 0),
                "invalid_schema_audit": errors.get("invalid_schema", 0),
                "craft_error_categories": dict(sorted(craft_errors.items())),
            }
        )
    return rows, inventory


def summarize(rows: list[dict]) -> dict:
    keys = (
        "success",
        "calls",
        "invalid_schema",
        "first_root_query",
        "ever_root_query",
        "nonexistent_query_calls",
        "repeated_nonexistent_mentions",
        "repeat_returned_static_recipe_calls",
        "repeat_returned_static_recipe_mentions",
        "native_action_error",
        "rejected_action",
        "invalid_schema_audit",
    )
    craft_errors = Counter()
    for row in rows:
        craft_errors.update(row["craft_error_categories"])
    return {
        "episodes": len(rows),
        **{key: sum(row[key] for row in rows) for key in keys},
        "craft_error_categories": dict(sorted(craft_errors.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("immutable report exists")
    source = read(REPORT)
    if sha(REPORT) != "574b545bfb6bd7b00c3e83758d7bc4007464afca86ca9a958d4679f96995e480":
        raise ValueError("authoritative fresh comparison report changed")
    old_plan = read(ARMS[0][1] / "PLAN.json")
    depths = task_depths(old_plan)
    if len(depths) != 16:
        raise ValueError("expected 16 fresh root tasks")
    all_rows, inventory = {}, []
    for label, directory, index in ARMS:
        rows, bound = collect(label, directory, source["arms"][index], depths)
        all_rows[label] = rows
        inventory.extend(bound)
    old = {(row["task_id"], row["repeat"]): row for row in all_rows["privileged_old"]}
    public = {(row["task_id"], row["repeat"]): row for row in all_rows["public_teacher"]}
    if set(old) != set(public) or len(old) != 32:
        raise ValueError("fresh paired slots differ")
    paired = [
        {
            "task_id": key[0],
            "repeat": key[1],
            "old": old[key]["success"],
            "public": public[key]["success"],
        }
        for key in sorted(old)
    ]
    by_depth = {}
    for depth in sorted(set(row["depth"] for row in all_rows["public_teacher"])):
        values = [row for row in all_rows["public_teacher"] if row["depth"] == depth]
        by_depth[str(depth)] = {
            **summarize(values),
            "failures": sum(not row["success"] for row in values),
        }
    result = {
        "schema": "textcraft_fresh007_behavior_audit_v1",
        "authoritative_result": {"path": str(REPORT), "sha256": sha(REPORT)},
        "scope_document": {
            "path": str(Path(__file__).with_name("TEXTCRAFT-FRESH-SCOPE.md")),
            "sha256": sha(Path(__file__).with_name("TEXTCRAFT-FRESH-SCOPE.md")),
            "exposed_root_tasks": sorted(EXPOSED_ROOT_TASKS),
        },
        "arms": {
            name: {"summary": summarize(rows), "rows": rows} for name, rows in all_rows.items()
        },
        "public_error_and_failure_by_depth": by_depth,
        "exposure_split": exposure_summary(paired, EXPOSED_ROOT_TASKS),
        "native_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_sha256": {
            str(Path(__file__)): sha(Path(__file__)),
            str(Path(query.__file__)): sha(Path(query.__file__)),
        },
        "caveat": (
            "Full 16-parent/32-slot panel is primary. The 13-task exact-identifier-absent split "
            "was established before outcome reading, but remains supplemental and does not remove "
            "shared prerequisite/world exposure. Query/error counts are descriptive, not causal "
            "mechanisms."
        ),
    }
    with args.report.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
