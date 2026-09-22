"""Order-insensitive provenance audit for paired TextCraft SFT targets."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _canonical_target(value: str) -> str:
    return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_contract(old_plan_path: Path, new_plan_path: Path) -> dict[str, Any]:
    """Compare fields shared by the two fixed-dose SFT plans and their step-zero adapter."""
    old_plan = json.loads(old_plan_path.read_text())
    new_plan = json.loads(new_plan_path.read_text())
    fields = sorted(set(old_plan).intersection(new_plan).difference({"prepared", "rows_sha256"}))
    equal = [field for field in fields if old_plan[field] == new_plan[field]]
    different = [field for field in fields if old_plan[field] != new_plan[field]]
    old_commit = json.loads((old_plan_path.parent / "checkpoint-0000" / "COMMIT.json").read_text())
    new_commit = json.loads((new_plan_path.parent / "checkpoint-0000" / "COMMIT.json").read_text())
    old_adapter = old_commit["files"]["adapter_model.safetensors"]
    new_adapter = new_commit["files"]["adapter_model.safetensors"]
    old_actual = _sha256(old_plan_path.parent / "checkpoint-0000" / "adapter_model.safetensors")
    new_actual = _sha256(new_plan_path.parent / "checkpoint-0000" / "adapter_model.safetensors")
    bytes_match = {"old": old_actual == old_adapter, "new": new_actual == new_adapter}
    if not all(bytes_match.values()):
        raise ValueError("adapter bytes do not match COMMIT")
    return {
        "old_plan_sha256": _sha256(old_plan_path),
        "new_plan_sha256": _sha256(new_plan_path),
        "equal_plan_fields": equal,
        "different_plan_fields": different,
        "step_zero_adapter_sha256": {"old": old_adapter, "new": new_adapter},
        "step_zero_adapter_equal": old_adapter == new_adapter,
        "step_zero_adapter_bytes_match_commit": bytes_match,
    }


def _by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    return grouped


def _action_counter(rows: list[dict[str, Any]], action: str) -> Counter[str]:
    return Counter(
        _canonical_target(str(row["target"]))
        for row in rows
        if json.loads(str(row["target"]))["action"] == action
    )


def _raw_target_counter(rows: list[dict[str, Any]], action: str) -> Counter[str]:
    return Counter(
        str(row["target"])
        for row in rows
        if json.loads(str(row["target"]))["action"] == action
    )


def _unmasked_label_counter(rows: list[dict[str, Any]], action: str) -> Counter[tuple[int, ...]]:
    return Counter(
        tuple(label for label in row.get("labels", []) if label != -100)
        for row in rows
        if json.loads(str(row["target"]))["action"] == action
    )


def audit_rows(old_path: Path, new_path: Path) -> dict[str, Any]:
    """Compare target multisets by task; JSON key order and row order are irrelevant."""
    old_by_task = _by_task(_rows(old_path))
    new_by_task = _by_task(_rows(new_path))
    if set(old_by_task) != set(new_by_task):
        raise ValueError("task IDs differ")

    task_ids = sorted(old_by_task)
    initial_same = []
    initial_different = []
    action_multisets: dict[str, dict[str, list[str]]] = {}
    raw_target_multisets: dict[str, dict[str, list[str]]] = {}
    unmasked_label_multisets: dict[str, dict[str, list[str]]] = {}
    target_differences: list[dict[str, Any]] = []
    for task_id in task_ids:
        old_task, new_task = old_by_task[task_id], new_by_task[task_id]
        old_initial = next(row["prompt"] for row in old_task if row["step"] == 0)
        new_initial = next(row["prompt"] for row in new_task if row["step"] == 0)
        (initial_same if old_initial == new_initial else initial_different).append(task_id)
        for action in ("get_info", "finish", "craft"):
            counters = (
                (
                    _action_counter(old_task, action),
                    _action_counter(new_task, action),
                    action_multisets,
                ),
                (
                    _raw_target_counter(old_task, action),
                    _raw_target_counter(new_task, action),
                    raw_target_multisets,
                ),
                (
                    _unmasked_label_counter(old_task, action),
                    _unmasked_label_counter(new_task, action),
                    unmasked_label_multisets,
                ),
            )
            for old_counter, new_counter, destination in counters:
                bucket = destination.setdefault(
                    action, {"identical_tasks": [], "different_tasks": []}
                )
                comparison = "identical_tasks" if old_counter == new_counter else "different_tasks"
                bucket[comparison].append(task_id)
        old_craft = _action_counter(old_task, "craft")
        new_craft = _action_counter(new_task, "craft")
        if old_craft != new_craft:
            target_differences.append(
                {
                    "task_id": task_id,
                    "old_only": sorted((old_craft - new_craft).elements()),
                    "new_only": sorted((new_craft - old_craft).elements()),
                }
            )
    return {
        "schema": "textcraft_target_multiset_audit_v1",
        "old_rows": str(old_path),
        "new_rows": str(new_path),
        "old_rows_sha256": _sha256(old_path),
        "new_rows_sha256": _sha256(new_path),
        "task_count": len(task_ids),
        "initial_prompts": {"identical_tasks": initial_same, "different_tasks": initial_different},
        "action_multisets": action_multisets,
        "raw_target_multisets": raw_target_multisets,
        "target_differences": target_differences,
        "unmasked_label_multisets": unmasked_label_multisets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old-plan", type=Path)
    parser.add_argument("--new-plan", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit_rows(args.old, args.new)
    if (args.old_plan is None) != (args.new_plan is None):
        raise ValueError("supply both plan paths or neither")
    if args.old_plan is not None:
        report["training_contract"] = training_contract(args.old_plan, args.new_plan)
    report["script_sha256"] = _sha256(Path(__file__))
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
