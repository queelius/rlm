"""Create the immutable all-TRAIN replay for the single known quantity correction."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
SOURCE = ROOT / "source-047-textcraft-action-inputs"
TASKS = ROOT / "textcraft-train-inputs-001/tasks.jsonl"
OLD_ROWS = ROOT / "textcraft-train-inputs-001/rows.jsonl"
TASKS_SHA = "390dff9bb19d0fe71c7bec0505c97608013c66c65aea90a821839f01b615ab30"
ROWS_SHA = "caa78390f9d4ac28e600674b26e56375203b72d8cdad1c3f9471da3fb25776a9"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct_task(task: dict) -> dict:
    """Return the sole predeclared sufficient-batch correction without mutating input."""
    fixed = copy.deepcopy(task)
    if fixed["id"] != "textcraft_synth.train.1029":
        return fixed
    first = fixed["misc"]["gold_trajectory"][0]
    expected = {
        "action": "craft",
        "target": ["t4_i1", 3],
        "ingredients": {"raw_t8": 6},
        "result_count": 12,
    }
    if first != expected:
        raise ValueError("unexpected train1029 source step")
    first.update(target=["t4_i1", 2], ingredients={"raw_t8": 4}, result_count=8)
    return fixed


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def replay(output: Path) -> None:
    if output.exists():
        raise FileExistsError("immutable quantity-matched output already exists")
    if sha(TASKS) != TASKS_SHA or sha(OLD_ROWS) != ROWS_SHA:
        raise ValueError("pinned original TRAIN input changed")
    sys.path.insert(0, str(SOURCE))
    import prepare_textcraft_sft as teacher
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        teacher.BASE, local_files_only=True, trust_remote_code=False
    )
    world = teacher.bridge.load_world()
    tasks = [json.loads(line) for line in TASKS.read_text().splitlines()]
    old_rows = [json.loads(line) for line in OLD_ROWS.read_text().splitlines()]
    corrected = [correct_task(task) for task in tasks]
    if len(tasks) != 32 or len({task["id"] for task in tasks}) != 32:
        raise ValueError("exact original TRAIN32 required")
    rows, receipts, diffs = [], [], []
    for old_task, new_task in zip(tasks, corrected, strict=True):
        old_receipt, old_trace = teacher.trajectory(old_task, world, tokenizer)
        receipt, trace = teacher.trajectory(new_task, world, tokenizer)
        if not old_receipt["eligible"] or not receipt["eligible"]:
            raise ValueError("all original/corrected native traces must be successful")
        action_diffs = sum(
            a["target"] != b["target"] for a, b in zip(old_trace, trace, strict=True)
        )
        prompt_diffs = sum(
            a["prompt"] != b["prompt"] for a, b in zip(old_trace, trace, strict=True)
        )
        expected = (1, 29) if old_task["id"] == "textcraft_synth.train.1029" else (0, 0)
        if (action_diffs, prompt_diffs) != expected:
            raise ValueError("unexpected action/history difference")
        rows.extend(trace)
        receipts.append(receipt)
        diffs.append(
            dict(
                task_id=old_task["id"],
                action_target_differences=action_diffs,
                prompt_differences=prompt_diffs,
            )
        )
    if len(rows) != 366 or len(old_rows) != 366:
        raise ValueError("whole 366-row replay required")
    output.mkdir(parents=True)
    task_bytes = "\n".join(json.dumps(task, sort_keys=True) for task in corrected) + "\n"
    row_bytes = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    (output / "tasks.jsonl").write_text(task_bytes)
    (output / "rows.jsonl").write_text(row_bytes)
    save(output / "TRACE-DIFFS.json", diffs)
    save(output / "RECEIPTS.json", receipts)
    save(
        output / "MANIFEST.json",
        dict(
            schema="textcraft-quantity-matched-privileged-replay-v1",
            base_tasks_sha256=TASKS_SHA,
            base_rows_sha256=ROWS_SHA,
            tasks_sha256=sha(output / "tasks.jsonl"),
            rows_sha256=sha(output / "rows.jsonl"),
            trace_diffs_sha256=sha(output / "TRACE-DIFFS.json"),
            receipt_sha256=sha(output / "RECEIPTS.json"),
            tasks=32,
            rows=366,
            eligible_task_count=32,
            model=str(teacher.BASE),
            model_manifest_sha256=sha(
                Path(teacher.BASE) / "local-research-manifest.json"
            ),
            correction=dict(
                task_id="textcraft_synth.train.1029",
                target_quantity=[3, 2],
                raw_t8=[6, 4],
                output_count=[12, 8],
            ),
            native_successful_tasks=32,
            action_target_differences=1,
            prompt_differences=29,
            source_sha256={
                str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                str(SOURCE / "prepare_textcraft_sft.py"): sha(SOURCE / "prepare_textcraft_sft.py"),
                str(SOURCE / "textcraft_bridge.py"): sha(SOURCE / "textcraft_bridge.py"),
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    replay(parser.parse_args().output.resolve())
