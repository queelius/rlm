"""Small conditional flat-action SFT input prototype; freeze before native gold replay."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import prepare_textcraft as original
import textcraft_bridge as bridge

SEED = 2026092206
TRAIN = original.TASKS.with_name("textcraft_synth_train.jsonl")
TRAIN_SHA = "685823fbca90aa89e061e61fef58d0122def4221eed14cf2c8c730341363a00f"
BASE = Path(
    "/project/alex_phd/research-cache/models/"
    "Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554"
)
sha, save = original.sha, original.save


def select(rows, validation):
    ids = {r["id"] for r in validation}
    roots = {item for r in validation for item in r["misc"]["target_items"]}
    eligible = [
        r for r in rows if r["id"] not in ids and not set(r["misc"]["target_items"]) & roots
    ]
    selected, counts = [], {}
    for depth, count in ((2, 11), (3, 11), (4, 10)):
        candidates = [r for r in eligible if r["misc"]["max_depth"] == depth]
        counts[str(depth)] = len(candidates)
        if len(candidates) < count:
            raise ValueError("insufficient fixed TRAIN stratum; no replacement")
        selected.extend(
            sorted(
                candidates, key=lambda r: hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest()
            )[:count]
        )
    return selected, {
        "eligible_by_depth": counts,
        "excluded_task_or_root_count": len(rows) - len(eligible),
        "validation_task_count": len(ids),
        "validation_root_count": len(roots),
    }


def freeze(root, output):
    if output.exists():
        raise FileExistsError("immutable selection already exists")
    if sha(TRAIN) != TRAIN_SHA or sha(original.TASKS) != original.TASK_SHA:
        raise ValueError("official TRAIN/VAL identity differs")
    paths = [original.TASKS, *sorted(root.glob("*textcraft*/tasks.jsonl"))]
    validation = []
    for p in paths:
        # New prospective TRAIN outputs must not accidentally become VAL exclusions.
        if p != original.TASKS and "train" in p.parent.name:
            continue
        validation.extend(json.loads(line) for line in p.read_text().splitlines())
    rows = [json.loads(line) for line in TRAIN.read_text().splitlines()]
    selected, counts = select(rows, validation)
    output.mkdir(parents=True)
    with (output / "tasks.jsonl").open("x") as stream:
        for task in selected:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    receipt = {
        "schema": "textcraft-flat-action-sft-frozen32-v1",
        "selection_seed": SEED,
        "selected_at": time.time(),
        "selected_before_replay": True,
        "selection": "first SHA256(seed:task_id),11 depth2/11 depth3/10 depth4; "
        "exclude known VAL task IDs/root items; no outcome filter",
        "task_ids": [t["id"] for t in selected],
        "task_count": 32,
        "depths": dict(Counter(t["misc"]["max_depth"] for t in selected)),
        "split": "official TRAIN",
        "upstream_commit": original.COMMIT,
        "official_train": str(TRAIN),
        "official_train_sha256": TRAIN_SHA,
        "exclusion_inputs_sha256": {str(p): sha(p) for p in paths},
        "tasks_sha256": sha(output / "tasks.jsonl"),
        "inventory": counts,
        "preparer_sha256": sha(Path(__file__)),
        "bridge_sha256": sha(Path(bridge.__file__)),
        "trusted_source": bridge.trusted_provenance(),
        "future_training_status": "CPU inputs only; no GPU accepted",
        "limitations": "Root-disjoint TRAIN/VAL, shared recipe world and potentially "
        "shared intermediates. Gold actions are supervision, not prompts. "
        "No failed/capped selected task may be replaced.",
    }
    save(output / "SELECTION.json", receipt)
    return receipt


def encode_row(prompt, action, tokenizer):
    target = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if tokenizer.eos_token_id is None:
        raise ValueError("native EOS absent")
    return {
        "prompt": prompt,
        "target": target,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def trajectory(task, world, tokenizer, max_calls=96):
    frame = bridge.Frame(
        world,
        dict(task["misc"]["initial_inventory"]),
        task["misc"]["target_items"],
        bridge.Budget(max_calls, 8192),
        max_depth=0,
    )
    history, rows, actions = [], [], []
    for step in task["misc"]["gold_trajectory"]:
        actions.extend(
            [
                {"action": "get_info", "items": [step["target"][0]]},
                {
                    "action": "craft",
                    "ingredients": step["ingredients"],
                    "target_item": step["target"][0],
                    "output_count": step["result_count"],
                },
            ]
        )
    actions.append({"action": "finish", "message": "done"})
    status, error = "finished", None
    for action in actions:
        try:
            cap = frame.budget.reserve()
        except bridge.BudgetExceeded:
            status = "global_call_cap" if frame.budget.calls >= max_calls else "global_token_cap"
            break
        prompt = bridge.public_prompt(frame, history, goal=task["goal"])
        row = encode_row(prompt, action, tokenizer)
        if row["prompt_tokens"] + cap > 8192:
            status = "context_cap"
            break
        if row["target_tokens"] > cap:
            status = "target_output_cap"
            break
        # Teacher actions are strict exact JSON; native quantities stay unchanged.
        bridge.parse_action(row["target"])
        frame.budget.charge(row["target_tokens"])
        try:
            feedback = frame.apply(action)
        except Exception as exc:
            status, error = "native_exception", f"{type(exc).__name__}: {exc}"
            break
        rows.append({"task_id": task["id"], "step": len(rows), **row, "feedback": feedback})
        history.append({"action": action, "feedback": feedback})
        if isinstance(feedback, str) and feedback.startswith("Error:"):
            status, error = "native_action_error", feedback
            break
    score, details = frame.score() if error is None else (None, None)
    eligible = status == "finished" and frame.finished and score == 1
    return {
        "task_id": task["id"],
        "status": status,
        "error": error,
        "native_score": score,
        "native_details": details,
        "eligible": eligible,
        "rows": len(rows),
        "calls_charged": frame.budget.calls,
        "output_tokens": frame.budget.output_tokens,
        "gold_craft_steps": len(task["misc"]["gold_trajectory"]),
        "finish_included": frame.finished,
    }, rows


def replay(output):
    selection = json.loads((output / "SELECTION.json").read_text())
    if sha(output / "tasks.jsonl") != selection["tasks_sha256"]:
        raise ValueError("frozen selected tasks changed")
    if (output / "MANIFEST.json").exists() or (output / "rows.jsonl").exists():
        raise FileExistsError("immutable replay exists")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
    world = bridge.load_world()
    tasks = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
    receipts, training_rows, started = [], [], time.time()
    for task in tasks:
        receipt, rows = trajectory(task, world, tokenizer)
        save(
            output / "traces" / (hashlib.sha256(task["id"].encode()).hexdigest()[:24] + ".json"),
            {"receipt": receipt, "rows": rows},
        )
        receipts.append(receipt)
        if receipt["eligible"]:
            training_rows.extend(rows)
    with (output / "rows.jsonl").open("x") as stream:
        for row in training_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "textcraft-flat-public-action-sft-rows-v1",
        "started": started,
        "ended": time.time(),
        "selection_sha256": sha(output / "SELECTION.json"),
        "tasks_sha256": sha(output / "tasks.jsonl"),
        "selected_task_count": len(tasks),
        "eligible_task_count": sum(t["eligible"] for t in receipts),
        "tasks": receipts,
        "failed_or_capped_tasks_retained": [t["task_id"] for t in receipts if not t["eligible"]],
        "eligibility_rule": "Only complete native-successful trajectories within exact044 "
        "budgets train; all selected failures retained, no replacement",
        "rows": len(training_rows),
        "rows_sha256": sha(output / "rows.jsonl"),
        "prompt_tokens": sum(r["prompt_tokens"] for r in training_rows),
        "supervised_tokens_including_eos": sum(r["target_tokens"] for r in training_rows),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in training_rows), default=0),
        "max_target_tokens_including_eos": max(
            (r["target_tokens"] for r in training_rows), default=0
        ),
        "max_prompt_plus_generation_cap": max(
            (r["prompt_tokens"] + 256 for r in training_rows), default=0
        ),
        "model": str(BASE),
        "model_manifest_sha256": sha(BASE / "local-research-manifest.json"),
        "eos_token_id": tokenizer.eos_token_id,
        "loss_mask": "all prompt tokens -100; "
        "strict action JSON and actual EOS supervised; feedback/host metadata never targets",
        "exact_interface": "043/044 public_prompt, flat maxdepth0,96calls/8192emitted, "
        "256percall,8192context, no truncation; public get_info before every craft",
        "preparer_sha256": sha(Path(__file__)),
        "bridge_sha256": sha(Path(bridge.__file__)),
        "trusted_source": bridge.trusted_provenance(),
        "status": "CPU proposed training inputs, not trained/model sampled",
        "trace_sha256": {
            str(p.relative_to(output)): sha(p) for p in sorted((output / "traces").glob("*.json"))
        },
    }
    save(output / "MANIFEST.json", manifest)
    return {
        k: manifest[k]
        for k in (
            "selected_task_count",
            "eligible_task_count",
            "rows",
            "prompt_tokens",
            "supervised_tokens_including_eos",
            "max_prompt_plus_generation_cap",
            "rows_sha256",
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("freeze", "replay"), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.root.resolve(), args.output.resolve())
            if args.stage == "freeze"
            else replay(args.output.resolve())
        )
    )
