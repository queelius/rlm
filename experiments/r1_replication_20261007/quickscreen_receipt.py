"""Validate prescribed quickscreen endpoints and save one terminal arm receipt."""

import argparse
import json
import re
import time
from pathlib import Path


def endpoint(log_path: Path, native_root: Path, updates: int) -> Path:
    log = log_path.read_text(errors="replace")
    if "Traceback (most recent call last)" in log or log.count("finish learn()") != 32:
        raise ValueError("Expected32 completed collections without a training traceback")
    counters = re.findall(r"misc/policy_sgd_step['\"]?:\s*([0-9]+(?:\.[0-9]+)?)", log)
    if not counters or float(counters[-1]) != updates:
        raise ValueError(f"Expected{updates} actual optimizer updates")
    models = list(native_root.glob("debug_*/saved_models/step_00033/model.safetensors"))
    if len(models) != 1:
        raise ValueError("Expected one prescribed natural-end checkpoint33")
    monitor = models[0].parents[2] / "eval_results/33_math.json"
    if not monitor.exists() or len(json.loads(monitor.read_text())) != 64:
        raise ValueError("Missing complete64-question terminal monitor")
    return models[0].parent


def evaluation(root: Path) -> dict:
    files = list(root.glob("model_eval_out_*.json"))
    if len(files) != 1:
        raise ValueError("Expected one saved heldout128 evaluation")
    rows = json.loads(files[0].read_text())
    if len(rows) != 128 or any(
        row.get("task_name") != "math"
        or len(row.get("reward", [])) != 1
        or row["reward"][0] not in (0, 1)
        for row in rows
    ):
        raise ValueError("Expected128 MATH rows with one binary reward each")
    return {"correct": int(sum(row["reward"][0] for row in rows)), "n": 128, "path": str(files[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("endpoint", "record"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--arm", default="baseline")
    parser.add_argument("--status", choices=("completed", "failed", "capped"), default="failed")
    parser.add_argument("--phase", default="unknown")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    if args.command == "endpoint":
        print(endpoint(args.root / "training.log", args.native, args.updates))
        return
    receipt = {
        "arm": args.arm,
        "status": args.status,
        "completed": args.status == "completed",
        "phase": args.phase,
        "exit_code": args.exit_code,
        "time_unix": time.time(),
        "expected_collections": 32 if args.updates else 0,
        "expected_optimizer_updates": args.updates,
        "terminal_label": "step_00033" if args.updates else "original_base",
        "model": args.model,
        "learner_seed": 42,
        "root": str(args.root),
        "native_root": str(args.native) if args.native else None,
        "claim": "Exploratory heldout128 arm selection; development evidence, not confirmation",
    }
    if args.status == "completed":
        if args.updates:
            receipt["model"] = str(endpoint(args.root / "training.log", args.native, args.updates))
        receipt["evaluations"] = {
            "chat": evaluation(args.root / "eval-chat"),
            "raw": evaluation(args.root / "eval-raw"),
        }
    with (args.root / "result.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
