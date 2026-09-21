"""Accepted canonical Hotpot128 replication after authenticated RL24 readout release."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import ANALYSIS_PYTHON, TRAIN_PYTHON, released


def commands(root):
    source, cases = root / "source-021", root / "hotpot-fresh-inputs-001/cases.jsonl"
    jobs = []
    for name, condition in (("planner", "sft"), ("direct", "base")):
        output = root / f"hotpot-fresh-{name}-001"
        collect = [
            TRAIN_PYTHON,
            str(source / "eval_planner.py"),
            "--cases",
            str(cases),
            "--adapter",
            str(root / "planner-sft-001/checkpoint-0048"),
            "--output",
            str(output),
            "--split",
            "transfer",
            "--start",
            "0",
            "--limit",
            "128",
            "--repeats",
            "2",
            "--execution",
            "isolated",
            "--hours",
            "1",
        ]
        if name == "planner":
            collect += [
                "--trained-condition",
                "sft",
                "--trained-only",
                "--helper-contract",
                "trained_helper",
                "--helper-adapter",
                str(root / "helper-sft-001/checkpoint-0036"),
            ]
        else:
            collect += ["--mode", "direct"]
        score = [
            ANALYSIS_PYTHON,
            str(source / "score_hotpot.py"),
            "--cases",
            str(cases),
            "--evaluation",
            str(output),
            "--output",
            str(root / f"analysis-hotpot-fresh-{name}-001"),
        ]
        jobs.append((condition, collect, score))
    return jobs


def complete_release(output, condition, count):
    if not released(output):
        return False
    summary = json.loads((output / "SUMMARY.json").read_text())
    group = summary.get("conditions", {}).get(condition, {})
    if (
        group.get("planned_episodes") != count
        or group.get("recorded_episodes") != count
        or group.get("missing_episodes") != 0
        or summary.get("unresolved_started_attempts") != 0
    ):
        raise RuntimeError("released predecessor incomplete; no silent advance: " + str(output))
    return True


def refuse_existing_attempt(root):
    for stem in (
        "hotpot-fresh-planner-001",
        "hotpot-fresh-direct-001",
        "analysis-hotpot-fresh-planner-001",
        "analysis-hotpot-fresh-direct-001",
    ):
        if (root / stem).exists():
            raise RuntimeError("existing attempt requires explicit review: " + stem)


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "HOTPOT-FRESH-DECISION-001.json").read_text())
    if decision.get("status") != "accepted" or decision.get("source") != "source-021":
        raise RuntimeError("accepted sealed Hotpot fresh decision required")
    if not decision.get("sha256"):
        raise RuntimeError("accepted source/input hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("accepted source/input changed: " + relative)
    print(
        json.dumps(
            {
                "validated": time.time(),
                "decision": decision["schema"],
                "maximum_calls": 2816,
                "maximum_gpu_seconds": 7200,
                "native_muSiQue_scores": "diagnostic only; official Hotpot regrade required",
            }
        ),
        flush=True,
    )
    refuse_existing_attempt(root)
    if args.validate_only:
        return 0
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 7800)
    while True:
        if time.time() >= deadline:
            raise RuntimeError("predecessor did not release within accepted wait/lease bound")
        if complete_release(root / "fresh-contract-rl24-001", "rl", 128):
            break
        time.sleep(5)
    refuse_existing_attempt(root)
    completed_scores, events, failed = [], [], False
    for condition, collect, score in commands(root):
        print(json.dumps({"started": time.time(), "command": collect}), flush=True)
        result = subprocess.run(collect, check=False)
        output = Path(collect[collect.index("--output") + 1])
        event = {"output": str(output), "returncode": result.returncode, "ended": time.time()}
        if (output / "PLAN.json").exists():
            completed_scores.append(score)
        try:
            if result.returncode or not complete_release(output, condition, 256):
                raise RuntimeError("collector failed/capped or owner not released")
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            event["error"] = str(exc)
            failed = True
        events.append(event)
        print(json.dumps(event), flush=True)
        if failed:
            break
    # Keep the architectures separate: strict scorer comparison mode is intentionally unused.
    for score in completed_scores:
        print(json.dumps({"started_cpu_regrade": time.time(), "command": score}), flush=True)
        result = subprocess.run(score, check=False)
        failed |= result.returncode != 0
        events.append(
            {
                "official_regrade": score[score.index("--output") + 1],
                "returncode": result.returncode,
                "ended": time.time(),
            }
        )
    output = root / "hotpot-fresh-planner-001"
    output.mkdir(exist_ok=True)
    receipt = {
        "status": "failed_or_incomplete" if failed else "collectors_and_regrades_returned",
        "events": events,
        "ended": time.time(),
        "native_muSiQue_scores": "diagnostic only; use official Hotpot reports",
        "paired_architecture_analysis": "separate completed-result task, not performed here",
    }
    with (output / "QUEUE-RESULT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
