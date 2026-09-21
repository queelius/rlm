"""Run the accepted frozen-prefix screen after the existing diagnostic owner releases."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TRAIN_PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)
CPU_PYTHON = "/project/alex_phd/envs/prime-rl-5990b1b/bin/python"


def commands(root):
    output = root / "next-question-screen-001"
    collect = [TRAIN_PYTHON, str(root / "source-017/next_question_probe.py")]
    for flag, relative in (
        ("source-output", "helper-transfer-musique-001"),
        ("cases", "inputs-001/cases.jsonl"),
        ("root-adapter", "planner-sft-001/checkpoint-0048"),
        ("helper-adapter", "helper-sft-001/checkpoint-0036"),
        ("output", "next-question-screen-001"),
    ):
        collect += ["--" + flag, str(root / relative)]
    collect += ["--hours", "1"]
    analyze = [
        CPU_PYTHON,
        str(root / "source-017/analyze_next_question.py"),
        "--output",
        str(output),
        "--cases",
        str(root / "inputs-001/cases.jsonl"),
        "--report",
        str(root / "analysis-next-question-screen-001.json"),
    ]
    return collect, analyze


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "NEXT-QUESTION-DECISION-001.json").read_text())
    if decision["status"] != "accepted" or decision["source"] != "source-017":
        raise RuntimeError("accepted sealed screen required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("accepted source/input changed: " + relative)
    print(json.dumps({"validated": time.time(), "decision": decision["schema"]}), flush=True)
    if args.validate_only:
        return 0
    sys.path.insert(0, str(root / "source-017"))
    from launch_rl_diagnostics import released

    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 4200)
    while not released(root / "frozen-execution-001"):
        if time.time() >= deadline:
            raise RuntimeError("predecessor did not release within accepted wait bound")
        time.sleep(5)
    if list((root / "next-question-screen-001").glob("OWNER-*.json")):
        raise RuntimeError("existing screen attempt requires explicit resume review")
    collect, analyze = commands(root)
    print(json.dumps({"started": time.time(), "command": collect}), flush=True)
    result = subprocess.run(collect, check=False)
    print(json.dumps({"ended": time.time(), "returncode": result.returncode}), flush=True)
    analysis = subprocess.run(analyze, check=False)
    return int(result.returncode != 0 or analysis.returncode != 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
