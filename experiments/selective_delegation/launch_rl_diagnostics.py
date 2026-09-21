"""Accepted frozen TRAIN-fit and execution-noise diagnostics after prior owners release."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import psutil

TRAIN_PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)
ANALYSIS_PYTHON = "/project/alex_phd/envs/prime-rl-5990b1b/bin/python"


def released(output):
    owners = list(output.glob("OWNER-*.json"))
    if not owners:
        return False
    if len(owners) != 1:
        raise RuntimeError("predecessor owner inventory needs review")
    owner = json.loads(owners[0].read_text())
    terminal = owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-"))
    if not terminal.exists():
        return False
    try:
        process = psutil.Process(owner["pid"])
        if (
            abs(process.create_time() - owner["create_time"]) < 0.01
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            return False
    except psutil.NoSuchProcess:
        pass
    receipt = json.loads(terminal.read_text())
    if receipt.get("failure") or receipt.get("stopped"):
        raise RuntimeError("predecessor failed/stopped; inspect before new owner")
    return True


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "RL-DIAGNOSTICS-DECISION-001.json").read_text())
    if decision["status"] != "accepted" or decision["source"] != "source-016":
        raise RuntimeError("accepted sealed diagnostic decision required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("accepted input/source changed: " + relative)
    print(json.dumps({"validated": time.time(), "decision": decision["schema"]}), flush=True)
    if args.validate_only:
        return 0
    predecessor = root / "direct-adapted-001"
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + 21600, lease - 600 - 5400)
    while not released(predecessor):
        if time.time() >= deadline:
            raise RuntimeError("predecessor did not release within accepted queued bound")
        time.sleep(5)
    jobs = (
        (
            "rl-trainfit-001",
            "eval_rl_trainfit.py",
            root / "rl-fullpass-001",
            "0.5",
            "analyze_rl_trainfit.py",
            "analysis-rl-trainfit-001.json",
        ),
        (
            "frozen-execution-001",
            "frozen_execution_probe.py",
            root / "rl-fullpass-001/batch-0001",
            "1",
            "analyze_frozen_execution.py",
            "analysis-frozen-execution-001.json",
        ),
    )
    analyses, failed = [], False
    for name, collector, source, hours, analyzer, report in jobs:
        output = root / name
        if output.exists():
            raise RuntimeError("existing attempt requires explicit resume review: " + name)
        command = [
            TRAIN_PYTHON,
            str(root / "source-016" / collector),
            "--source",
            str(source),
            "--cases",
            str(root / "inputs-001/cases.jsonl"),
            "--output",
            str(output),
            "--hours",
            hours,
        ]
        print(json.dumps({"started": time.time(), "command": command}), flush=True)
        result = subprocess.run(command, check=False)
        failed |= result.returncode != 0
        print(
            json.dumps({"ended": time.time(), "job": name, "returncode": result.returncode}),
            flush=True,
        )
        # The independent next GPU diagnostic need not wait for CPU analysis.
        if (output / "PLAN.json").exists():
            analyses.append(
                subprocess.Popen(
                    [
                        ANALYSIS_PYTHON,
                        str(root / "source-016" / analyzer),
                        "--output",
                        str(output),
                        "--cases",
                        str(root / "inputs-001/cases.jsonl"),
                        "--report",
                        str(root / report),
                    ]
                )
            )
    for process in analyses:
        failed |= process.wait() != 0
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
