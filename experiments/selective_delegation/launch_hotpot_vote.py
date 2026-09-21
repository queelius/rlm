"""Launch the accepted Hotpot direct-vote collector after sufficiency releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-025-hotpot-vote"


def commands(root: Path) -> tuple[list[str], list[str]]:
    output = root / "hotpot-fresh-direct-vote-001"
    common = [
        "--cases",
        str(root / "hotpot-fresh-inputs-001/cases.jsonl"),
        "--original",
        str(root / "hotpot-fresh-direct-001"),
        "--adapter",
        str(root / "planner-sft-001/checkpoint-0048"),
        "--output",
        str(output),
    ]
    return (
        [
            TRAIN_PYTHON,
            str(root / SOURCE / "hotpot_direct_vote.py"),
            *common,
            "--hours",
            str(1 / 3),
        ],
        [
            TRAIN_PYTHON,
            str(root / SOURCE / "hotpot_direct_vote.py"),
            *common,
            "--analyze",
            "--analysis-output",
            str(root / "analysis-hotpot-fresh-direct-vote-001"),
            "--planner-output",
            str(root / "hotpot-fresh-planner-001"),
        ],
    )


def _validate(root: Path, decision: dict) -> None:
    if (
        decision.get("schema") != "hotpot-direct-vote-decision-v1"
        or decision.get("source") != SOURCE
    ):
        raise RuntimeError("recognized sealed direct-vote decision required")
    if decision.get("status") not in {"proposed", "accepted"} or not decision.get("sha256"):
        raise RuntimeError("direct-vote decision lacks a status or source hashes")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("sealed source/input changed: " + relative)


def main(args) -> int:
    root = args.root.resolve()
    decision = json.loads((root / "HOTPOT-DIRECT-VOTE-DECISION-001.json").read_text())
    _validate(root, decision)
    if not args.validate_only and decision["status"] != "accepted":
        raise RuntimeError("accepted decision required; proposal cannot launch")
    output, report = (
        root / "hotpot-fresh-direct-vote-001",
        root / "analysis-hotpot-fresh-direct-vote-001",
    )
    if list(output.glob("OWNER-*.json")) or report.exists():
        raise RuntimeError("existing vote attempt requires explicit review")
    if args.validate_only:
        print(json.dumps({"validated": time.time(), "maximum_new_calls": 512}), flush=True)
        return 0
    deadline = min(time.time() + 21_600, int(os.environ["SLURM_JOB_END_TIME"]) - 1_800)
    while not released(root / "sufficiency-001"):
        if time.time() >= deadline:
            raise RuntimeError("sufficiency predecessor did not release within wait/lease bound")
        time.sleep(5)
    collect, analyze = commands(root)
    result = subprocess.run(collect, check=False)
    if result.returncode:
        return result.returncode
    return subprocess.run(analyze, check=False).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
