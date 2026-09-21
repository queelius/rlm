"""Conditionally read base and fixed sufficiency adapters on a fresh paired panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

BASE_SOURCE = "source-024-sufficiency"
READOUT_SOURCE = "source-031-sufficiency-readout"
PREDECESSOR = "alfworld-local-reason-001"
PANEL = "sufficiency-fresh-inputs-001"
ARMS = ("base", "joint", "positive_only")
PLANNED_NEW_CALLS = 384


def commands(root: Path) -> tuple[list[str], list[str], list[str]]:
    cases = root / PANEL / "cases.jsonl"
    base = [
        TRAIN_PYTHON,
        str(root / BASE_SOURCE / "sufficiency_probe.py"),
        "--cases",
        str(cases),
        "--output",
        str(root / "sufficiency-fresh-base-001"),
        "--hours",
        str(1 / 3),
    ]
    adapters = []
    for arm in ARMS[1:]:
        adapters.append(
            [
                TRAIN_PYTHON,
                str(root / READOUT_SOURCE / "eval_sufficiency.py"),
                "--cases",
                str(cases),
                "--baseline",
                str(root / "sufficiency-fresh-base-001"),
                "--adapter",
                str(root / f"sufficiency-sft-{arm}-001/checkpoint-0032"),
                "--arm",
                arm,
                "--output",
                str(root / f"sufficiency-fresh-{arm}-001"),
                "--hours",
                str(1 / 3),
            ]
        )
    return base, *adapters


def validate(root: Path, decision: dict) -> None:
    if (
        decision.get("schema") != "sufficiency-fresh-readout-decision-v1"
        or decision.get("status") not in {"proposed", "accepted"}
        or decision.get("predecessor") != PREDECESSOR
        or decision.get("planned_new_calls") != PLANNED_NEW_CALLS
        or decision.get("seeds") != [2026092181, 2026092182]
    ):
        raise ValueError("recognized fixed two-seed sufficiency proposal required")
    required = {
        f"{PANEL}/cases.jsonl",
        f"{PANEL}/MANIFEST.json",
        f"{BASE_SOURCE}/sufficiency_probe.py",
        f"{READOUT_SOURCE}/eval_sufficiency.py",
        "sufficiency-sft-joint-001/checkpoint-0032/COMMIT.json",
        "sufficiency-sft-positive_only-001/checkpoint-0032/COMMIT.json",
    }
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("decision does not bind panel, source, and fixed checkpoints")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed source/input changed: " + relative)


def _complete(output: Path) -> bool:
    if not released(output):
        return False
    summary = json.loads((output / "SUMMARY.json").read_text())
    return (
        summary.get("planned_variant_attempts") == 128
        and summary.get("missing_or_inference_unavailable") == 0
        and summary.get("returned_valid", 0) + summary.get("returned_protocol_invalid", 0) == 128
    )


def main(args) -> int:
    root = args.root.resolve()
    decision = json.loads((root / "SUFFICIENCY-FRESH-DECISION-001.json").read_text())
    validate(root, decision)
    if args.validate_only:
        print(json.dumps({"status": decision["status"], "planned_new_calls": PLANNED_NEW_CALLS}))
        return 0
    if decision["status"] != "accepted":
        raise RuntimeError("proposal is not authorized for GPU launch")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 5400)
    while not released(root / PREDECESSOR):
        if time.time() >= deadline:
            raise RuntimeError("predecessor did not release inside launcher budget")
        time.sleep(5)
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    outputs = [root / f"sufficiency-fresh-{arm}-001" for arm in ARMS]
    if any(list(output.glob("OWNER-*.json")) for output in outputs):
        raise RuntimeError("existing readout owner requires explicit review")
    for command, output in zip(commands(root), outputs, strict=True):
        if time.time() >= deadline:
            raise RuntimeError("insufficient allocation for fixed fresh readout")
        result = subprocess.run(command, check=False, env=environment)
        if result.returncode:
            return result.returncode
        if not _complete(output):
            raise RuntimeError("incomplete fixed readout; no retry/substitution")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
