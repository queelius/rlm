"""Conditional two-arm TRAIN SFT and fixed-step32 readout after QAMPARI release."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

TRAIN_SOURCE = "source-030-sufficiency-sft"
EVAL_SOURCE = "source-031-sufficiency-readout"
PREDECESSOR = "qampari-reading-001"
ARMS = ("joint", "positive_only")


def commands(root, arm):
    training = root / f"sufficiency-sft-{arm}-001"
    readout = root / f"sufficiency-readout-{arm}-001"
    return (
        [
            TRAIN_PYTHON,
            str(root / TRAIN_SOURCE / "train_sufficiency.py"),
            "--prepared",
            str(root / "sufficiency-sft-inputs-draft-001" / arm),
            "--output",
            str(training),
            "--hours",
            "0.5",
            "--learning-rate",
            "0.0001",
        ],
        [
            TRAIN_PYTHON,
            str(root / EVAL_SOURCE / "eval_sufficiency.py"),
            "--cases",
            str(root / "sufficiency-inputs-001/cases.jsonl"),
            "--baseline",
            str(root / "sufficiency-001"),
            "--adapter",
            str(training / "checkpoint-0032"),
            "--arm",
            arm,
            "--output",
            str(readout),
            "--hours",
            str(1 / 6),
        ],
    )


def require_endpoint(output):
    if not released(output):
        raise RuntimeError("training owner has not released")
    terminal = json.loads(next(output.glob("TERMINAL-*.json")).read_text())
    commit = output / "checkpoint-0032/COMMIT.json"
    state = output / "checkpoint-0032/STATE.json"
    if (
        terminal.get("step") != 32
        or not terminal.get("complete")
        or terminal.get("stopping")
        or not commit.exists()
        or not state.exists()
    ):
        raise RuntimeError("incomplete training: exact committed step32 required, no substitution")
    if json.loads(commit.read_text())["step"] != 32 or json.loads(state.read_text())["step"] != 32:
        raise RuntimeError("endpoint checkpoint mismatch")


def validate(root, decision):
    if (
        decision.get("schema") != "sufficiency-training-decision-v1"
        or decision.get("source") != TRAIN_SOURCE
        or decision.get("readout_source") != EVAL_SOURCE
        or decision.get("status") not in ("proposed", "accepted")
        or decision.get("predecessor") != PREDECESSOR
    ):
        raise ValueError("recognized immutable proposal/acceptance required")
    required = {
        f"{TRAIN_SOURCE}/train_sufficiency.py",
        f"{TRAIN_SOURCE}/launch_sufficiency_training.py",
        f"{EVAL_SOURCE}/eval_sufficiency.py",
        "sufficiency-sft-inputs-draft-001/MANIFEST.json",
        "sufficiency-inputs-001/cases.jsonl",
        "sufficiency-001/PLAN.json",
    }
    for arm in ARMS:
        required.update(
            f"sufficiency-sft-inputs-draft-001/{arm}/{name}"
            for name in ("MANIFEST.json", "examples.jsonl")
        )
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("source and data hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed source/input changed: " + relative)


def require_predecessor_complete(summary):
    conditions = summary.get("conditions", {})
    if (
        summary.get("planned_episodes") != 64
        or summary.get("recorded_episodes") != 64
        or summary.get("planned_calls") != 160
        or set(conditions) != {"direct200", "map50"}
        or any(
            g.get("observed") != 32 or g.get("missing_or_unavailable") != 0
            for g in conditions.values()
        )
        or summary.get("physical_cost", {}).get("calls") != 160
        or summary.get("unresolved_starts")
    ):
        raise RuntimeError("QAMPARI predecessor incomplete; no silent advancement")


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "SUFFICIENCY-SFT-DECISION-001.json").read_text())
    validate(root, decision)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": decision["status"],
                    "training_updates": 64,
                    "new_readout_calls": 256,
                    "maximum_seconds": 4800,
                }
            )
        )
        return 0
    if decision["status"] != "accepted":
        raise RuntimeError("main acceptance required; proposal cannot launch")
    outputs = [
        root / f"sufficiency-{kind}-{arm}-001" for arm in ARMS for kind in ("sft", "readout")
    ]
    if any(list(output.glob("OWNER-*.json")) for output in outputs):
        raise RuntimeError("prior attempt requires explicit review, no automatic retry")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 5400)
    while not released(root / PREDECESSOR):
        if time.time() >= deadline:
            raise RuntimeError("predecessor wait/allocation margin exhausted")
        time.sleep(5)
    summary = json.loads((root / PREDECESSOR / "SUMMARY.json").read_text())
    require_predecessor_complete(summary)
    if time.time() >= deadline:
        raise RuntimeError("insufficient allocation for two bounded arms")
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for arm in ARMS:
        train, evaluate = commands(root, arm)
        for cmd in (train, evaluate):
            print(json.dumps({"started": time.time(), "command": cmd}), flush=True)
            result = subprocess.run(cmd, check=False, env=environment)
            print(json.dumps({"ended": time.time(), "returncode": result.returncode}), flush=True)
            if result.returncode:
                return result.returncode
            if cmd == train:
                require_endpoint(root / f"sufficiency-sft-{arm}-001")
            else:
                output = root / f"sufficiency-readout-{arm}-001"
                summary = json.loads((output / "SUMMARY.json").read_text())
                if not released(output) or summary["missing_or_inference_unavailable"]:
                    raise RuntimeError("fixed endpoint readout incomplete")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
