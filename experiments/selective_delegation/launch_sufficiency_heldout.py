"""Conditional held32 readout after matched training or authenticated zero-update skip."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-039-sufficiency-readout"
OUTPUT = "sufficiency-heldout-readout-001"
DECISION = "SUFFICIENCY-HELDOUT-READOUT-DECISION-001.json"
RESOLVED = "SUFFICIENCY-HELDOUT-READOUT-RESOLVED-001.json"


def read(path):
    return json.loads(path.read_text())


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "eval_sufficiency_heldout.py"),
        "--cases",
        str(root / "sufficiency-heldout-inputs-003/cases.jsonl"),
        "--warm-adapter",
        str(root / "sufficiency-sft-joint-001/checkpoint-0032"),
        "--rl-output",
        str(root / "sufficiency-rl-001"),
        "--sft-output",
        str(root / "sufficiency-extra-sft-001"),
        "--output",
        str(root / OUTPUT),
        "--hours",
        "1",
    ]


def complete(summary):
    groups = summary.get("conditions", {})
    return (
        summary.get("planned_calls") == 384
        and set(groups) == {"warm_joint32", "rl_terminal", "matched_sft_terminal"}
        and all(
            g["planned_variant_attempts"] == 128
            and g["missing_or_inference_unavailable"] == 0
            and g["returned_valid"] + g["returned_protocol_invalid"] == 128
            and g["physical_cost"]["calls"] == 128
            and g["physical_cost"]["failed_calls"] == 0
            for g in groups.values()
        )
    )


def zero_skip_matches(skip, summary):
    return (
        skip.get("reason") == "zero_real_updates"
        and skip.get("chain_resolved") is True
        and skip.get("scientific_readout") is False
        and not summary.get("failure")
        and summary.get("actual_optimizer_steps") == 0
        and skip.get("rl_endpoint", {}).get("endpoint") == summary.get("endpoint")
        and skip["rl_endpoint"]["state"]["step"] == 0
        and skip["rl_endpoint"]["state"]["sample_cursor"] == summary.get("committed_sampled_blocks")
    )


def validate(root, decision):
    if (
        decision.get("source") != SOURCE
        or decision.get("output") != OUTPUT
        or decision.get("status") not in ("proposed", "accepted")
    ):
        raise ValueError("recognized conditional decision required")
    required = {
        f"{SOURCE}/eval_sufficiency_heldout.py",
        f"{SOURCE}/launch_sufficiency_heldout.py",
        "sufficiency-heldout-inputs-003/cases.jsonl",
        "sufficiency-heldout-inputs-003/MANIFEST.json",
        "sufficiency-rl-001/PLAN.json",
        "source-037-sufficiency-rl/rl_sufficiency.py",
        "source-038-sufficiency-extra-sft/rl_sufficiency.py",
        "sufficiency-sft-joint-001/checkpoint-0032/COMMIT.json",
    }
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("source, frozen panel, warmstart and training contract hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed source/input changed: " + relative)


def main(args):
    root = args.root.resolve()
    decision = read(root / DECISION)
    validate(root, decision)
    if args.validate_only:
        print(json.dumps({"status": decision["status"], "planned_calls": 384, "maximum_hours": 1}))
        return 0
    if decision["status"] != "accepted":
        raise ValueError("main acceptance required")
    if list((root / OUTPUT).glob("OWNER-*.json")) or (root / RESOLVED).exists():
        raise ValueError("existing readout/resolution; no implicit retry")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 4200)
    skip_path = root / "SUFFICIENCY-ZERO-UPDATE-SKIP-001.json"
    zero = False
    while True:
        if time.time() >= deadline:
            raise RuntimeError("bounded wait/allocation margin exhausted")
        if skip_path.exists():
            if not released(root / "sufficiency-rl-001"):
                time.sleep(5)
                continue
            if not zero_skip_matches(
                read(skip_path), read(root / "sufficiency-rl-001/SUMMARY.json")
            ):
                raise ValueError("zero-update skip differs from released RL")
            zero = True
            break
        if released(root / "sufficiency-extra-sft-001"):
            break
        time.sleep(5)
    cmd = command(root) + (["--prepare-only"] if zero else [])
    print(json.dumps({"started": time.time(), "command": cmd}), flush=True)
    result = subprocess.run(
        cmd,
        check=False,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    if result.returncode:
        raise RuntimeError("readout failed; no automatic downstream advance")
    if zero:
        receipt = read(root / OUTPUT / "SKIPPED.json")
        if receipt["endpoint"]["step"] != 0 or receipt["scientific_readout"]:
            raise ValueError("readout skip not authenticated")
    elif not released(root / OUTPUT) or not complete(read(root / OUTPUT / "SUMMARY.json")):
        raise ValueError("readout incomplete; no automatic downstream advance")
    with (root / RESOLVED).open("x") as stream:
        json.dump(
            {
                "chain_resolved": True,
                "scientific_readout": not zero,
                "output": OUTPUT,
                "reason": "zero_real_updates" if zero else "complete384",
                "ended": time.time(),
                "decision_sha256": hashlib.sha256((root / DECISION).read_bytes()).hexdigest(),
            },
            stream,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
