"""Conditional one-epoch ALFWorld action SFT after the resolved sufficiency chain."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-041b-alfworld-one-epoch-sft"
OUTPUT = "alfworld-action-sft-002"
DECISION = "ALFWORLD-ACTION-SFT-DECISION-001.json"
RESOLVED = "SUFFICIENCY-HELDOUT-READOUT-RESOLVED-001.json"
PREDECESSOR = "sufficiency-heldout-readout-001"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def command(root: Path) -> list[str]:
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "train_alfworld_sft.py"),
        "--prepared",
        str(root / "alfworld-train-sft-inputs-001"),
        "--output",
        str(root / OUTPUT),
        "--hours",
        "0.5",
        "--resume",
    ]


def complete(summary: dict) -> bool:
    groups = summary.get("conditions", {})
    return (
        summary.get("planned_calls") == 384
        and set(groups) == {"warm_joint32", "rl_terminal", "matched_sft_terminal"}
        and all(
            group["planned_variant_attempts"] == 128
            and group["missing_or_inference_unavailable"] == 0
            and group["returned_valid"] + group["returned_protocol_invalid"] == 128
            and group["physical_cost"]["calls"] == 128
            and group["physical_cost"]["failed_calls"] == 0
            for group in groups.values()
        )
    )


def zero_skip_matches(skip: dict, summary: dict) -> bool:
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


def predecessor_resolved(root: Path) -> bool:
    receipt_path = root / RESOLVED
    if not receipt_path.exists():
        return False
    receipt = read(receipt_path)
    if receipt.get("chain_resolved") is not True or receipt.get("output") != PREDECESSOR:
        return False
    output = root / PREDECESSOR
    if receipt.get("scientific_readout") is True:
        return (
            receipt.get("reason") == "complete384"
            and released(output)
            and complete(read(output / "SUMMARY.json"))
        )
    if (
        receipt.get("scientific_readout") is not False
        or receipt.get("reason") != "zero_real_updates"
    ):
        return False
    skip_path = output / "SKIPPED.json"
    zero_path = root / "SUFFICIENCY-ZERO-UPDATE-SKIP-001.json"
    if (
        not skip_path.exists()
        or not zero_path.exists()
        or not released(root / "sufficiency-rl-001")
    ):
        return False
    return zero_skip_matches(read(zero_path), read(root / "sufficiency-rl-001/SUMMARY.json"))


def validate(root: Path, decision: dict) -> None:
    if (
        decision.get("schema") != "alfworld-action-sft-decision-v1"
        or decision.get("source") != SOURCE
        or decision.get("predecessor") != RESOLVED
        or decision.get("status") not in ("proposed", "accepted")
    ):
        raise ValueError("recognized proposed/accepted action-SFT decision required")
    required = {
        f"{SOURCE}/train_alfworld_sft.py",
        f"{SOURCE}/launch_alfworld_sft.py",
        f"{SOURCE}/test_train_alfworld_sft.py",
        f"{SOURCE}/launch_rl_diagnostics.py",
        f"{OUTPUT}/PLAN.json",
        "alfworld-train-sft-inputs-001/examples.jsonl",
        "alfworld-train-sft-inputs-001/MANIFEST.json",
        "SUFFICIENCY-HELDOUT-READOUT-DECISION-001.json",
    }
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("trainer, lifecycle, frozen input, plan and predecessor hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed source/input changed: " + relative)


def main(args) -> int:
    root = args.root.resolve()
    decision = read(root / DECISION)
    validate(root, decision)
    if args.validate_only:
        print(
            json.dumps({"status": decision["status"], "planned_updates": 33, "maximum_hours": 0.5})
        )
        return 0
    if decision["status"] != "accepted":
        raise ValueError("proposal cannot launch; main acceptance required")
    output = root / OUTPUT
    if list(output.glob("OWNER-*.json")) or any(
        (output / name).exists() for name in ("steps", "checkpoint-0000", "FIRST-UPDATE.json")
    ):
        raise RuntimeError("existing scientific attempt requires explicit resume review")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 4200)
    while not predecessor_resolved(root):
        if time.time() >= deadline:
            raise RuntimeError("resolved predecessor/allocation bound exhausted")
        time.sleep(5)
    result = subprocess.run(
        command(root),
        check=False,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    terminal_ok = released(output)
    terminal = read(next(output.glob("TERMINAL-*.json"))) if terminal_ok else {}
    complete_run = (
        terminal.get("complete") is True
        and (output / "checkpoint-0033" / "COMMIT.json").exists()
        and (output / "FIRST-UPDATE.json").exists()
    )
    with (output / "QUEUE-RESULT.json").open("x") as stream:
        json.dump(
            {
                "returncode": result.returncode,
                "released": terminal_ok,
                "complete33": complete_run,
                "ended": time.time(),
                "status": "complete"
                if not result.returncode and complete_run
                else "failed_or_incomplete",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return int(bool(result.returncode) or not complete_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
