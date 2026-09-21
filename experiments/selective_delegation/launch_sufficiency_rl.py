"""Conditional finite paired RL and matched-dose SFT, never implicit GPU acceptance."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released
from qampari_sampling_control import complete
from rl_sufficiency import boundary_inventory

SOURCES = ("source-037-sufficiency-rl", "source-038-sufficiency-extra-sft")
OUTPUTS = ("sufficiency-rl-001", "sufficiency-extra-sft-001")
PREDECESSOR = "qampari-sampling-control-001"
DECISIONS = ("SUFFICIENCY-RL-DECISION-001.json", "SUFFICIENCY-EXTRA-SFT-DECISION-001.json")


def commands(root):
    result = []
    for i, mode in enumerate(("rl", "sft_control")):
        cmd = [
            TRAIN_PYTHON,
            str(root / SOURCES[i] / "rl_sufficiency.py"),
            "--mode",
            mode,
            "--prepared",
            str(root / "sufficiency-rl-input-proposal-001"),
            "--adapter",
            str(root / "sufficiency-sft-joint-001/checkpoint-0032"),
            "--output",
            str(root / OUTPUTS[i]),
            "--hours",
            ("0.75", "0.5")[i],
        ]
        if i:
            cmd += ["--rl-output", str(root / OUTPUTS[0])]
        result.append(cmd)
    return result


def endpoint_action(summary):
    if summary.get("failure") or not 0 <= summary.get("actual_optimizer_steps", -1) <= 8:
        raise ValueError("failed/unknown RL requires main review")
    if not 0 <= summary.get("committed_sampled_blocks", -1) <= 8:
        raise ValueError("missing finite sampling cursor")
    return (
        "run_matched_control"
        if summary["actual_optimizer_steps"]
        else "skip_zero_update_control_and_readout"
    )


def validate(root, decision, index):
    if (
        decision.get("schema") != "paired-sufficiency-training-decision-v1"
        or decision.get("source") != SOURCES[index]
        or decision.get("output") != OUTPUTS[index]
        or decision.get("status") not in ("proposed", "accepted")
    ):
        raise ValueError("recognized immutable proposal/acceptance required")
    required = {
        f"{SOURCES[index]}/rl_sufficiency.py",
        f"{SOURCES[index]}/launch_sufficiency_rl.py",
        "sufficiency-rl-input-proposal-001/cases.jsonl",
        "sufficiency-rl-input-proposal-001/MANIFEST.json",
        "sufficiency-sft-joint-001/checkpoint-0032/COMMIT.json",
    }
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("source, input and initializer hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed source/input changed: " + relative)
    if index == 0 and not decision.get("predecessor_plan_sha256"):
        raise ValueError("predecessor PLAN must be pinned")


def main(args):
    root = args.root.resolve()
    decisions = [json.loads((root / name).read_text()) for name in DECISIONS]
    for i, decision in enumerate(decisions):
        validate(root, decision, i)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": [d["status"] for d in decisions],
                    "max_seconds": 4500,
                    "max_native_calls": 1024,
                }
            )
        )
        return 0
    if any(d["status"] != "accepted" for d in decisions):
        raise ValueError("both proposed arms require main acceptance")
    for name in OUTPUTS:
        if list((root / name).glob("OWNER-*.json")):
            raise ValueError("existing owner requires explicit review, no implicit restart")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 5100)
    prior = root / PREDECESSOR
    while not released(prior):
        if time.time() >= deadline:
            raise RuntimeError("predecessor wait/allocation margin exhausted")
        time.sleep(5)
    if hashlib.sha256((prior / "PLAN.json").read_bytes()).hexdigest() != decisions[0][
        "predecessor_plan_sha256"
    ] or not complete(json.loads((prior / "SUMMARY.json").read_text())):
        raise ValueError("036 must be authenticated complete32, no silent advance")
    if time.time() >= deadline:
        raise RuntimeError("insufficient allocation margin for both bounded arms")
    results = []
    for index, cmd in enumerate(commands(root)):
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
        output = root / OUTPUTS[index]
        if result.returncode or not released(output):
            raise RuntimeError("failed/unreleased training owner requires explicit main review")
        summary = json.loads((output / "SUMMARY.json").read_text())
        boundaries = boundary_inventory(output)
        terminal_state = boundaries[-1]["state"]
        if (
            str(boundaries[-1]["checkpoint"]) != summary["endpoint"]
            or terminal_state["step"] != summary["actual_optimizer_steps"]
            or terminal_state["sample_cursor"] != summary["committed_sampled_blocks"]
        ):
            raise ValueError("summary and committed endpoint disagree")
        results.append(
            {
                "output": OUTPUTS[index],
                "endpoint": summary["endpoint"],
                "state": terminal_state,
                "ended": time.time(),
            }
        )
        if index == 0 and endpoint_action(summary) != "run_matched_control":
            skip = {
                "reason": "zero_real_updates",
                "rl_endpoint": results[-1],
                "skip_outputs": [OUTPUTS[1], "sufficiency-heldout-readout"],
                "chain_resolved": True,
                "scientific_readout": False,
                "ended": time.time(),
            }
            with (root / "SUFFICIENCY-ZERO-UPDATE-SKIP-001.json").open("x") as stream:
                json.dump(skip, stream, indent=2)
            break
        if index and (
            not summary["matched_control_complete"]
            or terminal_state["step"] != results[0]["state"]["step"]
            or terminal_state["sample_cursor"] != results[0]["state"]["sample_cursor"]
        ):
            raise ValueError("SFT control capped/incomplete; no automatic matched readout")
    with (root / "SUFFICIENCY-TRAINING-QUEUE-RESULT-001.json").open("x") as stream:
        json.dump(
            {"results": results, "chain_resolved": True, "ended": time.time()}, stream, indent=2
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
