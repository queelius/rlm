"""Explicit stopped-RL002 cp1 qualification; never changes failed-run success flags."""

import argparse
from pathlib import Path

import psutil
import train_textcraft_matched_sft as control

ROOT = control.ROOT
RL_OUTPUT = ROOT / "textcraft-terminal-rl-002"
COMMIT_SHA = "063785496989e1cc16b46bd701a92b71d6c8c50aac1a5fde8c2ed27b66f518c7"
FAILURE = "ValueError: generation/replay discrepancy exceeded declared tolerance"


def validate_stopped(terminal, summary, state, commit):
    if (
        terminal.get("complete") is not False
        or terminal.get("endpoint_usable") is not False
        or terminal.get("failure") != FAILURE
        or summary.get("complete") is not False
        or summary.get("failure") != FAILURE
        or any(
            row.get(key) != 1
            for row in (terminal, summary)
            for key in (
                "actual_optimizer_steps",
                "committed_optimizer_steps",
                "committed_sampled_batches",
            )
        )
        or state.get("step") != 1
        or state.get("sample_cursor") != 1
        or commit.get("step") != 1
    ):
        raise ValueError("only exact stopped RL002 one-step numerical failure is qualified")


def build():
    import torch

    read, sha = control.read, control.p.campaign.sha
    owners = list(RL_OUTPUT.glob("OWNER-*.json"))
    if len(owners) != 1:
        raise ValueError("exact released owner required")
    owner = read(owners[0])
    try:
        process = psutil.Process(owner["pid"])
        if abs(process.create_time() - owner["create_time"]) < 0.01 and process.is_running():
            raise ValueError("original owner still running")
    except psutil.NoSuchProcess:
        pass
    terminal_path = owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-"))
    endpoint = RL_OUTPUT / "boundaries/sample-0001/checkpoint-0001"
    terminal, summary = read(terminal_path), read(RL_OUTPUT / "SUMMARY.json")
    state, commit = read(endpoint / "STATE.json"), read(endpoint / "COMMIT.json")
    validate_stopped(terminal, summary, state, commit)
    if sha(endpoint / "COMMIT.json") != COMMIT_SHA or terminal["endpoint"] != str(endpoint):
        raise ValueError("fixed unique checkpoint1 identity changed")
    plan = read(RL_OUTPUT / "PLAN.json")
    if (
        plan["warm"]["sha256"] != control.WARM_SHA
        or plan["learning_rate"] != 2e-5
        or plan["first_batch"]["plan_sha256"] != control.READINESS_SHA
    ):
        raise ValueError("qualified source065 contract changed")
    boundary_path = endpoint.parent / "BOUNDARY.json"
    boundary = read(boundary_path)
    if (
        boundary["commit_sha256"] != COMMIT_SHA
        or boundary["state"] != state
        or state["plan_sha256"] != sha(RL_OUTPUT / "PLAN.json")
    ):
        raise ValueError("state/boundary mismatch")
    batch_path = RL_OUTPUT / "batches/sample-0001/BATCH.json"
    batch = read(batch_path)
    if sha(batch_path) != state["batch_sha256"]:
        raise ValueError("committed batch changed")
    pins = {
        str(p): sha(p)
        for p in (
            owners[0],
            terminal_path,
            RL_OUTPUT / "SUMMARY.json",
            RL_OUTPUT / "PLAN.json",
            endpoint / "COMMIT.json",
            boundary_path,
            batch_path,
            RL_OUTPUT / "batches/sample-0002/BATCH.json",
        )
    }
    for name in (
        "STATE.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng.pt",
    ):
        p = endpoint / name
        if sha(p) != commit["files"][name]:
            raise ValueError("committed checkpoint content changed: " + name)
        pins[str(p)] = commit["files"][name]
    optimizer = torch.load(
        endpoint / "optimizer.pt", map_location="cpu", weights_only=True, mmap=True
    )
    steps = sorted({int(row["step"].item()) for row in optimizer["state"].values()})
    if steps != [1]:
        raise ValueError("actual Adam step is not exactly1")
    calls = {}
    for name, digest in batch["native_receipt_sha256"].items():
        p = Path(name)
        if p.parent.name == "calls":
            if sha(p) != digest:
                raise ValueError("native first-batch call changed")
            calls[p.stem] = read(p)
    tokens = control.credited_budget(batch["episodes"], calls, state["update"])
    if tokens != 12074 or state["update"]["nonzero_action_calls"] != 388:
        raise ValueError("predeclared actual one-step credit dose differs")
    marker = batch_path.parent / "OPTIMIZER-STEP-STARTED.json"
    replay = batch_path.parent / "TRAIN-EVAL-REPLAY.json"
    if read(marker)["previous_step"] != 0 or not read(replay)["passed"]:
        raise ValueError("first optimizer boundary not qualified")
    if (RL_OUTPUT / "batches/sample-0002/OPTIMIZER-STEP-STARTED.json").exists():
        raise ValueError("a second optimizer attempt would require separate review")
    for p in (marker, replay):
        pins[str(p)] = sha(p)
    return dict(
        schema="textcraft-stopped-one-step-amendment-v1",
        status="PROPOSED_NOT_GPU_ACCEPTED",
        rl_output=str(RL_OUTPUT),
        endpoint=str(endpoint),
        original_complete=False,
        original_endpoint_usable=False,
        original_failure=FAILURE,
        actual_optimizer_steps=1,
        credited_rl_token_budgets=[tokens],
        actual_adam_steps=steps,
        pins=pins,
        selection="Only committed nonzero checkpoint; stopped before second optimizer. "
        "Not selected by evaluation quality; does not relabel failed RL002 as success.",
        planned_eval="Frozen fresh16 x2, BF16 native inference, same warm/control contract",
        numeric_caveat="First actual gradient was qualified; later batch failed max-only "
        "cached/full BF16 check. No trainer resume or tolerance change.",
    )


def qualify(path, output):
    amendment = control.read(path)
    if output.resolve() != RL_OUTPUT or amendment != build():
        raise ValueError("explicit immutable stopped-run qualification changed")
    pins = dict(amendment["pins"])
    pins[str(path.resolve())] = control.p.campaign.sha(path)
    return control.read(RL_OUTPUT / "PLAN.json"), amendment["credited_rl_token_budgets"], pins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-amendment", type=Path, required=True)
    args = parser.parse_args()
    control.p.runtime.save(args.write_amendment, build())
