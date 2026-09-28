"""Fixed teacher-dose identities and source048 continuation seams; no GPU launch."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
OUTPUT = ROOT / "textcraft-teaching-dose-20260928-001"
SOURCE = ROOT / "source-048-textcraft-action-sft"
SEED, ROWS, LABEL_TOKENS = 2026092208, 366, 8820
ORIGINALS = {
    "discovery": (
        "textcraft-public-discovery-sft-001",
        "5a34080562a39dc92dee2a813078e730b883fb60944dad96b613598c31f9624c",
    ),
    "known_recipe": (
        "textcraft-quantity-matched-seed2026092208-001",
        "368b0b3df7367129a78760010dbaf9664f780b41e123cf8902e077453a6cbb60",
    ),
}
SOURCE_PINS = {
    "train_textcraft_sft.py": "274daede3a32aab96f3ad7914ae22b00cb9c4301efb39f457eeb8c786e6ec439",
    "train_planner.py": "9ce66e6f2bc8774bf5f4a5ae0e4934fbad5fd04da8886cb799231b8582f562f8",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


@lru_cache(maxsize=1)
def recipe():
    for filename, expected in SOURCE_PINS.items():
        require(sha(SOURCE / filename) == expected, "source048 recipe changed")
    sys.path.insert(0, str(SOURCE))
    import train_textcraft_sft

    require(Path(train_textcraft_sft.__file__).parent == SOURCE, "mixed source048 import")
    require(
        Path(train_textcraft_sft.target_loss.__code__.co_filename).parent == SOURCE,
        "mixed source048 target loss",
    )
    return train_textcraft_sft


def verify_checkpoint(path: Path) -> tuple[dict, dict]:
    commit = read(path / "COMMIT.json")
    required = {
        "STATE.json",
        "optimizer.pt",
        "rng.pt",
        "adapter_model.safetensors",
        "adapter_config.json",
    }
    require(required <= set(commit["files"]), "incomplete checkpoint commit")
    for name in required:
        require(sha(path / name) == commit["files"][name], "checkpoint checksum changed: " + name)
    state = read(path / "STATE.json")
    require(commit["step"] == state["step"], "checkpoint step mismatch")
    return state, commit


def validate_state(state: dict) -> None:
    epoch, cursor, step = (state[k] for k in ("epoch", "cursor", "step"))
    require(
        type(epoch) is int
        and 1 <= epoch <= 3
        and type(cursor) is int
        and 0 <= cursor < ROWS
        and cursor % 16 == 0
        and step == epoch * 23 + cursor // 16
        and (epoch != 3 or cursor == 0),
        "inconsistent cumulative continuation state",
    )
    if "cumulative_microbatches" in state:
        require(
            state["cumulative_microbatches"] == epoch * ROWS + cursor,
            "cumulative microbatch state mismatch",
        )


def schedule(state: dict):
    validate_state(state)
    for epoch in range(state["epoch"], 3):
        start = state["cursor"] if epoch == state["epoch"] else 0
        order = recipe().epoch_order(ROWS, SEED, epoch)
        for cursor in range(start, ROWS, 16):
            yield epoch, cursor, order[cursor : cursor + 16]


def advance(state: dict, *, rows: int, tokens: int, seconds: float) -> dict:
    validate_state(state)
    require(rows == min(16, ROWS - state["cursor"]), "wrong microbatch count")
    cursor = state["cursor"] + rows
    result = {
        **state,
        "step": state["step"] + 1,
        "epoch": state["epoch"] + int(cursor == ROWS),
        "cursor": 0 if cursor == ROWS else cursor,
        "training_seconds": state["training_seconds"] + seconds,
        "cumulative_microbatches": state["cumulative_microbatches"] + rows,
        "cumulative_target_tokens": state["cumulative_target_tokens"] + tokens,
    }
    validate_state(result)
    return result


def validate_dose(state: dict, steps: list[dict], endpoint: int) -> None:
    require(endpoint in (46, 69), "only prospective fixed endpoints46/69")
    validate_state(state)
    require(
        state["step"] == endpoint
        and state["epoch"] == endpoint // 23
        and state["cursor"] == 0
        and state.get("cumulative_microbatches") == ROWS * (endpoint // 23)
        and state.get("cumulative_target_tokens") == LABEL_TOKENS * (endpoint // 23),
        "fixed endpoint state is incomplete",
    )
    require(
        [s["step"] for s in steps] == list(range(24, endpoint + 1))
        and [s["rows"] for s in steps] == ([16] * 22 + [14]) * (endpoint // 23 - 1)
        and sum(s["target_tokens"] for s in steps) == LABEL_TOKENS * (endpoint // 23 - 1),
        "all cumulative optimizer/row/token receipts required",
    )


def restore_optimizer_rng(optimizer, checkpoint: Path, expected_step: int, *, device: str):
    import torch

    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location=device, weights_only=True)
    )
    require(
        {int(v["step"]) for v in optimizer.state.values()} == {expected_step},
        "restored Adam step mismatch",
    )
    require(
        all(
            g["lr"] == 1e-4
            and g["weight_decay"] == 0.0
            and tuple(g["betas"]) == (0.9, 0.999)
            and g["eps"] == 1e-8
            for g in optimizer.param_groups
        ),
        "restored AdamW recipe differs",
    )
    rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        require(device.startswith("cuda"), "CUDA RNG requires the assigned GPU")
        torch.cuda.set_rng_state_all(rng["cuda"])


def original(teacher: str) -> tuple[Path, dict, dict]:
    name, digest = ORIGINALS[teacher]
    directory = ROOT / name
    require(sha(directory / "PLAN.json") == digest, "original SFT PLAN changed")
    plan = read(directory / "PLAN.json")
    require(
        plan["seed"] == SEED
        and plan["planned_updates"] == 23
        and plan["dtype"] == dict(base="bfloat16", lora="float32"),
        "original teacher optimization contract differs",
    )
    prepared = Path(plan["prepared"])
    for filename, key in (
        ("rows.jsonl", "rows_sha256"),
        ("MANIFEST.json", "prepared_manifest_sha256"),
    ):
        require(sha(prepared / filename) == plan[key], "original input changed")
    state, commit = verify_checkpoint(directory / "checkpoint-0023")
    require(
        {k: state[k] for k in ("step", "epoch", "cursor")} == dict(step=23, epoch=1, cursor=0),
        "original endpoint is not complete cp23",
    )
    steps = [read(directory / "steps" / f"{i:04d}.json") for i in range(1, 24)]
    require(
        [s["rows"] for s in steps] == [16] * 22 + [14]
        and sum(s["target_tokens"] for s in steps) == LABEL_TOKENS,
        "original dose differs",
    )
    terminals = [read(path) for path in directory.glob("TERMINAL-*.json")]
    require(
        terminals and all(t["complete"] and not t["failure"] for t in terminals),
        "original owner did not finish",
    )
    losses = [s["target_nll"] for s in steps]
    receipt = dict(
        teacher=teacher,
        source=str(directory),
        plan_sha256=digest,
        checkpoint=str(directory / "checkpoint-0023"),
        checkpoint_commit_sha256=sha(directory / "checkpoint-0023/COMMIT.json"),
        checkpoint_files=commit["files"],
        state=state,
        original_steps_sha256={
            str(directory / "steps" / f"{i:04d}.json"): sha(directory / "steps" / f"{i:04d}.json")
            for i in range(1, 24)
        },
        online_minibatch_nll=losses,
        first_five_online_nll_mean=sum(losses[:5]) / 5,
        last_five_online_nll_mean=sum(losses[-5:]) / 5,
        terminal_wall_seconds=sum(t["elapsed_seconds"] for t in terminals),
        new_two_epoch_seconds_estimate=2 * state["training_seconds"]
        + sum(t["elapsed_seconds"] for t in terminals)
        - state["training_seconds"],
        loss_caveat="Changing-minibatch pre-update NLL, not fixed-endpoint teacher-forced NLL.",
    )
    return directory, plan, receipt
