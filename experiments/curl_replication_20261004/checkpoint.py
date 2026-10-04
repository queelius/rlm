"""Full learning-state checkpoints for trusted, locally produced CURL runs."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

MODULE_NAMES = ("actor", "critic", "critic_target", "CURL")
OPTIMIZER_NAMES = (
    "actor_optimizer",
    "critic_optimizer",
    "encoder_optimizer",
    "cpc_optimizer",
    "log_alpha_optimizer",
)
REPLAY_ARRAY_NAMES = ("obses", "next_obses", "actions", "rewards", "not_dones")


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA checkpoint RNG requires a CUDA runtime")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    agent: Any,
    replay: Any,
    counters: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Atomically replace latest; do not copy unused multi-gigabyte replay capacity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    filled = replay.capacity if replay.full else replay.idx
    payload = {
        "schema_version": 1,
        "config": config,
        "counters": counters,
        "modules": {name: getattr(agent, name).state_dict() for name in MODULE_NAMES},
        "optimizers": {name: getattr(agent, name).state_dict() for name in OPTIMIZER_NAMES},
        "log_alpha": agent.log_alpha.detach(),
        "training": agent.training,
        "rng": capture_rng(),
        "replay": {
            "capacity": replay.capacity,
            "batch_size": replay.batch_size,
            "image_size": replay.image_size,
            "idx": replay.idx,
            "full": replay.full,
            "last_save": replay.last_save,
            "arrays": {name: getattr(replay, name)[:filled] for name in REPLAY_ARRAY_NAMES},
        },
    }
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle, pickle_protocol=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(
    path: str | Path,
    agent: Any,
    replay: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # NumPy and Python RNG require pickle; only use this with this campaign's own artifacts.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint version")
    saved = payload["replay"]
    for name in ("capacity", "batch_size", "image_size"):
        if getattr(replay, name) != saved[name]:
            raise ValueError(f"Replay {name} differs from checkpoint")
    # Copy into constructed modules/tensor: replacing objects breaks upstream parameter ties.
    for name in MODULE_NAMES:
        getattr(agent, name).load_state_dict(payload["modules"][name])
    with torch.no_grad():
        agent.log_alpha.copy_(payload["log_alpha"])
    for name in OPTIMIZER_NAMES:
        getattr(agent, name).load_state_dict(payload["optimizers"][name])
    for name in REPLAY_ARRAY_NAMES:
        array = saved["arrays"][name]
        getattr(replay, name)[: len(array)] = array
    replay.idx, replay.full, replay.last_save = saved["idx"], saved["full"], saved["last_save"]
    agent.train(payload["training"])
    restore_rng(payload["rng"])
    return payload["counters"], payload["config"]
