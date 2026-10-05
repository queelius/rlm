"""Deliberately wrong key correspondence; the pinned CURL learner remains unchanged."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch


def derangement(generator: Any, size: int) -> np.ndarray:
    """Reject uniform permutations with fixed points, yielding a uniform derangement."""
    if size < 2:
        raise ValueError("Shuffled correspondence requires batch size at least two")
    positions = np.arange(size)
    while True:
        permutation = generator.permutation(size)
        if not np.any(permutation == positions):
            return permutation


class ShuffledKeys:
    """Wrap only sampled-index diagnostics and encoded keys, not labels or optimizers."""

    def __init__(self, agent: Any, replay: Any, training_seed: int):
        if replay.batch_size < 2:
            raise ValueError("Shuffled correspondence requires batch size at least two")
        self.seed = training_seed + 1000000
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self._shadow = np.random.RandomState(0)
        self._replay = replay
        self._sample = replay.sample_cpc
        self._logits = agent.CURL.compute_logits
        self.replay_indices: np.ndarray | None = None
        self.permutation: np.ndarray | None = None
        self.diagnostics: dict[str, int] = {}
        replay.sample_cpc = self.sample_cpc
        agent.CURL.compute_logits = self.compute_logits

    def sample_cpc(self) -> Any:
        # Pinned sample_cpc's first draw is these indices; shadow it without advancing globals.
        self._shadow.set_state(np.random.get_state())
        self.replay_indices = self._shadow.randint(
            0,
            self._replay.capacity if self._replay.full else self._replay.idx,
            size=self._replay.batch_size,
        )
        return self._sample()  # Original replay, three crop draws and returned tensors unchanged.

    def compute_logits(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        size = keys.shape[0]
        if self.replay_indices is None or len(self.replay_indices) != size:
            raise ValueError("Shuffled keys require the current sampled replay indices")
        self.permutation = derangement(self.rng, size)
        self.diagnostics = {
            "batch_size": size,
            "fixed_positions": int(np.count_nonzero(self.permutation == np.arange(size))),
            "duplicate_indices": size - len(np.unique(self.replay_indices)),
            "residual_same_record_matches": int(
                np.count_nonzero(self.replay_indices == self.replay_indices[self.permutation])
            ),
        }
        indices = torch.as_tensor(self.permutation, dtype=torch.long, device=keys.device)
        return self._logits(query, keys.index_select(0, indices))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "bit_generator": "PCG64",
            "state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict) or (
            state.get("schema_version") != 1
            or state.get("seed") != self.seed
            or state.get("bit_generator") != "PCG64"
        ):
            raise ValueError("Shuffled resume requires matching private permutation RNG state")
        self.rng.bit_generator.state = copy.deepcopy(state["state"])
