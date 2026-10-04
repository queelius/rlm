"""Ensure panels retain true RGB frame ordering and avoid unfilled replay capacity."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def test_exports_filled_first_middle_last_entries_with_channel_order(tmp_path):
    source = Path(__file__).with_name("sample_observations.py")
    assert source.exists(), "Observation panel exporter has not been implemented"
    spec = importlib.util.spec_from_file_location("sample_observations", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observations = np.zeros((4, 9, 100, 100), dtype=np.uint8)
    for index in range(3):
        for channel in range(9):
            observations[index, channel] = np.arange(100)[None, :] + 10 * index + channel
    checkpoint = tmp_path / "trusted.pt"
    torch.save(
        {
            "config": {"seed": 123, "arm": "curl"},
            "counters": {"step": 3},
            "replay": {"idx": 3, "full": False, "capacity": 4, "arrays": {"obses": observations}},
        },
        checkpoint,
    )
    output = tmp_path / "output"
    metadata = module.export(checkpoint, output)
    assert metadata["selected_indices"] == [0, 1, 2]
    assert metadata["filled_entries"] == 3
    assert metadata["observation_shape"] == [9, 100, 100]
    assert metadata["dtype"] == "uint8"
    assert all(frame["nonblank"] for entry in metadata["samples"] for frame in entry["frames"])
    panel = np.asarray(Image.open(output / "observations.png"))
    np.testing.assert_array_equal(panel[metadata["raw_row_y"], 0], [0, 1, 2])
    np.testing.assert_array_equal(panel[metadata["raw_row_y"], 100], [3, 4, 5])
    assert json.loads((output / "observations.json").read_text())["selected_indices"] == [0, 1, 2]
