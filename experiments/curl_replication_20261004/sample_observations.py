"""Export actual replay pixels from trusted, locally produced campaign checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def export(checkpoint: Path, output: Path) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    replay = payload["replay"]
    filled = replay["capacity"] if replay["full"] else replay["idx"]
    observations = replay["arrays"]["obses"]
    if not 0 < filled <= len(observations):
        raise ValueError("Checkpoint has no valid filled replay observations")
    indices = [0, filled // 2, filled - 1]
    channels, height, width = observations.shape[1:]
    if observations.dtype != np.uint8 or channels != 9 or min(height, width) < 84:
        raise ValueError("Expected uint8 CHW observations containing three RGB frames")
    raw_y, crop_y, cell_width = 28, height + 56, width * 3
    panel = Image.new("RGB", (cell_width * 3, height * 2 + 56), "white")
    draw = ImageDraw.Draw(panel)
    samples = []
    for column, index in enumerate(indices):
        observation = observations[index]
        frames = observation.reshape(3, 3, height, width).transpose(0, 2, 3, 1)
        offset_x, offset_y = (width - 84) // 2, (height - 84) // 2
        crops = frames[:, offset_y : offset_y + 84, offset_x : offset_x + 84]
        x = column * cell_width
        draw.text((x + 3, 5), f"Replay entry {index}: raw; oldest -> newest", fill="black")
        draw.text((x + 3, height + 33), "Center crops (evaluation view)", fill="black")
        panel.paste(Image.fromarray(np.concatenate(frames, axis=1)), (x, raw_y))
        panel.paste(Image.fromarray(np.concatenate(crops, axis=1)), (x, crop_y))
        frame_stats = [
            {
                "min": int(f.min()),
                "max": int(f.max()),
                "std": float(f.std()),
                "nonblank": bool(f.std() > 0),
            }
            for f in frames
        ]
        samples.append(
            {
                "index": index,
                "sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                "frames": frame_stats,
            }
        )
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "config": payload["config"],
        "training_step": payload["counters"]["step"],
        "filled_entries": filled,
        "selected_indices": indices,
        "observation_shape": [channels, height, width],
        "dtype": str(observations.dtype),
        "channel_slices": [[0, 3], [3, 6], [6, 9]],
        "frame_order": "oldest, middle, newest",
        "raw_row_y": raw_y,
        "crop_row_y": crop_y,
        "samples": samples,
        "selection": "First/middle/last physical replay entries; no performance selection.",
        "layout": "Three columns are replay entries. Raw frame stacks above center-crop stacks.",
        "crop_note": "Training uses random crops; this panel's center crops illustrate evaluation.",
    }
    output.mkdir(parents=True, exist_ok=True)
    panel.save(output / "observations.png")
    (output / "observations.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Trusted local checkpoint only"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.checkpoint, args.output)
    print(json.dumps({key: result[key] for key in ("filled_entries", "selected_indices", "dtype")}))
