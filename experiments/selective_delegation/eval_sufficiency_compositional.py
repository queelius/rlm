"""Thin fixed-profile entry point; all endpoint/native logic remains in the shared reader."""

import argparse
import json
from pathlib import Path

import eval_sufficiency_heldout as shared


def profile():
    path = Path(__file__).with_name("COMPOSITIONAL-PROFILE.json")
    value = json.loads(path.read_text())
    return {
        **value,
        "entrypoint_sha256": shared.sha(Path(__file__)),
        "profile_sha256": shared.sha(path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("cases", "warm-adapter", "rl-output", "sft-output", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument(
        "--prepare-only", "--validate-only", dest="prepare_only", action="store_true"
    )
    shared.run(parser.parse_args(), profile=profile())
