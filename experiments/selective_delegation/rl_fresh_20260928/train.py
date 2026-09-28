"""Additive fresh TRAIN manifest admission around the unchanged signed RLOO update."""

import argparse
from pathlib import Path

import fresh_common as f


def run(args):
    trainer = f.load_legacy("train.py")
    trainer.prepare = f.prepare_training
    trainer.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
