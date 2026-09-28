"""Additive fresh TRAIN collector using the unchanged native collection loop."""

import argparse
from pathlib import Path

import fresh_common as f


def run(args):
    f.load_legacy("collect.py").run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "binder"), required=True)
    parser.add_argument("--phase", choices=("collect", "readout"), default="collect")
    parser.add_argument("--update", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=f.WARM)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=2.5)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
