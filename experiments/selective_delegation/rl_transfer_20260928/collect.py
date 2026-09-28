"""Readout only: a fixed familiar-goal RL actor through either frozen execution interface."""

import argparse
from pathlib import Path

import transfer_common as t

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", choices=("raw", "binder"), required=True)
    parser.add_argument("--mode", choices=("raw", "binder"), required=True)
    parser.add_argument("--study", type=Path, default=t.STUDY)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    t.load_collector().run(
        t.arguments(args.actor, args.mode, args.study, prepare_only=args.prepare_only)
    )
