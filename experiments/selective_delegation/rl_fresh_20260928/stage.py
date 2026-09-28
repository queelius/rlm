"""Resolve one prospectively fixed stage, preserving skips after an unusable boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import fresh_common as f


def endpoint(directory: Path) -> str | None:
    path = directory / "SUMMARY.json"
    if not path.exists():
        return None
    summary = f.read(path)
    return summary.get("endpoint") if summary.get("endpoint_usable") else None


def resolve(args) -> tuple[Path, list[str] | None, str | None]:
    study = args.study
    suffix = f"{args.update:04d}"
    if args.kind == "sft":
        out = study / "sft/train-0001"
        return out, [sys.executable, str(f.HERE / "extra_sft.py"), "--output", str(out)], None
    if args.kind == "collect":
        out = study / args.mode / f"collect-{suffix}"
        checkpoint = (
            str(f.WARM)
            if args.update == 1
            else endpoint(study / args.mode / f"train-{args.update - 1:04d}")
        )
        if not checkpoint:
            return out, None, "Previous update was absent, flat-reward, failed, or uncommitted"
        return (
            out,
            [
                sys.executable,
                str(f.HERE / "collect.py"),
                "--dataset",
                str(f.DATA / "train"),
                "--mode",
                args.mode,
                "--update",
                str(args.update),
                "--checkpoint",
                checkpoint,
                "--hours",
                "2.5",
                "--output",
                str(out),
            ],
            None,
        )
    if args.kind == "train":
        out = study / args.mode / f"train-{suffix}"
        collection = study / args.mode / f"collect-{suffix}"
        summary = collection / "SUMMARY.json"
        if not summary.exists() or not f.read(summary).get("complete"):
            return out, None, "No complete native-audited fresh collection; no partial-batch update"
        return (
            out,
            [
                sys.executable,
                str(f.HERE / "train.py"),
                "--collection",
                str(collection),
                "--hours",
                "2",
                "--output",
                str(out),
            ],
            None,
        )
    if args.kind == "readout":
        if args.actor == "warm":
            out, checkpoint = study / args.mode / "readout-warm", str(f.WARM)
        elif args.actor == "sft":
            out = study / "sft" / args.mode / "readout-0001"
            checkpoint = endpoint(study / "sft/train-0001")
        else:
            out = study / args.mode / f"readout-{suffix}"
            checkpoint = endpoint(study / args.mode / f"train-{suffix}")
        if not checkpoint:
            return out, None, "No usable committed actor endpoint; native success remains unknown"
        return (
            out,
            [
                sys.executable,
                str(f.HERE / "collect.py"),
                "--dataset",
                str(f.DATA / "diagnostic"),
                "--mode",
                args.mode,
                "--phase",
                "readout",
                "--update",
                str(args.update),
                "--hours",
                "1.5",
                "--checkpoint",
                checkpoint,
                "--output",
                str(out),
            ],
            None,
        )
    raise ValueError("undeclared stage")


def run(args):
    if args.update not in range(1, 5):
        raise ValueError("at most4 prospective fresh on-policy updates")
    out, argv, reason = resolve(args)
    if argv is None:
        record = dict(
            reason=reason,
            scientific_calls=0,
            GPU_loaded=False,
            update=args.update,
            kind=args.kind,
            mode=args.mode,
            diagnostic_outcome_selection=False,
        )
        f.c.save(out / "CONDITIONAL-SKIP.json", record)
        print(json.dumps(record), flush=True)
        return
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--kind", choices=("collect", "train", "readout", "sft"), required=True)
    parser.add_argument("--mode", choices=("raw", "binder"), default="raw")
    parser.add_argument("--actor", choices=("warm", "rl", "sft"), default="rl")
    parser.add_argument("--update", type=int, default=1)
    run(parser.parse_args())
