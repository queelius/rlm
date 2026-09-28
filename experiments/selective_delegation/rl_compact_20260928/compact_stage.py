"""Exactly four parent-dispatched GPU stages; compact warm readout is independent."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import compact_common as c


def resolve(kind):
    if kind == "train":
        output, batch = c.STUDY / "train-0001", c.STUDY / "collect-0001"
        summary = c.read(batch / "SUMMARY.json") if (batch / "SUMMARY.json").exists() else {}
        if not summary.get("complete") or summary.get("failure"):
            return output, None, "No complete compact collection; no partial-batch optimizer update"
        return output, [sys.executable, str(Path(__file__)), "--execute", "train"], None
    phase = "collect" if kind == "collect" else "readout"
    checkpoint = str(c.WARM) if kind in ("collect", "warm") else c.endpoint()
    output = (
        c.STUDY
        / {"collect": "collect-0001", "warm": "readout-warm", "updated": "readout-0001"}[kind]
    )
    if not checkpoint:
        return output, None, "No usable committed compact endpoint; updated readout remains unknown"
    return (
        output,
        [
            sys.executable,
            str(Path(__file__)),
            "--execute",
            phase,
            "--checkpoint",
            checkpoint,
            "--output",
            str(output),
        ],
        None,
    )


def main(args):
    if args.execute:
        if args.execute == "train":
            local = SimpleNamespace(
                collection=c.STUDY / "collect-0001",
                output=c.STUDY / "train-0001",
                hours=2,
                prepare_only=False,
            )
            c.load_legacy("train.py").run(local)
        else:
            local = SimpleNamespace(
                mode=c.MODE,
                phase=args.execute,
                update=1,
                checkpoint=args.checkpoint,
                output=args.output,
                hours=2.5 if args.execute == "collect" else 1.5,
                prepare_only=False,
            )
            c.load_legacy("collect.py").run(local)
        return
    output, argv, reason = resolve(args.kind)
    if argv is None:
        c.skip(output, reason)
        print(json.dumps(dict(skipped=True, reason=reason, GPU_loaded=False)))
    else:
        if not (c.WARM / "COMMIT.json").exists():
            reason = "Compact SFTcp23 absent; no real warm endpoint to bind"
            c.skip(output, reason)
            print(json.dumps(dict(skipped=True, reason=reason, GPU_loaded=False)))
            return
        os.execv(sys.executable, argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kind", choices=("collect", "train", "warm", "updated"))
    group.add_argument("--execute", choices=("collect", "train", "readout"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
