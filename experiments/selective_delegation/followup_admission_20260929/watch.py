"""Read-only first-response journal for one newly accepted follow-on receipt."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

MONITOR_SHA = "9ba54efccd15f67a55873340b645b357b5856f88f1b3e1da0e914494123b02aa"


def checked_receipt(path, expected_sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("accepted receipt changed")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "accepted" or not receipt.get("jobs"):
        raise ValueError("nonempty accepted receipt required")
    return path.resolve()


def main(args):
    path = checked_receipt(args.receipt, args.receipt_sha256)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import watch_research_20260928 as monitor

    if monitor.h.sha(Path(monitor.__file__)) != MONITOR_SHA:
        raise ValueError("reviewed read-only monitor changed")
    # Process-local selection only: the old live watcher's date glob is unchanged.
    monitor.operation_paths = lambda _root: [path]
    # The invocation identifies this wrapper; MONITOR_SHA pins the reused algorithm.
    monitor.__file__ = str(Path(__file__).resolve())
    monitor.main(
        argparse.Namespace(root=path.parent.parent, output=args.output, interval=15, once=args.once)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    main(parser.parse_args())
