"""Append per-step route counts for RL rollouts whose traces still exist (traces are pruned).

Usage: python route_shares.py RUN_DIR [RUN_DIR ...]; writes RUN_DIR/route-shares.jsonl, one
record per (step, source, length) with counts per route and correct counts, skipping steps already
recorded. Idempotent; meant to run periodically while a run is in progress.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import route_analysis as ra


def main(run_dirs: list[str]) -> None:
    for run in map(Path, run_dirs):
        out = run / "route-shares.jsonl"
        done = set()
        if out.exists():
            done = {json.loads(line)["step"] for line in out.read_text().splitlines()}
        for path in sorted((run / "rollouts").glob("step-*.jsonl")):
            step = int(path.stem.split("-")[1])
            if step in done:
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if not rows or not (run / "traces" / f"step-{step:04d}").exists():
                continue
            table = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
            for row in rows:
                route = ra.route_of(row.get("trace_dir"))
                cell = table[(row["source"], row["n"])][route]
                cell[0] += 1
                cell[1] += bool(row.get("correct"))
            if any("missing" in v for v in table.values()) and len(rows) < 64:
                continue  # step still being written
            with out.open("a") as fh:
                for (source, n), routes in sorted(table.items(), key=str):
                    fh.write(
                        json.dumps(dict(step=step, source=source, n=n, routes=dict(routes))) + "\n"
                    )
            print(run.name, "step", step, "recorded")


if __name__ == "__main__":
    main(sys.argv[1:])
