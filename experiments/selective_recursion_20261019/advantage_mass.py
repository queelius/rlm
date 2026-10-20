"""Where does the gradient on the direct route come from? Advantage mass by route, length and step.

Route inference without traces (traces are pruned after three steps): on aggregation tasks the
taught direct route is one delegated call after the peek, i.e. turns == 2 and calls == 3; lookups
(`lookup_day`) are excluded because their taught cell has the same shape. Validated against
trace-classified routes where traces still exist (reported as a confusion count).
Usage: python advantage_mass.py RUN_DIR [RUN_DIR ...] [--max-step N]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import route_analysis as ra


def route_heuristic(row: dict) -> str | None:
    if row.get("question_type") == "lookup_day":
        return None
    if row.get("error"):
        return None
    return "direct" if (row.get("turns") == 2 and row.get("calls") == 3) else "decompose"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--max-step", type=int, default=30)
    args = p.parse_args()
    for run in args.runs:
        rows = []
        for f in sorted((run / "rollouts").glob("step-*.jsonl")):
            rows += [json.loads(line) for line in f.read_text().splitlines()]
        rows = [r for r in rows if r["step"] <= args.max_step]
        confusion = collections.Counter()
        for r in rows:
            h = route_heuristic(r)
            if h is None or not r.get("trace_dir"):
                continue
            t = ra.route_of(r["trace_dir"])
            if t in ("direct", "decompose"):
                confusion[(t, h)] += 1
        mass = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0, 0]))
        for r in rows:
            h = route_heuristic(r)
            if h != "direct":
                continue
            key = (r["source"], r["n"])
            cell = mass[key][r["step"]]
            cell[0] += 1
            cell[1] += float(r.get("advantage") or 0.0)
            cell[2] += bool(r.get("correct"))
        print(f"== {run.name}  (heuristic vs trace where traces exist: {dict(confusion)})")
        print("   direct-route rollouts: count / summed advantage / correct, by length and step")
        for key in sorted(mass, key=str):
            total_n = sum(v[0] for v in mass[key].values())
            total_a = sum(v[1] for v in mass[key].values())
            steps = " ".join(
                f"s{s}:{v[0]}/{v[1]:+.2f}/{v[2]}" for s, v in sorted(mass[key].items())
            )
            print(f"   {key[0]:<8} {key[1]:>5}: n={total_n:<3} sumA={total_a:+.2f}  {steps}")
        all_direct = [r for r in rows if route_heuristic(r) == "direct"]
        last = max((r["step"] for r in all_direct), default=None)
        print(f"   total direct rollouts {len(all_direct)}, summed advantage "
              f"{sum(float(r.get('advantage') or 0) for r in all_direct):+.2f}, last step with any: {last}")


if __name__ == "__main__":
    main()
