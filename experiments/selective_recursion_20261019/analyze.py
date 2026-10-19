"""Tables for the selective-recursion study: accuracy and cost by policy, length and question type."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def load(paths: list[Path], label_override: dict | None = None) -> list[dict]:
    rows = []
    for path in paths:
        f = path / "rows.jsonl" if path.is_dir() else path
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            r["source"] = str(path)
            pol = r["policy"] + ("/" + r["harness"] if r.get("harness") else "")
            if r.get("controller") and r["controller"] not in ("qwen3-4b", None):
                pol += f"[{r['controller']}]"
            r["label"] = (label_override or {}).get(str(path), pol)
            rows.append(r)
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def table(rows: list[dict], by=("label", "n")) -> str:
    agg = collections.defaultdict(lambda: dict(n=0, c=0, tok=0, err=0, calls=0))
    for r in rows:
        k = tuple(r.get(b) for b in by)
        a = agg[k]
        a["n"] += 1; a["c"] += bool(r.get("correct")); a["tok"] += int(r.get("total_tokens") or 0)
        a["err"] += bool(r.get("error")); a["calls"] += int(r.get("calls") or 0)
    lines = [f"{'  '.join(by):<36} {'acc':>5} {'95% CI':>13} {'mean_tok':>9} {'calls':>5} {'cnt':>4} err"]
    for k in sorted(agg, key=lambda x: tuple(str(v) if not isinstance(v, (int, float)) else v for v in x)):
        a = agg[k]
        lo, hi = wilson(a["c"], a["n"])
        lines.append(f"{'  '.join(str(x) for x in k):<36} {a['c']/a['n']:5.2f} [{lo:4.2f},{hi:4.2f}] "
                     f"{a['tok']/a['n']:9.0f} {a['calls']/a['n']:5.1f} {a['n']:4d} {a['err']}")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--by", default="label,n")
    p.add_argument("--qt", default=None, help="restrict to one question type, or 'aggregate'/'lookup'")
    p.add_argument("--exclude-n", default="")
    args = p.parse_args()
    rows = load(args.paths)
    if args.qt == "aggregate":
        rows = [r for r in rows if r["question_type"] != "lookup_day"]
    elif args.qt == "lookup":
        rows = [r for r in rows if r["question_type"] == "lookup_day"]
    elif args.qt:
        rows = [r for r in rows if r["question_type"] == args.qt]
    if args.exclude_n:
        ex = {int(x) for x in args.exclude_n.split(",")}
        rows = [r for r in rows if r["n"] not in ex]
    print(table(rows, tuple(args.by.split(","))))
