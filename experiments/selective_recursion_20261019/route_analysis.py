"""Classify the route each controller took (direct / decompose / other) from its trace, by length.

The second controller cell decides the route: `model_complete(request)` is the taught direct route,
`ask_batch` is chunked decomposition, anything else (lookup filters, inspection) is `other`.
Usage: python route_analysis.py ROWS.jsonl [ROWS.jsonl ...]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def route_of(trace_dir: str | None) -> str:
    if not trace_dir:
        return "missing"
    path = Path(trace_dir) / "trace.jsonl"
    if not path.exists():
        return "missing"
    cells = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["type"] == "controller.action":
            cells.append(event["payload"].get("code", ""))
    if len(cells) < 2:
        return "short"
    code = cells[1]
    if "ask_batch" in code:
        return "decompose"
    if "model_complete(request)" in code:
        return "direct"
    return "other"


def main(paths: list[str]) -> None:
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        table = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
        for row in rows:
            route = route_of(row["trace_dir"])
            length = row.get("context_len") or row["n"]
            cell = table[length][route]
            cell[0] += 1
            cell[1] += bool(row.get("correct"))
            cell[2] += int(row.get("total_tokens") or 0)
        print(f"== {path}")
        for length in sorted(table):
            parts = []
            for route in ("direct", "decompose", "other", "short", "missing"):
                if route in table[length]:
                    n, k, tok = table[length][route]
                    parts.append(f"{route}={k}/{n}({tok // n // 1000}k)")
            print(f"  {length:>6}: " + " ".join(parts))


if __name__ == "__main__":
    main(sys.argv[1:])
