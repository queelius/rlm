"""Build a route-balanced teaching set: on short inputs, as many decompose as direct trajectories.

Trajectories without a taught `route` (older teaching sets) count as decompose. Lengths where the
direct route was taught: event-log 8 and 32 records, OOLONG 1024 and 2048 tokens. Decompose
trajectories at those lengths are subsampled (fixed seed) to max(direct count, floor); all other
rows are kept. Only correct trajectories are written.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

SHORT = {("eventlog", 8), ("eventlog", 32), ("oolong", 1024), ("oolong", 2048)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--floor", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--consistent", action="store_true",
                   help="teach a deterministic rule: direct only at 8 records, decomposition everywhere else")
    args = p.parse_args()
    rows = []
    for index, f in enumerate(args.data):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("correct"):
                # The same task can be taught in several sets; keep each trajectory distinct.
                r["_traj"] = f"{index}:{r['task_id']}"
                rows.append(r)
    traj = collections.defaultdict(list)
    for r in rows:
        traj[r["_traj"]].append(r)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, items in traj.items():
        r = items[0]
        source = r.get("source") or ("oolong" if r["task_id"].startswith("oolong") else "eventlog")
        route = (r.get("variant") or {}).get("route", "decompose")
        groups[(source, r["n"])][route].append(key)
    rng = random.Random(args.seed)
    keep = set()
    manifest = {}
    for key, by_route in sorted(groups.items(), key=str):
        direct = by_route.get("direct", [])
        decomp = sorted(by_route.get("decompose", []))
        if args.consistent:
            if key == ("eventlog", 8):
                decomp = []  # only the direct route is taught at 8 records
            else:
                direct = []  # decomposition everywhere else
        elif key in SHORT:
            target = max(len(direct), args.floor)
            rng.shuffle(decomp)
            decomp = decomp[:target]
        keep.update(direct)
        keep.update(decomp)
        manifest[f"{key[0]}:{key[1]}"] = {"direct": len(direct), "decompose": len(decomp)}
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    with (args.out / "sft.jsonl").open("w") as fh:
        for r in rows:
            if r.pop("_traj") in keep:
                fh.write(json.dumps(r) + "\n")
                n += 1
    manifest["pairs"] = n
    manifest["sources"] = [str(f) for f in args.data]
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
