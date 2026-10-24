"""E49: how many labels does a noisy family need to pick a good library? OOLONG counting candidates (every bootstrap
candidate with a non-zero teaching score) were scored on the 78 held-out counting rows (`e35c_heldout.py`). Here,
choosing by the mean over k random labelled teaching rows is simulated and the pick's held-out score compared with
the best candidate's; k = 30 is the bootstrapper's own choice (all teaching rows).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

RUNS = Path("/project/alex_phd/runs/selective-recursion-20261019")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heldout-log", type=Path, required=True, help="launch.log of the e35c_heldout run (one JSON line per module)")
    p.add_argument("--draws", type=int, default=2000)
    a = p.parse_args()
    held = {}
    for line in a.heldout_log.read_text().splitlines():
        if line.startswith('{"module"'):
            r = json.loads(line)
            held[Path(r["module"]).stem] = r["mean"]
    teach = {}
    for stem in held:
        m = re.match(r"(8b|14b)-s(\d)-r(\d)c(\d+)$", stem)
        run = RUNS / f"e35c-boot-{m.group(1)}-oolong-counting-s{m.group(2)}" / "candidates.jsonl"
        for line in run.read_text().splitlines():
            c = json.loads(line)
            if c["round"] == int(m.group(3)) and c["k"] == int(m.group(4)):
                teach[stem] = {d["task_id"]: d["score"] for d in c["details"]}
    cands = sorted(held)
    rows = sorted(next(iter(teach.values())))
    best = max(held.values())
    print(f"{len(cands)} candidates; held-out mean: best {best:.3f}, median {statistics.median(held.values()):.3f}; "
          f"shipped library 0.551 on the same rows")
    full = max(cands, key=lambda c: (sum(teach[c].values()), random.random()))
    print(f"choice by all 30 teaching rows: {full} teaching {sum(teach[full].values()) / 30:.3f}, held out {held[full]:.3f}")
    rng = random.Random(0)
    out = []
    for k in (1, 3, 5, 10, 20, 30):
        picks = []
        for _ in range(a.draws):
            s = rng.sample(rows, k)
            pick = max(cands, key=lambda c: (sum(teach[c][r] for r in s), rng.random()))
            picks.append(held[pick])
        rec = dict(k=k, mean_heldout_of_pick=round(statistics.mean(picks), 3), regret=round(best - statistics.mean(picks), 3),
                   within_005_of_best=round(sum(v >= best - 0.05 for v in picks) / a.draws, 3))
        out.append(rec)
        print(f"k={k:>2}: pick's held-out mean {rec['mean_heldout_of_pick']:.3f}, regret {rec['regret']:.3f}, "
              f"within 0.05 of best {rec['within_005_of_best']:.2f}")
    (a.heldout_log.parent / "e49_label_curve.json").write_text(json.dumps(dict(held=held, curve=out), indent=1) + "\n")


if __name__ == "__main__":
    main()
