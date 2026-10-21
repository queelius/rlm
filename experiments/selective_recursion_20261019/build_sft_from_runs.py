"""Build SFT pairs from logged controller runs (correct runs only), one pair per controller turn.

Used for S1d: distil the untrained 4B's one-cell library calls (teaching splits only) into a smaller
caller, instead of the scripted three-cell teacher.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_train  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("runs", nargs="+")
a = p.parse_args()
n_runs = n_pairs = 0
with open(a.out, "w") as f:
    for run in a.runs:
        for line in open(Path(run) / "rows.jsonl"):
            r = json.loads(line)
            if not r.get("correct") or not r.get("trace_dir"):
                continue
            n_runs += 1
            for k, t in enumerate(rl_train.controller_turns(r["trace_dir"])):
                f.write(json.dumps(dict(task_id=r.get("task_id") or f"oolong-{r.get('id')}", turn=k, correct=True,
                                        messages=t["messages"], completion=t["completion"], source=Path(run).name)) + "\n")
                n_pairs += 1
print(f"{n_runs} correct runs, {n_pairs} pairs -> {a.out}")
