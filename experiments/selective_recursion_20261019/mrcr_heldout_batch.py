"""Held-out check for every accepted MRCR library in the given bootstrap runs (E52b, E54 to E56): runs
`run_routine.py --family mrcr --split test` at 2 and 4 needles on each run's `accepted.py` and writes one JSON line per
run (accepted, held-out means, protocol, round and candidate of the accepted library). Correct = 0.9 or more at both.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parent.parent / ".venv" / "bin" / "python"


def heldout(module: Path, needles: int, work: Path) -> float | None:
    out = work / f"{module.parent.name}-{needles}"
    r = subprocess.run([str(PY), "run_routine.py", "--family", "mrcr", "--needles", str(needles), "--module", str(module),
                        "--entry", "answer_mrcr", "--out", str(out), "--split", "test", "--concurrency", "8"],
                       cwd=HERE, capture_output=True, text=True, timeout=1800)
    m = re.search(r"mean ([0-9.]+) n=", (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "")
    return float(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in a.runs:
        s = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
        row = dict(run=run.name, finished=s is not None, accepted=bool(s and s["accepted"]), protocol=s.get("protocol") if s else None,
                   best_round=s.get("best_round") if s else None, best_k=s.get("best_k") if s else None,
                   teaching_mean=s.get("best_mean") if s else None)
        if row["accepted"]:
            row["heldout_2"] = heldout(run / "accepted.py", 2, a.work)
            row["heldout_4"] = heldout(run / "accepted.py", 4, a.work)
            row["correct"] = all(v is not None and v >= 0.9 for v in (row["heldout_2"], row["heldout_4"]))
        else:
            row["correct"] = False
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
