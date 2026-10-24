"""Held-out check for every accepted MRCR library in the given bootstrap runs (E52b, E54 to E56): runs
`run_routine.py --family mrcr --split test` at 2 and 4 needles on each run's `accepted.py` and writes one JSON line per
run (accepted, held-out means, protocol, round and candidate of the accepted library). Correct = 0.9 or more at both.

Means are recomputed at full precision from the per-task rows (not the rounded log line), the row count is checked
against the tasks run, and errors and exact matches (score 1.0) are reported apart from the partial-credit mean.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parent.parent / ".venv" / "bin" / "python"


def stats(rows_path: Path, n_expected: int | None) -> dict:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    ids = {r["task_id"] for r in rows}
    ok = bool(rows) and len(ids) == len(rows) and (n_expected is None or len(rows) == n_expected)
    return dict(mean=sum(r["score"] for r in rows) / len(rows) if ok else None, n=len(rows), n_expected=n_expected,
                exact=sum(r["score"] >= 1.0 for r in rows), errors=sum(bool(r.get("error")) for r in rows), complete=ok)


def heldout(module: Path, needles: int, work: Path) -> dict:
    out = work / f"{module.parent.name}-{needles}"
    if (out / "rows.jsonl").exists():  # run_routine appends, so a rerun into the same directory would double count
        raise SystemExit(f"{out}/rows.jsonl exists; use a fresh --work directory")
    r = subprocess.run([str(PY), "run_routine.py", "--family", "mrcr", "--needles", str(needles), "--module", str(module),
                        "--entry", "answer_mrcr", "--out", str(out), "--split", "test", "--concurrency", "8"],
                       cwd=HERE, capture_output=True, text=True, timeout=1800)
    m = re.search(r"mean [0-9.]+ n=(\d+)", r.stdout + r.stderr)
    return stats(out / "rows.jsonl", int(m.group(1)) if m else None)


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
            h2, h4 = heldout(run / "accepted.py", 2, a.work), heldout(run / "accepted.py", 4, a.work)
            row.update(heldout_2=h2["mean"], heldout_4=h4["mean"], detail_2=h2, detail_4=h4)
            row["correct"] = all(v is not None and v >= 0.9 for v in (row["heldout_2"], row["heldout_4"]))
        else:
            row["correct"] = False
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
