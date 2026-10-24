"""E72: a label-free metamorphic check for day-span libraries.

A day-span answer depends only on calendar dates. The variant task keeps every record's date and the record order,
redraws the time of day (sorted within each date, from a separate RNG so the question and answer are unchanged), and
the check compares the library's outputs on the original and the variant: a library that subtracts full timestamps
changes its answer, a correct one does not. No labels are used; the fresh-task success recorded by the E58 analysis
says which libraries were actually correct.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap_library as B  # noqa: E402
import e39_analysis as E  # noqa: E402
import family_gen as F  # noqa: E402


class TimeShifted(F.Family):
    """Same records and question as the family; each record's time of day redrawn, order within a date kept."""

    def records(self, r, n):
        recs = super().records(r, n)
        rr = random.Random(f"timeshift:{self.fid}:{recs[0]['ts'].isoformat()}")
        by_day: dict = {}
        for x in recs:
            by_day.setdefault(x["ts"].date(), []).append(x)
        for day, xs in by_day.items():
            mins = sorted(rr.sample(range(24 * 60), len(xs)))
            for x, m in zip(xs, mins, strict=True):
                x["ts"] = dt.datetime.combine(day, dt.time(m // 60, m % 60))
        return recs


def outputs(code: str, tasks: list, work: Path, tag: str, helper: str, entry: str) -> list:
    v = B.verify(code, tasks, lambda out, t: F.grade(out, t), work, tag, helper, "qwen3-4b", 600, entry)
    return [d["output"] for d in v["details"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, help="RUN_DIR:RULE:ROWS (rows = e39_rows.jsonl of its analysis)")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8006/v1")
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    out = []
    for spec in a.run:
        run, rule, rows = spec.rsplit(":", 2)
        run = Path(run)
        truth = {r["family"]: r for r in map(json.loads, Path(rows).read_text().splitlines())}
        arm = "a" if rule == "a" else "b"
        for fid, r in sorted(truth.items()):
            if "-day_span-" not in fid or not r[f"{arm}_accepted"]:
                continue
            c = (E.pick_a if rule == "a" else E.pick_b)(run / fid)
            entry = "answer_" + re.sub(r"\W", "_", fid)
            orig = [F.Family(fid).task(i, "metamorphic", 400) for i in range(a.k)]
            shifted = [TimeShifted(fid).task(i, "metamorphic", 400) for i in range(a.k)]
            assert [t.answer for t in orig] == [t.answer for t in shifted]
            o1 = outputs(c["code"], orig, a.work, f"{run.name}-{fid}-orig", a.helper_base_url, entry)
            o2 = outputs(c["code"], shifted, a.work, f"{run.name}-{fid}-shift", a.helper_base_url, entry)
            changed = sum(x != y for x, y in zip(o1, o2, strict=True))
            row = dict(run=run.name, family=fid, fresh_success=r[f"{arm}_success"], fresh=r[f"{arm}_fresh"],
                       changed=changed, k=a.k, flagged=changed > 0)
            out.append(row)
            print(json.dumps(row), flush=True)
    a.out.write_text("".join(json.dumps(x) + "\n" for x in out))
    tp = sum(x["flagged"] and not x["fresh_success"] for x in out)
    fp = sum(x["flagged"] and x["fresh_success"] for x in out)
    print(f"flagged {tp} of {sum(not x['fresh_success'] for x in out)} wrong libraries; "
          f"{fp} of {sum(x['fresh_success'] for x in out)} correct libraries")


if __name__ == "__main__":
    main()
