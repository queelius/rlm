"""E59: do frozen bootstrapped libraries survive meaning-preserving shifts (stress_gen.py)?

For each run (with the rule that picks its accepted library per family, as in e39_analysis.py: "a" = first valid
candidate reproducing the 3 shown examples, "b" = labelled acceptance at teaching mean 0.9 or more), scores the
accepted library on 30 stress tasks under each condition and writes one JSON line per (run, family, condition).
Resumable: rows already in --out are skipped. Success = 0.9 or more, as everywhere else.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap_library as B  # noqa: E402
import e39_analysis as E  # noqa: E402
import family_gen as F  # noqa: E402
import stress_gen as S  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, help="RUN_DIR:RULE with RULE a or b; repeatable")
    p.add_argument("--families", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conditions", default=",".join(S.CONDITIONS))
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--prefix", default="", help="E60: run directories and entry names use PREFIX + family (e.g. mix:)")
    a = p.parse_args()
    fams = a.families.read_text().strip().split(",")
    conds = a.conditions.split(",")
    done = set()
    if a.out.exists():
        done = {(r["run"], r["family"], r["condition"]) for r in map(json.loads, a.out.read_text().splitlines())}
    jobs = []
    for spec in a.run:
        run, rule = spec.rsplit(":", 1)
        run = Path(run)
        for fid in fams:
            if not (run / (a.prefix + fid) / "summary.json").exists():
                continue
            c = (E.pick_a if rule == "a" else E.pick_b)(run / (a.prefix + fid))
            if c is None:
                continue
            jobs += [(run, rule, fid, c, cond) for cond in conds if (run.name, fid, cond) not in done]
    print(f"{len(jobs)} library-condition scorings to run", flush=True)
    lock = Lock()
    a.work.mkdir(parents=True, exist_ok=True)

    def one(job):
        run, rule, fid, c, cond = job
        tasks = S.make(fid, cond)
        entry = "answer_" + re.sub(r"\W", "_", a.prefix + fid)
        v = B.verify(c["code"], tasks, lambda out, t: F.grade(out, t), a.work, f"{run.name}-{fid}-r{c['round']}c{c['k']}-{cond}",
                     a.helper_base_url, "qwen3-4b", 900, entry)
        row = dict(run=run.name, rule=rule, family=fid, condition=cond, round=c["round"], k=c["k"], mean=v["mean"],
                   success=v["mean"] >= 0.9, n=v["n"], errors=sum(bool(d.get("error")) for d in v["details"]),
                   load_error=v.get("load_error"), timed_out=v.get("timed_out"), seconds=v["seconds"])
        with lock, a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
