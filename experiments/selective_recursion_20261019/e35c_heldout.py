"""E35c held-out check: score OOLONG libraries on one task group's test-window rows (the bootstrap verifier, same
helper), per task type, so a bootstrapped group library and the shipped library are compared on identical rows."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap_library as B  # noqa: E402
import oolong_eval as O  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", required=True, choices=("counting", "timeline", "user"))
    p.add_argument("--module", type=Path, nargs="+", required=True, help="library files exposing answer_oolong_<group> or answer_oolong")
    p.add_argument("--lens", default="4096,8192,16384")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--timeout", type=float, default=3600)
    a = p.parse_args()
    lens = {int(x) for x in a.lens.split(",")}
    rows = [json.loads(line) for line in O.ROWS.read_text().splitlines()]
    rows = [r for r in rows if not r["answer_type"].endswith("MONTH_YEAR")]
    split = O.window_split(rows)
    test = sorted((r for r in rows if r["task_group"] == a.group and r["context_len"] in lens
                   and split[(r["context_len"], r["context_window_id"])] == "test"), key=lambda r: r["id"])
    tasks = [SimpleNamespace(task_id=f"oolong-{r['id']}", prompt=r["context"] + "\n" + r["question"], row=r) for r in test]
    print(f"{a.group}: {len(tasks)} test rows at {sorted(lens)}", flush=True)
    a.work.mkdir(parents=True, exist_ok=True)
    for mod in a.module:
        code = mod.read_text()
        entry = f"answer_oolong_{a.group}" if f"def answer_oolong_{a.group}(" in code else "answer_oolong"
        v = B.verify(code, tasks, lambda out, t: float(O.verify(t.row, out or "")), a.work, mod.stem, a.helper_base_url, "qwen3-4b",
                     a.timeout, entry)
        by = collections.defaultdict(list)
        for t, d in zip(tasks, v["details"], strict=True):
            by[t.row["task"].split(".")[-1]].append(d["score"])
        print(json.dumps(dict(module=str(mod), mean=round(v["mean"], 3), exact=v["exact"], n=v["n"], timed_out=v["timed_out"],
                              load_error=bool(v["load_error"]), by_type={k: f"{sum(x) / len(x):.2f}/{len(x)}" for k, x in sorted(by.items())})),
              flush=True)


if __name__ == "__main__":
    main()
