"""Evaluate RLM controllers and the direct answer on MRCR (E21). Writes rows.jsonl with the official grade.

Policies: `direct` (one call to the public model with the full message list; inputs beyond the
model's window fail and score 0) and `rlm` (the kernel with the chosen harness: `default` = no
library, `skills` = the learned library for the family, retrieved by the task's family).
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mrcr_tasks  # noqa: E402
import run_sweep  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test", choices=("teach", "test"))
    p.add_argument("--needles", type=int, default=2)
    p.add_argument("--per-bin", type=int, default=8)
    p.add_argument("--max-bin", type=int, default=None)
    p.add_argument("--policies", default="rlm")
    p.add_argument("--harness", default="skills", choices=("default", "skills"))
    p.add_argument("--controller-model", default="qwen3-4b")
    p.add_argument("--controller-base-url", default=None)
    p.add_argument("--public-model", default="qwen3-4b")
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-subcalls", type=int, default=2000)
    p.add_argument("--max-model-calls", type=int, default=2100)
    p.add_argument("--max-total-tokens", type=int, default=8_000_000)
    p.add_argument("--deadline", type=float, default=3600.0)
    p.add_argument("--exec-timeout", type=float, default=3500.0)
    a = p.parse_args()
    rows = mrcr_tasks.load(a.split, a.needles, per_bin=a.per_bin, max_bin=a.max_bin)
    ra = SimpleNamespace(controller_base_url=a.controller_base_url, base_url=a.base_url, api_key=a.api_key,
                         public_model=a.public_model, controller_model=a.controller_model, temperature=a.temperature,
                         harness=a.harness, direct_max_output=8192, controller_max_output=1024, max_turns=10,
                         max_model_calls=a.max_model_calls, max_subcalls=a.max_subcalls,
                         max_total_tokens=a.max_total_tokens, deadline=a.deadline, exec_timeout=a.exec_timeout)
    a.out.mkdir(parents=True, exist_ok=True)
    trace_dir = a.out / "traces"
    rows_path = a.out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {(json.loads(l)["task_id"], json.loads(l)["policy"]) for l in rows_path.read_text().splitlines()}
    jobs = [(r, pol) for r in rows for pol in a.policies.split(",") if (r["row_id"], pol) not in done]
    print(f"{len(rows)} rows, {len(jobs)} jobs", flush=True)
    lock = threading.Lock()

    def work(job):
        r, policy = job
        task = mrcr_tasks.as_task(r)
        row = dict(task_id=task.task_id, policy=policy, harness=a.harness if policy == "rlm" else None,
                   controller=a.controller_model if policy == "rlm" else None, needles=r["needles"], bin=r["bin"],
                   n_chars=r["n_chars"], total_messages=r["total_messages"])
        t0 = time.time()
        try:
            fn = run_sweep.run_direct if policy == "direct" else run_sweep.run_rlm
            row.update(fn(ra, task, trace_dir))
            row["error"] = None
        except Exception as e:  # noqa: BLE001
            row.update(output_text=None, error=f"{type(e).__name__}: {e}"[:400])
        row["score"] = mrcr_tasks.grade(row.get("output_text") or "", task.answer, task.prefix)
        row["seconds"] = round(time.time() - t0, 1)
        with lock:
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    with ThreadPoolExecutor(a.concurrency) as ex:
        for k, row in enumerate(ex.map(work, jobs), 1):
            print(f"[{k}/{len(jobs)}] {row['policy']} bin={row['bin']} score={row['score']:.3f} err={bool(row['error'])}", flush=True)
    res = [json.loads(l) for l in rows_path.read_text().splitlines()]
    for pol in a.policies.split(","):
        rs = [r for r in res if r["policy"] == pol]
        by = {}
        for r in rs:
            by.setdefault(r["bin"], []).append(r["score"])
        print(pol, f"mean {sum(r['score'] for r in rs) / max(len(rs), 1):.3f} n={len(rs)}",
              {b: round(sum(v) / len(v), 3) for b, v in sorted(by.items())})


if __name__ == "__main__":
    main()
