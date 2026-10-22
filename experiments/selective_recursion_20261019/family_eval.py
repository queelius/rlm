"""Evaluate direct answering and RLM controllers on any registered task family (E21 onwards).

A family adapter provides `tasks(split, args) -> list[task]` and `score(output, task) -> dict` with a
`score` key in [0, 1] (MRCR: official grade; MuSiQue: answer F1, with EM alongside). Rows go to
rows.jsonl with the family's scores, tokens, calls, turns, trace directory and errors.
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
import run_sweep  # noqa: E402


def adapter(family: str):
    if family == "mrcr":
        import mrcr_tasks as F

        def tasks(split, a):
            return [F.as_task(r) for r in F.load(split, a.needles, per_bin=a.per_bin, max_bin=a.max_bin)]

        def score(out, t):
            return dict(score=F.grade(out or "", t.answer, t.prefix))

        return tasks, score, lambda t: dict(bin=t.bin, needles=t.question_type)
    if family == "musique":
        import musique_tasks as F

        def tasks(split, a):
            return [t for size in map(int, a.sizes.split(",")) for t in F.make(split, size, a.per_size, seed=a.seed)]

        def score(out, t):
            g = F.grade(out or "", t)
            return dict(score=g["f1"], em=g["em"])

        return tasks, score, lambda t: dict(size=t.n_records, hops=t.question_type)
    raise ValueError(family)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--needles", type=int, default=2)
    p.add_argument("--per-bin", type=int, default=8)
    p.add_argument("--max-bin", type=int, default=None)
    p.add_argument("--sizes", default="8192,32768,131072")
    p.add_argument("--per-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policies", default="rlm")
    p.add_argument("--harness", default="skills", choices=("default", "skills"))
    p.add_argument("--controller-model", default="qwen3-4b")
    p.add_argument("--controller-base-url", default=None)
    p.add_argument("--public-model", default="qwen3-4b")
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-depth", type=int, default=0)
    p.add_argument("--max-subcalls", type=int, default=2000)
    p.add_argument("--max-model-calls", type=int, default=2100)
    p.add_argument("--max-total-tokens", type=int, default=8_000_000)
    p.add_argument("--deadline", type=float, default=3600.0)
    p.add_argument("--exec-timeout", type=float, default=3500.0)
    a = p.parse_args()
    get_tasks, score, meta = adapter(a.family)
    tasks = get_tasks(a.split, a)
    ra = SimpleNamespace(controller_base_url=a.controller_base_url, base_url=a.base_url, api_key=a.api_key,
                         public_model=a.public_model, controller_model=a.controller_model, temperature=a.temperature,
                         harness=a.harness, direct_max_output=8192, controller_max_output=1024, max_turns=10,
                         max_model_calls=a.max_model_calls, max_subcalls=a.max_subcalls, max_depth=a.max_depth,
                         max_total_tokens=a.max_total_tokens, deadline=a.deadline, exec_timeout=a.exec_timeout)
    a.out.mkdir(parents=True, exist_ok=True)
    trace_dir = a.out / "traces"
    rows_path = a.out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {(json.loads(l)["task_id"], json.loads(l)["policy"]) for l in rows_path.read_text().splitlines()}
    jobs = [(t, pol) for t in tasks for pol in a.policies.split(",") if (t.task_id, pol) not in done]
    print(f"{len(tasks)} tasks, {len(jobs)} jobs", flush=True)
    lock = threading.Lock()

    def work(job):
        task, policy = job
        row = dict(task_id=task.task_id, family=a.family, policy=policy, harness=a.harness if policy == "rlm" else None,
                   controller=a.controller_model if policy == "rlm" else None, **meta(task))
        t0 = time.time()
        try:
            fn = run_sweep.run_direct if policy == "direct" else run_sweep.run_rlm
            row.update(fn(ra, task, trace_dir))
            row["error"] = None
        except Exception as e:  # noqa: BLE001
            row.update(output_text=None, error=f"{type(e).__name__}: {e}"[:400])
        row.update(score(row.get("output_text"), task))
        row["seconds"] = round(time.time() - t0, 1)
        with lock:
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    with ThreadPoolExecutor(a.concurrency) as ex:
        for k, row in enumerate(ex.map(work, jobs), 1):
            print(f"[{k}/{len(jobs)}] {row['policy']} score={row['score']:.3f} err={bool(row['error'])}", flush=True)
    res = [json.loads(l) for l in rows_path.read_text().splitlines()]
    for pol in a.policies.split(","):
        rs = [r for r in res if r["policy"] == pol]
        print(pol, f"mean {sum(r['score'] for r in rs) / max(len(rs), 1):.3f} n={len(rs)}")


if __name__ == "__main__":
    main()
