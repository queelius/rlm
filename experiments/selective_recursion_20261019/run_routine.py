"""Run a library routine directly (no controller) on a family's tasks: a scripted caller, the library's
own ceiling for that family. Binds ask/ask_batch to a helper server and logs helper calls and tokens.

Usage: run_routine.py --family musique --module skills/reference/musique_handwritten.py --entry answer_musique --out DIR
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
import family_eval  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--entry", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--needles", type=int, default=2)
    p.add_argument("--per-bin", type=int, default=8)
    p.add_argument("--max-bin", type=int, default=None)
    p.add_argument("--sizes", default="8192,32768,131072")
    p.add_argument("--per-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--concurrency", type=int, default=4)
    a = p.parse_args()
    from openai import OpenAI

    get_tasks, score, meta = family_eval.adapter(a.family)
    tasks = get_tasks(a.split, a)
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=600)
    pool = ThreadPoolExecutor(32)
    local = threading.local()

    def _one(prompt, system, options):
        options = options or {}
        r = client.responses.create(model=a.model, input=prompt, instructions=system,
                                    temperature=options.get("temperature", 0),
                                    max_output_tokens=options.get("max_output_tokens", 1024))
        st = local.stats
        st["calls"] += 1
        st["tokens"] += r.usage.input_tokens + r.usage.output_tokens
        return r.output_text or ""

    def ask(prompt, system=None, options=None):
        return _one(prompt, system, options)

    def ask_batch(prompts, system=None, options=None):
        st = local.stats

        def run(pp):
            local.stats = st
            return _one(pp, system, options)
        return SimpleNamespace(texts=list(pool.map(run, prompts)))

    src = Path(a.module).read_text()
    a.out.mkdir(parents=True, exist_ok=True)
    rows_path = a.out / "rows.jsonl"
    lock = threading.Lock()

    def work(t):
        local.stats = {"calls": 0, "tokens": 0}
        ns = {"ask": ask, "ask_batch": ask_batch}
        exec(compile(src, "routine", "exec"), ns)
        t0 = time.time()
        try:
            out, err = ns[a.entry]({"input": t.prompt}), None
        except Exception as e:  # noqa: BLE001
            out, err = None, f"{type(e).__name__}: {e}"[:300]
        row = dict(task_id=t.task_id, family=a.family, policy="routine", module=Path(a.module).name, output_text=out,
                   error=err, helper_calls=local.stats["calls"], total_tokens=local.stats["tokens"],
                   seconds=round(time.time() - t0, 1), **meta(t))
        row.update(score(out, t))
        with lock:
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(work, tasks))
    by = {}
    for r in res:
        k = r.get("size") or r.get("bin")
        by.setdefault(k, []).append(r["score"])
    print(f"mean {sum(r['score'] for r in res) / len(res):.3f} n={len(res)}", {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())},
          f"tokens/task {sum(r['total_tokens'] for r in res) / len(res) / 1000:.1f}k", f"errors {sum(bool(r['error']) for r in res)}")


if __name__ == "__main__":
    main()
