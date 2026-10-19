"""Scripted OOLONG expert controller run through the real kernel (teaching trajectories + expert baseline).

Split: within each context length, distinct context windows are sorted by id and alternate
train/test, so every length has held-out windows. Label questions classify every line with helper calls; user/date counting
questions are answered by code alone.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oolong_eval  # noqa: E402
import rl_train  # noqa: E402
import run_sweep  # noqa: E402
from teach import response_object  # noqa: E402

from rlm import RLM, response_output_text  # noqa: E402

import variants  # noqa: E402

CELLS = HERE / "oolong_cells"
NEEDS_LABELS = re.compile(r"label|classified", re.I)


class OolongExpert:
    def __init__(self, variant: dict | None = None):
        self.variant = variant or variants.pick("", False)
        self.cells = {n: variants.render(CELLS, n, self.variant) for n in ("inspect", "classify", "aggregate", "count")}

    def complete(self, request, *, timeout, context):
        items = [it for it in request["input"] if it.get("role") == "user"]
        if len(items) == 1:
            return response_object(f"```python\n{self.cells['inspect']}\n```")
        last = json.loads(items[-1]["content"])
        execution = last.get("execution") or {}
        if execution.get("exception") or last.get("submission", {}).get("status") == "rejected":
            raise RuntimeError("expert cell failed: " + json.dumps(execution)[:500])
        namespace = execution.get("namespace") or {}
        if "batch" in namespace:
            return response_object(f"```python\n{self.cells['aggregate']}\n```")
        if "question" in namespace:
            stdout = execution.get("stdout", "")
            question = stdout.split("\n", 1)[1].strip() if "\n" in stdout else ""
            if NEEDS_LABELS.search(question):
                return response_object(f"```python\n{self.cells['classify']}\n```")
            return response_object(f"```python\n{self.cells['count']}\n```")
        raise RuntimeError("expert cannot determine its stage: " + json.dumps(namespace)[:300])


def load_rows(split: str, max_len: int, lens: set[int] | None = None) -> list[dict]:
    rows = [json.loads(l) for l in oolong_eval.ROWS.read_text().splitlines()]
    rows = [r for r in rows if r["context_len"] <= max_len and not r["answer_type"].endswith("MONTH_YEAR")]
    if lens:
        rows = [r for r in rows if r["context_len"] in lens]
    assignment = oolong_eval.window_split(rows)
    return [r for r in rows if assignment[(r["context_len"], r["context_window_id"])] == split]


def as_task(r: dict) -> SimpleNamespace:
    return SimpleNamespace(task_id=f"oolong-{r['id']}", prompt=r["context"] + "\n" + r["question"], answer=str(r["answer"]),
                           answer_type=r["answer_type"], n_records=r["context_len"], question_type=r["task"], seed=r["id"], events=[], row=r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--public-model", default="qwen3-4b")
    p.add_argument("--harness", default="default", choices=("default", "hinted"))
    p.add_argument("--diversify", action="store_true")
    p.add_argument("--unified", action="store_true", help="family-agnostic peek first (expert.UnifiedExpert)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.split, args.max_len)
    random.Random(0).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    ra = SimpleNamespace(base_url=args.base_url, api_key=args.api_key, public_model=args.public_model, harness=args.harness,
                         controller_max_output=2048, max_turns=6, max_model_calls=96, max_subcalls=80, max_total_tokens=600_000,
                         deadline=1200.0, exec_timeout=900.0, temperature=0.0)
    trace_dir = args.out / "traces"
    rows_path, sft_path = args.out / "rows.jsonl", args.out / "sft.jsonl"
    done = {json.loads(l)["id"] for l in rows_path.read_text().splitlines()} if rows_path.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows to run", flush=True)
    lock = threading.Lock()

    def one(r):
        task = as_task(r)
        variant = variants.pick(task.task_id, args.diversify)
        row = dict(id=r["id"], policy="expert", dataset=r["dataset"], context_len=r["context_len"], task_group=r["task_group"],
                   task=r["task"], answer_type=r["answer_type"], gold=r["answer"], n=r["context_len"], question_type=r["task"],
                   variant=variant)
        rlm = run_sweep.make_rlm(ra, trace_dir, "expert-controller")
        rlm = RLM(rlm.backend, config=rlm.config, controller_backend=(__import__("expert").UnifiedExpert(variant) if args.unified else OolongExpert(variant)))
        try:
            result = rlm.run(run_sweep.public_request(task, args.public_model, 0.0, 4096))
            row.update(output_text=response_output_text(result.response), total_tokens=result.usage.total_tokens,
                       prompt_tokens=result.usage.input_tokens, completion_tokens=result.usage.output_tokens,
                       calls=result.usage.calls, turns=result.turns, duration=result.duration_seconds,
                       trace_dir=str(result.trace_directory), error=None)
            row["parsed"] = oolong_eval.parse(r["answer_type"], row["output_text"])
            row["correct"] = oolong_eval.verify(r, row["output_text"])
        except Exception as exc:  # noqa: BLE001
            row.update(correct=False, output_text=None, error=f"{type(exc).__name__}: {exc}"[:600], trace_dir=None)
        finally:
            rlm.close()
        pairs = []
        if row.get("trace_dir"):
            for k, t in enumerate(rl_train.controller_turns(row["trace_dir"])):
                pairs.append(dict(task_id=task.task_id, n=r["context_len"], question_type=r["task"], turn=k, correct=row["correct"],
                                  messages=t["messages"], completion=t["completion"], source="oolong", variant=variant))
        with lock:
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            with sft_path.open("a") as f:
                for pr in pairs:
                    f.write(json.dumps(pr) + "\n")
        return row

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        out_rows = list(pool.map(one, rows))
    import collections
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in out_rows:
        a = agg[(r["context_len"])]; a[0] += 1; a[1] += bool(r["correct"]); a[2] += int(r.get("total_tokens") or 0)
    for k in sorted(agg):
        a = agg[k]; print(f"len {k:>6}  acc {a[1]/a[0]:.2f}  mean_tok {a[2]/a[0]:.0f}  n={a[0]}")
    errs = collections.Counter((r.get("error") or "")[:90] for r in out_rows if r.get("error"))
    print(errs.most_common(5))


if __name__ == "__main__":
    main()
