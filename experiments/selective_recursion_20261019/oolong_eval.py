"""Zero-shot transfer evaluation on OOLONG-synth (validation, <=32k tokens) for direct and RLM policies.

Uses the standard OOLONG input (context_window_text + newline + question) as the public request,
and a format-based exact-match verifier for the OOLONG answer conventions. This is not the official
OOLONG scorer; it is a conservative exact match on the requested answer line.
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
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sweep  # noqa: E402

ROWS = Path("/project/alex_phd/research-cache/datasets/oolong-synth-20261019/validation-le32k.jsonl")
def window_split(rows: list[dict]) -> dict:
    """Map (context_len, context_window_id) -> 'train' | 'test': even/odd window id per length, or alternating
    sorted ids where a length has only one parity (4096 tokens)."""
    by_len: dict = {}
    for r in rows:
        by_len.setdefault(r["context_len"], set()).add(r["context_window_id"])
    out = {}
    for n, ids in by_len.items():
        ordered = sorted(ids)
        if {wid % 2 for wid in ordered} == {0, 1}:
            # even ids train, odd ids test (matches the first teaching run)
            for wid in ordered:
                out[(n, wid)] = "train" if wid % 2 == 0 else "test"
        else:
            for rank, wid in enumerate(ordered):
                out[(n, wid)] = "train" if rank % 2 == 0 else "test"
    return out


COMPARISON_CANON = {"same frequency as": "same frequency", "the same frequency": "same frequency",
                    "more common than": "more common", "less common than": "less common"}


KEY = r"(?:Label|Answer|ANSWER|User|Date)\s*:\s*\**\s*"


def parse(answer_type: str, text: str):
    """Value of the last answer line; the key prefix is not scored (Label:/Answer:/User:/Date:/ANSWER:)."""
    if not text:
        return None
    t = text.strip()
    if answer_type.endswith("LABEL"):
        m = re.findall(KEY + r"\[?([A-Za-z][A-Za-z _\-]*)", t)
        return m[-1].strip().lower().rstrip(".") if m else None
    if answer_type.endswith("NUMERIC"):
        m = re.findall(KEY + r"\[?(-?\d[\d,]*)", t)
        return int(m[-1].replace(",", "")) if m else None
    if answer_type.endswith("COMPARISON"):
        m = re.findall(r"(more common than|less common than|same frequency as|more common|less common|the same frequency|same frequency)", t)
        return COMPARISON_CANON.get(m[-1], m[-1]) if m else None
    if answer_type.endswith("USER"):
        m = re.findall(KEY + r"\[?(\d+)", t)
        return int(m[-1]) if m else None
    if answer_type.endswith("DATE"):
        m = re.findall(KEY + r"\[?(\d{1,2})/(\d{1,2})/(\d{4})", t)
        return f"{int(m[-1][2]):04d}-{int(m[-1][0]):02d}-{int(m[-1][1]):02d}" if m else None
    return None


def gold_value(answer_type: str, gold: list):
    g = gold[0]
    if answer_type.endswith("LABEL"):
        return str(g).strip().lower()
    if answer_type.endswith("NUMERIC") or answer_type.endswith("USER"):
        return int(g)
    if answer_type.endswith("COMPARISON"):
        return COMPARISON_CANON.get(str(g), str(g))
    if answer_type.endswith("DATE"):
        return str(g)[:10]
    return str(g)


def verify(row: dict, text: str) -> bool:
    try:
        return parse(row["answer_type"], text) == gold_value(row["answer_type"], row["answer"])
    except Exception:  # noqa: BLE001
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policies", default="direct,rlm")
    p.add_argument("--controller-model", default="qwen3-4b")
    p.add_argument("--harness", default="default", choices=("default", "hinted", "skills"))
    p.add_argument("--lens", default="1024,2048,4096,8192,16384,32768")
    p.add_argument("--per-cell", type=int, default=8, help="rows per (dataset, context_len)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="test", choices=("test", "train", "all"))
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--public-model", default="qwen3-4b")
    args = p.parse_args()
    lens = {int(x) for x in args.lens.split(",")}
    rows = [json.loads(l) for l in ROWS.read_text().splitlines()]
    rows = [r for r in rows if r["context_len"] in lens and not r["answer_type"].endswith("MONTH_YEAR")]
    split = window_split(rows)
    if args.split != "all":
        rows = [r for r in rows if split[(r["context_len"], r["context_window_id"])] == args.split]
    rng = random.Random(args.seed)
    by_cell: dict[tuple, list] = {}
    for r in rows:
        by_cell.setdefault((r["dataset"], r["context_len"]), []).append(r)
    selected = []
    for k in sorted(by_cell):
        cell = sorted(by_cell[k], key=lambda r: r["id"])
        rng.shuffle(cell)
        selected += cell[: args.per_cell]
    ra = SimpleNamespace(base_url=args.base_url, api_key=args.api_key, public_model=args.public_model,
                         controller_model=args.controller_model, temperature=args.temperature, harness=args.harness,
                         direct_max_output=4096, controller_max_output=1024, max_turns=10, max_model_calls=96,
                         max_subcalls=80, max_total_tokens=600_000, deadline=1200.0, exec_timeout=900.0)
    args.out.mkdir(parents=True, exist_ok=True)
    trace_dir = args.out / "traces"
    rows_path = args.out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {(json.loads(l)["id"], json.loads(l)["policy"]) for l in rows_path.read_text().splitlines()}
    jobs = [(r, pol) for r in selected for pol in args.policies.split(",") if (r["id"], pol) not in done]
    print(f"{len(selected)} rows, {len(jobs)} jobs", flush=True)
    lock = threading.Lock()

    def work(job):
        r, policy = job
        task = SimpleNamespace(task_id=f"oolong-{r['id']}", prompt=r["context"] + "\n" + r["question"], answer=str(r["answer"]),
                               answer_type=r["answer_type"], n_records=r["context_len"], question_type=r["task"], seed=r["id"], events=[])
        row = dict(id=r["id"], policy=policy, controller=args.controller_model if policy == "rlm" else None, harness=args.harness if policy == "rlm" else None,
                   dataset=r["dataset"], context_len=r["context_len"], task_group=r["task_group"], task=r["task"], answer_type=r["answer_type"],
                   gold=r["answer"], n=r["context_len"], question_type=r["task"])
        try:
            fn = run_sweep.run_direct if policy == "direct" else run_sweep.run_rlm
            row.update(fn(ra, task, trace_dir))
            row["parsed"] = parse(r["answer_type"], row["output_text"])
            row["correct"] = verify(r, row["output_text"])
            row["error"] = None
        except Exception as exc:  # noqa: BLE001
            row.update(correct=False, output_text=None, error=f"{type(exc).__name__}: {exc}"[:600], traceback=traceback.format_exc()[-1500:])
        with lock:
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for k, row in enumerate(pool.map(work, jobs), 1):
            if k % 10 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {row['policy']} len={row['context_len']} {row['task']} correct={row['correct']} tok={row.get('total_tokens')} err={bool(row.get('error'))}", flush=True)
    import collections
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for l in rows_path.read_text().splitlines():
        r = json.loads(l); a = agg[(r["policy"], r["context_len"])]; a[0] += 1; a[1] += bool(r["correct"]); a[2] += int(r.get("total_tokens") or 0)
    for k in sorted(agg):
        a = agg[k]; print(f"{k[0]:<8} {k[1]:>6}  acc {a[1]/a[0]:.2f}  mean_tok {a[2]/a[0]:.0f}  n={a[0]}")


if __name__ == "__main__":
    main()
