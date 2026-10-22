"""E22c: component ablation of the MuSiQue filter-then-read routine (scripted, no controller).

Per question the helper filter runs once; every arm then reads its own kept set with the same reader:
  bm25_15, bm25_60   BM25 top k against the question
  filter             helper filter only
  both               filter united with BM25 top 15, first 60 by position (the reference routine)
  both_rank          the same union, but trimmed by BM25 rank instead of position
  oracle             the gold supporting paragraphs only (reader ceiling)
  random60           60 random paragraphs (floor)
Logs F1/EM, kept-set size, gold-support recall and helper tokens per arm.
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import musique_tasks as M  # noqa: E402

ROUTINE = (HERE / "skills" / "reference" / "musique_filter_read.py").read_text()
READER = ("Answer the multi-step question from the paragraphs. Work out each step briefly, then finish with one line "
          "'ANSWER: <short answer>' giving a short phrase (a name, place, date or number). Always give your best answer.")
ARMS = ("bm25_15", "bm25_60", "filter", "both", "both_rank", "oracle", "random60")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--sizes", default="8192,32768,131072")
    p.add_argument("--per-size", type=int, default=30)
    p.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--read-max-tokens", type=int, default=600)
    p.add_argument("--fallback", action="store_true",
                   help="when the reader gives no ANSWER line, ask once more for the short answer only")
    a = p.parse_args()
    arms = tuple(a.arms.split(","))
    from openai import OpenAI

    client = OpenAI(base_url=a.base_url, api_key="local", timeout=900)
    pool = ThreadPoolExecutor(32)
    local = threading.local()

    def _one(prompt, system, options):
        options = options or {}
        r = client.responses.create(model=a.model, input=prompt, instructions=system, temperature=options.get("temperature", 0),
                                    max_output_tokens=options.get("max_output_tokens", 1024))
        local.stats["tokens"] += r.usage.input_tokens + r.usage.output_tokens
        return r.output_text or ""

    def ask(prompt, system=None, options=None):
        return _one(prompt, system, options)

    def ask_batch(prompts, system=None, options=None):
        st = local.stats

        def go(pp):
            local.stats = st
            return _one(pp, system, options)
        return SimpleNamespace(texts=list(pool.map(go, prompts)))

    items = {x["id"]: x for x in M._items(a.split)}
    tasks = [t for s in a.sizes.split(",") for t in M.make(a.split, int(s), a.per_size)]
    a.out.mkdir(parents=True, exist_ok=True)
    rows_path = a.out / "rows.jsonl"
    done = {(r["task_id"], r["arm"]) for r in map(json.loads, rows_path.read_text().splitlines())} if rows_path.exists() else set()
    lock = threading.Lock()

    def work(t):
        if all((t.task_id, arm) in done for arm in arms):
            return
        ns = {"ask": ask, "ask_batch": ask_batch}
        exec(compile(ROUTINE, "routine", "exec"), ns)
        paras, q = ns["parse"](t.prompt)
        it = items[t.task_id.split("-", 3)[3]]
        sup_text = {f"{x['title']}: {x['paragraph_text']}".strip() for x in it["paragraphs"] if x["is_supporting"]}
        gold = [i for i, para in enumerate(paras) if para.strip() in sup_text]
        local.stats = {"tokens": 0}
        filt = ns["select_relevant"](paras, q)
        filter_tokens = local.stats["tokens"]
        rank = ns["bm25"](paras, q, k=len(paras))
        union = set(filt) | set(rank[:15])
        keeps = dict(
            bm25_15=sorted(rank[:15]), bm25_60=sorted(rank[:60]), filter=sorted(filt),
            both=sorted(union)[:60], both_rank=sorted(sorted(union, key=rank.index)[:60]), oracle=sorted(gold),
            random60=sorted(random.Random(hash(t.task_id) % 2**32).sample(range(len(paras)), min(60, len(paras)))))
        for arm in arms:
            if (t.task_id, arm) in done:
                continue
            keep = keeps[arm]
            local.stats = {"tokens": 0}
            ctx = "\n\n".join(f"[{i + 1}] {paras[i]}" for i in keep) or "(no paragraphs)"
            t0 = time.time()
            out = ask(f"Paragraphs:\n{ctx}\n\nQuestion: {q}", system=READER,
                      options={"temperature": 0, "max_output_tokens": a.read_max_tokens})
            m = re.findall(r"ANSWER:\s*(.+)", out or "")
            fell_back = False
            if not m and a.fallback:
                fell_back = True
                out2 = ask(f"Question: {q}\n\nNotes from reading the evidence:\n{(out or '')[-3000:]}\n\n"
                           "Give the final answer as a short phrase (a name, place, date or number) on one line "
                           "of the form 'ANSWER: <answer>'.", options={"temperature": 0, "max_output_tokens": 40})
                m = re.findall(r"ANSWER:\s*(.+)", out2 or "") or ([out2.strip().splitlines()[0]] if (out2 or "").strip() else [])
            ans = "ANSWER: " + (m[-1].strip() if m else (out or "unknown").strip().splitlines()[-1] if out else "unknown")
            g = M.grade(ans, t)
            row = dict(task_id=t.task_id, size=t.n_records, arm=arm, hops=len(gold), n_paragraphs=len(paras), kept=len(keep),
                       support_recall=(len(set(keep) & set(gold)) / len(gold)) if gold else None,
                       read_tokens=local.stats["tokens"],
                       filter_tokens=filter_tokens if arm in ("filter", "both", "both_rank") else 0,
                       output_text=ans, fell_back=fell_back, seconds=round(time.time() - t0, 1), **g)
            with lock:
                with rows_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(work, tasks))
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    for arm in arms:
        for s in sorted({r["size"] for r in rows}):
            rs = [r for r in rows if r["arm"] == arm and r["size"] == s]
            if rs:
                rec = [r["support_recall"] for r in rs if r["support_recall"] is not None]
                print(f"{arm:<10} {s // 1024:>4}k F1 {sum(r['f1'] for r in rs) / len(rs):.3f} recall {sum(rec) / max(len(rec), 1):.2f} "
                      f"kept {sum(r['kept'] for r in rs) / len(rs):.0f} tokens {sum(r['read_tokens'] + r['filter_tokens'] for r in rs) / len(rs) / 1000:.0f}k n {len(rs)}")


if __name__ == "__main__":
    main()
