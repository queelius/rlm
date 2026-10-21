"""E18: the RLM paper's released RLM-Qwen3-8B in its own scaffold (alexzhang13/rlm @ 854e688) on the
same 293 held-out OOLONG-synth rows. Runs inside that scaffold's environment and only records raw
outputs; scoring is done afterwards with this repository's verifier (oolong_eval.verify).

Settings follow the scaffold's OOLONG training environment: the context goes to the REPL, the root
prompt is a dataset instruction plus the question, max 12 iterations, sub-call max_tokens 4096,
local REPL, recursion depth 1; root and sub-calls both use RLM-Qwen3-8B.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlm import RLM

TREC = ("The context contains thousands of general-knowledge questions, one per line. Each line has a User ID "
        "and a question, and each question's answer falls into one of 6 categories: 'numeric value', 'entity', "
        "'location', 'description and abstract concept', 'abbreviation', 'human being'. Answer the following "
        "aggregate question.")  # verbatim from the scaffold's OOLONG environment
SPAM = ("The context contains thousands of text messages, one per line. Each line has a User ID and a text "
        "message, and each message can be classified as 'spam' or 'ham' (not spam). Answer the following "
        "aggregate question.")  # same wording adapted to the spam label set


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="/project/alex_phd/research-cache/datasets/oolong-synth-20261019/validation-le32k.jsonl")
    p.add_argument("--ids-from", default="/project/alex_phd/runs/selective-recursion-20261019/oolong-lib-base-full/rows.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    p.add_argument("--model", default="rlm-qwen3-8b")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    ids = [json.loads(l)["id"] for l in open(a.ids_from)]
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(a.rows)}
    todo = [rows[i] for i in ids]
    if a.limit:
        todo = todo[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    rp = out / "raw.jsonl"
    if rp.exists():
        done = {json.loads(l)["id"] for l in open(rp)}
    todo = [r for r in todo if r["id"] not in done]
    print(len(todo), "rows to run", flush=True)

    def one(r):
        t0 = time.time()
        inst = TREC if r["dataset"] == "trec_coarse" else SPAM
        rec = dict(id=r["id"], dataset=r["dataset"], context_len=r["context_len"], task=r["task"])
        try:
            rlm = RLM(backend="openai", backend_kwargs={"model_name": a.model, "base_url": a.base_url, "api_key": "local"},
                      environment="local", max_depth=1, max_iterations=12, sub_sampling_args={"max_tokens": 4096},
                      max_timeout=1800)
            res = rlm.completion(r["context"], root_prompt=f"{inst}\n\nQuestion: {r['question']}")
            rec.update(output_text=getattr(res, "response", None), error=None,
                       usage=str(getattr(res, "usage_summary", ""))[:500])
        except Exception as e:  # noqa: BLE001
            rec.update(output_text=None, error=f"{type(e).__name__}: {e}"[:400])
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

    with ThreadPoolExecutor(a.concurrency) as ex, open(rp, "a") as f:
        for k, rec in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{k}/{len(todo)}] {rec['dataset']} {rec['context_len']} {str(rec['output_text'])[:60]!r} err={bool(rec['error'])}", flush=True)


if __name__ == "__main__":
    main()
