"""E15: per-row classification accuracy of the library's classify_rows against OOLONG gold labels.

Uses OOLONG-synth's labelled context (`context_window_text_with_labels`) for teaching windows
(train split, up to 32k) to measure how often the helper's label for a row disagrees with the
gold label, per dataset and label, and what that implies for exact counts over N rows.
"""

import argparse
import collections
import glob
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oolong_eval  # noqa: E402
import oolong_variants  # noqa: E402

RAW = "/project/alex_phd/research-cache/datasets/oolong-synth-20261021-long/raw/data/validation-*.parquet"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--per-len", type=int, default=4)
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--split", default="train", choices=("train", "test"))
    a = p.parse_args()
    import pyarrow.parquet as pq
    from openai import OpenAI

    wins = {}
    for f in sorted(glob.glob(RAW)):
        for r in pq.read_table(f, columns=["context_len", "dataset", "context_window_id", "context_window_text",
                                            "context_window_text_with_labels"]).to_pylist():
            if int(r["context_len"]) <= 32768 and r["dataset"] in ("spam", "trec_coarse"):
                wins.setdefault((int(r["context_len"]), r["dataset"], int(r["context_window_id"])), r)
    split = oolong_eval.window_split([dict(context_len=k[0], context_window_id=k[2]) for k in wins])
    chosen = collections.defaultdict(list)
    for k in sorted(wins):
        if split[(k[0], k[2])] == a.split and len(chosen[(k[0], k[1])]) < a.per_len:
            chosen[(k[0], k[1])].append(wins[k])
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=600)
    pool = ThreadPoolExecutor(16)
    ns = {"ask_batch": oolong_variants.make_ask_batch(client, a.model, pool), "ask": None}
    exec(compile((HERE / "skills" / "oolong.py").read_text(), "oolong", "exec"), ns)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []

    def one(item):
        (L, ds), r = item
        rows, labels, _ = ns["parse_oolong"](r["context_window_text"] + "\nQ")
        gold = re.findall(r"\|\| Label: (.+)$", r["context_window_text_with_labels"], re.M)
        ns["classify_rows"](rows, labels, chunk=a.chunk)
        pairs = [(g.strip().lower(), (x["label"] or "none")) for g, x in zip(gold, rows)]
        return dict(context_len=L, dataset=ds, window=int(r["context_window_id"]), n_rows=len(rows), n_gold=len(gold),
                    agree=sum(g == h for g, h in pairs), pairs=pairs)

    items = [((L, ds), r) for (L, ds), rs in chosen.items() for r in rs]
    with ThreadPoolExecutor(4) as ex:
        for rec in ex.map(one, items):
            recs.append(rec)
            print(rec["dataset"], rec["context_len"], f"{rec['agree']}/{rec['n_rows']}", flush=True)
    (out / "windows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    summary = {}
    for ds in ("spam", "trec_coarse"):
        rs = [r for r in recs if r["dataset"] == ds]
        tot = sum(r["n_rows"] for r in rs)
        agree = sum(r["agree"] for r in rs)
        conf = collections.Counter(p for r in rs for p in map(tuple, r["pairs"]))
        summary[ds] = dict(rows=tot, per_row_accuracy=round(agree / max(tot, 1), 4),
                           top_confusions=[(f"{g} -> {h}", c) for (g, h), c in conf.most_common() if g != h][:6],
                           by_len={L: round(sum(r["agree"] for r in rs if r["context_len"] == L) /
                                            max(sum(r["n_rows"] for r in rs if r["context_len"] == L), 1), 4)
                                   for L in sorted({r["context_len"] for r in rs})})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
