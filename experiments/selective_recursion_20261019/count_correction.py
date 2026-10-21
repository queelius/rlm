"""E16: confusion-corrected counting for OOLONG numeric-one-class questions.

The helper's per-row confusion matrix per dataset is estimated on teaching windows only (E15).
On held-out numeric-one-class questions the library classifies rows as usual, then the predicted
label counts in the question's subset are corrected by solving M^T x = c (least squares clipped at
zero, rescaled to the subset size, rounded). Scored against gold on the held-out rows.
"""

import argparse
import collections
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oolong_eval  # noqa: E402
import oolong_variants  # noqa: E402
import teach_oolong  # noqa: E402


def confusion(windows_path):
    conf = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for line in open(windows_path):
        w = json.loads(line)
        for g, h in w["pairs"]:
            conf[w["dataset"]][g][h] += 1
    return conf


def correct(counts: dict, labels: list, conf_ds) -> dict:
    L = list(labels)
    M = np.zeros((len(L), len(L)))
    for i, g in enumerate(L):
        tot = sum(conf_ds[g].values()) or 1
        for j, h in enumerate(L):
            M[i, j] = conf_ds[g][h] / tot
    c = np.array([counts.get(l, 0) for l in L], dtype=float)
    x, *_ = np.linalg.lstsq(M.T, c, rcond=None)
    x = np.clip(x, 0, None)
    if x.sum() > 0:
        x = x * c.sum() / x.sum()
    return {l: int(round(v)) for l, v in zip(L, x)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--windows", default="/project/alex_phd/runs/selective-recursion-20261019/e15-row-accuracy/windows.jsonl")
    p.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--model", default="qwen3-4b")
    a = p.parse_args()
    from openai import OpenAI

    conf = confusion(a.windows)
    rows_all = [r for r in teach_oolong.load_rows("test", 32768) if r["task"].endswith("NUMERIC_ONE_CLASS")]
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=600)
    pool = ThreadPoolExecutor(16)
    ns = {"ask_batch": oolong_variants.make_ask_batch(client, a.model, pool), "ask": None}
    exec(compile((HERE / "skills" / "oolong.py").read_text(), "oolong", "exec"), ns)

    def one(r):
        rows, labels, question = ns["parse_oolong"](r["context"] + "\n" + r["question"])
        ns["classify_rows"](rows, labels)
        plain = ns["answer_oolong"](rows, labels, question)
        sub = ns["_subset"](rows, question)
        counts = collections.Counter(x["label"] for x in sub if x["label"])
        m = re.search(r"classified as label '([^']+)'", question)
        corrected = plain
        if m:
            fixed = correct(counts, labels, conf[r["dataset"]])
            corrected = "Answer: " + str(fixed.get(m.group(1), 0))
        return dict(id=r["id"], dataset=r["dataset"], context_len=r["context_len"], gold=r["answer"],
                    plain=plain, corrected=corrected, plain_ok=oolong_eval.verify(r, plain),
                    corrected_ok=oolong_eval.verify(r, corrected), n_subset=len(sub))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, rows_all))
    (out / "rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in res))
    for ds in ("spam", "trec_coarse", None):
        rs = [x for x in res if ds is None or x["dataset"] == ds]
        print(ds or "all", len(rs), "plain", sum(x["plain_ok"] for x in rs), "corrected", sum(x["corrected_ok"] for x in rs),
              "wins", sum(x["corrected_ok"] and not x["plain_ok"] for x in rs), "losses", sum(x["plain_ok"] and not x["corrected_ok"] for x in rs))


if __name__ == "__main__":
    main()
