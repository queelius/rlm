"""Score E18 raw outputs with this repository's OOLONG verifier (strict) and a lenient bare-value parse."""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oolong_eval  # noqa: E402

S = Path("/project/alex_phd/runs/selective-recursion-20261019")
rows = {json.loads(l)["id"]: json.loads(l) for l in open(oolong_eval.ROWS)}
raw = [json.loads(l) for l in open(S / "e18-official-rlm8b" / "raw.jsonl")]


def lenient(r, text):
    if oolong_eval.verify(r, text):
        return True
    t = (text or "").strip().splitlines()[-1] if (text or "").strip() else ""
    return oolong_eval.verify(r, "Answer: " + t) or oolong_eval.verify(r, "Label: " + t)


by = collections.defaultdict(lambda: [0, 0, 0, 0])
for x in raw:
    r = rows[x["id"]]
    s, l = oolong_eval.verify(r, x["output_text"] or ""), lenient(r, x["output_text"])
    for k in (x["dataset"], x["context_len"], "all"):
        b = by[k]; b[0] += 1; b[1] += s; b[2] += l; b[3] += bool(x["error"])
lib = {json.loads(l)["id"]: json.loads(l)["correct"] for l in open(S / "oolong-lib-base-full" / "rows.jsonl")}
ids = [x["id"] for x in raw]
print("rows", len(raw), "| library (same rows)", sum(lib[i] for i in ids))
for k, v in sorted(by.items(), key=lambda kv: str(kv[0])):
    print(f"{str(k):<12} n={v[0]:<4} strict {v[1]:<4} lenient {v[2]:<4} errors {v[3]}")
