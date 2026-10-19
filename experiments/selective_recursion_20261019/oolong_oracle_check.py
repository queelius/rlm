"""Offline check of the OOLONG expert's parsing/aggregation logic using the dataset's gold labels.

Runs the expert's inspect, aggregate and count cells with labels taken from
`context_window_text_with_labels` (no model calls) and scores against the gold answers.
"""
import glob, json, re, sys, collections
from pathlib import Path
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oolong_eval as o

files = sorted(glob.glob('/home/atowell/research-cache/hf/hub/datasets--oolongbench--oolong-synth/snapshots/*/data/validation-*.parquet'))
cells = {n: (HERE / "oolong_cells" / f"{n}.py").read_text() for n in ("inspect", "aggregate", "count")}
import expert as EX
USE_COMPOSER = True
NEEDS_LABELS = re.compile(r"label|classified", re.I)
stats = collections.defaultdict(lambda: [0, 0])
failures = []
for f in files:
    t = pq.read_table(f).to_pandas()
    t = t[t["context_len"] <= 32768]
    for _, r in t.iterrows():
        if str(r["answer_type"]).endswith("MONTH_YEAR"):
            continue
        gold_labels = {}
        for line in r["context_window_text_with_labels"].split("\n"):
            m = re.match(r"Date: (.*?) \|\| User: (\d+) \|\| Instance: (.*) \|\| Label: (.*)$", line)
            if m:
                gold_labels[(m.group(1).strip(), m.group(2), m.group(3).strip())] = m.group(4).strip().lower()
        ns = {"request": {"input": r["context_window_text"] + "\n" + r["question"]}}
        out = {}
        ns["FINAL_TEXT"] = lambda s: out.__setitem__("text", s)
        try:
            exec(cells["inspect"], ns)
            for row in ns["rows"]:
                row["label"] = gold_labels.get((row["date"].strftime("%b %d, %Y"), row["user"], row["text"].strip()))
            if USE_COMPOSER:
                exec(EX.oolong_aggregate_cell(ns["question"]), ns)
            else:
                needs = bool(NEEDS_LABELS.search(ns["question"]))
                exec(cells["aggregate"] if needs else cells["count"], ns)
            text = out.get("text", "")
        except Exception as exc:  # noqa: BLE001
            text = f"EXC {type(exc).__name__}: {exc}"
        gold = o.gold_value(str(r["answer_type"]), __import__("oolong_export").parse_gold(r["answer"]))
        ok = o.parse(str(r["answer_type"]), text) == gold
        key = (r["task_group"], r["task"][10:])
        stats[key][0] += 1; stats[key][1] += ok
        if not ok and len(failures) < 12:
            failures.append((key, r["context_len"], r["question"][:140], "->", text[:80], "gold", gold))
for k, (n, c) in sorted(stats.items()):
    print(f"{k[0]:<9} {k[1]:<20} {c}/{n}")
print("TOTAL", sum(c for n, c in stats.values()), "/", sum(n for n, c in stats.values()))
for fl in failures:
    print(fl)
