"""Export OOLONG-synth validation rows (context <= 32k tokens) to JSONL for the transfer evaluation."""
import ast, glob, json, re, sys, datetime
import pyarrow.parquet as pq


def parse_gold(value):
    """The parquet answer column is a Python-literal string such as "['spam']", "[4]" or
    "[datetime.date(2025, 3, 20)]"; convert to a JSON list with ISO dates."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = re.sub(r"datetime\.date\((\d+), (\d+), (\d+)\)", lambda m: f"'{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}'", str(value))
        items = ast.literal_eval(text)
        if not isinstance(items, (list, tuple)):
            items = [items]
    out = []
    for x in items:
        if isinstance(x, (datetime.date, datetime.datetime)):
            out.append(x.isoformat())
        elif hasattr(x, "item"):
            out.append(x.item())
        else:
            out.append(x)
    return out

OUT = "/project/alex_phd/research-cache/datasets/oolong-synth-20261019/validation-le32k.jsonl"
files = sorted(glob.glob('/project/alex_phd/cache/huggingface/hub/datasets--oolongbench--oolong-synth/snapshots/*/data/validation-*.parquet'))
n = 0
with open(OUT, "w") as out:
    for f in files:
        t = pq.read_table(f).to_pandas()
        t = t[t["context_len"] <= 32768]
        for _, r in t.iterrows():
            gold = parse_gold(r["answer"])
            out.write(json.dumps(dict(id=int(r["id"]), context_len=int(r["context_len"]), dataset=r["dataset"], question=r["question"],
                                      task_group=r["task_group"], task=r["task"], answer=gold, answer_type=str(r["answer_type"]),
                                      input_subset=str(r["input_subset"]), num_labels=int(r["num_labels"]), context_window_id=int(r["context_window_id"]),
                                      context=r["context_window_text"])) + "\n")
            n += 1
print("exported", n, "rows to", OUT)
