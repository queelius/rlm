"""Record primitives for labelled-instance logs (E35b: give the proposer parsing and classification, see what it builds).

parse_oolong(text) -> (rows, labels, question)   # rows: list of {date (datetime.date), user (str), text, label (None)}; labels from the header; question = last non-empty line
classify_rows(rows, labels, chunk=12) -> None    # asks the helper to label each row's text with one of the labels; sets r["label"] in place
"""
import json
import re
from collections import Counter
from datetime import datetime

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


def parse_oolong(text):
    rows = []
    for line in text.split("\n"):
        m = re.match(r"Date: (.*?) \|\| User: (\d+) \|\| Instance: (.*)$", line)
        if m:
            rows.append({"date": datetime.strptime(m.group(1).strip(), "%b %d, %Y").date(), "user": m.group(2), "text": m.group(3), "label": None})
    question = [l for l in text.split("\n") if l.strip()][-1].strip()
    header = text.split("\nDate:", 1)[0]
    m = re.search(r"classified as (\w+) or (\w+)", header)
    labels = list(m.groups()) if m else (re.findall(r"'([^']+)'", header.split("categories:", 1)[1].split("\n", 1)[0]) if "categories:" in header else [])
    return rows, labels, question


def classify_rows(rows, labels, chunk=12):
    system = ('You label short texts. For EVERY numbered input line output exactly one JSON object, in the same order, '
              'inside one JSON array: {"line": <line number>, "label": <one of the allowed labels>}. Allowed labels: '
              + ", ".join(repr(l) for l in labels) + ". Output only the JSON array.")
    numbered = [f"{i + 1}. {r['text']}" for i, r in enumerate(rows)]
    chunks = ["\n".join(numbered[i:i + chunk]) for i in range(0, len(numbered), chunk)]
    batch = ask_batch(["Lines:\n" + c + "\n\nJSON array:" for c in chunks], system=system,
                      options={"temperature": 0, "max_output_tokens": 1024})
    for out in batch.texts:
        m = re.search(r"\[.*\]", out or "", re.S)
        if not m:
            continue
        try:
            for e in json.loads(m.group(0)):
                if isinstance(e, dict) and isinstance(e.get("line"), int) and 1 <= e["line"] <= len(rows):
                    lab = str(e.get("label") or "").strip().lower()
                    rows[e["line"] - 1]["label"] = lab if lab in labels else next((l for l in labels if l in lab or lab in l), lab)
        except json.JSONDecodeError:
            pass
