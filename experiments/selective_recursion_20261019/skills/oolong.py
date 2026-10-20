"""OOLONG-synth skills (verified on labelled-instance aggregation questions, 1k to 32k tokens).

parse_oolong(text) -> (rows, labels, question)   # rows: {date, user, text}; labels from the header
classify_rows(rows, labels, chunk=12) -> None    # helper calls per chunk; sets r["label"] in place
answer_oolong(rows, labels, question) -> str     # exact aggregation; returns "Answer: ...", "Label: ..." or "User: ..."
Counting questions about users or dates need no classification; label questions need classify_rows first.
Typical use: rows, labels, question = parse_oolong(request["input"]); classify_rows(rows, labels);
FINAL_TEXT(answer_oolong(rows, labels, question)).
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


def _subset(rows, q):
    if m := re.search(r"associated with user IDs? ([\d, ]+)\.", q):
        ids = set(re.findall(r"\d+", m.group(1)))
        return [r for r in rows if r["user"] in ids]
    if m := re.search(r"occur in (\w+) of any year", q):
        month = MONTHS.index(m.group(1).lower()) + 1
        return [r for r in rows if r["date"].month == month]
    if m := re.search(r"occur between (\w+ \d+, \d{4}) and (\w+ \d+, \d{4}), inclusive", q):
        lo, hi = (datetime.strptime(s, "%b %d, %Y").date() for s in m.groups())
        return [r for r in rows if lo <= r["date"] <= hi]
    return rows


def answer_oolong(rows, labels, question):
    q = question
    sub = _subset(rows, q)
    counts = Counter(r["label"] for r in sub if r["label"])
    if m := re.search(r"was label '([^']+)' more common, less common, or the same frequency before (\d{4}-\d{2}-\d{2})", q):
        lab, d = m.group(1), datetime.strptime(m.group(2), "%Y-%m-%d").date()
        before = sum(1 for r in sub if r["label"] == lab and r["date"] < d) / (sum(1 for r in sub if r["date"] < d) or 1)
        after = sum(1 for r in sub if r["label"] == lab and r["date"] > d) / (sum(1 for r in sub if r["date"] > d) or 1)
        x = "more common" if before > after else "less common" if before < after else "the same frequency"
        return f"Answer: {lab} is {x} before {d.isoformat()}"
    if m := re.search(r"label '([^']+)' occur more frequently than the label '([^']+)'", q):
        a, b = m.groups()
        by_month = Counter((r["date"].year, r["date"].month, r["label"]) for r in sub if r["label"] in (a, b))
        months = {(y, mo) for (y, mo, _) in by_month}
        return "Answer: " + str(sum(1 for (y, mo) in months if by_month[(y, mo, a)] > by_month[(y, mo, b)]))
    if m := re.search(r"label '([^']+)' the single most frequently occur?ring label", q):
        lab = m.group(1)
        by_month = {}
        for r in sub:
            if r["label"]:
                by_month.setdefault((r["date"].year, r["date"].month), Counter())[r["label"]] += 1
        wins = sum(1 for c in by_month.values() if c.most_common(1)[0][0] == lab and (len(c) == 1 or c.most_common(2)[0][1] > c.most_common(2)[1][1]))
        return "Answer: " + str(wins)
    if m := re.search(r"is label '([^']+)' more common, less common, or the same frequency as label '([^']+)'", q):
        a, b = m.groups()
        x = "more common than" if counts[a] > counts[b] else "less common than" if counts[a] < counts[b] else "same frequency as"
        return f"Answer: {a} is {x} {b}"
    if m := re.search(r"classified as label '([^']+)'", q):
        return "Answer: " + str(counts[m.group(1)])
    if m := re.search(r"which user has more instances with the label (\w[\w ]*?): User (\d+) or User (\d+)\?", q):
        lab, u1, u2 = m.groups()
        c1 = sum(1 for r in sub if r["user"] == u1 and r["label"] == lab.strip())
        c2 = sum(1 for r in sub if r["user"] == u2 and r["label"] == lab.strip())
        return "User: " + (u1 if c1 >= c2 else u2)
    if m := re.search(r"which user has the most instances with the label (\w[\w ]*?)\?", q):
        users = Counter(r["user"] for r in sub if r["label"] == m.group(1).strip())
        return "User: " + users.most_common(1)[0][0]
    if "which of the labels is the most common" in q or "most common label" in q:
        return "Label: " + counts.most_common(1)[0][0]
    if "least common" in q:
        present = [l for l in labels if counts[l] > 0] or labels
        return "Label: " + min(present, key=lambda l: counts[l])
    if re.search(r"which user is represented (the )?(second )?most often", q):
        rank = 1 if "second most" in q else 0
        return "User: " + Counter(r["user"] for r in sub).most_common()[rank][0]
    if m := re.search(r"represented exactly (\d+) times", q):
        k = int(m.group(1))
        return "Answer: " + str(sum(1 for c in Counter(r["date"] for r in sub).values() if c == k))
    if re.search(r"which date is represented (the )?(second )?most often", q):
        rank = 1 if "second most" in q else 0
        d = Counter(r["date"] for r in sub).most_common()[rank][0]
        return f"Date: {d.month:02d}/{d.day:02d}/{d.year:04d}"
    return "Answer: unknown"
