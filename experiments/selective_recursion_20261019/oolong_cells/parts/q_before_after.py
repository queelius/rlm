lab, d = re.search(r"was label '([^']+)' more common, less common, or the same frequency before (\d{4}-\d{2}-\d{2})", question).groups()
d = datetime.strptime(d, "%Y-%m-%d").date()
before = sum(1 for r in sub if r["label"] == lab and r["date"] < d) / (sum(1 for r in sub if r["date"] < d) or 1)
after = sum(1 for r in sub if r["label"] == lab and r["date"] > d) / (sum(1 for r in sub if r["date"] > d) or 1)
x = "more common" if before > after else "less common" if before < after else "the same frequency"
FINAL_TEXT(f"Answer: {lab} is {x} before {d.isoformat()}")
