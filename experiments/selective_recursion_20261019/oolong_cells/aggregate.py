from collections import Counter
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
q = question
sub = rows
if m := re.search(r"associated with user IDs? ([\d, ]+)\.", q):
    ids = set(re.findall(r"\d+", m.group(1)))
    sub = [r for r in rows if r["user"] in ids]
elif m := re.search(r"occur in (\w+) of any year", q):
    month = MONTHS.index(m.group(1).lower()) + 1
    sub = [r for r in rows if r["date"].month == month]
elif m := re.search(r"occur between (\w+ \d+, \d{4}) and (\w+ \d+, \d{4}), inclusive", q):
    lo, hi = (datetime.strptime(s, "%b %d, %Y").date() for s in m.groups())
    sub = [r for r in rows if lo <= r["date"] <= hi]
counts = Counter(r["label"] for r in sub if r["label"])
def cmp_phrase(a, b, suffix):
    return ("more common" if a > b else "less common" if a < b else ("the same frequency" if suffix == "" else "same frequency")) + suffix
if m := re.search(r"was label '([^']+)' more common, less common, or the same frequency before (\d{4}-\d{2}-\d{2})", q):
    lab, d = m.group(1), datetime.strptime(m.group(2), "%Y-%m-%d").date()
    # OOLONG compares the label's share of instances before versus after the date, not raw counts.
    nb = sum(1 for r in sub if r["date"] < d) or 1
    na = sum(1 for r in sub if r["date"] > d) or 1
    before = sum(1 for r in sub if r["label"] == lab and r["date"] < d) / nb
    after = sum(1 for r in sub if r["label"] == lab and r["date"] > d) / na
    x = "more common" if before > after else "less common" if before < after else "the same frequency"
    answer = f"Answer: {lab} is {x} before {m.group(2)}"
elif m := re.search(r"For how many months does the label '([^']+)' occur more frequently than the label '([^']+)'", q):
    a, b = m.groups()
    by_month = Counter()
    for r in sub:
        if r["label"] in (a, b):
            by_month[(r["date"].year, r["date"].month, r["label"])] += 1
    months = {k[:2] for k in by_month}
    answer = "Answer: " + str(sum(1 for mo in months if by_month[(mo[0], mo[1], a)] > by_month[(mo[0], mo[1], b)]))
elif m := re.search(r"is label '([^']+)' more common, less common, or the same frequency as label '([^']+)'", q):
    a, b = m.groups()
    x = "more common than" if counts[a] > counts[b] else "less common than" if counts[a] < counts[b] else "same frequency as"
    answer = f"Answer: {a} is {x} {b}"
elif m := re.search(r"how many data points should be classified as label '([^']+)'", q):
    answer = "Answer: " + str(counts[m.group(1)])
elif m := re.search(r"which user has more instances with the label (\w[\w ]*?): User (\d+) or User (\d+)\?", q):
    lab, u1, u2 = m.group(1).strip(), m.group(2), m.group(3)
    c1 = sum(1 for r in sub if r["user"] == u1 and r["label"] == lab)
    c2 = sum(1 for r in sub if r["user"] == u2 and r["label"] == lab)
    answer = "User: " + (u1 if c1 >= c2 else u2)
elif m := re.search(r"For how many months is the label '([^']+)' the single most frequently occur?ring label", q):
    lab = m.group(1)
    by_month = {}
    for r in sub:
        if r["label"]:
            by_month.setdefault((r["date"].year, r["date"].month), Counter())[r["label"]] += 1
    wins = 0
    for c in by_month.values():
        top = c.most_common()
        if top and top[0][0] == lab and (len(top) == 1 or top[0][1] > top[1][1]):
            wins += 1
    answer = "Answer: " + str(wins)
elif m := re.search(r"which user has the most instances with the label (\w[\w ]*?)\?", q):
    lab = m.group(1).strip()
    users = Counter(r["user"] for r in sub if r["label"] == lab)
    answer = "User: " + (users.most_common(1)[0][0] if users else "none")
elif "which of the labels is the most common" in q:
    answer = "Label: " + (counts.most_common(1)[0][0] if counts else "none")
elif "which of the labels is the least common" in q:
    present = [l for l in labels if counts[l] > 0] or labels
    answer = "Label: " + min(present, key=lambda l: counts[l])
else:
    answer = "Answer: unknown"
FINAL_TEXT(answer)
