lab = re.search(r"label '([^']+)' the single most frequently occur?ring label", question).group(1)
by_month = {}
for r in sub:
    if r["label"]:
        by_month.setdefault((r["date"].year, r["date"].month), Counter())[r["label"]] += 1
wins = sum(1 for c in by_month.values() if c.most_common(1)[0][0] == lab and (len(c) == 1 or c.most_common(2)[0][1] > c.most_common(2)[1][1]))
FINAL_TEXT("Answer: " + str(wins))
