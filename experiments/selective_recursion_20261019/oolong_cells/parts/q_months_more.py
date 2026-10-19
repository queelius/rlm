a, b = re.search(r"label '([^']+)' occur more frequently than the label '([^']+)'", question).groups()
by_month = Counter((r["date"].year, r["date"].month, r["label"]) for r in sub if r["label"] in (a, b))
months = {(y, m) for (y, m, _) in by_month}
FINAL_TEXT("Answer: " + str(sum(1 for (y, m) in months if by_month[(y, m, a)] > by_month[(y, m, b)])))
