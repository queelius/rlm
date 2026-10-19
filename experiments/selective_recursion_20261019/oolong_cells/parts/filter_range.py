lo, hi = (datetime.strptime(s, "%b %d, %Y").date() for s in re.search(r"occur between (\w+ \d+, \d{4}) and (\w+ \d+, \d{4}), inclusive", question).groups())
sub = [r for r in rows if lo <= r["date"] <= hi]
