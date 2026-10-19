MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
month = MONTHS.index(re.search(r"occur in (\w+) of any year", question).group(1).lower()) + 1
sub = [r for r in rows if r["date"].month == month]
