ids = set(re.findall(r"\d+", re.search(r"associated with user IDs? ([\d, ]+)\.", question).group(1)))
sub = [r for r in rows if r["user"] in ids]
