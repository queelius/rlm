lab, u1, u2 = re.search(r"which user has more instances with the label (\w[\w ]*?): User (\d+) or User (\d+)\?", question).groups()
c1 = sum(1 for r in sub if r["user"] == u1 and r["label"] == lab.strip())
c2 = sum(1 for r in sub if r["user"] == u2 and r["label"] == lab.strip())
FINAL_TEXT("User: " + (u1 if c1 >= c2 else u2))
