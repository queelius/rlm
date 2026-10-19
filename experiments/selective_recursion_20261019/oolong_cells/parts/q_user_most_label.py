lab = re.search(r"which user has the most instances with the label (\w[\w ]*?)\?", question).group(1).strip()
users = Counter(r["user"] for r in sub if r["label"] == lab)
FINAL_TEXT("User: " + users.most_common(1)[0][0])
