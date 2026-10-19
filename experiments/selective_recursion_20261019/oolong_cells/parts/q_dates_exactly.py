k = int(re.search(r"represented exactly (\d+) times", question).group(1))
FINAL_TEXT("Answer: " + str(sum(1 for c in Counter(r["date"] for r in sub).values() if c == k)))
