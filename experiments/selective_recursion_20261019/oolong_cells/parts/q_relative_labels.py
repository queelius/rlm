a, b = re.search(r"is label '([^']+)' more common, less common, or the same frequency as label '([^']+)'", question).groups()
counts = Counter(r["label"] for r in sub if r["label"])
x = "more common than" if counts[a] > counts[b] else "less common than" if counts[a] < counts[b] else "same frequency as"
FINAL_TEXT(f"Answer: {a} is {x} {b}")
