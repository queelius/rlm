lab = re.search(r"classified as label '([^']+)'", question).group(1)
FINAL_TEXT("Answer: " + str(sum(1 for r in sub if r["label"] == lab)))
