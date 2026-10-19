counts = Counter(r["label"] for r in sub if r["label"])
FINAL_TEXT("Label: " + counts.most_common(1)[0][0])
