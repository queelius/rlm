counts = Counter(r["label"] for r in sub if r["label"])
present = [l for l in labels if counts[l] > 0] or labels
FINAL_TEXT("Label: " + min(present, key=lambda l: counts[l]))
