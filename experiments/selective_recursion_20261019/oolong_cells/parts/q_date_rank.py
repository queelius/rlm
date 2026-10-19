rank = 1 if "second most" in question else 0
d = Counter(r["date"] for r in sub).most_common()[rank][0]
FINAL_TEXT(f"Date: {d.month:02d}/{d.day:02d}/{d.year:04d}")
