rank = 1 if "second most" in question else 0
FINAL_TEXT("User: " + Counter(r["user"] for r in sub).most_common()[rank][0])
