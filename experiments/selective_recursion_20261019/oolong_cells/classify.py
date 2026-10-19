SYSTEM = ("You label short texts. For EVERY numbered input line output exactly one JSON object, in the same "
          "order, inside one JSON array: {\"line\": <line number>, \"label\": <one of the allowed labels>}. "
          "Allowed labels: " + ", ".join(repr(l) for l in labels) + ". Output only the JSON array.")
numbered = [f"{i + 1}. {r['text']}" for i, r in enumerate(rows)]
chunks = ["\n".join(numbered[i:i + 12]) for i in range(0, len(numbered), 12)]
batch = ask_batch(["Lines:\n" + c + "\n\nJSON array:" for c in chunks], system=SYSTEM,
                  options={"temperature": 0, "max_output_tokens": 1024})
for r in rows:
    r["label"] = None
for text_out in batch.texts:
    m = re.search(r"\[.*\]", text_out or "", re.S)
    if not m:
        continue
    try:
        for e in json.loads(m.group(0)):
            if isinstance(e, dict) and isinstance(e.get("line"), int) and 1 <= e["line"] <= len(rows):
                lab = str(e.get("label") or "").strip().lower()
                rows[e["line"] - 1]["label"] = lab if lab in labels else next((l for l in labels if l in lab or lab in l), lab)
    except json.JSONDecodeError:
        pass
print(len(chunks), "chunks;", sum(r["label"] is None for r in rows), "unlabeled;", len(batch.failed_indexes), "failed")
