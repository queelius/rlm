SYSTEM = """You classify store log lines. For EVERY numbered input line output exactly one JSON object, in the same order, inside one JSON array. Object format: {"line": <line number>, "kind": "buy" | "return" | "none", "person": <name or null>, "item": <singular item noun or null>, "qty": <integer or null>, "day": <integer or null>}. Use "buy" only for a completed purchase (bought, purchased, paid for, picked up per a receipt, a sale logged by the register, or an order that went through). Use "return" only for a completed return or refund of items. Everything else (complaints, reviews, recommendations, wishlists, considering or declining a purchase, almost returning, stock questions, store notices, deliveries) is "none". Output only the JSON array."""
chunks = ["\n".join(lines[i:i + 12]) for i in range(0, len(lines), 12)]
batch = ask_batch(["Lines:\n" + c + "\n\nJSON array:" for c in chunks], system=SYSTEM,
                  options={"temperature": 0, "max_output_tokens": 2048})
events = []
for text in batch.texts:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        continue
    try:
        events += [e for e in json.loads(m.group(0)) if isinstance(e, dict) and e.get("kind") in ("buy", "return")]
    except json.JSONDecodeError:
        pass
print(len(chunks), "chunks,", len(events), "events,", len(batch.failed_indexes), "failed")
