SYSTEM = """{{SYSTEM_BODY}}"""
chunks = ["\n".join(lines[i:i + {{CHUNK}}]) for i in range(0, len(lines), {{CHUNK}})]
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
