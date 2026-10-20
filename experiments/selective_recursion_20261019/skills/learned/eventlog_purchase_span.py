# families: eventlog
# question_kind: purchase_span
# question_pattern: How\s+many\s+days\s+passed\s+between\s+the\s+earliest\s+purchase\s+of\s+a\s+\S+\s+and\s+the\s+latest\s+purchase\s+of\s+a\s+\S+
"""Learned skill for purchase span questions (E08 librarian, qwen3-4b; verified 1.00 on 48 teaching tasks with model-extracted events).

answer_purchase_span(events, question) -> str      # exact aggregation over extract_events output; returns "ANSWER: ..."
Typical use: lines, question = split_log(request["input"]); events = extract_events(lines); FINAL_TEXT(answer_purchase_span(events, question)).
"""
def answer_purchase_span(events, question):
    buys = [e for e in events if e.get("kind") == "buy"]
    if m := re.search(r"between the earliest purchase of a (\w+) and the latest purchase of a (\w+)", question):
        item1, item2 = m.groups()
        if item1 == item2:
            days = sorted(_day(e) for e in buys if _is_item(e, item1) and _day(e) >= 0)
            earliest = days[0] if days else None
            latest = days[-1] if days else None
            return "ANSWER: " + str(latest - earliest) if earliest is not None and latest is not None else "ANSWER: 0"
        else:
            return "ANSWER: unknown"
    else:
        return "ANSWER: unknown"
