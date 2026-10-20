# families: eventlog
# question_kind: top_buyer
# question_pattern: Which\s+single\s+person\s+bought\s+the\s+most\s+units\s+of\s+\S+\s+in\s+total\s+\(returns\s+do\s+not\s+count\)\?
"""Learned skill for top buyer questions (E08 librarian, qwen3-4b; verified 0.98 on 48 teaching tasks with model-extracted events).

answer_top_buyer(events, question) -> str      # exact aggregation over extract_events output; returns "ANSWER: ..."
Typical use: lines, question = split_log(request["input"]); events = extract_events(lines); FINAL_TEXT(answer_top_buyer(events, question)).
"""
def answer_top_buyer(events, question):
    buys = [e for e in events if e.get("kind") == "buy"]
    if m := re.search(r"Which single person bought the most units of (\w+) in total", question):
        item = m.group(1)
        # Count total units bought for the item by each person
        person_units = {}
        for e in buys:
            if _is_item(e, item):
                person = _who(e)
                qty = _qty(e)
                person_units[person] = person_units.get(person, 0) + qty
        
        # Find the person with the maximum units
        if not person_units:
            answer = "none"
        else:
            top_person = max(person_units, key=person_units.get)
            answer = top_person
    else:
        answer = "unknown"
    return "ANSWER: " + answer
