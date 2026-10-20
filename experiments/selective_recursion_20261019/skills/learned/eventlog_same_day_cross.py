# families: eventlog
# question_kind: same_day_cross
# question_pattern: Which\s+people\s+bought\s+a\s+\S+\s+on\s+a\s+day\s+when\s+a\s+different\s+person\s+returned\s+a\s+\S+
"""Learned skill for same day cross questions (E08 librarian, qwen3-4b; verified 0.90 on 48 teaching tasks with model-extracted events).

answer_same_day_cross(events, question) -> str      # exact aggregation over extract_events output; returns "ANSWER: ..."
Typical use: lines, question = split_log(request["input"]); events = extract_events(lines); FINAL_TEXT(answer_same_day_cross(events, question)).
"""
def answer_same_day_cross(events, question):
    buys = [e for e in events if e.get("kind") == "buy"]
    rets = [e for e in events if e.get("kind") == "return"]
    
    # Extract the item from the question
    m = re.search(r"bought a (\w+)", question)
    if not m:
        return "ANSWER: unknown"
    
    item = m.group(1)
    
    # Find all buy events for the item
    buy_events = [e for e in buys if _is_item(e, item)]
    return_events = [e for e in rets if _is_item(e, item)]
    
    # If no relevant events, return unknown
    if not buy_events or not return_events:
        return "ANSWER: unknown"
    
    # For each person who bought the item, check if there's a return by a different person on the same day
    # We want people who bought on a day when someone else returned the item
    result = set()
    
    for buy_event in buy_events:
        buy_day = _day(buy_event)
        # Check if any return event happened on the same day
        same_day_returns = [e for e in return_events if _day(e) == buy_day]
        # If there's at least one return on that day, and it's by a different person than the buyer
        if same_day_returns:
            # Get the person who bought on that day
            buyer = _who(buy_event)
            # Check if any returner is not the buyer
            for return_event in same_day_returns:
                returner = _who(return_event)
                if returner != buyer:
                    result.add(buyer)
                    break
    
    # Format the answer
    answer = ", ".join(sorted(result)) if result else "none"
    return "ANSWER: " + answer
