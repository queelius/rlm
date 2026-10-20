# families: eventlog
# question_kind: busiest_day
# question_pattern: On\s+which\s+day\s+were\s+the\s+most\s+units\s+of\s+\S+\s+bought\s+in\s+total,\s+counting\s+every\s+purchase\s+that\s+day\s+\(returns\s+do\s+not\s+count\)\?\s+If\s+several\s+days\s+tie,\s+give\s+the\s+earliest\.\s+Answer\s+with\s+the\s+day\s+number\s+only\.
"""Learned skill for busiest day questions (E08 librarian, qwen3-8b-awq; verified 1.00 on 48 teaching tasks with model-extracted events).

answer_busiest_day(events, question) -> str      # exact aggregation over extract_events output; returns "ANSWER: ..."
Typical use: lines, question = split_log(request["input"]); events = extract_events(lines); FINAL_TEXT(answer_busiest_day(events, question)).
"""
import re
from collections import defaultdict

def answer_busiest_day(events, question):
    m = re.search(r"On which day were the most units of (\w+) bought in total", question)
    if not m:
        return "ANSWER: unknown"
    target = m.group(1)
    day_totals = defaultdict(int)
    for e in events:
        if e.get("kind") == "buy":
            if _is_item(e, target):
                day = _day(e)
                qty = _qty(e)
                if day >= 0:
                    day_totals[day] += qty
    if not day_totals:
        return "ANSWER: unknown"
    max_total = max(day_totals.values())
    candidates = [day for day, total in day_totals.items() if total == max_total]
    answer_day = min(candidates)
    return f"ANSWER: {answer_day}"
