"""Store-log skills (verified on the synthetic event-log family, 8 to 512 records).

split_log(text) -> (lines, question)          # numbered log lines and the question
extract_events(lines, chunk=12) -> list[dict]  # helper calls per chunk; dicts: kind buy/return, person, item, qty, day
answer_eventlog(events, question) -> str      # exact aggregation in Python for every question type, lookups included; returns "ANSWER: ..."
lookup_day(lines, question) -> str            # cheaper alternative for "On which day did X buy a Y?": one helper call on the lines naming the person
Typical use: lines, question = split_log(request["input"]); events = extract_events(lines);
FINAL_TEXT(answer_eventlog(events, question)). For "On which day did X buy a Y?" use lookup_day.
"""
import json
import re

EXTRACT_SYSTEM = (
    'You classify store log lines. For EVERY numbered input line output exactly one JSON object, in the same '
    'order, inside one JSON array. Object format: {"line": <line number>, "kind": "buy" | "return" | "none", '
    '"person": <name or null>, "item": <singular item noun or null>, "qty": <integer or null>, "day": <integer or null>}. '
    'Use "buy" only for a completed purchase (bought, purchased, paid for, picked up per a receipt, a sale logged by '
    'the register, or an order that went through). Use "return" only for a completed return or refund of items. '
    'Everything else (complaints, reviews, recommendations, wishlists, considering or declining a purchase, almost '
    'returning, stock questions, store notices, deliveries) is "none". Output only the JSON array.'
)


def split_log(text):
    log = text.split("LOG (", 1)[1].split("\n", 1)[1].split("\n\nQUESTION:", 1)[0]
    question = text.split("QUESTION:", 1)[1].split("\n", 1)[0].strip()
    return log.split("\n"), question


def extract_events(lines, chunk=12):
    chunks = ["\n".join(lines[i:i + chunk]) for i in range(0, len(lines), chunk)]
    batch = ask_batch(["Lines:\n" + c + "\n\nJSON array:" for c in chunks], system=EXTRACT_SYSTEM,
                      options={"temperature": 0, "max_output_tokens": 2048})
    events = []
    for out in batch.texts:
        m = re.search(r"\[.*\]", out or "", re.S)
        if not m:
            continue
        try:
            events += [e for e in json.loads(m.group(0)) if isinstance(e, dict) and e.get("kind") in ("buy", "return")]
        except json.JSONDecodeError:
            pass
    return events


def _is_item(e, target):
    x = str(e.get("item") or "").strip().lower()
    return x in (target, target + "s", target + "es")


def _who(e):
    return str(e.get("person") or "").strip()


def _qty(e):
    try:
        return int(e.get("qty") or 0)
    except (TypeError, ValueError):
        return 0


def _day(e):
    m = re.search(r"\d+", str(e.get("day") or ""))
    return int(m.group(0)) if m else -1


def answer_eventlog(events, question):
    buys = [e for e in events if e.get("kind") == "buy"]
    rets = [e for e in events if e.get("kind") == "return"]
    if m := re.search(r"did (\w+) buy an? (\w+)\?", question):
        # lookup question answered from the extracted events: the earliest matching completed purchase
        person, item = m.groups()
        days = sorted(_day(e) for e in buys if _who(e).lower() == person.lower() and _is_item(e, item) and _day(e) >= 0)
        return "ANSWER: " + (str(days[0]) if days else "unknown")
    if m := re.search(r"at least one (\w+) and also bought at least one (\w+)\?", question):
        a, b = m.groups()
        both = sorted({_who(e) for e in buys if _is_item(e, a)} & {_who(e) for e in buys if _is_item(e, b)})
        answer = ", ".join(both) if both else "none"
    elif m := re.search(r"bought at least one (\w+)\?", question):
        a = m.group(1)
        answer = str(len({_who(e) for e in buys if _is_item(e, a)}))
    elif m := re.search(r"How many units of (\w+) were bought", question):
        a = m.group(1)
        answer = str(sum(_qty(e) for e in buys if _is_item(e, a)) - sum(_qty(e) for e in rets if _is_item(e, a)))
    elif m := re.search(r"Which people bought a (\w+) and later", question):
        a = m.group(1)
        first = {}
        for e in buys:
            if _is_item(e, a):
                first[_who(e)] = min(first.get(_who(e), 10**9), _day(e))
        people = sorted({_who(e) for e in rets if _is_item(e, a) and _who(e) in first and _day(e) > first[_who(e)]})
        answer = ", ".join(people) if people else "none"
    else:
        answer = "unknown"
    return "ANSWER: " + answer


def lookup_day(lines, question):
    person, item = re.search(r"did (\w+) buy an? (\w+)\?", question).groups()
    hits = [l for l in lines if person.lower() in l.lower() and item.lower()[:4] in l.lower()] or lines
    reply = ask(
        "Lines:\n" + "\n".join(hits) + f"\n\nOn which day did {person} buy a {item}? Count only a "
        "completed purchase (bought, purchased, paid for, picked up per a receipt, a logged sale, or an "
        "order that went through); ignore complaints, reviews, wishlists, considered or declined "
        "purchases, and returns. Finish with one line of the form ANSWER: <day number>.",
        options={"temperature": 0, "max_output_tokens": 512},
    )
    m = re.search(r"ANSWER:\s*(\d+)", reply)
    nums = re.findall(r"\d+", reply)
    return "ANSWER: " + (m.group(1) if m else (nums[-1] if nums else reply.strip()))
