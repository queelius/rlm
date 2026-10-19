def is_item(e, target):
    x = str(e.get("item") or "").strip().lower()
    return x in (target, target + "s", target + "es")
def who(e):
    return str(e.get("person") or "").strip()
def qty(e):
    try:
        return int(e.get("qty") or 0)
    except (TypeError, ValueError):
        return 0
def day(e):
    m = re.search(r"\d+", str(e.get("day") or ""))
    return int(m.group(0)) if m else -1
buys = [e for e in events if e.get("kind") == "buy"]
rets = [e for e in events if e.get("kind") == "return"]
if m := re.search(r"at least one (\w+) and also bought at least one (\w+)\?", question):
    a, b = m.groups()
    both = sorted({who(e) for e in buys if is_item(e, a)} & {who(e) for e in buys if is_item(e, b)})
    answer = ", ".join(both) if both else "none"
elif m := re.search(r"bought at least one (\w+)\?", question):
    a = m.group(1)
    answer = str(len({who(e) for e in buys if is_item(e, a)}))
elif m := re.search(r"How many units of (\w+) were bought", question):
    a = m.group(1)
    answer = str(sum(qty(e) for e in buys if is_item(e, a)) - sum(qty(e) for e in rets if is_item(e, a)))
elif m := re.search(r"Which people bought a (\w+) and later", question):
    a = m.group(1)
    first = {}
    for e in buys:
        if is_item(e, a):
            first[who(e)] = min(first.get(who(e), 10**9), day(e))
    people = sorted({who(e) for e in rets if is_item(e, a) and who(e) in first and day(e) > first[who(e)]})
    answer = ", ".join(people) if people else "none"
else:
    answer = "unknown"
FINAL_TEXT("ANSWER: " + answer)
