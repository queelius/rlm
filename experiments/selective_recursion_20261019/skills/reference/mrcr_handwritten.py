"""Hand-written MRCR routine (reference ceiling for E21; not part of the learned library).

answer_mrcr(request) -> str    # the prepended string plus the assistant reply to the requested i-th ask
Typical use: FINAL_TEXT(answer_mrcr(request)).
"""
import re


def answer_mrcr(request):
    msgs = request["input"]
    last = msgs[-1]["content"]
    m = re.search(r"Prepend (\S+) to the (\d+)\w* \(1 indexed\) (.+?)\. Do not include", last)
    prefix, nth, what = m.group(1), int(m.group(2)), m.group(3).strip().lower()
    seen = 0
    for i, msg in enumerate(msgs[1:-1], start=1):  # message 0 holds few-shot examples
        if msg["role"] != "user":
            continue
        ask = msg["content"].strip().lower()
        ask = re.sub(r"^write an? ", "", ask)
        if ask == what:
            seen += 1
            if seen == nth and i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                return prefix + msgs[i + 1]["content"]
    return prefix
