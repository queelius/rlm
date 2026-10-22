"""Two-round retrieval for multi-hop questions over long paragraph haystacks, v4b (E22f; tolerant of helper faults).

answer_musique(request) -> str    # BM25 top 20 for the question; the helper names the missing fact as a query; BM25 top 20 more for it; one read of the union; "ANSWER: <answer>"
bm25(paragraphs, query, k=15) -> list    # indices of the k best paragraphs for the query
Typical use: FINAL_TEXT(answer_musique(request)).
"""
import re


def parse(text):
    body, q = text.split("\n\nQUESTION:", 1)
    question = q.split("\n", 1)[0].strip()
    paras = re.findall(r"^\[(\d+)\] (.*)$", body, re.M)
    return [p for _, p in paras], question


import math as _math
from collections import Counter as _Counter

_STOP = set("a an the of in on at to for from by with and or is was were are be been who what which when where how why "
            "whose whom that this these those did does do its it his her their as than into".split())


def _toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


def bm25(paragraphs, query, k=15):
    docs = [_toks(p) for p in paragraphs]
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(n, 1)
    df = _Counter(w for d in docs for w in set(d))
    q = _toks(query)
    sc = []
    for i, d in enumerate(docs):
        tf = _Counter(d)
        s = sum(_math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg))
                for w in q if w in tf)
        sc.append((s, i))
    return [i for _, i in sorted(sc, reverse=True)[:k]]


_FOLLOWUP = ("You help answer a multi-step question from a large collection of paragraphs. You are shown the paragraphs "
             "found so far. Name the one fact that is still missing (usually about an entity found in these paragraphs) "
             "and write a short search query for it, on one line 'QUERY: <query>'. If nothing is missing, write "
             "'QUERY: DONE'.")


def _ask_or_none(prompt, system=None, options=None):
    """The kernel's ask raises when a reply is cut off or empty; here that means: no text."""
    try:
        return ask(prompt, system=system, options=options)
    except Exception:  # noqa: BLE001
        return None


def answer_musique(request):
    text = request["input"] if isinstance(request["input"], str) else request["input"][-1]["content"]
    paragraphs, question = parse(text)
    first = bm25(paragraphs, question, k=20)
    ctx1 = "\n\n".join(f"[{i + 1}] {paragraphs[i]}" for i in sorted(first))
    fq = _ask_or_none(f"Paragraphs found so far:\n{ctx1}\n\nQuestion: {question}", system=_FOLLOWUP,
                      options={"temperature": 0, "max_output_tokens": 800})
    mq = re.findall(r"QUERY:\s*(.+)", fq or "")
    q2 = mq[-1].strip() if mq else ""
    second = [i for i in bm25(paragraphs, q2 + " " + question, k=60) if i not in set(first)][:20] \
        if q2 and q2.upper() != "DONE" else []
    keep = sorted(set(first) | set(second))
    ctx = "\n\n".join(f"[{i + 1}] {paragraphs[i]}" for i in keep)
    sys_p = ("Answer the multi-step question from the paragraphs. Work out each step briefly, then finish with one line "
             "'ANSWER: <short answer>' giving a short phrase (a name, place, date or number). Always give your best answer.")
    out = _ask_or_none(f"Paragraphs:\n{ctx}\n\nQuestion: {question}", system=sys_p,
                       options={"temperature": 0, "max_output_tokens": 1500})
    m = re.findall(r"ANSWER:\s*(.+)", out or "")
    if not m:  # the reading was cut off (or had no ANSWER line): ask once for the short answer alone
        short = ("Answer the question from the paragraphs with a short phrase only (a name, place, date or number), "
                 "on one line 'ANSWER: <answer>'.")
        out2 = _ask_or_none(f"Paragraphs:\n{ctx}\n\nQuestion: {question}", system=short,
                            options={"temperature": 0, "max_output_tokens": 60})
        m = re.findall(r"ANSWER:\s*(.+)", out2 or "") or ([out2.strip().splitlines()[0]] if (out2 or "").strip() else [])
    return "ANSWER: " + (m[-1].strip() if m else "unknown")
