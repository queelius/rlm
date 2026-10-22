"""Hand-written multi-hop routine for long paragraph haystacks (reference for E22; not part of the learned library).

answer_musique(request) -> str    # decompose, retrieve per hop with BM25, read with the helper; "ANSWER: <answer>"
decompose(question) -> list       # helper call: chain of single-hop sub-questions using #1, #2 for earlier answers
retrieve(paragraphs, query, k=8) -> list    # BM25 over the numbered paragraphs, best k (index, text)
read_hop(question, passages) -> str         # helper call: short answer from the passages, or "unknown"
Typical use: FINAL_TEXT(answer_musique(request)).
"""
import json
import math
import re
from collections import Counter

_STOP = set("a an the of in on at to for from by with and or is was were are be been who what which when where "
            "how why whose whom that this these those did does do its it his her their as than into".split())


def _toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


def parse(text):
    body, q = text.split("\n\nQUESTION:", 1)
    question = q.split("\n", 1)[0].strip()
    paras = re.findall(r"^\[(\d+)\] (.*)$", body, re.M)
    return [p for _, p in paras], question


def retrieve(paragraphs, query, k=8):
    docs = [_toks(p) for p in paragraphs]
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(n, 1)
    df = Counter(w for d in docs for w in set(d))
    q = _toks(query)
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append((s, i))
    scores.sort(reverse=True)
    return [(i, paragraphs[i]) for _, i in scores[:k]]


def decompose(question):
    sys_p = ("Break a multi-hop question into the shortest chain of simple single-hop questions. Later questions refer "
             "to earlier answers as #1, #2, ... Return only a JSON list of strings.")
    out = ask(f"Question: {question}\nJSON list:", system=sys_p, options={"temperature": 0, "max_output_tokens": 300})
    m = re.search(r"\[.*\]", out or "", re.S)
    try:
        hops = [h for h in json.loads(m.group(0)) if isinstance(h, str) and h.strip()] if m else []
    except json.JSONDecodeError:
        hops = []
    return hops or [question]


def read_hop(question, passages):
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in passages)
    sys_p = ("Answer the question using only the passages. Reply with a short phrase only (a name, place, date or "
             "number), or 'unknown' if the passages do not contain the answer.")
    out = ask(f"Passages:\n{ctx}\n\nQuestion: {question}\nShort answer:", system=sys_p,
              options={"temperature": 0, "max_output_tokens": 40})
    return (out or "unknown").strip().strip(".").splitlines()[0] if out else "unknown"


def answer_musique(request):
    paragraphs, question = parse(request["input"] if isinstance(request["input"], str) else request["input"][-1]["content"])
    hops = decompose(question)
    answers = []
    for h in hops:
        q = h
        for j, a in enumerate(answers, start=1):
            q = q.replace(f"#{j}", a)
        ans = "unknown"
        for k in (8, 16):
            ans = read_hop(q, retrieve(paragraphs, q + " " + question, k=k))
            if ans.lower() != "unknown":
                break
        answers.append(ans)
    return "ANSWER: " + (answers[-1] if answers else "unknown")
