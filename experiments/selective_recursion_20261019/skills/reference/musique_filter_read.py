"""Filter-then-read routine for long paragraph haystacks (reference for E22; not part of the learned library).

answer_musique(request) -> str    # helper filter (chunks of 15) united with BM25 top 15, then one read of the kept paragraphs; "ANSWER: <answer>"
select_relevant(paragraphs, question, chunk=15) -> list    # indices of paragraphs that may help with any step of the question
bm25(paragraphs, query, k=15) -> list    # indices of the k best paragraphs by BM25
Typical use: FINAL_TEXT(answer_musique(request)).
"""
import re


def parse(text):
    body, q = text.split("\n\nQUESTION:", 1)
    question = q.split("\n", 1)[0].strip()
    paras = re.findall(r"^\[(\d+)\] (.*)$", body, re.M)
    return [p for _, p in paras], question


def select_relevant(paragraphs, question, chunk=15):
    sys_p = ("You select evidence for a multi-step question. A paragraph is relevant if it states a fact needed for ANY "
             "step of the reasoning, including intermediate entities the question does not name. Reply with the "
             "paragraph numbers only, comma-separated, or 'none'.")
    starts = list(range(0, len(paragraphs), chunk))
    prompts = []
    for s in starts:
        block = "\n".join(f"[{s + i + 1}] {p}" for i, p in enumerate(paragraphs[s:s + chunk]))
        prompts.append(f"Question: {question}\n\nParagraphs:\n{block}\n\nRelevant paragraph numbers:")
    outs = ask_batch(prompts, system=sys_p, options={"temperature": 0, "max_output_tokens": 120}).texts
    keep = set()
    for s, out in zip(starts, outs):
        for n in re.findall(r"\d+", out or ""):
            i = int(n) - 1
            if s <= i < s + chunk:
                keep.add(i)
    return sorted(keep)


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


def answer_musique(request):
    text = request["input"] if isinstance(request["input"], str) else request["input"][-1]["content"]
    paragraphs, question = parse(text)
    keep = sorted(set(select_relevant(paragraphs, question)) | set(bm25(paragraphs, question)))
    if len(keep) > 60:
        keep = keep[:60]
    ctx = "\n\n".join(f"[{i + 1}] {paragraphs[i]}" for i in keep) or "\n\n".join(paragraphs[:40])
    sys_p = ("Answer the multi-step question from the paragraphs. Work out each step briefly, then finish with one line "
             "'ANSWER: <short answer>' giving a short phrase (a name, place, date or number). Always give your best answer.")
    out = ask(f"Paragraphs:\n{ctx}\n\nQuestion: {question}", system=sys_p, options={"temperature": 0, "max_output_tokens": 600})
    m = re.findall(r"ANSWER:\s*(.+)", out or "")
    return "ANSWER: " + (m[-1].strip() if m else (out or "unknown").strip().splitlines()[-1])
