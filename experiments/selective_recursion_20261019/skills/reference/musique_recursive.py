"""Recursive multi-hop routine for long paragraph haystacks (reference for E23, question 4).

answer_musique(request) -> str    # decompose the question; each hop is solved by a child RLM on the whole haystack (depth 1), falling back to filter-then-read where recursion is disabled; "ANSWER: <answer>"
answer_single(request) -> str     # filter-then-read for one question (the depth-0 routine)
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


def answer_single(request):
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


def _decompose(question):
    import json as _json
    sys_p = ("Break a multi-hop question into the shortest chain of simple single-hop questions. Later questions refer "
             "to earlier answers as #1, #2, ... Return only a JSON list of strings.")
    out = ask(f"Question: {question}\nJSON list:", system=sys_p, options={"temperature": 0, "max_output_tokens": 300})
    m = re.search(r"\[.*\]", out or "", re.S)
    try:
        hops = [h for h in _json.loads(m.group(0)) if isinstance(h, str) and h.strip()] if m else []
    except ValueError:
        hops = []
    return hops or [question]


def _child_text(resp):
    return "".join(c.get("text", "") for o in resp.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []) if c.get("type") == "output_text")


def answer_musique(request):
    text = request["input"] if isinstance(request["input"], str) else request["input"][-1]["content"]
    body, rest = text.split("\n\nQUESTION:", 1)
    question = rest.split("\n", 1)[0].strip()
    hops = _decompose(question)
    if len(hops) <= 1:
        return answer_single(request)
    answers = []
    for h in hops:
        q = h
        for j, a in enumerate(answers, start=1):
            q = q.replace(f"#{j}", a)
        sub = (f"{body}\n\nQUESTION: {q}\nAnswer with a short phrase. Finish with one final line of the form ANSWER: <answer>.")
        try:
            resp = rlm_complete({"model": request.get("model", "qwen3-4b"), "input": sub, "max_output_tokens": 4096})
            out = _child_text(resp)
        except Exception:  # recursion disabled at this depth, or the child failed: solve the hop here
            out = answer_single({"input": sub})
        m = re.findall(r"ANSWER:\s*(.+)", out or "")
        answers.append((m[-1] if m else (out or "unknown")).strip().strip("."))
    return "ANSWER: " + answers[-1]
