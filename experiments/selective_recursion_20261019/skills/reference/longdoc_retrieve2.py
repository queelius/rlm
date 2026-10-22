"""Two-round retrieve-then-read for multiple-choice questions over a long document (E32d).

answer_longdoc(request) -> str    # BM25 over 1,000-character passages for the question and each option (20 best), the helper names what is still missing as a query, 20 more for it; one read of the union; "ANSWER: <letter>"
Typical use: FINAL_TEXT(answer_longdoc(request)).
"""
import math as _math
import re
from collections import Counter as _Counter

_STOP = set("a an the of in on at to for from by with and or is was were are be been who what which when where how why "
            "whose whom that this these those did does do its it his her their as than into which following".split())


def _toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


def _bm25_scores(docs, query):
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(n, 1)
    df = _Counter(w for d in docs for w in set(d))
    q = _toks(query)
    out = []
    for d in docs:
        tf = _Counter(d)
        out.append(sum(_math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / max(avg, 1)))
                       for w in q if w in tf))
    return out


def answer_longdoc(request):
    text = request["input"] if isinstance(request["input"], str) else request["input"][-1]["content"]
    body, rest = text.rsplit("\n\nQUESTION:", 1)
    qblock = "QUESTION:" + rest
    question = rest.split("\n", 1)[0]
    options = re.findall(r"^([ABCD])\) (.*)$", qblock, re.M)
    passages = [body[i:i + 1000] for i in range(0, len(body), 1000)] or [""]
    docs = [_toks(p) for p in passages]
    score = [0.0] * len(passages)
    for query in [question] + [question + " " + o for _, o in options]:  # the question alone, then with each option
        s = _bm25_scores(docs, query)
        top = max(s) or 1.0
        score = [a + b / top for a, b in zip(score, s)]
    first = sorted(range(len(passages)), key=lambda i: -score[i])[:20]
    ctx1 = "\n...\n".join(passages[i] for i in sorted(first))
    try:
        fq = ask(f"Passages found so far:\n{ctx1}\n\n{qblock}", system=("You help answer a multiple-choice question about a "
                 "long document. Name the one piece of information still missing to decide between the options, as a short "
                 "search query on one line 'QUERY: <query>', or 'QUERY: DONE' if nothing is missing."),
                 options={"temperature": 0, "max_output_tokens": 800})
    except Exception:  # noqa: BLE001
        fq = ""
    mq = re.findall(r"QUERY:\s*(.+)", fq or "")
    q2 = mq[-1].strip() if mq else ""
    second = []
    if q2 and q2.upper() != "DONE":
        s2 = _bm25_scores(docs, q2)
        second = [i for i in sorted(range(len(passages)), key=lambda i: -s2[i]) if i not in set(first)][:20]
    keep = sorted(set(first) | set(second))
    ctx = "\n...\n".join(passages[i] for i in keep)
    sys_r = ("Answer the multiple-choice question using the passages taken from a long document (in document order). "
             "Reason briefly, then finish with one line 'ANSWER: <letter>'.")
    try:
        out = ask(f"Passages:\n{ctx}\n\n{qblock}", system=sys_r, options={"temperature": 0, "max_output_tokens": 1500})
    except Exception:  # noqa: BLE001  the kernel raises on a cut-off reply
        out = ""
    m = re.findall(r"ANSWER:\s*\(?([ABCD])\b", out or "")
    if not m:
        try:
            out2 = ask(f"Passages:\n{ctx}\n\n{qblock}", system="Reply with one line 'ANSWER: <letter>' only.",
                       options={"temperature": 0, "max_output_tokens": 20})
        except Exception:  # noqa: BLE001
            out2 = ""
        m = re.findall(r"\b([ABCD])\b", out2 or "")
    return "ANSWER: " + (m[-1] if m else "unknown")
