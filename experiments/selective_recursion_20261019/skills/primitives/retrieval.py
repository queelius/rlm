"""Retrieval primitives preloaded for library proposers (E34: give the proposer parts, see what it builds).

numbered_paragraphs(text) -> list    # the paragraphs of lines of the form "[n] ..." in order (the text after the number)
chunks(text, size=20000) -> list    # the text cut into consecutive pieces of at most `size` characters
bm25(texts, query, k=15) -> list    # indices of the k texts that best match the query (BM25 over lowercased words), best first
"""
import math as _math
import re as _re
from collections import Counter as _Counter

_STOP = set("a an the of in on at to for from by with and or is was were are be been who what which when where how why "
            "whose whom that this these those did does do its it his her their as than into".split())


def _toks(s):
    return [w for w in _re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


def numbered_paragraphs(text):
    return [p for _, p in _re.findall(r"^\[(\d+)\] (.*)$", text, _re.M)]


def chunks(text, size=20000):
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def bm25(texts, query, k=15):
    docs = [_toks(t) for t in texts]
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(n, 1)
    df = _Counter(w for d in docs for w in set(d))
    q = _toks(query)
    scores = []
    for i, d in enumerate(docs):
        tf = _Counter(d)
        s = sum(_math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / max(avg, 1)))
                for w in q if w in tf)
        scores.append((s, i))
    return [i for _, i in sorted(scores, reverse=True)[:k]]
