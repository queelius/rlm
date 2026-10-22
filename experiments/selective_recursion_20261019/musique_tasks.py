"""Long-haystack multi-hop QA from MuSiQue (E22): a second new family, and the recursion testbed.

Source: MuSiQue v1.0 answerable split (CC BY 4.0), cached zip below. A task embeds one question's 20
paragraphs (2 to 4 supporting, the rest MuSiQue's own distractors) among paragraphs drawn from other
questions of the same split until the haystack reaches a target size in approximate tokens
(characters / 4), shuffles them, numbers them, and appends the question. Teaching tasks use the
training split, held-out tasks the dev split, each with distractors from its own split only.
Score: MuSiQue's answer F1 and exact match, maximised over the answer and its aliases.
"""

from __future__ import annotations

import collections
import json
import random
import re
import string
import zipfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ZIP = Path("/project/alex_phd/research-cache/datasets/musique-v1.0-922ac98f19a201998dbdae6d7f2887a5258dbdeb/musique_data_v1.0.zip")
FILES = {"teach": "data/musique_ans_v1.0_train.jsonl", "test": "data/musique_ans_v1.0_dev.jsonl"}
SIZES = (8192, 32768, 131072, 524288)


@lru_cache(maxsize=2)
def _items(split: str) -> list[dict]:
    with zipfile.ZipFile(ZIP) as z, z.open(FILES[split]) as f:
        return [json.loads(l) for l in f]


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    k = sum(common.values())
    if not p or not g or k == 0:
        return float(p == g)
    prec, rec = k / len(p), k / len(g)
    return 2 * prec * rec / (prec + rec)


def final_answer(text: str) -> str:
    """The value after the last 'Answer:' line, or the whole text."""
    text = text or ""
    m = re.findall(r"answer\s*:\s*(.+)", text, re.I)
    return (m[-1] if m else text).strip().strip(".")


def grade(output: str, task) -> dict:
    ans = final_answer(output)
    golds = [task.answer] + list(task.aliases)
    return dict(f1=max(_f1(ans, g) for g in golds), em=float(max(_norm(ans) == _norm(g) for g in golds)))


def _para_text(p: dict) -> str:
    return f"{p['title']}: {p['paragraph_text']}"


def make(split: str, size: int, k: int, seed: int = 0, hops: str = "") -> list[SimpleNamespace]:
    """k tasks of the given haystack size from the split; `hops` filters by prefix, e.g. '2hop'."""
    items = [x for x in _items(split) if x["answerable"] and (not hops or x["id"].startswith(hops))]
    rng = random.Random(f"{seed}:{split}:{size}:{hops}")
    chosen = rng.sample(items, k)
    pool = [p for x in items for p in x["paragraphs"]]
    tasks = []
    for it in chosen:
        own = list(it["paragraphs"])
        own_titles = {p["title"] for p in own}
        paras = list(own)
        chars = sum(len(_para_text(p)) for p in paras)
        r2 = random.Random(f"{seed}:{split}:{size}:{it['id']}")
        while chars / 4 < size * 0.9:
            p = pool[r2.randrange(len(pool))]
            if p["title"] in own_titles:
                continue
            paras.append(p)
            chars += len(_para_text(p))
        r2.shuffle(paras)
        body = "\n\n".join(f"[{i + 1}] {_para_text(p)}" for i, p in enumerate(paras))
        prompt = (f"Below are {len(paras)} numbered paragraphs, followed by a question that needs facts from several of them.\n\n"
                  f"{body}\n\nQUESTION: {it['question']}\nAnswer with a short phrase. Finish with one final line of the form "
                  f"ANSWER: <answer>.")
        tasks.append(SimpleNamespace(task_id=f"musique-{split}-{size}-{it['id']}", prompt=prompt, answer=it["answer"],
                                     aliases=it.get("answer_aliases", []), family="musique", n_records=size,
                                     question_type=it["id"].split("__")[0], seed=seed, events=[],
                                     decomposition=it["question_decomposition"], question=it["question"]))
    return tasks


def describe(t, head: int = 200) -> str:
    """A compact rendering of one teaching task for a library proposer."""
    text = t.prompt
    q = text.split("QUESTION:", 1)[1]
    paras = re.findall(r"^\[(\d+)\] (.*)$", text, re.M)
    lines = [f"Input: request['input'] is one string of {len(text)} characters: a header line, {len(paras)} numbered "
             f"paragraphs like the ones below, then the question.",
             f"  header: {text.splitlines()[0]!r}"]
    for n, p in paras[:3]:
        lines.append(f"  [{n}] {p[:head]!r}")
    lines.append("  ...")
    lines.append(f"  question part: {('QUESTION:' + q)[:400]!r}")
    lines.append(f"Gold answer: {t.answer!r}")
    lines.append(f"(For reference, the gold reasoning chain: {json.dumps([{'question': d['question'], 'answer': d['answer']} for d in t.decomposition])})")
    return "\n".join(lines)
