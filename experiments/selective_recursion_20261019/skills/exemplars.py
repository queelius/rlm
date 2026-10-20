"""Exemplar retrieval (H1'): the cells of one successful teaching trajectory for an input like this one.

Keyed by family, question type and nearest length; drawn only from the teaching sets (never from
held-out windows). Returns a prompt block the harness injects as a demonstration.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

RUNS = Path("/project/alex_phd/runs/selective-recursion-20261019")
SOURCES = {"eventlog": ["teach-003/sft.jsonl"], "oolong": ["teach-oolong-004/sft.jsonl"]}


@lru_cache(maxsize=None)
def _trajectories(family: str) -> list[dict]:
    by_task: dict[str, dict] = {}
    for rel in SOURCES[family]:
        for line in (RUNS / rel).read_text().splitlines():
            r = json.loads(line)
            if not r.get("correct"):
                continue
            t = by_task.setdefault(r["task_id"], dict(n=r["n"], question_type=r["question_type"], cells={}))
            t["cells"][r["turn"]] = r["completion"]
    return [t for t in by_task.values() if 0 in t["cells"] and 1 in t["cells"]]


def retrieve(sig: dict) -> dict | None:
    pool = _trajectories(sig["family"])
    same = [t for t in pool if t["question_type"] == sig.get("question_type")] or pool
    if not same:
        return None
    return min(same, key=lambda t: abs(t["n"] - sig["length"]))


def hint_block(sig: dict) -> str:
    t = retrieve(sig)
    if t is None:
        return ""
    cells = "\n\n".join(f"Turn {k + 1}:\n{t['cells'][k]}" for k in sorted(t["cells"]))
    return (f"Example of a successful approach on an input of the same kind ({t['n']} records or rows, "
            f"question type {t['question_type']}); adapt it to this input:\n\n{cells}")
