"""LongBench-v2 as an RLM task family (E32, external validity).

Source: zai-org/LongBench-v2 (Apache-2.0), revision 2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9, cached under
/project/alex_phd/research-cache/datasets/. 503 four-option multiple-choice questions over contexts of about
8k to 2M words in six domains; length classes short / medium / long. The prompt is the full context followed
by the question and options; `truncated=True` keeps the head and tail of the context (the middle-truncation
LongBench uses) so direct answering fits a 64k-token window. Accuracy against the gold letter.
"""

import json
import random
import re
from functools import lru_cache
from types import SimpleNamespace

DATA = ("/project/alex_phd/research-cache/datasets/zai-org--LongBench-v2--2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/"
        "data.json")
TRUNC = "/project/alex_phd/research-cache/derived/longbench-v2-2b48e494-trunc48k-qwen3.json"  # make_longbench_trunc.py


@lru_cache(maxsize=1)
def _truncated() -> dict:
    """Contexts cut to the first and last 24k Qwen3 tokens when longer than 48k (exact, precomputed)."""
    return json.load(open(TRUNC))


@lru_cache(maxsize=1)
def _rows() -> list[dict]:
    return json.load(open(DATA))


def _prompt(x: dict, truncated: bool) -> str:
    ctx = _truncated()[x["_id"]] if truncated else x["context"]
    opts = "\n".join(f"{k}) {x['choice_' + k]}" for k in "ABCD")
    return (f"{ctx}\n\nQUESTION: {x['question']}\n{opts}\n"
            "Choose the correct option. End with one line of the form 'ANSWER: <letter>'.")


def make(per_length: int = 50, seed: int = 0, truncated: bool = False, domain: str = "") -> list[SimpleNamespace]:
    """A stratified sample: `per_length` questions from each length class (all of a class if fewer)."""
    rows = [x for x in _rows() if not domain or x["domain"] == domain]
    out = []
    for length in ("short", "medium", "long"):
        pool = sorted((x for x in rows if x["length"] == length), key=lambda x: x["_id"])
        pick = random.Random(f"{seed}:{length}:{domain}").sample(pool, min(per_length, len(pool)))
        for x in pick:
            out.append(SimpleNamespace(task_id=f"longbench-{x['_id']}", family="longbench", prompt=_prompt(x, truncated),
                                       answer=x["answer"], length_class=length, domain=x["domain"],
                                       difficulty=x["difficulty"], n_records=len(x["context"]), question_type=x["domain"]))
    return out


def letter(output: str) -> str | None:
    m = re.findall(r"ANSWER:\s*\(?([ABCD])\b", output or "")
    if m:
        return m[-1]
    m = re.fullmatch(r"\s*\(?([ABCD])\)?[.\s]*", output or "")
    return m.group(1) if m else None


def grade(output: str, task) -> float:
    return float(letter(output) == task.answer)
