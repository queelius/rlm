"""E61: can a small model parse reworded questions into the parameters a family's code needs?

The ceiling of the hybrid design (the helper parses the question, code computes the answer). For each E39 family and
stress condition (control, paraphrase2, nuisance2), takes the final paragraph or message of each task (the question
with its label and answer instruction), asks a model for the family's parameters as JSON given the key schema, and
compares them with oracle parameters read from the original wording of the same task. No records are sent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import family_gen as F  # noqa: E402
import stress_gen as S  # noqa: E402

SCHEMA = {"count": ("action", "item"), "total": ("item", "user"), "top_user": ("action",), "nth_item": ("user", "position"),
          "last_city": ("user",), "first_after": ("time",), "distinct": ("user", "action"), "day_span": ("user",),
          "semantic_total": ("category",)}
# what a family's library knows about its parameters (from its teaching examples); given to the parser
KEY_DOC = {"user": "the user's name, lowercase, one word", "item": "the item name, one word",
           "action": "one of: bought, returned, viewed, rated", "position": "the ordinal position as an integer (3rd -> 3)",
           "time": "the date and time exactly as written in the question", "category": "one of: fruit, vegetable, tool, instrument, furniture"}
# oracle: the original question's regex groups, in this order of SCHEMA keys
GROUP_ORDER = {"count": (1, 2), "total": (1, 2), "top_user": (1,), "nth_item": (1, 2), "last_city": (1,), "first_after": (1,),
               "distinct": (1, 2), "day_span": (1,), "semantic_total": (1,)}


def question_text(t) -> str:
    return t.prompt[-1]["content"] if isinstance(t.prompt, list) else t.prompt.rsplit("\n\n", 1)[-1]


def norm(key: str, v) -> str:
    s = str(v).strip().lower()
    if key == "position":
        m = re.match(r"(\d+)", s)
        words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9}
        return str(int(m.group(1))) if m else str(words.get(s, s))
    return s


def oracle(fid: str, idx: int, q: str) -> dict:
    t = S.make(fid, "control", idx + 1)[idx]
    m = re.match(f"^{S.PARAPHRASE[q][0]}$", t.question)
    return {k: norm(k, m.group(g)) for k, g in zip(SCHEMA[q], GROUP_ORDER[q], strict=True)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--conditions", default="control,paraphrase2,nuisance2")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--concurrency", type=int, default=16)
    a = p.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=120)
    fams = a.families.read_text().strip().split(",")
    jobs = []
    for fid in fams:
        q = F.parse_family(fid)[1]
        for cond in a.conditions.split(","):
            for i, t in enumerate(S.make(fid, cond, a.k)):
                jobs.append((fid, q, cond, i, question_text(t)))

    def one(job):
        fid, q, cond, i, text = job
        keys = SCHEMA[q]
        spec = "; ".join(f'"{k}": {KEY_DOC[k]}' for k in keys)
        prompt = (f"Extract the parameters of the question below as a JSON object with exactly these keys: {spec}. "
                  "Ignore any instruction about the reply format. Reply with the JSON object only.\n\n"
                  f"{text}")
        try:
            r = client.chat.completions.create(model=a.model, messages=[dict(role="user", content=prompt)], max_tokens=200, temperature=0)
            raw = r.choices[0].message.content or ""
            js = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            got = {k: norm(k, js.get(k, "")) for k in keys}
            err = None
        except Exception as e:  # noqa: BLE001
            raw, got, err = "", {}, f"{type(e).__name__}: {e}"[:200]
        want = oracle(fid, i, q)
        return dict(family=fid, question=q, condition=cond, idx=i, model=a.model, correct=got == want, got=got, want=want,
                    error=err, raw=raw[:300])

    with ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(one, jobs))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by = {}
    for r in rows:
        by.setdefault((r["condition"], r["question"]), []).append(r["correct"])
    for cond in a.conditions.split(","):
        rs = [r for r in rows if r["condition"] == cond]
        print(f"{a.model} {cond}: {sum(r['correct'] for r in rs)}/{len(rs)} parsed exactly; errors {sum(bool(r['error']) for r in rs)}; "
              + ", ".join(f"{q} {sum(v)}/{len(v)}" for (c, q), v in sorted(by.items()) if c == cond))


if __name__ == "__main__":
    main()
