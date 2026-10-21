"""E12: improve the OOLONG library's classification routine by verified selection.

Runs the library (parse_oolong, classify_rows variant, answer_oolong) as a scripted caller on
OOLONG-synth teaching windows (never the held-out ones), with helper calls to the 4B server, and
scores each variant end to end against gold. The best variant that beats the shipped routine is
written as a library variant (`skills/oolong_v2.py`) for a held-out evaluation through the RLM.
"""

import argparse
import collections
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oolong_eval  # noqa: E402
import teach_oolong  # noqa: E402

LABEL_TASKS = ("MOST_FREQ", "LEAST_FREQ", "RELATIVE_FREQ", "NUMERIC_ONE_CLASS", "SECOND_MOST_FREQ")


def make_ask_batch(client, model, pool):
    def ask_batch(prompts, system=None, options=None):
        options = options or {}

        def one(p):
            try:
                r = client.responses.create(model=model, input=p, instructions=system,
                                            temperature=options.get("temperature", 0),
                                            max_output_tokens=options.get("max_output_tokens", 1024))
                return r.output_text or ""
            except Exception:
                return ""
        return SimpleNamespace(texts=list(pool.map(one, prompts)))
    return ask_batch


VARIANTS = {}


def variant(name):
    def deco(f):
        VARIANTS[name] = f
        return f
    return deco


def _apply(rows, labels, outs):
    votes = [collections.Counter() for _ in rows]
    for out in outs:
        m = re.search(r"\[.*\]", out or "", re.S)
        if not m:
            continue
        try:
            for e in json.loads(m.group(0)):
                if isinstance(e, dict) and isinstance(e.get("line"), int) and 1 <= e["line"] <= len(rows):
                    lab = str(e.get("label") or "").strip().lower()
                    lab = lab if lab in labels else next((l for l in labels if l in lab or lab in l), lab)
                    votes[e["line"] - 1][lab] += 1
        except json.JSONDecodeError:
            pass
    return votes


def _classify(ns, rows, labels, chunk, offset=0, header=""):
    system = ('You label short texts. For EVERY numbered input line output exactly one JSON object, in the same order, '
              'inside one JSON array: {"line": <line number>, "label": <one of the allowed labels>}. Allowed labels: '
              + ", ".join(repr(l) for l in labels) + ". Output only the JSON array.")
    if header:
        system = "Task description: " + header.strip() + "\n\n" + system
    numbered = [f"{i + 1}. {r['text']}" for i, r in enumerate(rows)]
    starts = list(range(0, len(numbered), chunk))
    if offset:
        starts = [0] + list(range(offset, len(numbered), chunk))
    bounds = [(s, (starts[k + 1] if k + 1 < len(starts) else len(numbered))) for k, s in enumerate(starts)]
    chunks = ["\n".join(numbered[a:b]) for a, b in bounds if b > a]
    batch = ns["ask_batch"](["Lines:\n" + c + "\n\nJSON array:" for c in chunks], system=system,
                            options={"temperature": 0, "max_output_tokens": 1024})
    return batch.texts


def _set(rows, votes):
    for r, v in zip(rows, votes):
        r["label"] = v.most_common(1)[0][0] if v else None


@variant("base12")
def v_base(ns, rows, labels, header):
    _set(rows, _apply(rows, labels, _classify(ns, rows, labels, 12)))


@variant("chunk6")
def v_c6(ns, rows, labels, header):
    _set(rows, _apply(rows, labels, _classify(ns, rows, labels, 6)))


@variant("chunk24")
def v_c24(ns, rows, labels, header):
    _set(rows, _apply(rows, labels, _classify(ns, rows, labels, 24)))


@variant("header12")
def v_h(ns, rows, labels, header):
    _set(rows, _apply(rows, labels, _classify(ns, rows, labels, 12, header=header)))


@variant("vote3")
def v_vote(ns, rows, labels, header):
    outs = []
    for chunk, off in ((12, 0), (12, 6), (8, 0)):
        outs += _classify(ns, rows, labels, chunk, offset=off)
    _set(rows, _apply(rows, labels, outs))


@variant("header_vote3")
def v_hvote(ns, rows, labels, header):
    outs = []
    for chunk, off in ((12, 0), (12, 6), (8, 0)):
        outs += _classify(ns, rows, labels, chunk, offset=off, header=header)
    _set(rows, _apply(rows, labels, outs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--per-len", type=int, default=16)
    p.add_argument("--split", default="train")
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    from openai import OpenAI

    rng = random.Random(0)
    rows_all = teach_oolong.load_rows(a.split, 32768)
    pick = []
    for L in (1024, 2048, 4096, 8192, 16384, 32768):
        num = [r for r in rows_all if int(r["context_len"]) == L and r["task"].endswith("NUMERIC_ONE_CLASS")]
        other = [r for r in rows_all if int(r["context_len"]) == L and r["task"].split(".")[-1] in LABEL_TASKS
                 and not r["task"].endswith("NUMERIC_ONE_CLASS")]
        rng.shuffle(num)
        rng.shuffle(other)
        k = (a.per_len * 5) // 8  # weight the weak task type
        pick += num[:k] + other[: a.per_len - min(k, len(num))]
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=600)
    pool = ThreadPoolExecutor(a.workers)
    ns = {"ask_batch": make_ask_batch(client, a.model, pool), "ask": None}
    exec(compile((HERE / "skills" / "oolong.py").read_text(), "oolong", "exec"), ns)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "rows.jsonl", "a")
    for name in a.variants.split(","):
        f = VARIANTS[name]

        def run(r):
            rows, labels, question = ns["parse_oolong"](r["context"] + "\n" + r["question"])
            header = r["context"].split("\nDate:", 1)[0]
            f(ns, rows, labels, header)
            ans = ns["answer_oolong"](rows, labels, question)
            ok = oolong_eval.verify(r, ans)
            return dict(variant=name, id=r["id"], context_len=r["context_len"], task=r["task"], answer=ans, gold=r["answer"], correct=ok)
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(run, pick))
        for x in res:
            log.write(json.dumps(x) + "\n")
        log.flush()
        acc = sum(bool(x["correct"]) for x in res) / len(res)
        print(f"{name}: {acc:.3f} on {len(res)} teaching rows", flush=True)


if __name__ == "__main__":
    main()
