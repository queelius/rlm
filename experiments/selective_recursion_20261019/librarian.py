"""The librarian (E08): grow the skill library from verified experience.

For one held-out question type it asks the proposer model for candidate implementations of
`answer_<type>(events, question) -> str` (given the event schema, the shipped aggregator as a style
reference and a few teaching examples with their gold answers), verifies every candidate on the
cached model-extracted events of the teaching split, keeps the best candidate that clears the
threshold, writes it to `skills/learned/`, and emits a pseudo-run of verification rows so the
routine enters the outcome statistics with the same bookkeeping as any logged route.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import taskgen  # noqa: E402

EVENT_SCHEMA = ('Each event is a dict: {"line": int, "kind": "buy" | "return", "person": str, '
                '"item": str (singular or plural noun), "qty": int, "day": int}. Items may appear '
                'in singular or plural form; match them with the helper `_is_item(e, target)`, and '
                'use `_who(e)`, `_qty(e)`, `_day(e)` for the other fields.')


def shipped_helpers() -> str:
    src = (HERE / "skills" / "eventlog.py").read_text()
    start = src.index("def _is_item")
    end = src.index("def answer_eventlog")
    return src[start:end].strip()


def shipped_aggregator() -> str:
    src = (HERE / "skills" / "eventlog.py").read_text()
    start = src.index("def answer_eventlog")
    end = src.index("\ndef lookup_day")
    return src[start:end].strip()


def load_cache(cache_dir: Path, qt: str) -> list[dict]:
    recs = [json.loads(p.read_text()) for p in sorted(cache_dir.glob("*.json"))]
    return [r for r in recs if r["question_type"] == qt]


def build_prompt(qt: str, examples: list[dict], fn: str) -> str:
    ex_text = []
    for r in examples:
        sample = [e for e in r["events"]][:10]
        ex_text.append(f"Question: {r['question']}\nExtracted events (first {len(sample)} of {len(r['events'])}): "
                       f"{json.dumps(sample)}\nGold answer: {r['answer']}")
    return (
        f"You maintain a small Python library used by a controller that answers questions about store logs.\n"
        f"The controller already has `extract_events(lines)` which returns a list of events. {EVENT_SCHEMA}\n\n"
        f"Helpers already defined in the library (do not redefine them):\n```python\n{shipped_helpers()}\n```\n\n"
        f"The shipped aggregator for other question types, as a style reference:\n```python\n{shipped_aggregator()}\n```\n\n"
        f"Write a new function `{fn}(events, question)` for questions of this form, returning the string "
        f"\"ANSWER: <answer>\" exactly as the question's answer format requires. Parse the item (and any other "
        f"argument) from the question text with a regular expression. Count only completed purchases "
        f"(kind == \"buy\") unless the question says otherwise. Use day numbers exactly as they appear in the "
        f"events. Examples from the teaching split:\n\n" + "\n\n".join(ex_text) +
        f"\n\nReturn only one ```python fenced block containing the complete function `{fn}` and nothing else."
    )


def verify(code: str, fn: str, recs: list[dict], use_gold: bool = False) -> tuple[float, int, list[bool]]:
    ns = {"ask_batch": None, "ask": None}
    exec(compile((HERE / "skills" / "eventlog.py").read_text(), "eventlog", "exec"), ns)
    try:
        exec(compile(code, fn, "exec"), ns)
    except Exception:
        return 0.0, len(recs), [False] * len(recs)
    f = ns.get(fn)
    if not callable(f):
        return 0.0, len(recs), [False] * len(recs)
    results = []
    for r in recs:
        events = r["gold_events"] if use_gold else r["events"]
        if use_gold:
            events = [dict(e, day=e["day"] + 1) for e in events]  # gold days are 0-based
        try:
            out = f(events, r["question"])
            results.append(bool(taskgen.verify(dict(answer=r["answer"], answer_type=r["answer_type"]), str(out))))
        except Exception:
            results.append(False)
    return sum(results) / max(len(results), 1), sum(not x for x in results), results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True, help="librarian run dir (candidates, rows.jsonl)")
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--register", action="store_true", help="write the accepted routine to skills/learned/")
    p.add_argument("--max-output-tokens", type=int, default=2048, help="raise for thinking models that reason before the code")
    p.add_argument("--repair-rounds", type=int, default=0,
                   help="after a round with no accepted candidate, show the proposer the best candidate and its failing cases and ask for a revision")
    a = p.parse_args()
    from openai import OpenAI

    client = OpenAI(base_url=a.base_url, api_key="local")
    recs = load_cache(Path(a.cache), a.qt)
    assert recs, "no cached tasks for this question type"
    rng = random.Random(a.seed)
    fn = f"answer_{a.qt}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt_examples = rng.sample([r for r in recs if r["n"] <= 32], min(a.examples, len(recs)))
    prompt = build_prompt(a.qt, prompt_examples, fn)
    (out / "prompt.txt").write_text(prompt)
    best = None

    def failing_cases(code: str, limit: int = 3) -> list[dict]:
        ns = {"ask_batch": None, "ask": None}
        exec(compile((HERE / "skills" / "eventlog.py").read_text(), "eventlog", "exec"), ns)
        exec(compile(code, fn, "exec"), ns)
        cases = []
        for r in recs:
            try:
                got = str(ns[fn](r["events"], r["question"]))
            except Exception as e:
                got = f"raised {type(e).__name__}: {e}"
            if not taskgen.verify(dict(answer=r["answer"], answer_type=r["answer_type"]), got):
                cases.append(dict(question=r["question"], expected=r["answer"], got=got,
                                  events=[e for e in r["events"] if ns["_is_item"](e, re.findall(r"\b(\w+)\b", r["question"])[-1]) or True][:14]))
            if len(cases) >= limit:
                break
        return cases

    with open(out / "candidates.jsonl", "a") as log:
        for rnd in range(1 + a.repair_rounds):
            if rnd == 0:
                round_prompt = prompt
            else:
                if best is None or best["acc"] >= a.threshold:
                    break
                cases = failing_cases(best["code"])
                round_prompt = (prompt + "\n\nA previous attempt was verified on the teaching split and failed. Its code:\n"
                                f"```python\n{best['code'].strip()}\n```\nIt scored {best['acc']:.2f}. Failing cases "
                                "(question, expected answer, what the code returned, the relevant extracted events):\n"
                                + "\n".join(f"- Q: {c['question']}\n  expected: {c['expected']}; got: {c['got']}\n  events: {json.dumps(c['events'])}" for c in cases)
                                + "\n\nFind the mistake, fix it, and return the complete corrected function in one ```python block.")
                (out / f"repair-prompt-{rnd}.txt").write_text(round_prompt)
            for k in range(a.candidates):
                t0 = time.time()
                r = client.responses.create(model=a.model, input=round_prompt, temperature=a.temperature,
                                            max_output_tokens=a.max_output_tokens)
                text = r.output_text or ""
                blocks = re.findall(r"```python\n(.*?)```", text, re.S)
                code = blocks[-1] if blocks else text.split("</think>")[-1]  # the last block, after any thinking
                acc, fails, results = verify(code, fn, recs)
                acc_gold, _, _ = verify(code, fn, recs, use_gold=True)
                rec = dict(round=rnd, k=k, acc=acc, acc_gold=acc_gold, n=len(recs), fails=fails, seconds=round(time.time() - t0, 1),
                           tokens=dict(input=r.usage.input_tokens, output=r.usage.output_tokens), code=code)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"round {rnd} candidate {k}: verified {acc:.3f} on extracted events ({acc_gold:.3f} on gold), "
                      f"{r.usage.output_tokens} output tokens", flush=True)
                if best is None or acc > best["acc"]:
                    best = dict(rec, results=results)
    accepted = best is not None and best["acc"] >= a.threshold
    summary = dict(question_type=a.qt, candidates=a.candidates, repair_rounds=a.repair_rounds,
                   best_acc=best["acc"] if best else None, best_k=best["k"] if best else None,
                   best_round=best.get("round", 0) if best else None, accepted=accepted, threshold=a.threshold, model=a.model)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if accepted and a.register:
        path = HERE / "skills" / "learned" / f"eventlog_{a.qt}.py"
        from skills import stats as _stats

        pattern = _stats.pattern_from_examples([r["question"] for r in recs])
        header = ('# families: eventlog\n'
                  f'# question_kind: {a.qt}\n'
                  f'# question_pattern: {pattern}\n'
                  f'"""Learned skill for {a.qt.replace("_", " ")} questions (E08 librarian, {a.model}; verified '
                  f'{best["acc"]:.2f} on {len(recs)} teaching tasks with model-extracted events).\n\n'
                  f'{fn}(events, question) -> str      # exact aggregation over extract_events output; returns "ANSWER: ..."\n'
                  f'Typical use: lines, question = split_log(request["input"]); events = extract_events(lines); '
                  f'FINAL_TEXT({fn}(events, question)).\n"""\n')
        path.write_text(header + best["code"].strip() + "\n")
        # pseudo-run rows: the verification outcomes as logged routes for the statistics table
        with open(out / "rows.jsonl", "w") as rows:
            for r, ok in zip(recs, best["results"]):
                rows.write(json.dumps(dict(task_id=r["task_id"], policy="rlm", harness="skills", n=r["n"], seed=r["seed"],
                                           question_type=r["question_type"], correct=ok,
                                           total_tokens=r["prompt_tokens"] + r["completion_tokens"] + 1200)) + "\n")
        print(f"registered {path}")


if __name__ == "__main__":
    main()
