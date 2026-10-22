"""Bootstrap a whole library for a new task family from teaching examples (E21, question 1).

The proposer model sees only: the RLM library contract (what library code may call, the module
format the registry parses), three rendered teaching examples with their gold answers, and, in
repair rounds, its best candidate with failing teaching cases. Each candidate module is executed in
a separate process on every teaching task (real helper calls), scored with the family's grader, and
the best one is registered under skills/learned/ only if its mean teaching score clears the threshold.
Held-out rows are never shown or used.
"""

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONTRACT = '''You are writing a small Python library for a Recursive Language Model (RLM). In an RLM, a small
controller model never reads the long input itself; it writes Python in a persistent interpreter
where the caller's request is bound as `request` (a dict; the input is `request["input"]`). Library
functions are preloaded into that interpreter so the controller can solve a whole task with one call.

Your library will be used for inputs like the examples below. Write ONE Python module that:
- defines `{entry}(request) -> str`, returning exactly the final answer text the task requires
  (no extra words, no explanation); it must work on inputs of up to several million characters;
- may use only the Python standard library (re, json, collections, difflib, math, itertools, ...)
  and two preloaded helper functions, which call a separate 4B instruction-tuned model with a
  64k-token window: `ask(prompt, system=None, options=None) -> str` and
  `ask_batch(prompts, system=None, options=None)` which returns an object whose `.texts` is a list
  of strings, one per prompt (calls run in parallel). Use them only for judgements code cannot make;
  prefer exact Python wherever the structure allows it. Never send more than about 150,000
  characters in one prompt. Do not read files, use the network, or print;
- starts with a module docstring in exactly this format (it is parsed by a registry):

"""<One line describing the input family and what the library computes.>

{entry}(request) -> str    # <one-line note on what it returns>
<any other public helper, same format: name(args) -> type    # note>
Typical use: FINAL_TEXT({entry}(request)).
"""

Return only one ```python fenced block containing the complete module.'''


def family_api(family: str):
    if family == "mrcr":
        import mrcr_tasks as F

        teach = F.load("teach", 2)
        return dict(teach=teach, describe=F.describe, as_task=F.as_task,
                    grade=lambda out, t: F.grade(out or "", t.answer, t.prefix))
    if family == "musique":
        import musique_tasks as F

        teach = F.make("teach", 8192, 30, seed=0)
        return dict(teach=teach, describe=F.describe, as_task=lambda t: t,
                    grade=lambda out, t: F.grade(out or "", t)["f1"])
    if family.startswith("gen-"):
        import family_gen as F

        teach = F.make(family, "teach", 30)
        return dict(teach=teach, describe=F.describe, as_task=lambda t: t, grade=lambda out, t: F.grade(out, t))
    raise ValueError(family)


def build_prompt(entry: str, examples: list[str]) -> str:
    ex = "\n\n".join(f"=== Example {k + 1} ===\n{e}" for k, e in enumerate(examples))
    return CONTRACT.format(entry=entry) + "\n\nTeaching examples (inputs abbreviated):\n\n" + ex


def extract_code(text: str) -> str:
    blocks = re.findall(r"```python\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else (text or "").split("</think>")[-1]


def verify(code: str, tasks: list, grade, workdir: Path, tag: str, base_url: str, model: str, timeout: float, entry: str):
    cand = workdir / f"{tag}.py"
    cand.write_text(code)
    tj = workdir / f"{tag}.tasks.json"
    tj.write_text(json.dumps(dict(entry=entry, tasks=[dict(task_id=t.task_id, input=t.prompt) for t in tasks])))
    oj = workdir / f"{tag}.out.json"
    py = str(HERE.parent.parent / ".venv" / "bin" / "python")
    t0 = time.time()
    try:
        subprocess.run([py, str(HERE / "bootstrap_verify_worker.py"), str(cand), str(tj), str(oj), base_url, model],
                       timeout=timeout, capture_output=True)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
    tj.unlink(missing_ok=True)
    res = json.load(open(oj)) if oj.exists() else {"load_error": "no output", "results": []}
    by = {r["task_id"]: r for r in res.get("results", [])}
    scores, details = [], []
    for t in tasks:
        r = by.get(t.task_id)
        s = grade(r["output"], t) if r and r["output"] is not None else 0.0
        scores.append(s)
        details.append(dict(task_id=t.task_id, score=s, output=(r or {}).get("output"), error=(r or {}).get("error"),
                            helper_calls=(r or {}).get("helper_calls")))
    return dict(mean=sum(scores) / len(scores), exact=sum(s >= 0.999 for s in scores), n=len(scores),
                load_error=res.get("load_error"), timed_out=timed_out, seconds=round(time.time() - t0, 1), details=details)


def _question_of(prompt) -> str:
    """The part of a task input that states what is asked: the last message of a chat input, or the
    text from QUESTION: onwards (else the tail) of a plain-text input."""
    if isinstance(prompt, list):
        return str(prompt[-1].get("content", ""))
    text = str(prompt)
    return text[text.rfind("QUESTION:"):] if "QUESTION:" in text else text[-300:]


def normalize_doc(code: str, entry: str) -> tuple[str, bool]:
    """Insert the contract's typical-use line into the module docstring when the proposer left it out.
    E25c: without it the untrained 4B caller often bypassed the library (0.60 vs 0.95 on MRCR)."""
    m = re.match(r'(\s*""")(.*?)(""")', code, re.S)
    if not m or "Typical use:" in m.group(2):
        return code, False
    doc = m.group(2).rstrip("\n") + f"\nTypical use: FINAL_TEXT({entry}(request)).\n"
    return code[: m.start(2)] + doc + code[m.end(2):], True


def failing_cases(details, tasks, k=3):
    by = {t.task_id: t for t in tasks}
    out = []
    for d in sorted(details, key=lambda d: d["score"])[:k]:
        if d["score"] >= 0.999:
            continue
        t = by[d["task_id"]]
        out.append(f"- Final message: {_question_of(t.prompt)[:200]!r}\n  expected (first 150 chars): {str(t.answer)[:150]!r}\n"
                   f"  your output (first 150 chars): {str(d['output'])[:150]!r}\n  error: {(d['error'] or '')[-400:]!r}\n"
                   f"  score {d['score']:.3f}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1", help="proposer server")
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--helper-model", default="qwen3-4b")
    p.add_argument("--max-output-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--repair-rounds", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--timeout", type=float, default=1200.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--register", action="store_true")
    p.add_argument("--prompt-suffix", default="", help='appended to every proposer prompt, e.g. " /no_think" for Qwen3 hybrids')
    a = p.parse_args()
    from openai import OpenAI

    api = family_api(a.family)
    entry = "answer_" + re.sub(r"\W", "_", a.family)
    rng = random.Random(a.seed)
    teach_rows = api["teach"]
    tasks = [api["as_task"](r) for r in teach_rows]
    shown = rng.sample(teach_rows, a.n_examples)
    prompt = build_prompt(entry, [api["describe"](r) for r in shown])
    shown_ids = [getattr(r, "task_id", None) or r["row_id"] for r in shown]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "prompt.txt").write_text(prompt)
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=1800)
    best = None
    done = {}
    if (a.out / "candidates.jsonl").exists():  # resume: keep verified candidates, continue after them
        for line in (a.out / "candidates.jsonl").read_text().splitlines():
            rec = json.loads(line)
            done[(rec["round"], rec["k"])] = rec
            if best is None or rec["mean"] > best["mean"]:
                best = rec
        print(f"resuming with {len(done)} verified candidates, best {best['mean'] if best else None}", flush=True)
    log = open(a.out / "candidates.jsonl", "a")
    for rnd in range(1 + a.repair_rounds):
        if rnd == 0:
            rp = prompt
        else:
            if best is None or best["mean"] >= a.threshold:
                break
            rp = (prompt + "\n\nA previous module was run on the teaching tasks and scored "
                  f"{best['mean']:.2f} on average ({best['exact']}/{best['n']} exact). Its code:\n```python\n"
                  f"{best['code'].strip()}\n```\nWorst teaching cases:\n{failing_cases(best['details'], tasks)}\n"
                  + (f"Load error: {best['load_error'][-600:]}\n" if best.get("load_error") else "")
                  + "\nFind the mistake, fix it, and return the complete corrected module in one ```python block.")
            (a.out / f"repair-prompt-{rnd}.txt").write_text(rp)
        for k in range(a.candidates):
            if (rnd, k) in done:
                continue
            t0 = time.time()
            try:
                r = client.responses.create(model=a.model, input=rp + a.prompt_suffix, temperature=a.temperature,
                                            max_output_tokens=a.max_output_tokens)
                text, usage = r.output_text or "", dict(input=r.usage.input_tokens, output=r.usage.output_tokens)
            except Exception as e:  # noqa: BLE001
                text, usage = "", dict(error=f"{type(e).__name__}: {e}"[:300])
            code = extract_code(text)
            v = verify(code, tasks, api["grade"], a.out, f"r{rnd}c{k}", a.helper_base_url, a.helper_model, a.timeout, entry)
            rec = dict(round=rnd, k=k, mean=v["mean"], exact=v["exact"], n=v["n"], load_error=v["load_error"],
                       timed_out=v["timed_out"], verify_seconds=v["seconds"], propose_seconds=round(time.time() - t0 - v["seconds"], 1),
                       usage=usage, code=code, details=v["details"])
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"round {rnd} candidate {k}: mean {v['mean']:.3f} exact {v['exact']}/{v['n']} "
                  f"load_error={bool(v['load_error'])} timeout={v['timed_out']} verify {v['seconds']}s", flush=True)
            if best is None or v["mean"] > best["mean"]:
                best = dict(rec, code=code)
            if best["mean"] >= 0.999:
                break
        if best and best["mean"] >= 0.999:
            break
    accepted = best is not None and best["mean"] >= a.threshold
    summary = dict(family=a.family, proposer=a.model, best_mean=best["mean"] if best else None,
                   best_exact=best["exact"] if best else None, best_round=best["round"] if best else None,
                   best_k=best["k"] if best else None, accepted=accepted, threshold=a.threshold,
                   shown_examples=shown_ids, teaching_tasks=len(tasks))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if accepted:
        code, added = normalize_doc(best["code"], entry)
        (a.out / "accepted.py").write_text(code.strip() + "\n")
        print(f"accepted library written to {a.out / 'accepted.py'} (typical-use line added: {added})")
    if accepted and a.register:
        path = HERE / "skills" / "learned" / f"{a.family}.py"
        header = (f"# families: {a.family}\n# bootstrapped by bootstrap_library.py (proposer {a.model}; "
                  f"teaching mean {best['mean']:.3f}, {best['exact']}/{best['n']} exact)\n")
        path.write_text(header + normalize_doc(best["code"], entry)[0].strip() + "\n")
        print(f"registered {path}")


if __name__ == "__main__":
    main()
