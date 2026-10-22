"""E24: GEPA-style reflective evolution of a routine's configuration (prompts and numeric parameters).

The routine's code is fixed (MuSiQue filter-plus-BM25-then-read); a candidate is a JSON configuration:
filter_system, reader_system (text), chunk, bm25_k, keep_cap, read_max_tokens (numbers). Each
iteration picks a parent from the population, runs it on a fresh teaching minibatch, shows a reflector
model the failures (question, gold, prediction, how many supporting paragraphs the filter kept), and asks
for one improved configuration. A child that beats its parent on that minibatch is scored on a fixed
teaching selection set and joins the population. Only teaching (MuSiQue train) tasks are used; the best
configuration is then run on held-out tasks separately. After GEPA (Agrawal et al., arXiv 2507.19457).
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import musique_tasks as M  # noqa: E402

BASE = dict(
    filter_system=("You select evidence for a multi-step question. A paragraph is relevant if it states a fact needed "
                   "for ANY step of the reasoning, including intermediate entities the question does not name. Reply "
                   "with the paragraph numbers only, comma-separated, or 'none'."),
    reader_system=("Answer the multi-step question from the paragraphs. Work out each step briefly, then finish with one "
                   "line 'ANSWER: <short answer>' giving a short phrase (a name, place, date or number). Always give your "
                   "best answer."),
    chunk=15, bm25_k=15, keep_cap=60, read_max_tokens=600)
BOUNDS = dict(chunk=(4, 60), bm25_k=(0, 60), keep_cap=(10, 120), read_max_tokens=(100, 1500))

ROUTINE = open(HERE / "skills" / "reference" / "musique_filter_read.py").read()


def module_source(cfg: dict) -> str:
    """The reference routine with the configuration substituted in."""
    src = ROUTINE
    src = src.replace('def select_relevant(paragraphs, question, chunk=15):', f'def select_relevant(paragraphs, question, chunk={int(cfg["chunk"])}):')
    old_sys = src[src.index('    sys_p = ("You select evidence'):src.index("    starts = list(range(0, len(paragraphs), chunk))")]
    src = src.replace(old_sys, f"    sys_p = {cfg['filter_system']!r}\n")
    src = src.replace("set(bm25(paragraphs, question))", f"set(bm25(paragraphs, question, k={int(cfg['bm25_k'])}))")
    src = src.replace("    if len(keep) > 60:\n        keep = keep[:60]", f"    if len(keep) > {int(cfg['keep_cap'])}:\n        keep = keep[:{int(cfg['keep_cap'])}]")
    old_r = src[src.index('    sys_p = ("Answer the multi-step question'):src.index('    out = ask(f"Paragraphs:')]
    src = src.replace(old_r, f"    sys_p = {cfg['reader_system']!r}\n")
    src = src.replace('"max_output_tokens": 600}', f'"max_output_tokens": {int(cfg["read_max_tokens"])}}}')
    return src


class Runner:
    def __init__(self, base_url, model, workers=32):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="local", timeout=600)
        self.model = model
        self.pool = ThreadPoolExecutor(workers)
        self.local = threading.local()

    def _one(self, prompt, system, options):
        options = options or {}
        r = self.client.responses.create(model=self.model, input=prompt, instructions=system,
                                         temperature=options.get("temperature", 0),
                                         max_output_tokens=options.get("max_output_tokens", 1024))
        st = self.local.stats
        st["tokens"] += r.usage.input_tokens + r.usage.output_tokens
        return r.output_text or ""

    def run(self, cfg, task, items):
        self.local.stats = {"tokens": 0}
        st = self.local.stats

        def ask(prompt, system=None, options=None):
            return self._one(prompt, system, options)

        def ask_batch(prompts, system=None, options=None):
            def go(pp):
                self.local.stats = st
                return self._one(pp, system, options)
            return SimpleNamespace(texts=list(self.pool.map(go, prompts)))

        ns = {"ask": ask, "ask_batch": ask_batch}
        exec(compile(module_source(cfg), "routine", "exec"), ns)
        try:
            out = ns["answer_musique"]({"input": task.prompt})
        except Exception as e:  # noqa: BLE001
            out = f"error {type(e).__name__}: {e}"
        paras, q = ns["parse"](task.prompt)
        it = items[task.task_id.split("-", 3)[3]]
        sup = {f"{p['title']}: {p['paragraph_text']}" for p in it["paragraphs"] if p["is_supporting"]}
        try:
            kept = {paras[i] for i in ns["select_relevant"](paras, q)} if False else None
        except Exception:  # noqa: BLE001
            kept = None
        return dict(task_id=task.task_id, size=task.n_records, question=task.question, gold=task.answer,
                    output=out, f1=M.grade(out, task)["f1"], tokens=st["tokens"], n_support=len(sup), kept=kept)


def evaluate(runner, cfg, tasks, items, conc=6):
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda t: runner.run(cfg, t, items), tasks))
    return dict(f1=sum(r["f1"] for r in res) / len(res), tokens=sum(r["tokens"] for r in res) / len(res), rows=res)


def reflect(client, model, cfg, res, max_tokens):
    fails = [r for r in res["rows"] if r["f1"] < 1.0][:6]
    lines = [f"- Q: {r['question']}\n  gold: {r['gold']!r}; routine answered: {str(r['output'])[:80]!r}; F1 {r['f1']:.2f}; "
             f"haystack {r['size'] // 1024}k tokens" for r in fails]
    prompt = (
        "You are improving the configuration of a routine that answers multi-hop questions over a long haystack of "
        "numbered paragraphs. The routine (fixed code) works in two stages:\n"
        "1. FILTER: the paragraphs are split into chunks of `chunk` paragraphs; for each chunk a small 4B helper model "
        "receives `filter_system` as its system prompt plus the question and the chunk, and replies with relevant "
        "paragraph numbers. The kept set is the union over chunks plus the `bm25_k` best paragraphs by BM25 against "
        "the question, truncated to `keep_cap` paragraphs.\n"
        "2. READ: the kept paragraphs and the question go to the same helper with `reader_system`; it may use up to "
        "`read_max_tokens` output tokens and must end with 'ANSWER: <short answer>'. The routine returns that answer.\n"
        "Score: answer F1 against the gold answer (short phrases score best), and fewer helper tokens is better.\n\n"
        f"Current configuration (JSON):\n{json.dumps(cfg, indent=1)}\n\n"
        f"On a batch of teaching questions it scored mean F1 {res['f1']:.3f} using {res['tokens'] / 1000:.0f}k helper "
        f"tokens per question. Failures:\n" + "\n".join(lines) + "\n\n"
        f"Numeric bounds: {json.dumps(BOUNDS)}. Diagnose why it fails (filter missing bridge paragraphs, reader "
        "confused by distractors, answers too long, ...) and propose ONE improved configuration. Reply with your brief "
        "diagnosis, then the complete new configuration as one ```json block with the same keys.")
    r = client.responses.create(model=model, input=prompt, temperature=0.8, max_output_tokens=max_tokens)
    text = r.output_text or ""
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)
    try:
        new = json.loads(m[-1]) if m else None
    except json.JSONDecodeError:
        new = None
    if not isinstance(new, dict):
        return None, text
    child = dict(cfg)
    for k in BASE:
        if k in new:
            v = new[k]
            if k in BOUNDS:
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    continue
                lo, hi = BOUNDS[k]
                v = max(lo, min(hi, v))
            elif not isinstance(v, str) or len(v) < 20:
                continue
            child[k] = v
    return child, text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--minibatch", type=int, default=8)
    p.add_argument("--select-per-size", type=int, default=12)
    p.add_argument("--sizes", default="32768,131072")
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--reflector-base-url", default="http://127.0.0.1:8002/v1")
    p.add_argument("--reflector-model", default="qwen3-8b-awq")
    p.add_argument("--reflector-max-tokens", type=int, default=12000)
    p.add_argument("--lam", type=float, default=0.05, help="fitness = F1 - lam * tokens / 100k")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    from openai import OpenAI

    rng = random.Random(a.seed)
    sizes = [int(s) for s in a.sizes.split(",")]
    items = {x["id"]: x for x in M._items("teach")}
    select = [t for s in sizes for t in M.make("teach", s, a.select_per_size, seed=101)]
    pool_tasks = [t for s in sizes for t in M.make("teach", s, 60, seed=202)]
    runner = Runner(a.helper_base_url, "qwen3-4b")
    reflector = OpenAI(base_url=a.reflector_base_url, api_key="local", timeout=3600)
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "log.jsonl", "a")

    def fitness(e):
        return e["f1"] - a.lam * e["tokens"] / 100_000

    base_eval = evaluate(runner, BASE, select, items)
    population = [dict(id=0, parent=None, cfg=BASE, select_f1=base_eval["f1"], select_tokens=base_eval["tokens"],
                       fitness=fitness(base_eval))]
    log.write(json.dumps(dict(event="seed", **{k: v for k, v in population[0].items()})) + "\n")
    log.flush()
    print(f"seed: select F1 {base_eval['f1']:.3f} tokens {base_eval['tokens'] / 1000:.0f}k", flush=True)
    for it in range(1, a.iterations + 1):
        t0 = time.time()
        parent = rng.choice(sorted(population, key=lambda c: -c["fitness"])[:4])
        batch = rng.sample(pool_tasks, a.minibatch)
        pres = evaluate(runner, parent["cfg"], batch, items)
        child_cfg, text = reflect(reflector, a.reflector_model, parent["cfg"], pres, a.reflector_max_tokens)
        rec = dict(event="iteration", it=it, parent=parent["id"], parent_mb_f1=pres["f1"], parent_mb_tokens=pres["tokens"],
                   child=child_cfg, reflection=text[-3000:])
        if child_cfg is None or child_cfg == parent["cfg"]:
            rec["outcome"] = "no valid child"
        else:
            cres = evaluate(runner, child_cfg, batch, items)
            rec.update(child_mb_f1=cres["f1"], child_mb_tokens=cres["tokens"])
            if fitness(cres) > fitness(pres):
                sres = evaluate(runner, child_cfg, select, items)
                node = dict(id=len(population), parent=parent["id"], cfg=child_cfg, select_f1=sres["f1"],
                            select_tokens=sres["tokens"], fitness=fitness(sres))
                population.append(node)
                rec.update(outcome="accepted", select_f1=sres["f1"], select_tokens=sres["tokens"], node=node["id"])
            else:
                rec["outcome"] = "rejected on minibatch"
        rec["seconds"] = round(time.time() - t0)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        best = max(population, key=lambda c: c["fitness"])
        print(f"it {it}: {rec['outcome']}; parent mb {pres['f1']:.3f}" + (f", child mb {rec.get('child_mb_f1', 0):.3f}" if 'child_mb_f1' in rec else "")
              + (f", select {rec['select_f1']:.3f}" if 'select_f1' in rec else "") + f"; best select {best['select_f1']:.3f} ({rec['seconds']}s)", flush=True)
        (a.out / "population.json").write_text(json.dumps(population, indent=1))
    best = max(population, key=lambda c: c["fitness"])
    (a.out / "best.json").write_text(json.dumps(best, indent=1))
    (a.out / "best_routine.py").write_text(module_source(best["cfg"]))
    print("best:", json.dumps({k: best[k] for k in ("id", "select_f1", "select_tokens", "fitness")}))


if __name__ == "__main__":
    main()
