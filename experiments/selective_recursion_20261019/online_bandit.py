"""Online routine selection over the statistics table (L3c).

A stream of tasks from a fresh seed namespace is processed in small batches. For each task the
driver picks a routine (Thompson sampling over Beta posteriors of per-signature accuracy minus a
token cost; or greedy; or uniform random), forces it through the harness note, runs the RLM,
verifies the answer, and updates the table. Rewards, choices and the routine actually taken (from
the trace) are logged so cumulative reward and empirical regret against the best fixed routine per
signature can be computed afterwards. The controller weights never change.
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sweep  # noqa: E402
import taskgen  # noqa: E402
from skills import registry, stats  # noqa: E402


def candidate_routines(family: str) -> list[str]:
    names = registry.retrieve(dict(family=family, length=None, question_type=None))
    exports = [e[0] for n in names for e in registry.exports_of(registry.all_skills()[n]["path"])]
    rts = ["direct"]
    if family == "eventlog":
        rts += ["lib:extract_events+answer_eventlog", "lib:lookup_day"]
        rts += [f"lib:{e}" for e in exports if e.startswith("answer_") and e != "answer_eventlog"]
    else:
        rts += ["lib:classify_rows+answer_oolong"]
    return rts


def cell(table: dict, sig: dict, rt: str) -> dict:
    return table.get(stats._key(sig), {}).get(rt, dict(n=0, correct=0, tokens=0))


def choose(policy: str, table: dict, sig: dict, routines: list[str], lam: float, rng: random.Random) -> str:
    if policy == "random":
        return rng.choice(routines)
    scores = {}
    for rt in routines:
        c = cell(table, sig, rt)
        mean_tokens = c["tokens"] / c["n"] if c["n"] else 0.0
        if policy == "thompson":
            acc = rng.betavariate(1 + c["correct"], 1 + c["n"] - c["correct"])
        else:  # greedy with an optimistic prior for untried routines
            acc = (c["correct"] + 1) / (c["n"] + 1)
        scores[rt] = acc - lam * mean_tokens / 10_000
    best = max(scores.values())
    return rng.choice([rt for rt, v in scores.items() if v == best])


def update(table: dict, sig: dict, rt: str, correct: bool, tokens: int) -> None:
    key = stats._key(sig)
    c = table.setdefault(key, {}).setdefault(rt, dict(n=0, correct=0, tokens=0))
    c["n"] += 1
    c["correct"] += int(correct)
    c["tokens"] += tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--policy", choices=("thompson", "greedy", "random"), default="thompson")
    p.add_argument("--warm", default=None, help="statistics table to start from (default: empty)")
    p.add_argument("--split", default="stream")
    p.add_argument("--seeds", default="2000-2019")
    p.add_argument("--ns", default="8,32,64,128,256,512")
    p.add_argument("--qts", default=",".join(taskgen.QUESTION_TYPES + taskgen.HELDOUT_TYPES))
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--controller-model", default="qwen3-4b")
    p.add_argument("--max-turns", type=int, default=10)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    lo, hi = a.seeds.split("-")
    grid = [(s, n, qt) for s in range(int(lo), int(hi) + 1) for n in map(int, a.ns.split(",")) for qt in a.qts.split(",")]
    rng.shuffle(grid)
    grid = grid[: a.steps]
    table = stats.load(a.warm) if a.warm else {}
    run_sweep.STATS_TABLE = table
    run_sweep.STATS_LAM = a.lam
    args = SimpleNamespace(harness="skills", max_turns=a.max_turns, max_model_calls=96, max_subcalls=80,
                           max_total_tokens=400_000, deadline=900.0, exec_timeout=540.0,
                           controller_model=a.controller_model, public_model="qwen3-4b", temperature=0.0,
                           direct_max_output=4096, controller_max_output=2048,
                           base_url="http://127.0.0.1:8001/v1", api_key="local")
    trace_dir = out / "traces"
    log = open(out / "rows.jsonl", "a")
    done = 0
    t0 = time.time()

    def one(item):
        k, (seed, n, qt) = item
        task = taskgen.make_task(seed, n, qt, split=a.split)
        sig = stats.signature_from_input(task.prompt)
        routines = candidate_routines(sig["family"])
        rt = choose(a.policy, table, sig, routines, a.lam, random.Random(a.seed * 100_003 + k))
        task.forced_routine = rt
        try:
            res = run_sweep.run_rlm(args, task, trace_dir)
            error = None
        except Exception as e:  # keep the stream going; a failed run is a zero-reward observation
            res, error = dict(output_text=None, total_tokens=0, trace_dir=None), f"{type(e).__name__}: {e}"[:200]
        correct = bool(taskgen.verify(task, res.get("output_text") or ""))
        taken = stats.routine_of(stats._cells(res.get("trace_dir")), sig["family"]) if res.get("trace_dir") else rt
        tokens = int(res.get("total_tokens") or 0)
        return dict(step=k, task_id=task.task_id, n=n, question_type=qt, seed=seed, sig=stats._key(sig), chosen=rt,
                    taken=taken, correct=correct, tokens=tokens, reward=int(correct) - a.lam * tokens / 10_000,
                    error=error, trace_dir=res.get("trace_dir"), output_text=(res.get("output_text") or "")[:80],
                    candidates=routines)

    for b in range(0, len(grid), a.batch):
        batch = list(enumerate(grid))[b: b + a.batch]
        with ThreadPoolExecutor(len(batch)) as ex:
            rows = list(ex.map(one, batch))
        for r in rows:
            # credit the arm the policy chose (what it controls); the route actually taken is logged as a diagnostic
            fam, length, qt = r["sig"].split("|")
            update(table, dict(family=fam, length=int(length), question_type=qt), r["chosen"], r["correct"], r["tokens"])
            log.write(json.dumps(r) + "\n")
        log.flush()
        done += len(rows)
        acc = sum(r["correct"] for r in rows)
        print(f"[{done}/{len(grid)}] batch acc {acc}/{len(rows)} elapsed {time.time() - t0:.0f}s", flush=True)
        (out / "table.json").write_text(json.dumps(dict(table=table), indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
