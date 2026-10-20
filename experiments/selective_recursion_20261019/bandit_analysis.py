"""Summarize the online-selection streams (E10): accuracy, tokens, cumulative reward, empirical regret.

Regret is against the best fixed routine per signature in hindsight, estimated from every arm's
observations pooled (same stream, same seeds), so all arms are judged against one reference.
"""

import argparse
import collections
import json
from pathlib import Path


def load(d: Path) -> list[dict]:
    return [json.loads(l) for l in open(d / "rows.jsonl")] if (d / "rows.jsonl").exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--lam", type=float, default=0.05)
    a = p.parse_args()
    arms = {Path(r).name: load(Path(r)) for r in a.runs}
    # hindsight reference: mean reward of each (signature, chosen routine) over all arms
    pool = collections.defaultdict(list)
    for rows in arms.values():
        for r in rows:
            pool[(r["sig"], r["chosen"])].append(r["reward"])
    best = {}
    for (sig, rt), rewards in pool.items():
        m = sum(rewards) / len(rewards)
        if sig not in best or m > best[sig][1]:
            best[sig] = (rt, m)
    print(f"{'arm':<26} {'n':>4} {'acc':>6} {'tokens':>7} {'reward':>8} {'regret':>8}  routes chosen")
    for name, rows in arms.items():
        if not rows:
            print(f"{name:<26} (no rows)")
            continue
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n
        tok = sum(r["tokens"] for r in rows) / n
        rew = sum(r["reward"] for r in rows)
        regret = sum(best[r["sig"]][1] - r["reward"] for r in rows)
        routes = collections.Counter(r["chosen"].replace("lib:", "") for r in rows)
        print(f"{name:<26} {n:>4} {acc:>6.3f} {tok / 1000:>6.1f}k {rew:>8.2f} {regret:>8.2f}  {dict(routes.most_common(4))}")
    # per-arm learning curve: accuracy in quarters of the stream
    print("\naccuracy by quarter of the stream:")
    for name, rows in arms.items():
        if not rows:
            continue
        q = max(len(rows) // 4, 1)
        print(f"  {name:<24}", [f"{sum(r['correct'] for r in rows[i:i + q]) / max(len(rows[i:i + q]), 1):.2f}" for i in range(0, len(rows), q)][:4])
    # what the hindsight-best routine is per question form and length bucket
    print("\nhindsight-best routine by signature (from pooled observations):")
    by_form = collections.defaultdict(dict)
    for sig, (rt, m) in sorted(best.items(), key=lambda kv: (kv[0].split("|")[2], int(kv[0].split("|")[1]))):
        fam, length, qt = sig.split("|")
        by_form[qt][int(length)] = rt.replace("lib:", "")
    for qt, d in by_form.items():
        print(f"  {qt:<22}", {k: v for k, v in sorted(d.items())})


if __name__ == "__main__":
    main()
