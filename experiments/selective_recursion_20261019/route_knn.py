"""E13: route by retrieval over the controller's own representation (learned key) versus the hand
signature, evaluated offline on held-out tasks whose outcomes under both routes are logged.

Index: teaching-side tasks of both families with logged outcomes for the direct and library
routes. Query: held-out tasks. A policy picks a route per task; its accuracy and tokens are read
from the held-out logs for that route. Policies: always-direct, always-library, hand-signature
table, kNN on hidden state + log length, kNN on hidden state only, and the per-task oracle.
"""

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

S = Path("/project/alex_phd/runs/selective-recursion-20261019")


def outcomes(run: str, family: str, key=None) -> dict:
    out = {}
    for line in open(S / run / "rows.jsonl"):
        r = json.loads(line)
        tid = f"oolong-{r['id']}" if family == "oolong" else r["task_id"]
        out[tid] = (bool(r["correct"]), int(r.get("total_tokens") or 0))
    return out


def load(teach_runs, eval_runs):
    tr = {route: {} for route in ("direct", "library")}
    ev = {route: {} for route in ("direct", "library")}
    for route, run, fam in teach_runs:
        tr[route].update(outcomes(run, fam))
    for route, run, fam in eval_runs:
        ev[route].update(outcomes(run, fam))
    return tr, ev


def score(acc, tokens, lam):
    return acc - lam * tokens / 10_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default=str(S / "e13-router" / "embeddings.npz"))
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--len-weight", type=float, default=4.0, help="weight of log2(tokens) relative to unit-norm embeddings")
    p.add_argument("--out", default=str(S / "e13-router" / "results.json"))
    a = p.parse_args()
    teach_runs = [("direct", "e13-teach-direct", "eventlog"), ("library", "teach-lib-base-stats", "eventlog"),
                  ("direct", "e13-oolong-teach-direct", "oolong"), ("library", "e13-oolong-teach-lib", "oolong")]
    eval_runs = [("direct", "e13-dev-direct", "eventlog"), ("library", "dev-lib-base-v2", "eventlog"),
                 ("direct", "oolong-direct-full", "oolong"), ("library", "oolong-lib-base-full", "oolong")]
    tr, ev = load(teach_runs, eval_runs)
    z = np.load(a.emb)
    ids = list(z["ids"])
    vec = z["vec"] / np.linalg.norm(z["vec"], axis=1, keepdims=True)
    loglen = np.log2(np.maximum(z["ntok"], 1))[:, None]
    fam = list(z["family"])
    pos = {t: i for i, t in enumerate(ids)}
    index = [t for t in ids if t in tr["direct"] and t in tr["library"]]
    queries = [t for t in ids if t in ev["direct"] and t in ev["library"]]
    print(f"index {len(index)} teaching tasks with both routes; {len(queries)} held-out queries")
    I = np.array([pos[t] for t in index])

    # hand signature: family, power-of-two token bucket, question form
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    def sig(t):
        f = fam[pos[t]]
        b = 1 << int(math.log2(max(int(z["ntok"][pos[t]]), 1)))
        return (f, b, qform.get(t, "?"))
    qform = {}
    import taskgen, teach_oolong  # noqa: E401
    for split, seeds in (("teach", range(16, 24)), ("dev", range(1000, 1004))):
        for n in (8, 32, 64, 128, 256, 512):
            for qt in taskgen.QUESTION_TYPES:
                for s in seeds:
                    qform[taskgen.make_task(s, n, qt, split=split).task_id] = qt
    for split in ("train", "test"):
        for r in teach_oolong.load_rows(split, 32768):
            qform[f"oolong-{r['id']}"] = r["task"]
    table = collections.defaultdict(lambda: {"direct": [], "library": []})
    for t in index:
        for route in ("direct", "library"):
            table[sig(t)][route].append(tr[route][t])

    def pick_table(t):
        cell = table.get(sig(t))
        if not cell:  # nearest bucket in the same family and form
            cands = [k for k in table if k[0] == sig(t)[0] and k[2] == sig(t)[2]]
            if not cands:
                return "library"
            cell = table[min(cands, key=lambda k: abs(math.log2(k[1]) - math.log2(sig(t)[1])))]
        s = {r: score(np.mean([c for c, _ in v]), np.mean([x for _, x in v]), a.lam) for r, v in cell.items() if v}
        return max(s, key=s.get)

    def pick_knn(t, use_len=True):
        q = vec[pos[t]]
        d = 1 - vec[I] @ q
        if use_len:
            d = d + a.len_weight * (loglen[I, 0] - loglen[pos[t], 0]) ** 2 / 100
        nn = [index[j] for j in np.argsort(d)[: a.k]]
        s = {r: score(np.mean([tr[r][u][0] for u in nn]), np.mean([tr[r][u][1] for u in nn]), a.lam) for r in ("direct", "library")}
        return max(s, key=s.get)

    policies = {
        "always-direct": lambda t: "direct",
        "always-library": lambda t: "library",
        "hand-signature": pick_table,
        "knn-state+length": lambda t: pick_knn(t, True),
        "knn-state-only": lambda t: pick_knn(t, False),
        "oracle": lambda t: max(("direct", "library"), key=lambda r: score(ev[r][t][0], ev[r][t][1], a.lam)),
    }
    res = {}
    for name, f in policies.items():
        per = collections.defaultdict(lambda: [0, 0, 0, 0])
        for t in queries:
            r = f(t)
            c, x = ev[r][t]
            for key in (fam[pos[t]], "all"):
                v = per[key]; v[0] += 1; v[1] += c; v[2] += x; v[3] += r == "direct"
        res[name] = {k: dict(n=v[0], acc=round(v[1] / v[0], 3), tokens_k=round(v[2] / v[0] / 1000, 1), direct_share=round(v[3] / v[0], 2)) for k, v in per.items()}
        print(f"{name:<18}", "  ".join(f"{k}: acc {d['acc']:.3f} tok {d['tokens_k']}k direct {d['direct_share']:.2f} (n={d['n']})" for k, d in sorted(res[name].items())))
    Path(a.out).write_text(json.dumps(dict(args=vars(a), index=len(index), queries=len(queries), results=res), indent=1))


if __name__ == "__main__":
    main()
