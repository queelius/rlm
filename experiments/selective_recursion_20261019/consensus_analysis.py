"""E37: can agreement among independently sampled libraries replace gold labels when accepting a library?

Reads a discovery run made with no early stopping (``discovery_bench.py --threshold 2 --repair-rounds 0``), so every
family has all its round-0 candidates, each with its outputs on the 30 teaching inputs. Two label-free rules, both
ignoring candidates that error, return nothing, or give the same output for every input (constant outputs agree
trivially):

- exact: accept the largest group of candidates with identical output vectors, if it has at least ``--min-group``;
- mbr: accept the candidate with the highest mean agreement with the others (MBR-exec style), if it agrees with at
  least one other candidate on at least ``--min-agree`` of the inputs;
- cross (with ``--also``, runs of other proposers on the same families): accept the largest group of identical
  output vectors that contains candidates from at least two different proposers.

Gold scores are used only to score the rules: precision (accepted libraries that are correct, teaching mean >= 0.9),
recall against an oracle that accepts any correct candidate, and the chance a random valid candidate is correct.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def vec(c: dict):
    out = []
    for d in c["details"]:
        o = d.get("output")
        out.append(None if d.get("error") or o is None or not str(o).strip() else str(o).strip())
    return tuple(out)


def valid(v) -> bool:
    return all(o is not None for o in v) and len(set(v)) > 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--min-group", type=int, default=2)
    p.add_argument("--min-agree", type=float, default=1.0)
    p.add_argument("--correct", type=float, default=0.9)
    p.add_argument("--also", type=Path, nargs="*", default=[], help="runs of other proposers for the cross rule")
    a = p.parse_args()
    stats = {k: collections.Counter() for k in ("code", "semantic")}
    rows = []
    for fam in sorted(a.run.glob("gen-*")):
        f = fam / "candidates.jsonl"
        if not f.exists():
            continue
        cands = [json.loads(line) for line in f.read_text().splitlines()]
        cands = [c for c in cands if c["round"] == 0 and not c.get("load_error") and not c.get("timed_out")]
        kind = "semantic" if "semantic" in fam.name else "code"
        s = stats[kind]
        vs = [(vec(c), c) for c in cands]
        ok = [(v, c) for v, c in vs if valid(v)]
        s["families"] += 1
        s["oracle"] += any(c["mean"] >= a.correct for c in cands)
        s["valid_candidates"] += len(ok)
        s["valid_correct"] += sum(c["mean"] >= a.correct for _, c in ok)
        groups = collections.defaultdict(list)
        for v, c in ok:
            groups[v].append(c)
        big = max(groups.values(), key=len, default=[])
        exact_acc = len(big) >= a.min_group
        exact_ok = exact_acc and big[0]["mean"] >= a.correct
        s["exact_accept"] += exact_acc
        s["exact_correct"] += exact_ok
        mbr_acc = mbr_ok = False
        if len(ok) >= 2:
            def agree(u, w):
                return sum(x == y for x, y in zip(u, w, strict=True)) / len(u)
            scores = [(sum(agree(v, w) for w, _ in ok if w is not v) / (len(ok) - 1), max(agree(v, w) for w, _ in ok if w is not v), c)
                      for v, c in ok]
            best = max(scores, key=lambda t: (t[0], t[1]))
            mbr_acc = best[1] >= a.min_agree
            mbr_ok = mbr_acc and best[2]["mean"] >= a.correct
        s["mbr_accept"] += mbr_acc
        s["mbr_correct"] += mbr_ok
        cross_acc = cross_ok = False
        if a.also:
            pooled = collections.defaultdict(list)
            for src, run in enumerate([a.run, *a.also]):
                g = run / fam.name / "candidates.jsonl"
                for c in ([json.loads(line) for line in g.read_text().splitlines()] if g.exists() else []):
                    if c["round"] == 0 and not c.get("load_error") and not c.get("timed_out") and valid(vec(c)):
                        pooled[vec(c)].append((src, c))
            shared = [m for m in pooled.values() if len({src for src, _ in m}) >= 2]
            if shared:
                cross_acc = True
                cross_ok = max(shared, key=len)[0][1]["mean"] >= a.correct
        s["cross_accept"] += cross_acc
        s["cross_correct"] += cross_ok
        rows.append(dict(family=fam.name, kind=kind, candidates=len(cands), valid=len(ok), largest_group=len(big),
                         oracle=any(c["mean"] >= a.correct for c in cands), exact_accept=exact_acc, exact_correct=exact_ok,
                         mbr_accept=mbr_acc, mbr_correct=mbr_ok, cross_accept=cross_acc, cross_correct=cross_ok))
    (a.run / "consensus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for kind, s in list(stats.items()) + [("all", sum(stats.values(), collections.Counter()))]:
        if not s["families"]:
            continue
        print(f"{kind}: families {s['families']}; oracle (any correct round-0 candidate) {s['oracle']}; "
              f"random valid candidate correct {s['valid_correct']}/{s['valid_candidates']}")
        for rule in ("exact", "mbr") + (("cross",) if a.also else ()):
            acc, cor = s[f"{rule}_accept"], s[f"{rule}_correct"]
            print(f"  {rule:<5} accepts {acc}, correct {cor} (precision {cor / max(acc, 1):.2f}; recall vs oracle "
                  f"{cor / max(s['oracle'], 1):.2f})")


if __name__ == "__main__":
    main()
