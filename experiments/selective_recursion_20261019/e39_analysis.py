"""E39 analysis, as pre-registered (docs/notebook/assets-e39-preregistration.md).

Arm A: per family, the first round-0 candidate (sampling order) that is valid (runs on all 30 teaching inputs with
varied outputs) and reproduces the 3 examples shown in its prompt. Arm B: the bootstrapper's own acceptance (teaching
mean >= 0.9 over 30 labels, best candidate). Each accepted library is scored on 30 fresh tasks (split "fresh", 400
records); success = fresh mean >= 0.9. Reports successes, false acceptances, tokens per family, exact McNemar.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap_library as B  # noqa: E402
import family_gen as F  # noqa: E402


def valid(c: dict) -> bool:
    outs = [None if d.get("error") or d.get("output") is None or not str(d["output"]).strip() else str(d["output"]).strip()
            for d in c["details"]]
    return all(o is not None for o in outs) and len(set(outs)) > 1


def pick_a(fam: Path):
    shown = json.loads((fam / "summary.json").read_text())["shown_examples"]
    for line in (fam / "candidates.jsonl").read_text().splitlines():
        c = json.loads(line)
        if c["round"] or c.get("load_error") or c.get("timed_out") or not valid(c):
            continue
        sc = {d["task_id"]: d["score"] for d in c["details"]}
        if all(sc[t] >= 0.999 for t in shown):
            return c
    return None


def pick_b(fam: Path):
    cands = [json.loads(line) for line in (fam / "candidates.jsonl").read_text().splitlines()]
    ok = [c for c in cands if c["mean"] >= 0.9 and not c.get("load_error")]
    return max(ok, key=lambda c: (c["mean"], -len(c["code"]))) if ok else None


def fresh_score(c: dict, fid: str, work: Path, helper: str) -> float:
    tasks = F.make(fid, "fresh", 30, n_records=400)
    entry = "answer_" + re.sub(r"\W", "_", fid)
    v = B.verify(c["code"], tasks, lambda out, t: F.grade(out, t), work, f"{fid}-r{c['round']}c{c['k']}", helper, "qwen3-4b", 600, entry)
    return v["mean"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--families", type=Path, required=True, help="comma-separated family list (families.txt)")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--margin", type=float, default=None, help="E58: non-inferiority margin for A - B (e.g. 0.10)")
    p.add_argument("--ci-draws", type=int, default=10000, help="family bootstrap draws for the 95%% interval")
    a = p.parse_args()
    fams = a.families.read_text().strip().split(",")
    rows = []
    for fid in fams:
        row = dict(family=fid)
        for arm, run, pick in (("a", a.a, pick_a), ("b", a.b, pick_b)):
            fam = run / fid
            if not (fam / "summary.json").exists():
                print(f"WARNING: {arm} has no finished run for {fid}; counted as not accepted", flush=True)
            cands = [json.loads(line) for line in (fam / "candidates.jsonl").read_text().splitlines()] if (fam / "candidates.jsonl").exists() else []
            row[f"{arm}_tokens"] = sum(c.get("usage", {}).get("output") or 0 for c in cands)
            c = pick(fam) if (fam / "summary.json").exists() else None
            row[f"{arm}_accepted"] = c is not None
            row[f"{arm}_fresh"] = None
            if c is not None:
                (a.work / arm).mkdir(parents=True, exist_ok=True)
                row[f"{arm}_fresh"] = fresh_score(c, fid, a.work / arm, a.helper_base_url)
            row[f"{arm}_success"] = bool(row[f"{arm}_fresh"] is not None and row[f"{arm}_fresh"] >= 0.9)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.work / "e39_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    for arm in ("a", "b"):
        acc = sum(r[f"{arm}_accepted"] for r in rows)
        suc = sum(r[f"{arm}_success"] for r in rows)
        print(f"arm {arm}: accepted {acc}/{n}, fresh success {suc}/{n}, false acceptances {acc - suc}, "
              f"proposer tokens per family {sum(r[f'{arm}_tokens'] for r in rows) / n / 1000:.1f}k")
    x = sum(r["a_success"] and not r["b_success"] for r in rows)
    y = sum(r["b_success"] and not r["a_success"] for r in rows)
    m = x + y
    pval = min(1.0, 2 * sum(comb(m, i) for i in range(min(x, y) + 1)) / 2 ** m) if m else 1.0
    print(f"A vs B: +{x}/-{y}, exact McNemar p {pval:.3f}")
    if a.margin is not None:  # E58: paired difference in success rate, 95% percentile interval over families
        import random
        d = [int(r["a_success"]) - int(r["b_success"]) for r in rows]
        rng = random.Random(0)
        boots = sorted(sum(rng.choice(d) for _ in d) / n for _ in range(a.ci_draws))
        lo, hi = boots[int(0.025 * a.ci_draws)], boots[int(0.975 * a.ci_draws) - 1]
        est = sum(d) / n
        verdict = "non-inferior" if lo > -a.margin else "non-inferiority not shown"
        print(f"A - B success rate {est:+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}] over {n} families; margin -{a.margin:.2f}: {verdict}")


if __name__ == "__main__":
    main()
