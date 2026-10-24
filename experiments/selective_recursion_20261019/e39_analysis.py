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


def fresh_score(c: dict, fid: str, work: Path, helper: str) -> dict:
    tasks = F.make(fid, "fresh", 30, n_records=400)
    entry = "answer_" + re.sub(r"\W", "_", fid)
    v = B.verify(c["code"], tasks, lambda out, t: F.grade(out, t), work, f"{fid}-r{c['round']}c{c['k']}", helper, "qwen3-4b", 600, entry)
    errors = sum(bool(d.get("error")) for d in v["details"])  # kept apart from wrong answers (E58 amendment)
    return dict(mean=v["mean"], n=v["n"], errors=errors, load_error=v.get("load_error"), timed_out=v.get("timed_out"))


def template(fid: str) -> str:
    """Format and question of a generated family (gen-<format>-<question>-<answer format>-s<seed>): the cluster
    for the E58 sensitivity interval, since families sharing a template are not independent."""
    return "-".join(fid.split("-")[1:3])


def interval(d: list, clusters: list, draws: int) -> tuple:
    """95% percentile interval of mean(d), resampling clusters (one cluster per element = family bootstrap)."""
    import random
    groups = {}
    for x, g in zip(d, clusters, strict=True):
        groups.setdefault(g, []).append(x)
    gs = list(groups.values())
    rng = random.Random(0)
    boots = []
    for _ in range(draws):
        pick = [rng.choice(gs) for _ in gs]
        boots.append(sum(map(sum, pick)) / sum(map(len, pick)))
    boots.sort()
    return boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--families", type=Path, required=True, help="comma-separated family list (families.txt)")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--margin", type=float, default=None, help="E58: non-inferiority margin for A - B (e.g. 0.10)")
    p.add_argument("--ci-draws", type=int, default=10000, help="family bootstrap draws for the 95%% interval")
    p.add_argument("--require-complete", action="store_true",
                   help="E58: refuse to analyse unless every family has a finished run in both arms")
    a = p.parse_args()
    fams = a.families.read_text().strip().split(",")
    missing = [(arm, f) for arm, run in (("a", a.a), ("b", a.b)) for f in fams if not (run / f / "summary.json").exists()]
    if missing and a.require_complete:
        sys.exit(f"incomplete: {len(missing)} unfinished (arm, family) pairs, e.g. {missing[:5]}; not analysing")
    rows = []
    for fid in fams:
        row = dict(family=fid)
        for arm, run, pick in (("a", a.a, pick_a), ("b", a.b, pick_b)):
            fam = run / fid
            if not (fam / "summary.json").exists():
                print(f"WARNING: {arm} has no finished run for {fid}; counted as not accepted", flush=True)
            cands = [json.loads(line) for line in (fam / "candidates.jsonl").read_text().splitlines()] if (fam / "candidates.jsonl").exists() else []
            row[f"{arm}_tokens"] = sum(c.get("usage", {}).get("output") or 0 for c in cands)
            row[f"{arm}_input_tokens"] = sum(c.get("usage", {}).get("input") or 0 for c in cands)
            row[f"{arm}_usage_missing"] = sum((c.get("usage") or {}).get("output") is None for c in cands)
            c = pick(fam) if (fam / "summary.json").exists() else None
            row[f"{arm}_accepted"] = c is not None
            row[f"{arm}_fresh"] = None
            if c is not None:
                (a.work / arm).mkdir(parents=True, exist_ok=True)
                fs = fresh_score(c, fid, a.work / arm, a.helper_base_url)
                row[f"{arm}_fresh"] = fs["mean"]
                row[f"{arm}_fresh_detail"] = fs
            row[f"{arm}_success"] = bool(row[f"{arm}_fresh"] is not None and row[f"{arm}_fresh"] >= 0.9)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.work / "e39_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    for arm in ("a", "b"):
        acc = sum(r[f"{arm}_accepted"] for r in rows)
        suc = sum(r[f"{arm}_success"] for r in rows)
        errs = sum(1 for r in rows if (r.get(f"{arm}_fresh_detail") or {}).get("errors"))
        print(f"arm {arm}: accepted {acc}/{n}, fresh success {suc}/{n}, false acceptances {acc - suc}, "
              f"proposer tokens per family {sum(r[f'{arm}_tokens'] for r in rows) / n / 1000:.1f}k output, "
              f"{sum(r[f'{arm}_input_tokens'] for r in rows) / n / 1000:.1f}k input "
              f"(candidates without usage: {sum(r[f'{arm}_usage_missing'] for r in rows)}); "
              f"accepted libraries with fresh-task errors: {errs}")
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
        # sensitivity analyses frozen before unblinding (E58 amendment); the primary result above is unchanged by them
        tl, th = interval(d, [template(r["family"]) for r in rows], a.ci_draws)
        print(f"sensitivity, resampling {len({template(r['family']) for r in rows})} format-question templates: "
              f"[{tl:+.3f}, {th:+.3f}]")
        ds = [(r["a_fresh"] or 0.0) - (r["b_fresh"] or 0.0) for r in rows]
        sl, sh = interval(ds, [r["family"] for r in rows], a.ci_draws)
        print(f"sensitivity, fresh score (unaccepted = 0) A - B {sum(ds) / n:+.3f}, 95% interval [{sl:+.3f}, {sh:+.3f}]")


if __name__ == "__main__":
    main()
