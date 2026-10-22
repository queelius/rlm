"""E26: SFT pairs for a library proposer from verified discovery runs.

For every family in the given discovery directories whose best candidate was accepted (teaching mean >= the
threshold), emit one pair: the family's original proposer prompt (round 0, as the proposer first saw it) ->
the verified module as one ```python block. The thinking text is dropped, so the target is direct code
writing; repair-round successes are paired with the round-0 prompt (hindsight relabelling, as in SOAR).

    build_proposer_sft.py --runs DIR [DIR ...] --out pairs.jsonl [--exclude-families gen-...,...]
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--exclude-families", default="")
    a = p.parse_args()
    exclude = set(filter(None, a.exclude_families.split(",")))
    pairs, seen = [], set()
    for run in a.runs:
        for fam in sorted(d for d in run.iterdir() if d.is_dir() and (d / "candidates.jsonl").exists()):
            if fam.name in exclude or fam.name in seen:
                continue
            cands = [json.loads(l) for l in (fam / "candidates.jsonl").read_text().splitlines()]
            ok = [c for c in cands if c["mean"] >= a.threshold and not c.get("load_error")]
            if not ok:
                continue
            best = max(ok, key=lambda c: (c["mean"], -len(c["code"])))
            prompt = (fam / "prompt.txt").read_text()
            pairs.append(dict(family=fam.name, source=str(fam), teaching_mean=best["mean"], round=best["round"],
                              messages=[dict(role="user", content=prompt)],
                              completion="```python\n" + best["code"].strip() + "\n```", correct=True))
            seen.add(fam.name)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in pairs))
    print(json.dumps(dict(pairs=len(pairs), families=sorted(seen))))


if __name__ == "__main__":
    main()
