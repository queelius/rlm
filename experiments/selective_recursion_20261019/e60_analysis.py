"""E60 analysis, as pre-registered (docs/notebook/assets-e39-preregistration.md, section E60).

Reads the stress rows written by e59_stress.py for each arm's run and reports, over all 30 families (non-acceptance
counts as failure): acceptance; control correctness; robust yield (accepted and 0.9 or more on control and on
paraphrase 2: the primary outcome); the same with nuisance 2; known-variant correctness (paraphrase 1, nuisance 1);
long; execution faults. Each arm is compared with the baseline paired by family (discordances, exact McNemar).
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def mcnemar(x: int, y: int) -> float:
    m = x + y
    return min(1.0, 2 * sum(comb(m, i) for i in range(min(x, y) + 1)) / 2 ** m) if m else 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--families", type=Path, required=True)
    p.add_argument("--arm", action="append", required=True, help="LABEL=RUN_NAME; the first is the baseline")
    a = p.parse_args()
    fams = a.families.read_text().strip().split(",")
    rows = [json.loads(line) for line in a.rows.read_text().splitlines()]
    arms = [s.split("=", 1) for s in a.arm]
    ok = {}  # (label, family, condition) -> success
    faults = {}
    for label, run in arms:
        for r in rows:
            if r["run"] == run:
                ok[(label, r["family"], r["condition"])] = r["success"]
                faults[(label, r["family"], r["condition"])] = bool(r["errors"] or r["load_error"] or r["timed_out"])

    def outcome(label, fid, conds):
        return all(ok.get((label, fid, c), False) for c in conds)

    measures = [("accepted", None), ("control", ("control",)), ("robust: control + paraphrase2 (primary)", ("control", "paraphrase2")),
                ("robust: control + nuisance2", ("control", "nuisance2")),
                ("robust: control + paraphrase2 + nuisance2", ("control", "paraphrase2", "nuisance2")),
                ("known variants: paraphrase1, nuisance1", ("control", "paraphrase", "nuisance")), ("long", ("control", "long"))]
    base = arms[0][0]
    for name, conds in measures:
        print(f"\n{name}")
        for label, _ in arms:
            if conds is None:
                vals = {f: (label, f, "control") in ok for f in fams}
            else:
                vals = {f: outcome(label, f, conds) for f in fams}
            line = f"  {label:4s} {sum(vals.values()):2d}/{len(fams)}"
            if label != base:
                bv = {f: ((base, f, "control") in ok) if conds is None else outcome(base, f, conds) for f in fams}
                x = sum(vals[f] and not bv[f] for f in fams)
                y = sum(bv[f] and not vals[f] for f in fams)
                line += f"   vs {base}: +{x}/-{y}, exact McNemar p {mcnemar(x, y):.3f}"
            print(line)
    print("\nexecution faults (library-condition scorings with errors, load errors or timeouts)")
    for label, _ in arms:
        print(f"  {label:4s} {sum(v for k, v in faults.items() if k[0] == label)} of {sum(1 for k in faults if k[0] == label)}")


if __name__ == "__main__":
    main()
