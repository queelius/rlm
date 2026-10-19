"""Assemble the study's comparison tables from every run directory that exists.

Event-log dev split: direct / base controller / programmed split / expert / SFT / RL controllers.
OOLONG test split: direct / base controller / expert / SFT / RL controllers, by context length.
Prints Markdown tables with Wilson intervals and mean total tokens; writes results.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

S = Path("/project/alex_phd/runs/selective-recursion-20261019")


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rows_of(path: Path, label: str, source: str, filt=None):
    f = path / "rows.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if filt and not filt(r):
            continue
        r["label"], r["source"] = label, source
        out.append(r)
    return out


def cell(rows):
    n = len(rows); c = sum(bool(r.get("correct")) for r in rows)
    tok = sum(int(r.get("total_tokens") or 0) for r in rows) / max(n, 1)
    lo, hi = wilson(c, n)
    return dict(n=n, correct=c, acc=c / n if n else float("nan"), lo=lo, hi=hi, tokens=tok,
                errors=sum(bool(r.get("error")) for r in rows))


def table(rows, title, key="n"):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["label"], r[key])].append(r)
    labels = sorted({k[0] for k in by}, key=lambda s: ORDER.index(s) if s in ORDER else 99)
    ns = sorted({k[1] for k in by})
    lines = [f"### {title}", "", "| Policy | " + " | ".join(f"{key}={n}" for n in ns) + " | all |", "|---|" + "---|" * (len(ns) + 1)]
    for lab in labels:
        cells = []
        for n in ns + ["all"]:
            rs = by.get((lab, n), []) if n != "all" else [r for (l, _), v in by.items() if l == lab for r in v]
            c = cell(rs)
            cells.append(f"{c['acc']:.2f} ({c['tokens']/1000:.1f}k, n={c['n']})" if c["n"] else "–")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cells: accuracy (mean total tokens in thousands, number of tasks).")
    return "\n".join(lines)


ORDER = ["direct", "base controller", "base controller (hinted)", "programmed split", "expert in harness",
         "SFT (event-log)", "SFT (event-log + OOLONG)", "SFT + RL (OOLONG)"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl-dir", default="rl-oolong-001")
    args = p.parse_args()
    ev = []
    ev += rows_of(S / "dev-base", "direct", "eventlog", lambda r: r["policy"] == "direct")
    ev += rows_of(S / "dev-base", "base controller", "eventlog", lambda r: r["policy"] == "rlm")
    ev += rows_of(S / "dev-base", "programmed split", "eventlog", lambda r: r["policy"] == "fixed")
    ev += rows_of(S / "dev-expert", "expert in harness", "eventlog")
    ev += rows_of(S / "dev-sft1", "SFT (event-log)", "eventlog")
    ev += rows_of(S / "dev-sft2", "SFT (event-log + OOLONG)", "eventlog")
    oo = []
    oo += rows_of(S / "oolong-base", "direct", "oolong", lambda r: r["policy"] == "direct")
    oo += rows_of(S / "oolong-base", "base controller", "oolong", lambda r: r["policy"] == "rlm")
    oo += rows_of(S / "oolong-expert-test", "expert in harness", "oolong")
    oo += rows_of(S / "oolong-sft1", "SFT (event-log)", "oolong")
    oo += rows_of(S / "oolong-sft2", "SFT (event-log + OOLONG)", "oolong")
    rl = S / args.rl_dir / "dev"
    if rl.exists():
        files = sorted(rl.glob("step-*.jsonl"))
        if files:
            last = files[-1]
            for line in last.read_text().splitlines():
                r = json.loads(line); r["label"] = f"SFT + RL (OOLONG) @{last.stem}"; r["source"] = r.get("source", "oolong")
                (oo if r["source"] == "oolong" else ev).append(r)
            ORDER.append(f"SFT + RL (OOLONG) @{last.stem}")
    out = []
    if ev:
        out.append(table([r for r in ev if r.get("question_type") != "lookup_day"], "Event-log dev split, aggregation questions, by records"))
        out.append(table([r for r in ev if r.get("question_type") == "lookup_day"], "Event-log dev split, lookup questions, by records"))
    if oo:
        out.append(table(oo, "OOLONG-synth held-out contexts, by context length (tokens)"))
        out.append(table(oo, "OOLONG-synth held-out contexts, by task group", key="task_group"))
    text = "\n\n".join(out) if out else "(no results yet)"
    print(text)
    (S / "RESULTS.md").write_text(text + "\n")
    json.dump(dict(eventlog=len(ev), oolong=len(oo)), open(S / "results.json", "w"))


if __name__ == "__main__":
    main()
