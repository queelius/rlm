"""One table for every run touched in the last N hours: rows, mean score per arm, errors, rate, last update.

    status.py [--hours 24]
"""
import argparse
import collections
import json
import time
from pathlib import Path

STORE = Path("/project/alex_phd/runs/selective-recursion-20261019")


def summarize(d: Path):
    for name in ("rows.jsonl", "candidates.jsonl"):
        f = d / name
        if f.exists():
            break
    else:
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    age = (time.time() - f.stat().st_mtime) / 60
    first = min(r.get("started") or 0 for r in rows) or None
    if name == "candidates.jsonl":
        best = max(r.get("mean", 0) for r in rows)
        return dict(kind="bootstrap", n=len(rows), arms=f"best teaching {best:.3f}", errors=sum(bool(r.get("load_error")) for r in rows), age=age)
    arms = collections.defaultdict(lambda: [0, 0.0, 0])
    for r in rows:
        k = r.get("policy", "?") + ("/" + r["harness"] if r.get("harness") else "")
        v = r.get("score", r.get("correct", 0))
        arms[k][0] += 1
        arms[k][1] += float(v or 0)
        arms[k][2] += bool(r.get("error"))
    return dict(kind="eval", n=len(rows), arms="  ".join(f"{k}: {s / n:.3f} (n {n})" for k, (n, s, e) in arms.items()),
                errors=sum(e for _, _, e in arms.values()), age=age)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=24)
    a = p.parse_args()
    now = time.time()
    out = []
    for d in sorted(STORE.iterdir()):
        try:
            if not d.is_dir() or now - d.stat().st_mtime > a.hours * 3600 and not any(
                    (d / n).exists() and now - (d / n).stat().st_mtime < a.hours * 3600 for n in ("rows.jsonl", "candidates.jsonl")):
                continue
            s = summarize(d)
        except OSError:
            continue
        if s:
            flag = " FAILED" if (d / "FAILED.txt").exists() else (" stopped" if (d / "STOPPED.txt").exists() else "")
            out.append((s["age"], f"{d.name:<34} {s['n']:>5} rows  err {s['errors']:<4} updated {s['age']:5.0f} min ago{flag}  {s['arms']}"))
    for _, line in sorted(out):
        print(line)


if __name__ == "__main__":
    main()
