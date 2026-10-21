"""Score OOLONG runs two ways: exact match (this repository's verifier) and OOLONG's official partial
credit for numeric answers (0.75 ** |prediction - gold|; exact match for every other answer type).

Usage: oolong_score.py RUN_DIR [RUN_DIR ...]   (rows.jsonl from oolong_eval, or raw.jsonl from E18)
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oolong_eval  # noqa: E402

SOURCES = [Path("/project/alex_phd/research-cache/datasets/oolong-synth-20261019/validation-le32k.jsonl"),
           Path("/project/alex_phd/research-cache/datasets/oolong-synth-20261021-long/validation-64k-512k.jsonl")]


def gold_rows():
    out = {}
    for p in SOURCES:
        for line in open(p):
            r = json.loads(line)
            r.pop("context", None)
            out[r["id"]] = r
    return out


def partial(r, text):
    if r["answer_type"].endswith("NUMERIC"):
        try:
            pred = oolong_eval.parse(r["answer_type"], text or "")
            gold = oolong_eval.gold_value(r["answer_type"], r["answer"])
            return 0.75 ** abs(pred - gold) if pred is not None else 0.0
        except Exception:  # noqa: BLE001
            return 0.0
    return float(oolong_eval.verify(r, text or ""))


def main():
    gold = gold_rows()
    for run in sys.argv[1:]:
        d = Path(run)
        f = d / "rows.jsonl" if (d / "rows.jsonl").exists() else d / "raw.jsonl"
        by = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for line in open(f):
            x = json.loads(line)
            r = gold[x["id"]]
            text = x.get("output_text") or ""
            em, pc = float(oolong_eval.verify(r, text)), partial(r, text)
            for k in ("all", r["dataset"], f"{r['dataset']}@{r['context_len']}"):
                b = by[k]; b[0] += 1; b[1] += em; b[2] += pc
        print(d.name)
        for k, (n, em, pc) in sorted(by.items()):
            print(f"  {k:<24} n={n:<4} exact {em / n:.3f}  oolong-score {pc / n:.3f}")


if __name__ == "__main__":
    main()
