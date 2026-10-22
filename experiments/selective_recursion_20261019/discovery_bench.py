"""Library-discovery benchmark (E26): run bootstrap_library.py on many generated families with one
proposer and report the success rate, candidates to first success and proposer tokens.

    discovery_bench.py --out DIR --model qwen3-8b-awq --base-url http://127.0.0.1:8002/v1 --split heldout
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import family_gen as G  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--split", default="heldout")
    p.add_argument("--seeds", default="0")
    p.add_argument("--families", default=None, help="comma-separated subset")
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--repair-rounds", type=int, default=1)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--prompt-suffix", default="")
    p.add_argument("--concurrency", type=int, default=4)
    a = p.parse_args()
    fams = a.families.split(",") if a.families else G.family_ids(a.split, tuple(int(s) for s in a.seeds.split(",")))
    a.out.mkdir(parents=True, exist_ok=True)
    py = str(HERE.parent.parent / ".venv" / "bin" / "python")

    def run(fid):
        d = a.out / fid
        if (d / "summary.json").exists():
            return fid
        cmd = [py, str(HERE / "bootstrap_library.py"), "--family", fid, "--out", str(d), "--model", a.model, "--base-url", a.base_url,
               "--helper-base-url", a.helper_base_url, "--max-output-tokens", str(a.max_output_tokens), "--candidates", str(a.candidates),
               "--repair-rounds", str(a.repair_rounds), "--threshold", "0.9", "--timeout", "600"]
        if a.prompt_suffix:
            cmd += ["--prompt-suffix", a.prompt_suffix]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "run.log", "a") as log:
            subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        return fid

    with ThreadPoolExecutor(a.concurrency) as ex:
        for fid in ex.map(run, fams):
            print("done", fid, flush=True)
    rows = []
    for fid in fams:
        d = a.out / fid
        cands = [json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines()] if (d / "candidates.jsonl").exists() else []
        first = next((i + 1 for i, c in enumerate(cands) if c["mean"] >= 0.9), None)
        toks = sum(c.get("usage", {}).get("output", 0) for c in cands)
        fmt, q, ans, s = G.parse_family(fid)
        rows.append(dict(family=fid, format=fmt, question=q, answer_format=ans, n_candidates=len(cands),
                         best=max((c["mean"] for c in cands), default=0.0), success=first is not None,
                         candidates_to_success=first, proposer_output_tokens=toks))
    (a.out / "bench.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    print(f"{a.model}{a.prompt_suffix}: success {sum(r['success'] for r in rows)}/{n}; "
          f"mean best {sum(r['best'] for r in rows) / n:.3f}; proposer tokens {sum(r['proposer_output_tokens'] for r in rows) / 1000:.0f}k")
    for key in ("format", "question", "answer_format"):
        by = {}
        for r in rows:
            by.setdefault(r[key], []).append(r["success"])
        print(key, {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())})


if __name__ == "__main__":
    main()
