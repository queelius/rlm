"""Write a resume manifest for the running campaign: everything a new session needs to restart where this one stopped,
without the conversation that produced it.

    python experiments/ops/checkpoint.py --out RUNS/CHECKPOINT-<stamp>.json --runs NAME ... [--note TEXT]

Records the git state (worktree, branch, HEAD, remote, dirty files), the Slurm job (from the environment, since client
tools fail inside jobs), every vLLM server (pid, device, full command line) and the LoRA adapters each one has
loaded, the queue (pending and running specs in full, slot capacities), environment and package versions, and per
named run its progress (finished families or rows, candidates, RL steps, FINAL, EXIT, PAUSED/STOPPED notes).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/project/alex_phd/runs/selective-recursion-20261019")
HOME = Path("/home/atowell/research-runs/selective-recursion-20261019")
ENVS = {"serving": "/project/alex_phd/envs/vllm/bin/python", "training": "/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python",
        "evaluation": str(REPO / ".venv" / "bin" / "python")}


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def servers() -> list[dict]:
    out = []
    for line in sh("ps", "-eo", "pid,args").splitlines()[1:]:
        pid, _, args = line.strip().partition(" ")
        if "vllm serve" not in args or "ps -eo" in args:
            continue
        try:
            env = dict(kv.split("=", 1) for kv in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0") if "=" in kv)
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
        except OSError:
            continue
        port = cmd.split("--port", 1)[1].split()[0] if "--port" in cmd else None
        models = None
        if port:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": "Bearer local"})
                models = sorted(m["id"] for m in json.load(urllib.request.urlopen(req, timeout=10))["data"])
            except Exception as e:  # noqa: BLE001 - a dead server is recorded, not fatal
                models = f"unreachable: {e}"
        out.append(dict(pid=int(pid), port=port, cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"), command=cmd, served_models=models))
    return sorted(out, key=lambda s: s["port"] or "")


def versions() -> dict:
    out = {}
    for name, py in ENVS.items():
        out[name] = sh(py, "-c", "import sys, importlib\nv={'python': sys.version.split()[0]}\nfor m in ('vllm','torch','transformers','peft','openai'):\n"
                                 "    try: v[m]=importlib.import_module(m).__version__\n    except Exception: pass\nprint(v)")
    out["uv"] = sh("uv", "--version")
    out["nvidia_driver"] = sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines()[:1]
    out["gpus"] = sh("nvidia-smi", "-L").splitlines()
    return out


def run_progress(name: str) -> dict:
    h = HOME / name
    d = RUNS / name if (RUNS / name).exists() else h  # pipeline RL runs live only in the home store
    rec: dict = dict(run=name, path=str(d), exists=d.exists())
    if not d.exists():
        return rec
    fams = [f for f in d.glob("gen-*") if f.is_dir()]
    if fams:
        rec["families_started"] = len(fams)
        rec["families_finished"] = sum((f / "summary.json").exists() for f in fams)
        rec["candidates"] = sum(len((f / "candidates.jsonl").read_text().splitlines()) for f in fams if (f / "candidates.jsonl").exists())
    for fname in ("candidates.jsonl", "rows.jsonl", "steps.jsonl"):
        if (d / fname).exists():
            rec[fname] = len((d / fname).read_text().splitlines())
    for fname in ("bench.jsonl", "summary.json", "FINAL", "EXIT", "PAUSED-not-failed.txt", "STOPPED.txt", "NOTE.txt", "FAILED.txt"):
        for base in (d, h):
            if (base / fname).exists():
                rec[fname] = (base / fname).read_text().strip()[:600]
                break
    steps = sorted(d.glob("step-*"), key=lambda p: int(p.name.split("-")[1])) if d.exists() else []
    if steps:
        rec["last_adapter"] = str(steps[-1])
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--note", default="")
    a = p.parse_args()
    q = RUNS / "queue"
    manifest = dict(
        written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), note=a.note,
        git=dict(worktree=str(REPO), branch=sh("git", "-C", str(REPO), "branch", "--show-current"),
                 head=sh("git", "-C", str(REPO), "rev-parse", "HEAD"), remote=sh("git", "-C", str(REPO), "remote", "get-url", "origin"),
                 pushed_to="origin/main", dirty=sh("git", "-C", str(REPO), "status", "--porcelain").splitlines()),
        slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_JOB") or k in ("SLURM_NODELIST", "SLURMD_NODENAME")},
        host=os.uname().nodename,
        servers=servers(),
        queue=dict(pending=[json.loads(f.read_text()) for f in sorted((q / "pending").glob("*.json"))],
                   running=[json.loads(f.read_text()) for f in sorted((q / "running").glob("*.json"))],
                   slots=json.loads((q / "slots.json").read_text()) if (q / "slots.json").exists() else None,
                   dispatcher=[ln for ln in sh("ps", "-eo", "pid,args").splitlines() if "queue.py run" in ln and "ps -eo" not in ln]),
        versions=versions(),
        runs=[run_progress(r) for r in a.runs],
        stores=dict(runs=str(RUNS), homes=str(HOME), adapters=str(HOME / "adapters"), sft_data=str(HOME / "sft-data"),
                    models="/project/alex_phd/research-cache/models", datasets="/project/alex_phd/research-cache/datasets"),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {a.out}: {len(manifest['servers'])} servers, {len(manifest['queue']['pending'])} pending, "
          f"{len(manifest['queue']['running'])} running, {len(manifest['runs'])} runs")


if __name__ == "__main__":
    main()
