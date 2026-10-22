"""Launch an experiment with the checks that would have caught this project's launch failures.

    launch.py NAME -- COMMAND ...        (COMMAND runs from the selective-recursion experiment directory)

1. Dry-runs argparse by appending --help to the command (an invalid argument fails here, not later).
2. Creates the run directory under the store, symlinks it, and writes MANIFEST.json (name, command,
   git commit, start time, host, GPU lanes from CUDA_VISIBLE_DEVICES).
3. Starts the command detached, with its log in the run directory.
4. Watches for the first rows (default 15 minutes): reports the error fraction of the first rows and
   writes FAILED.txt and stops the job if every early row is an error or the process died.
Prints one verdict line. Use --no-wait to skip step 4.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

STORE = Path("/project/alex_phd/runs/selective-recursion-20261019")
HOME = Path("/home/atowell/research-runs/selective-recursion-20261019")
REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "selective_recursion_20261019"


def rows_of(d: Path):
    for name in ("rows.jsonl", "candidates.jsonl", "log.jsonl"):
        f = d / name
        if f.exists():
            return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return []


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: launch.py NAME [--wait-min M] [--no-wait] [--env K=V ...] -- COMMAND ...")
    i = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--wait-min", type=float, default=15)
    p.add_argument("--min-rows", type=int, default=3)
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--env", action="append", default=[])
    a = p.parse_args(argv[:i])
    cmd = argv[i + 1:]
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in a.env))
    # 1. argument dry run
    dry = subprocess.run(cmd + ["--help"], cwd=EXP, env=env, capture_output=True, text=True, timeout=120)
    if dry.returncode != 0:
        print(f"LAUNCH REFUSED {a.name}: argument check failed:\n{dry.stderr[-800:]}")
        sys.exit(2)
    # 2. run directory and manifest
    run = HOME / a.name
    run.mkdir(parents=True, exist_ok=True)
    link = STORE / a.name
    if not link.exists():
        link.symlink_to(run)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    manifest = dict(name=a.name, command=cmd, env={k: v for k, v in (kv.split("=", 1) for kv in a.env)}, commit=commit,
                    dirty=dirty, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=os.uname().nodename,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    (run / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    # 3. start detached
    log = open(run / "launch.log", "a")
    # The job's exit code goes to EXIT in the run directory, so a crash after the early check is visible
    # (status.py flags nonzero codes); a detached job otherwise ends silently.
    wrapped = ["bash", "-c", f'{shlex.join(cmd)}; echo $? > {shlex.quote(str(run / "EXIT"))}']
    (run / "EXIT").unlink(missing_ok=True)
    proc = subprocess.Popen(wrapped, cwd=EXP, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (run / "PID").write_text(str(proc.pid))
    if a.no_wait:
        print(f"LAUNCHED {a.name} pid {proc.pid} (no wait)")
        return
    # 4. early health check
    deadline = time.time() + a.wait_min * 60
    while time.time() < deadline:
        time.sleep(15)
        rows = rows_of(run)
        dead = proc.poll() is not None
        if len(rows) >= a.min_rows or dead:
            break
    rows = rows_of(run)
    errs = sum(1 for r in rows if r.get("error") or r.get("load_error"))
    dead = proc.poll() is not None
    if dead and not rows:
        verdict = f"FAILED: process exited with code {proc.returncode} before any row"
    elif rows and errs == len(rows) and len(rows) >= a.min_rows:
        verdict = f"FAILED: all {len(rows)} early rows are errors: {str(rows[0].get('error') or rows[0].get('load_error'))[:200]}"
        os.killpg(proc.pid, signal.SIGTERM)
    elif not rows:
        verdict = f"SLOW: no rows after {a.wait_min} min (still running); check {run / 'launch.log'}"
    else:
        verdict = f"OK: {len(rows)} early rows, {errs} errors"
    if verdict.startswith("FAILED"):
        (run / "FAILED.txt").write_text(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {verdict}\n"
                                        f"log tail:\n{(run / 'launch.log').read_text()[-1500:]}\n")
    print(f"{a.name}: {verdict}")


if __name__ == "__main__":
    main()
