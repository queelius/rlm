"""A small job queue so the GPUs keep working while no one is watching.

Jobs are JSON files in QUEUE/pending: {"name", "lane": "gpu0"|"gpu1", "priority": int (higher first),
"cmd": [...], "env": {...}, "after": ["run-name", ...] (optional: wait until those runs have a DONE or are
not running)}. The dispatcher (`queue.py run`) starts the highest-priority eligible job whenever its lane
has fewer than SLOTS running jobs, via launch.py (argument check, manifest, early health check), and
moves the spec to running/ then done/. `queue.py add FILE`, `queue.py ls`.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

QUEUE = Path("/project/alex_phd/runs/selective-recursion-20261019/queue")
SLOTS = {"gpu0": 3, "gpu1": 4}
LAUNCH = Path(__file__).resolve().parent / "launch.py"
PY = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running():
    out = []
    for f in sorted((QUEUE / "running").glob("*.json")):
        j = json.loads(f.read_text())
        if j.get("pid") and alive(j["pid"]):
            out.append(j)
        else:
            j["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            (QUEUE / "done" / f.name).write_text(json.dumps(j, indent=1))
            f.unlink()
    return out


def dispatch_once(log):
    run = running()
    busy = {lane: sum(1 for j in run if j["lane"] == lane) for lane in SLOTS}
    pending = []
    for f in (QUEUE / "pending").glob("*.json"):
        try:
            pending.append((f, json.loads(f.read_text())))
        except json.JSONDecodeError:
            continue
    pending.sort(key=lambda fj: -fj[1].get("priority", 0))
    done_names = {json.loads(f.read_text())["name"] for f in (QUEUE / "done").glob("*.json")}
    for f, j in pending:
        if busy.get(j["lane"], 0) >= SLOTS.get(j["lane"], 1):
            continue
        if any(dep not in done_names for dep in j.get("after", [])):
            continue
        env = sum((["--env", f"{k}={v}"] for k, v in j.get("env", {}).items()), [])
        p = subprocess.Popen([str(PY), str(LAUNCH), j["name"], "--wait-min", "20", *env, "--", *j["cmd"]],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        j.update(pid=p.pid, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (QUEUE / "running" / f.name).write_text(json.dumps(j, indent=1))
        f.unlink()
        busy[j["lane"]] = busy.get(j["lane"], 0) + 1
        log.write(f"[{j['started']}] started {j['name']} on {j['lane']}\n")
        log.flush()


def main():
    for d in ("pending", "running", "done"):
        (QUEUE / d).mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ls"
    if cmd == "add":
        j = json.loads(Path(sys.argv[2]).read_text())
        (QUEUE / "pending" / f"{j['name']}.json").write_text(json.dumps(j, indent=1))
        print("queued", j["name"])
    elif cmd == "ls":
        for d in ("running", "pending", "done"):
            for f in sorted((QUEUE / d).glob("*.json")):
                j = json.loads(f.read_text())
                print(f"{d:<8} {j['lane']:<5} p{j.get('priority', 0):<3} {j['name']}  {j.get('started', '')}")
    elif cmd == "run":
        log = open(QUEUE / "dispatcher.log", "a")
        while True:
            dispatch_once(log)
            time.sleep(60)


if __name__ == "__main__":
    main()
