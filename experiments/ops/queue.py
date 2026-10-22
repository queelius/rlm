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
HOME = Path("/home/atowell/research-runs/selective-recursion-20261019")
SLOTS = {"gpu0": 3, "gpu1": 4}
LAUNCH = Path(__file__).resolve().parent / "launch.py"
PY = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"


def alive(pid: int) -> bool:
    """True for a live process; a zombie (state Z) counts as exited."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def job_pid(j):
    """The experiment's own PID (written by launch.py), not the short-lived launch wrapper's."""
    f = HOME / j["name"] / "PID"
    return int(f.read_text()) if f.exists() else j.get("pid")


def reap():
    """Collect exited launch wrappers so they do not linger as zombies that still look alive."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def running():
    reap()
    out = []
    for f in sorted((QUEUE / "running").glob("*.json")):
        j = json.loads(f.read_text())
        pid = job_pid(j)
        if pid and alive(pid):
            out.append(j)
        else:
            j["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            (QUEUE / "done" / f.name).write_text(json.dumps(j, indent=1))
            f.unlink()
    return out


def slots():
    """Lane capacities, re-read every loop from QUEUE/slots.json so they change without a restart."""
    try:
        return {**SLOTS, **json.loads((QUEUE / "slots.json").read_text())}
    except (OSError, json.JSONDecodeError):
        return SLOTS


IDLE_SINCE: dict = {}
IDLE_ALARM_MIN = 10


def idle_alarm(run, lanes, log):
    """Lab suggestion I-007: alarm when a lane has run nothing for IDLE_ALARM_MIN minutes (log line and IDLE-ALARM file).
    Jobs launched outside the queue are invisible here, so the alarm means 'nothing queued is running', a prompt to look."""
    now = time.time()
    active = {j["lane"] for j in run}
    lines = []
    for lane in lanes:
        if lane in active:
            IDLE_SINCE.pop(lane, None)
            continue
        since = IDLE_SINCE.setdefault(lane, now)
        if now - since >= IDLE_ALARM_MIN * 60:
            lines.append(f"{lane} idle for {int((now - since) / 60)} min")
    alarm = QUEUE / "IDLE-ALARM"
    if lines:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        alarm.write_text(f"[{stamp}] " + "; ".join(lines) + "\n")
        if int(now) % 600 < 60:  # at most about one log line per ten minutes
            log.write(f"[{stamp}] IDLE ALARM: " + "; ".join(lines) + "\n")
            log.flush()
    elif alarm.exists():
        alarm.unlink()


def dispatch_once(log):
    SLOTS = slots()  # noqa: N806
    run = running()
    idle_alarm(run, list(SLOTS), log)
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
