"""Bound one isolated Linux session, including nested GNU timeout groups.

Per-job timeout commands remain responsible for their own deadlines. This owner
cleans all members of its new session on whole-sequence expiry, TERM/INT, or exit.
Children must not deliberately escape the owned session with setsid().
"""

import argparse
import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
LIBC.pidfd_open.restype = ctypes.c_int
LIBC.pidfd_send_signal.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint)
LIBC.pidfd_send_signal.restype = ctypes.c_int


def members(session_id: int) -> list[tuple[int, int]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[3]) == session_id and stat[0] != "Z":
                found.append((int(entry.name), int(stat[19])))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def signal_member(pid: int, ticks: int, session_id: int, signum: int) -> None:
    # pidfd pins this identity while /proc validates ownership; never signal a reusedPID.
    fd = LIBC.pidfd_open(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            return
        raise OSError(error, os.strerror(error))
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if (
            int(stat[19]) == ticks
            and int(stat[3]) == session_id
            and LIBC.pidfd_send_signal(fd, signum, None, 0) < 0
        ):
            error = ctypes.get_errno()
            if error != errno.ESRCH:
                raise OSError(error, os.strerror(error))
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        os.close(fd)


def cleanup(session_id: int, grace: float) -> tuple[int, list[tuple[int, int]]]:
    initial = members(session_id)
    for pid, ticks in initial:
        signal_member(pid, ticks, session_id, signal.SIGTERM)
    term_end = time.monotonic() + grace
    while members(session_id) and time.monotonic() < term_end:
        time.sleep(0.02)
    kill_end = time.monotonic() + 5
    remaining = members(session_id)
    while remaining and time.monotonic() < kill_end:
        for pid, ticks in remaining:
            signal_member(pid, ticks, session_id, signal.SIGKILL)
        time.sleep(0.02)
        remaining = members(session_id)
    return len(initial), remaining


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--grace", type=float, default=55)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.seconds <= 0 or args.grace < 0:
        parser.error("Supply a command, positive cap, and nonnegative grace")
    stopped = {"signal": None}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda sig, frame: stopped.update(signal=sig))
    child = subprocess.Popen(command, start_new_session=True)
    deadline = time.monotonic() + args.seconds
    timed_out = False
    try:
        while child.poll() is None and stopped["signal"] is None:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.02)
    finally:
        cleaned, remaining = cleanup(child.pid, args.grace)
        if not remaining:
            child.wait(timeout=5)
        receipt = {
            "owned_session": child.pid,
            "timed_out": timed_out,
            "signal": stopped["signal"],
            "cleanup_initial_members": cleaned,
            "remaining_owned_members": remaining,
            "child_returncode": child.returncode,
        }
        print(json.dumps(receipt), file=sys.stderr, flush=True)
    if remaining:
        return 125
    if stopped["signal"] is not None:
        return 128 + stopped["signal"]
    if timed_out:
        return 124
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    sys.exit(main())
