"""CPU integration checks for owned cleanup across nested GNU timeout groups."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HELPER = Path(__file__).with_name("scoped_session_owner.py")


class ScopedSessionOwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="scoped-owner-test-")
        self.root = Path(self.tmp.name)
        self.leaf = self.root / "leaf.json"
        self.owner = None

    def tearDown(self):
        # Fixture cleanup only: authenticate the exact PID/startticks it created.
        if self.leaf.exists():
            pid, ticks = json.loads(self.leaf.read_text())
            self.signal_fixture(pid, ticks, signal.SIGKILL)
        if self.owner is not None and self.owner.poll() is None:
            self.owner.terminate()
            self.owner.wait(timeout=3)
        self.tmp.cleanup()

    def signal_fixture(self, pid, ticks, signum):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[19]) == ticks and stat[0] != "Z":
                os.kill(pid, signum)
        except (FileNotFoundError, ProcessLookupError):
            pass

    def alive(self):
        pid, ticks = json.loads(self.leaf.read_text())
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return int(stat[19]) == ticks and stat[0] != "Z"
        except FileNotFoundError:
            return False

    def command(self, inner_cap="5s", orphan=False):
        worker = (
            "import json,os,pathlib,signal,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "s=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split();"
            f"pathlib.Path({str(self.leaf)!r}).write_text(json.dumps([os.getpid(),int(s[19])]));"
            "time.sleep(20)"
        )
        parent = (
            "import pathlib,subprocess,sys,time;"
            f"p=subprocess.Popen(['timeout','--kill-after=0.1s',{inner_cap!r},"
            f"sys.executable,'-c',{worker!r}]);"
        )
        if orphan:
            parent += f"\nwhile not pathlib.Path({str(self.leaf)!r}).exists(): time.sleep(0.01)\n"
        else:
            parent += "code=p.wait();sys.exit(code if code>=0 else 128-code)"
        return [sys.executable, "-c", parent]

    def start(self, seconds, command):
        self.owner = subprocess.Popen(
            [
                sys.executable,
                str(HELPER),
                "--seconds",
                str(seconds),
                "--grace",
                "0.1",
                "--",
                *command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def finish(self):
        out, err = self.owner.communicate(timeout=4)
        self.assertTrue(self.leaf.exists(), f"CPU fixture did not start: {out} {err}")
        self.assertFalse(self.alive(), "Owned CPU worker survived owner return")
        return self.owner.returncode, json.loads(err.strip().splitlines()[-1])

    def test_whole_cap_kills_term_ignoring_child_in_nested_group(self):
        # Removing cleanup beyond the leader process group would leave this worker alive.
        self.start(0.4, self.command())
        code, receipt = self.finish()
        self.assertEqual(code, 124)
        self.assertTrue(receipt["timed_out"])

    def test_inner_cap_still_cleans_child_before_whole_cap(self):
        # Replacing the inner timeout with foreground mode would lose its group kill.
        self.start(3, self.command(inner_cap="0.3s"))
        code, receipt = self.finish()
        self.assertIn(code, (124, 137))
        self.assertFalse(receipt["timed_out"])

    def test_external_term_cleans_nested_child(self):
        self.start(3, self.command())
        deadline = time.monotonic() + 2
        while not self.leaf.exists() and self.owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.leaf.exists(), "CPU fixture did not start")
        self.owner.terminate()
        code, receipt = self.finish()
        self.assertEqual(code, 143)
        self.assertEqual(receipt["signal"], signal.SIGTERM)

    def test_cleanup_excludes_unrelated_session(self):
        outsider = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
        )
        try:
            self.start(0.4, self.command())
            self.finish()
            self.assertIsNone(outsider.poll(), "Unrelated session was signaled")
        finally:
            outsider.terminate()
            outsider.wait(timeout=2)

    def test_normal_return_also_cleans_orphaned_nested_child(self):
        self.start(3, self.command(orphan=True))
        code, receipt = self.finish()
        self.assertEqual(code, 0)
        self.assertFalse(receipt["timed_out"])


if __name__ == "__main__":
    unittest.main()
