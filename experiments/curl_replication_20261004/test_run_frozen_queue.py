"""Exercise the serial owner with CPU children and tiny native parent receipts."""

import fcntl
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import run_frozen_queue
except ModuleNotFoundError:
    run_frozen_queue = None


class FrozenQueueTest(unittest.TestCase):
    def fixture(self, base, defect=None):
        parent = base / "parents"
        parent.mkdir()
        lock = base / "GPU.lock"
        snapshot = base / "snapshot"
        snapshot.mkdir()
        declaration = {
            "task": {"domain": "walker", "task": "walk", "action_repeat": 2},
            "training_seeds": [123, 456, 789],
            "arms": ["curl", "no_curl"],
            "parent_env_steps": 100000,
            "parent_decisions": 50000,
            "parent_updates": 49000,
            "evaluation_seed_start": 20000,
            "evaluation_episodes_per_policy": 50,
            "additional_training_steps": 0,
            "per_policy_cap_seconds": 1200,
            "batch_cap_seconds": 7200,
            "allocation_deadline_epoch": time.time() + 10000,
            "parent_campaign": str(parent),
            "output_campaign": str(base / "panels"),
            "gpu_lock": str(lock),
        }
        declaration_path = base / "declaration.json"
        declaration_path.write_text(json.dumps(declaration))
        for seed in (123, 456, 789):
            for arm in ("curl", "no_curl"):
                job_id = f"{arm}-seed{seed}-100k-v1"
                job_root = parent / job_id
                output = job_root / "run"
                output.mkdir(parents=True)
                config = {
                    "arm": arm,
                    "seed": seed,
                    "domain_name": "walker",
                    "task_name": "walk",
                    "action_repeat": 2,
                    "num_train_steps": 50000,
                    "init_steps": 1000,
                }
                (output / "config.json").write_text(json.dumps(config))
                (output / "latest.pt").write_bytes(b"never loaded or hashed by queue")
                rows = [
                    {
                        "type": "train_episode",
                        "step": 50000,
                        "env_steps": 100000,
                        "truncated_by_budget": False,
                    },
                    {
                        "type": "checkpoint",
                        "step": 50000,
                        "env_steps": 100000,
                        "reason": "completed",
                        "path": str(output / "latest.pt"),
                        "bytes": (output / "latest.pt").stat().st_size,
                    },
                    {
                        "type": "end",
                        "reason": "completed",
                        "step": 50000,
                        "env_steps": 100000,
                        "updates": 49000,
                    },
                ]
                (output / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                (job_root / "result.json").write_text(
                    json.dumps(
                        {
                            "id": job_id,
                            "pid": 1234,
                            "exit_code": 0,
                            "complete": True,
                            "reason": None,
                        }
                    )
                )
        (parent / "queue.jsonl").write_text(
            json.dumps({"type": "queue_end", "time_unix": time.time()}) + "\n"
        )
        for name in (
            "env_adapter.py",
            "train_reference.py",
            "checkpoint.py",
            "requirements.lock.txt",
        ):
            (snapshot / name).write_text("# small sealed fixture\n")
        (snapshot / "evaluate_frozen.py").write_text(
            "import argparse,fcntl,json,os,pathlib,signal,time\n"
            "p=argparse.ArgumentParser();p.add_argument('--parent-run');p.add_argument('--output');"
            "p.add_argument('--seed-start',type=int);p.add_argument('--episodes',type=int);"
            "p.add_argument('--max-seconds',type=int);a,_=p.parse_known_args()\n"
            "o=pathlib.Path(a.output);o.mkdir();c=json.loads((pathlib.Path(a.parent_run)/'config.json').read_text())\n"
            f"lock=open({str(lock)!r},'a');locked=False\n"
            "try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "except BlockingIOError:locked=True\n"
            "assert locked and a.seed_start==20000 and a.episodes==50 and a.max_seconds==1200\n"
            "assert all(os.environ[k]=='4' for k in "
            "('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))\n"
            "assert os.environ['MUJOCO_GL']=='egl' and os.environ['PYOPENGL_PLATFORM']=='egl'\n"
            "receipt={'evaluation_seeds':list(range(20000,20050)),'parent':{'run':a.parent_run,'config':c}}\n"
            "(o/'config.json').write_text(json.dumps(receipt));h=(o/'metrics.jsonl').open('w',buffering=1)\n"
            "h.write(json.dumps({'type':'start','time_unix':time.time()})+'\\n')\n"
            + (
                "signal.signal(signal.SIGTERM,lambda *args:None);time.sleep(10)\n"
                if defect == "stall"
                else ""
            )
            + "for s in range(20000,20050):h.write(json.dumps({'type':'eval_episode',"
            "'eval_seed':s,'return':float(s-20000),"
            "'step':50000,'env_steps':100000,'eval_decisions':500,'eval_env_steps':1000})+'\\n')\n"
            "result={'evaluation_only':True,'complete':True,'reason':'completed','episodes_requested':50,"
            "'episodes_completed':50,'training_steps':0,'optimizer_updates':0,'eval_decisions':25000,"
            "'eval_env_steps':50000,'mean_return':24.5}\n"
            + (
                "if c['seed']==456 and c['arm']=='curl':result['mean_return']=999\n"
                if defect == "mean"
                else ""
            )
            + "(o/'result.json').write_text(json.dumps(result));"
            "h.write(json.dumps({'type':'end',**result})+'\\n');h.close()\n"
        )
        return {
            "declaration": str(declaration_path),
            "python": sys.executable,
            "source": str(snapshot),
            "snapshot": str(snapshot),
            "device": "cpu",
        }, declaration

    def run_owner(self, config):
        self.assertIsNotNone(run_frozen_queue, "Frozen queue entrypoint must exist")
        with patch.object(run_frozen_queue, "POLL_SECONDS", 0.01):
            return run_frozen_queue.run_queue(config)

    def test_waits_without_gpu_then_serializes_six_declared_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            config, declaration = self.fixture(Path(directory))
            parent = Path(declaration["parent_campaign"])
            (parent / "queue.jsonl").write_text(json.dumps({"type": "launch"}) + "\n")
            waiting_lock_available = []

            def finish_original():
                root = Path(declaration["output_campaign"])
                for _ in range(1000):
                    if (root / "queue.jsonl").exists() and "waiting_parents" in (
                        root / "queue.jsonl"
                    ).read_text():
                        break
                    time.sleep(0.005)
                with Path(declaration["gpu_lock"]).open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    waiting_lock_available.append(True)
                    (parent / "queue.jsonl").write_text(json.dumps({"type": "queue_end"}) + "\n")
                    time.sleep(0.03)

            thread = threading.Thread(target=finish_original)
            thread.start()
            try:
                result = self.run_owner(config)
            finally:
                thread.join()
            self.assertTrue(result["complete"])
            self.assertEqual(result["policies_completed"], 6)
            self.assertEqual(waiting_lock_available, [True])
            root = Path(declaration["output_campaign"])
            events = [json.loads(line) for line in (root / "queue.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["type"] == "actual_return" for row in events), 300)
            launches = [row for row in events if row["type"] == "launch"]
            self.assertEqual(len(launches), 6)
            self.assertEqual(len({row["pid"] for row in launches}), 6)
            receipts = list(root.glob("*/result.json"))
            self.assertEqual(len(receipts), 6)
            self.assertTrue(all(json.loads(path.read_text())["complete"] for path in receipts))
            self.assertFalse(list(root.glob("**/latest.pt")))
            self.assertEqual(events[-1]["type"], "queue_end")

    def test_wrong_mean_with_exit_zero_is_preserved_and_other_policies_advance(self):
        with tempfile.TemporaryDirectory() as directory:
            config, declaration = self.fixture(Path(directory), defect="mean")
            result = self.run_owner(config)
            self.assertFalse(result["complete"])
            self.assertEqual(result["policies_completed"], 5)
            root = Path(declaration["output_campaign"])
            failed = json.loads((root / "curl-seed456-fresh-starts-v1" / "result.json").read_text())
            self.assertEqual(failed["exit_code"], 0)
            self.assertFalse(failed["complete"])
            with self.assertRaisesRegex(ValueError, "attempt"):
                self.run_owner(config)

    def test_duplicate_owner_is_rejected_before_artifact_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            config, declaration = self.fixture(Path(directory))
            root = Path(declaration["output_campaign"])
            root.mkdir()
            with (root / "owner.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "owner"):
                    self.run_owner(config)
            self.assertFalse((root / "queue.jsonl").exists())

    def test_stop_terminates_only_owned_child_then_kills_with_bounded_grace(self):
        with tempfile.TemporaryDirectory() as directory:
            config, declaration = self.fixture(Path(directory), defect="stall")
            root = Path(declaration["output_campaign"])

            def request_stop():
                for _ in range(1000):
                    metrics = list(root.glob("*/run/metrics.jsonl")) if root.exists() else []
                    if metrics and "start" in metrics[0].read_text():
                        (root / "STOP").touch()
                        return
                    time.sleep(0.005)

            thread = threading.Thread(target=request_stop)
            thread.start()
            try:
                self.assertIsNotNone(run_frozen_queue)
                with patch.object(run_frozen_queue, "GRACE_SECONDS", 0.03):
                    result = self.run_owner(config)
            finally:
                thread.join()
            self.assertFalse(result["complete"])
            receipts = list(root.glob("*/result.json"))
            self.assertEqual(len(receipts), 1)
            child = json.loads(receipts[0].read_text())
            self.assertEqual(child["exit_code"], -signal.SIGKILL)
            events = [json.loads(line) for line in (root / "queue.jsonl").read_text().splitlines()]
            kill = [row for row in events if row["type"] == "kill_owned_child"]
            self.assertEqual([row["pid"] for row in kill], [child["pid"]])

    def test_completed_original_queue_with_bad_parent_is_rejected_without_launch(self):
        for defect in ("seed", "receipt", "natural"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                config, declaration = self.fixture(Path(directory))
                parent = Path(declaration["parent_campaign"]) / "curl-seed123-100k-v1"
                if defect == "seed":
                    path = parent / "run" / "config.json"
                    value = json.loads(path.read_text())
                    value["seed"] = 456
                elif defect == "receipt":
                    path = parent / "result.json"
                    value = json.loads(path.read_text())
                    value["complete"] = False
                else:
                    path = parent / "run" / "metrics.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["truncated_by_budget"] = True
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                if defect != "natural":
                    path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    self.run_owner(config)
                root = Path(declaration["output_campaign"])
                self.assertFalse(list(root.glob("*/console.log")))

    def test_no_real_episode_terminates_each_stalled_child_and_advances(self):
        with tempfile.TemporaryDirectory() as directory:
            config, declaration = self.fixture(Path(directory), defect="stall")
            self.assertIsNotNone(run_frozen_queue)
            with (
                patch.object(run_frozen_queue, "FIRST_EPISODE_SECONDS", 0.02),
                patch.object(run_frozen_queue, "GRACE_SECONDS", 0.02),
            ):
                result = self.run_owner(config)
            self.assertFalse(result["complete"])
            root = Path(declaration["output_campaign"])
            receipts = [json.loads(path.read_text()) for path in root.glob("*/result.json")]
            self.assertEqual(len(receipts), 6)
            self.assertTrue(all(row["reason"] == "first_episode_stalled" for row in receipts))
            self.assertTrue(all(row["exit_code"] == -signal.SIGKILL for row in receipts))

    def test_expired_allocation_stops_waiting_without_gpu_or_children(self):
        with tempfile.TemporaryDirectory() as directory:
            config, declaration = self.fixture(Path(directory))
            declaration["allocation_deadline_epoch"] = time.time() - 1
            Path(config["declaration"]).write_text(json.dumps(declaration))
            result = self.run_owner(config)
            self.assertEqual(result["reason"], "allocation_deadline")
            root = Path(declaration["output_campaign"])
            self.assertFalse(Path(declaration["gpu_lock"]).exists())
            self.assertFalse(list(root.glob("*/result.json")))

    def test_polling_exception_reaps_child_before_shared_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            config, declaration = self.fixture(Path(directory), defect="stall")
            root = Path(declaration["output_campaign"])
            released_while_child_alive = []
            self.assertIsNotNone(run_frozen_queue)
            original_records = run_frozen_queue.records

            def injected_poll(path):
                rows = original_records(path)
                if path.name == "metrics.jsonl" and root in path.parents and rows:
                    time.sleep(0.02)  # The controlled child has installed its TERM handler.
                    raise RuntimeError("controlled post-launch polling failure")
                return rows

            def probe_release():
                for _ in range(2000):
                    events = original_records(root / "queue.jsonl")
                    launch = next((row for row in events if row.get("type") == "launch"), None)
                    if launch:
                        break
                    time.sleep(0.002)
                else:
                    return
                with Path(declaration["gpu_lock"]).open("a") as lock:
                    for _ in range(2000):
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            time.sleep(0.002)
                            continue
                        try:
                            os.kill(launch["pid"], 0)
                            released_while_child_alive.append(True)
                        except ProcessLookupError:
                            released_while_child_alive.append(False)
                        return

            probe = threading.Thread(target=probe_release)
            probe.start()
            try:
                with (
                    patch.object(run_frozen_queue, "records", side_effect=injected_poll),
                    patch.object(run_frozen_queue, "GRACE_SECONDS", 0.1),
                    self.assertRaisesRegex(RuntimeError, "controlled post-launch"),
                ):
                    self.run_owner(config)
            finally:
                probe.join()
            self.assertEqual(released_while_child_alive, [False])


if __name__ == "__main__":
    unittest.main()
