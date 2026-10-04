"""Run the real queue against tiny CPU subprocesses, never a GPU."""

import fcntl
import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import run_queue


class QueueTest(unittest.TestCase):
    def continuation_fixture(self, base, mismatch=None, truncated=False, child_env_steps=500000):
        source = base / "snapshot"
        source.mkdir()
        names = ("curl_sac.py", "utils.py", "encoder.py", "train.py")
        for name in names:
            (source / name).write_text("# frozen official fixture\n")
        shared_lock = base / "initial-campaign" / "GPU.lock"
        shared_lock.parent.mkdir()
        reference = {
            "domain_name": "cartpole",
            "task_name": "swingup",
            "action_repeat": 8,
            "init_steps": 1000,
            "batch_size": 128,
            "num_train_steps": 12500,
            "eval_freq": 500,
            "num_eval_episodes": 10,
        }
        (source / "manifest.json").write_text(
            json.dumps({"reference_configuration": reference, "upstream": {"commit": "fixture"}})
        )
        parent = base / "parent"
        parent.mkdir()
        checkpoint = parent / "latest.pt"
        checkpoint.write_bytes(b"completed-parent")
        parent_config = {**reference, "seed": 123, "arm": "curl", "upstream_commit": "fixture"}
        parent_config["upstream_sha256"] = {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names
        }
        if mismatch:
            parent_config.update(mismatch)
        (parent / "config.json").write_text(json.dumps(parent_config))
        rows = [
            {"type": "train_episode", "step": 12500, "truncated_by_budget": truncated},
            {
                "type": "checkpoint",
                "reason": "completed",
                "step": 12500,
                "env_steps": 100000,
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
            },
            {
                "type": "end",
                "reason": "completed",
                "step": 12500,
                "env_steps": 100000,
                "updates": 11500,
            },
        ]
        (parent / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (source / "train_reference.py").write_text(
            "import argparse,fcntl,json,pathlib\n"
            "p=argparse.ArgumentParser();p.add_argument('--output');"
            "p.add_argument('--resume',required=True);p.add_argument('--steps',type=int);"
            "a,_=p.parse_known_args();o=pathlib.Path(a.output);o.mkdir()\n"
            f"locked=False;lock=open({str(shared_lock)!r},'a')\n"
            "try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "except BlockingIOError:locked=True\n"
            "(o/'received.json').write_text(json.dumps({'resume':a.resume,'shared_lock':locked}))\n"
            "(o/'latest.pt').write_bytes(b'child');"
            "(o/'metrics.jsonl').write_text(json.dumps({'type':'eval_episode','return':1})+'\\n'+"
            f"json.dumps({{'type':'end','reason':'completed','step':a.steps,'env_steps':{child_env_steps}}})+'\\n')\n"
        )
        job = {
            "id": "extended-123",
            "arm": "curl",
            "seed": 123,
            "steps": 62500,
            "env_steps": 500000,
            "cap": 10,
            "eval_every": 500,
            "eval_episodes": 10,
            "resume": str(checkpoint),
        }
        config = {
            "source_snapshot": str(source),
            "source": str(source),
            "python": sys.executable,
            "campaign_root": str(base / "continuations"),
            "gpu_lock": str(shared_lock),
            "allocation_deadline": time.time() + 1000,
            "jobs": [job],
        }
        return config, checkpoint

    def test_continuation_receives_parent_and_shared_lock_and_records_provenance(self):
        # Ignoring resume silently fresh-trains; using a new lock permits competing GPU jobs.
        with tempfile.TemporaryDirectory() as directory:
            config, parent = self.continuation_fixture(Path(directory))
            original = parent.read_bytes()
            checkpoint_reads = []
            real_digest = run_queue.stream_sha256

            def digest(path):
                if path == parent:
                    # Acquiring the same lock here proves large hashing precedes GPU ownership.
                    with Path(config["gpu_lock"]).open("a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        checkpoint_reads.append(path)
                return real_digest(path)

            with patch.object(run_queue, "stream_sha256", side_effect=digest):
                run_queue.run_queue(config)
            self.assertEqual(checkpoint_reads, [parent])
            child = Path(config["campaign_root"]) / "extended-123"
            self.assertTrue(json.loads((child / "result.json").read_text())["complete"])
            self.assertEqual(
                json.loads((child / "run" / "received.json").read_text()),
                {"resume": str(parent), "shared_lock": True},
            )
            provenance = json.loads((child / "job.json").read_text())["parent"]
            self.assertEqual(provenance["checkpoint_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(provenance["config"]["num_train_steps"], 12500)
            self.assertEqual(provenance["terminal_proof"]["end"]["env_steps"], 100000)
            self.assertEqual(parent.read_bytes(), original)

    def test_continuation_rejects_mismatched_or_truncated_parent_before_launch(self):
        for mismatch, truncated in (
            ({"seed": 456}, False),
            ({"arm": "no_curl"}, False),
            ({"batch_size": 64}, False),
            (None, True),
        ):
            with (
                self.subTest(mismatch=mismatch, truncated=truncated),
                tempfile.TemporaryDirectory() as directory,
            ):
                config, _ = self.continuation_fixture(Path(directory), mismatch, truncated)
                with self.assertRaisesRegex(ValueError, "parent"):
                    run_queue.run_queue(config)
                self.assertFalse(
                    (Path(config["campaign_root"]) / "extended-123" / "console.log").exists()
                )

    def test_continuation_rejects_wrong_simulator_endpoint_despite_successful_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.continuation_fixture(Path(directory), child_env_steps=499992)
            run_queue.run_queue(config)
            result = Path(config["campaign_root"]) / "extended-123" / "result.json"
            self.assertEqual(json.loads(result.read_text())["exit_code"], 0)
            self.assertFalse(json.loads(result.read_text())["complete"])

    def test_serial_completion_failure_preservation_and_immutable_ids(self):
        # Silent retries, accepting a wrong endpoint, or reusing an ID breaks this test.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            snapshot = base / "snapshot"
            snapshot.mkdir()
            (snapshot / "train_reference.py").write_text(
                "import argparse,json,pathlib,sys\n"
                "p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--seed',type=int);"
                "p.add_argument('--steps',type=int);a,_=p.parse_known_args()\n"
                "o=pathlib.Path(a.output);o.mkdir();"
                "(o/'metrics.jsonl').write_text(json.dumps({'type':'eval_episode','return':1})+'\\n'+"
                "json.dumps({'type':'end','reason':'completed','step':a.steps})+'\\n');"
                "(o/'latest.pt').write_bytes(b'fixture');sys.exit(3 if a.seed==456 else 0)\n"
            )
            jobs = [
                {
                    "id": f"seed-{seed}",
                    "arm": "curl",
                    "seed": seed,
                    "steps": 1,
                    "cap": 1,
                    "eval_every": 1,
                    "eval_episodes": 1,
                }
                for seed in (123, 456, 789)
            ]
            config = {
                "source_snapshot": str(snapshot),
                "source": str(snapshot),
                "python": sys.executable,
                "campaign_root": str(base / "campaign"),
                "allocation_deadline": time.time() + 1000,
                "jobs": jobs,
            }
            run_queue.run_queue(config)
            root = base / "campaign"
            results = [json.loads((root / job["id"] / "result.json").read_text()) for job in jobs]
            self.assertEqual([result["complete"] for result in results], [True, False, True])
            self.assertEqual([result["exit_code"] for result in results], [0, 3, 0])
            self.assertEqual([result["scientific_records"] for result in results], [1, 1, 1])
            run_queue.run_queue(config)
            events = [json.loads(line) for line in (root / "queue.jsonl").read_text().splitlines()]
            self.assertEqual(sum(event["type"] == "launch" for event in events), 3)
            self.assertEqual(sum(event["type"] == "skip_complete" for event in events), 2)
            self.assertEqual(sum(event["type"] == "preserve_attempt" for event in events), 1)
            config["jobs"][0] = {**jobs[0], "steps": 2}
            with self.assertRaisesRegex(ValueError, "immutable"):
                run_queue.run_queue(config)


if __name__ == "__main__":
    unittest.main()
