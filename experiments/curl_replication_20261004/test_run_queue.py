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
    def test_fresh_task_options_reach_the_child_and_completed_job_is_skipped(self):
        # Losing any forwarding flag silently substitutes the driver's cartpole defaults.
        for overrides, expected in (
            ({}, {"domain": "cartpole", "task": "swingup", "action_repeat": 8}),
            (
                {"domain": "walker", "task": "walk", "action_repeat": 2},
                {"domain": "walker", "task": "walk", "action_repeat": 2},
            ),
        ):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                source = base / "snapshot"
                source.mkdir()
                (source / "train_reference.py").write_text(
                    "import argparse,json,pathlib,sys\n"
                    "p=argparse.ArgumentParser();p.add_argument('--output');"
                    "p.add_argument('--domain',default='cartpole');"
                    "p.add_argument('--task',default='swingup');"
                    "p.add_argument('--action-repeat',type=int,default=8);"
                    "a,_=p.parse_known_args();o=pathlib.Path(a.output);o.mkdir()\n"
                    "(o/'received.json').write_text(json.dumps({'domain':a.domain,"
                    "'task':a.task,'action_repeat':a.action_repeat,'argv':sys.argv[1:]}))\n"
                    "(o/'config.json').write_text(json.dumps({'arm':'curl','seed':123,"
                    "'num_train_steps':1,'max_seconds':1,'eval_freq':1,'num_eval_episodes':1,"
                    "'domain_name':a.domain,'task_name':a.task,'action_repeat':a.action_repeat}))\n"
                    "(o/'latest.pt').write_bytes(b'fixture');"
                    "(o/'metrics.jsonl').write_text(json.dumps({'type':'eval_episode',"
                    "'return':1})+'\\n'+json.dumps({'type':'end','reason':'completed',"
                    "'step':1})+'\\n')\n"
                )
                job = {
                    "id": "fresh-task",
                    "arm": "curl",
                    "seed": 123,
                    "steps": 1,
                    "cap": 1,
                    "eval_every": 1,
                    "eval_episodes": 1,
                    **overrides,
                }
                config = {
                    "source_snapshot": str(source),
                    "source": str(source),
                    "python": sys.executable,
                    "campaign_root": str(base / "campaign"),
                    "allocation_deadline": time.time() + 1000,
                    "jobs": [job],
                }
                run_queue.run_queue(config)
                child = base / "campaign" / "fresh-task"
                received = json.loads((child / "run" / "received.json").read_text())
                self.assertEqual({key: received[key] for key in expected}, expected)
                for key, value in expected.items():
                    flag = "--" + key.replace("_", "-")
                    self.assertEqual(received["argv"][received["argv"].index(flag) + 1], str(value))
                self.assertTrue(json.loads((child / "result.json").read_text())["complete"])
                run_queue.run_queue(config)
                events = run_queue.records(base / "campaign" / "queue.jsonl")
                self.assertEqual(sum(row["type"] == "launch" for row in events), 1)
                self.assertEqual(sum(row["type"] == "skip_complete" for row in events), 1)

    def test_completed_rejects_different_effective_task_and_repeat(self):
        # Omitting effective defaults from identity checks admits another scientific task.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = {
                "id": "same-id",
                "arm": "curl",
                "seed": 123,
                "steps": 1,
                "cap": 1,
                "eval_every": 1,
                "eval_episodes": 1,
            }
            config = {
                "arm": "curl",
                "seed": 123,
                "num_train_steps": 1,
                "max_seconds": 1,
                "eval_freq": 1,
                "num_eval_episodes": 1,
                "domain_name": "cartpole",
                "task_name": "swingup",
                "action_repeat": 8,
            }
            (output / "latest.pt").write_bytes(b"fixture")
            (output / "metrics.jsonl").write_text(
                json.dumps({"type": "end", "reason": "completed", "step": 1}) + "\n"
            )
            (output / "config.json").write_text(json.dumps(config))
            self.assertTrue(run_queue.completed(output, job))
            for key, different in (
                ("domain_name", "walker"),
                ("task_name", "balance"),
                ("action_repeat", 2),
            ):
                with self.subTest(key=key):
                    (output / "config.json").write_text(json.dumps({**config, key: different}))
                    with self.assertRaisesRegex(ValueError, "immutable"):
                        run_queue.completed(output, job)

    def test_nondefault_task_without_config_is_not_complete(self):
        # Terminal counters alone cannot authenticate an unknown nondefault scientific task.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "latest.pt").write_bytes(b"fixture")
            (output / "metrics.jsonl").write_text(
                json.dumps({"type": "end", "reason": "completed", "step": 1}) + "\n"
            )
            for override in ({"domain": "walker"}, {"task": "balance"}, {"action_repeat": 2}):
                with self.subTest(override=override):
                    self.assertFalse(run_queue.completed(output, {"steps": 1, **override}))

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

    def recovery_fixture(self, base, stopped=False):
        config, initial = self.continuation_fixture(base)
        origin = base / "interrupted"
        output = origin / "run"
        output.mkdir(parents=True)
        checkpoint = output / "latest.pt"
        checkpoint.write_bytes(b"latest-successful-natural-checkpoint")
        original_config = json.loads((initial.parent / "config.json").read_text())
        parent_config = {**original_config, "num_train_steps": 62500}
        (output / "config.json").write_text(json.dumps(parent_config))
        origin_receipt = {
            "job": dict(config["jobs"][0]),
            "source_hashes": {"train_reference.py": "sealed-fixture"},
            "parent": {"checkpoint": str(initial), "config": original_config},
        }
        (origin / "job.json").write_text(json.dumps(origin_receipt))
        step = 25000 if stopped else 38375
        rows = [
            {"type": "resume", "step": 12500, "resume_path": str(initial)},
            {
                "type": "train_episode",
                "step": step,
                "env_steps": step * 8,
                "truncated_by_budget": False,
            },
            {
                "type": "checkpoint",
                "reason": "signal_15" if stopped else "periodic",
                "step": step,
                "env_steps": step * 8,
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
            },
        ]
        if stopped:
            rows.append(
                {
                    "type": "end",
                    "reason": "signal_15",
                    "step": step,
                    "env_steps": step * 8,
                    "updates": step - 1000,
                }
            )
        else:
            rows.extend(
                [
                    {
                        "type": "train_episode",
                        "step": 51125,
                        "env_steps": 409000,
                        "truncated_by_budget": False,
                    },
                    {"type": "failure", "step": 51125, "error": "OverflowError"},
                ]
            )
        (output / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (origin / "result.json").write_text(
            json.dumps(
                {
                    "id": "extended-123",
                    "complete": False,
                    "exit_code": 0 if stopped else 1,
                    "reason": "stop" if stopped else None,
                }
            )
        )
        config["jobs"][0].update(resume=str(checkpoint), recovery=True)
        return config, checkpoint, rows

    def test_recovery_retains_latest_saved_state_and_receives_shared_lock(self):
        # Hardcoded 100k admission, losing resume, or selecting the last failed episode breaks this.
        for stopped, expected_step in ((False, 38375), (True, 25000)):
            with self.subTest(stopped=stopped), tempfile.TemporaryDirectory() as directory:
                config, checkpoint, _ = self.recovery_fixture(Path(directory), stopped)
                original = checkpoint.read_bytes()
                checkpoint_reads = []
                real_digest = run_queue.stream_sha256

                def digest(
                    path,
                    checkpoint=checkpoint,
                    config=config,
                    checkpoint_reads=checkpoint_reads,
                    real_digest=real_digest,
                ):
                    if path == checkpoint:
                        with Path(config["gpu_lock"]).open("a") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            checkpoint_reads.append(path)
                    return real_digest(path)

                with patch.object(run_queue, "stream_sha256", side_effect=digest):
                    run_queue.run_queue(config)
                child = Path(config["campaign_root"]) / "extended-123"
                received = json.loads((child / "run" / "received.json").read_text())
                self.assertEqual(received, {"resume": str(checkpoint), "shared_lock": True})
                proof = json.loads((child / "job.json").read_text())["parent"]
                self.assertEqual(proof["kind"], "recovery")
                self.assertEqual(proof["restore_step"], expected_step)
                self.assertEqual(proof["restore_env_steps"], expected_step * 8)
                self.assertEqual(proof["terminal_proof"]["train_episode"]["step"], expected_step)
                self.assertEqual(proof["terminal_proof"]["end"] is not None, stopped)
                origin_job = checkpoint.parent.parent / "job.json"
                self.assertEqual(proof["origin_job_path"], str(origin_job))
                self.assertEqual(
                    proof["origin_job_sha256"], hashlib.sha256(origin_job.read_bytes()).hexdigest()
                )
                self.assertEqual(checkpoint_reads, [checkpoint])
                self.assertEqual(checkpoint.read_bytes(), original)

    def test_recovery_rejects_live_untrusted_or_nonlatest_natural_parent(self):
        # Weakening termination, scientific matching or latest-save admission permits invalid forks.
        for defect in (
            "live",
            "no_native_terminal",
            "truncated",
            "missing_origin",
            "wrong_seed",
            "wrong_budget",
            "later_save",
            "boolean_exit",
            "wrong_origin_seed",
            "wrong_result_id",
            "wrong_native_resume",
        ):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                config, checkpoint, rows = self.recovery_fixture(Path(directory))
                origin = checkpoint.parent.parent
                if defect == "live":
                    (origin / "result.json").unlink()
                elif defect == "missing_origin":
                    (origin / "job.json").unlink()
                elif defect == "boolean_exit":
                    receipt = json.loads((origin / "result.json").read_text())
                    receipt["exit_code"] = False
                    (origin / "result.json").write_text(json.dumps(receipt))
                elif defect == "wrong_result_id":
                    receipt = json.loads((origin / "result.json").read_text())
                    receipt["id"] = "another-owner"
                    (origin / "result.json").write_text(json.dumps(receipt))
                elif defect == "wrong_origin_seed":
                    receipt = json.loads((origin / "job.json").read_text())
                    receipt["job"]["seed"] = 456
                    (origin / "job.json").write_text(json.dumps(receipt))
                elif defect == "wrong_native_resume":
                    rows[0]["step"] = 13000
                elif defect in ("wrong_seed", "wrong_budget"):
                    settings = json.loads((checkpoint.parent / "config.json").read_text())
                    settings["seed" if defect == "wrong_seed" else "num_train_steps"] = 456
                    (checkpoint.parent / "config.json").write_text(json.dumps(settings))
                elif defect == "no_native_terminal":
                    rows.pop()
                elif defect == "truncated":
                    rows[1]["truncated_by_budget"] = True
                else:
                    rows.insert(
                        -1,
                        {
                            **rows[2],
                            "step": 50000,
                            "env_steps": 400000,
                            "path": str(checkpoint.parent / "other.pt"),
                        },
                    )
                (checkpoint.parent / "metrics.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                with self.assertRaises((ValueError, FileNotFoundError)):
                    run_queue.run_queue(config)
                self.assertFalse(
                    (Path(config["campaign_root"]) / "extended-123" / "console.log").exists()
                )

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
