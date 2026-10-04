"""Run the real queue against tiny CPU subprocesses, never a GPU."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import run_queue


class QueueTest(unittest.TestCase):
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
