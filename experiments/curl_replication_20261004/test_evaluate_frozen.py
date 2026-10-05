"""CPU checkpoint/real evaluation seam tests; no rendering, GPU, or replay construction."""

import hashlib
import importlib.metadata
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import train_reference

try:
    import evaluate_frozen
except ModuleNotFoundError:
    evaluate_frozen = None


class FrozenAgent:
    def __init__(self, obs_shape, action_shape, device):
        assert obs_shape == (9, 84, 84) and action_shape == (1,)
        self.actor = torch.nn.Linear(1, 1).to(device)
        self.training = True

    def train(self, training=True):
        self.training = training

    def select_action(self, observation):
        assert observation.shape == (9, 84, 84)
        assert np.all(observation == 2), "Policy must receive the deterministic center crop"
        with torch.no_grad():
            return self.actor(torch.ones(1)).numpy()

    def update(self, *args):
        raise AssertionError("Frozen evaluation cannot update the learner")


class FrozenEvaluationTest(unittest.TestCase):
    def fixture(self, base):
        source, parent = base / "source", base / "parent"
        source.mkdir()
        parent.mkdir()
        names = ("curl_sac.py", "utils.py", "encoder.py", "train.py")
        for name in names:
            (source / name).write_text("# pinned upstream fixture\n")
        config = {
            "domain_name": "walker",
            "task_name": "walk",
            "action_repeat": 2,
            "num_train_steps": 50000,
            "init_steps": 1000,
            "seed": 123,
            "arm": "curl",
            "encoder_type": "pixel",
            "frame_stack": 3,
            "image_size": 84,
            "evaluation_seeds": list(range(10000, 10010)),
            "num_eval_episodes": 10,
            "upstream_commit": train_reference.UPSTREAM_COMMIT,
            "upstream_sha256": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names
            },
        }
        (parent / "config.json").write_text(json.dumps(config))
        counters = {
            "step": 50000,
            "env_steps": 100000,
            "updates": 49000,
            "episode": 100,
            "episode_boundary": True,
            "boundary_kind": "natural",
            "last_eval_step": 50000,
        }
        payload = {
            "schema_version": 1,
            "config": config,
            "counters": counters,
            "modules": {"actor": {"weight": torch.tensor([[3.0]]), "bias": torch.zeros(1)}},
            "optimizers": {},
            "replay": {"arrays": {"obses": np.ones((1, 9, 100, 100), dtype=np.uint8)}},
            "rng": {},
        }
        checkpoint = parent / "latest.pt"
        torch.save(payload, checkpoint, pickle_protocol=4)
        rows = [{"type": "start", "step": 0, "config": config}]
        rows += [
            {
                "type": "eval_episode",
                "step": 50000,
                "env_steps": 100000,
                "eval_seed": seed,
                "return": 7,
                "eval_decisions": 500,
                "eval_env_steps": 1000,
            }
            for seed in range(10000, 10010)
        ]
        rows += [
            {
                "type": "train_episode",
                "step": 50000,
                "env_steps": 100000,
                "episode": 100,
                "truncated_by_budget": False,
            },
            {
                "type": "checkpoint",
                "step": 50000,
                "env_steps": 100000,
                "reason": "completed",
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
            },
            {
                "type": "end",
                "reason": "completed",
                **{k: counters[k] for k in ("step", "env_steps", "updates")},
            },
        ]
        self.write_rows(parent, rows)
        args = SimpleNamespace(
            source=source,
            parent_run=parent,
            output=base / "evaluation",
            seed_start=20000,
            episodes=2,
            device="cpu",
            max_seconds=1200,
        )
        return args, payload, rows

    def write_rows(self, parent, rows):
        (parent / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def run_main(self, args, fail_seed=None, expired=False):
        self.assertIsNotNone(evaluate_frozen, "Frozen evaluation entrypoint must exist")
        output = args.output

        class Env:
            action_space = SimpleNamespace(shape=(1,))

            def seed(self, seed):
                if seed > args.seed_start:
                    prior = [
                        json.loads(line)
                        for line in (output / "metrics.jsonl").read_text().splitlines()
                    ]
                    assert any(row.get("eval_seed") == seed - 1 for row in prior)
                if seed == fail_seed:
                    raise RuntimeError("controlled episode failure")

            def reset(self):
                self.steps = 0
                observation = np.zeros((9, 100, 100), dtype=np.uint8)
                observation[:, 8:92, 8:92] = 2
                return observation

            def step(self, action):
                self.steps += 1
                observation = np.full((9, 100, 100), 2, dtype=np.uint8)
                return observation, float(action[0]), self.steps == 2, {"env_steps": 2}

            def close(self):
                pass

        clock = iter([0, 0, 1201, 1201, 1201, 1201]) if expired else None
        with (
            patch.object(evaluate_frozen, "parse_args", return_value=args),
            patch("subprocess.check_output", return_value=train_reference.UPSTREAM_COMMIT + "\n"),
            patch.dict(
                sys.modules,
                {
                    "curl_sac": SimpleNamespace(CurlSacAgent=FrozenAgent),
                    "env_adapter": SimpleNamespace(make_env=lambda *unused: Env()),
                },
            ),
            patch.object(sys, "path", list(sys.path)),
            patch.object(
                evaluate_frozen.time,
                "monotonic",
                side_effect=(lambda: next(clock, 1201))
                if expired
                else __import__("time").monotonic,
            ),
        ):
            evaluate_frozen.main()

    def test_actor_only_checkpoint_restore_streams_fresh_center_crop_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(Path(directory))
            original = (args.parent_run / "latest.pt").read_bytes()
            checkpoint_reads = []
            digest = evaluate_frozen.stream_sha256 if evaluate_frozen else None

            def counted_digest(path):
                checkpoint_reads.append(path)
                return digest(path)

            self.assertIsNotNone(evaluate_frozen)
            with patch.object(evaluate_frozen, "stream_sha256", side_effect=counted_digest):
                self.run_main(args)
            self.assertEqual(checkpoint_reads, [args.parent_run / "latest.pt"])
            rows = [
                json.loads(line)
                for line in (args.output / "metrics.jsonl").read_text().splitlines()
            ]
            episodes = [row for row in rows if row["type"] == "eval_episode"]
            self.assertEqual([row["eval_seed"] for row in episodes], [20000, 20001])
            self.assertEqual([row["return"] for row in episodes], [6.0, 6.0])
            self.assertEqual([row["eval_env_steps"] for row in episodes], [4, 4])
            result = json.loads((args.output / "result.json").read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(result["mean_return"], 6.0)
            self.assertEqual(result["training_steps"], 0)
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertEqual(result["episodes_completed"], 2)
            receipt = json.loads((args.output / "config.json").read_text())
            self.assertEqual(receipt["runtime"]["python_version"], sys.version)
            self.assertEqual(
                receipt["runtime"]["package_versions"]["torch"],
                importlib.metadata.version("torch"),
            )
            for name in (
                "evaluate_frozen.py",
                "env_adapter.py",
                "train_reference.py",
                "checkpoint.py",
            ):
                self.assertEqual(
                    receipt["evaluation_source_sha256"][name],
                    hashlib.sha256(
                        Path(evaluate_frozen.__file__).with_name(name).read_bytes()
                    ).hexdigest(),
                )
            self.assertEqual(receipt["evaluation_seeds"], [20000, 20001])
            self.assertEqual(
                receipt["parent"]["checkpoint_sha256"], hashlib.sha256(original).hexdigest()
            )
            self.assertEqual(
                receipt["parent"]["config"],
                json.loads((args.parent_run / "config.json").read_text()),
            )
            self.assertEqual((args.parent_run / "latest.pt").read_bytes(), original)
            self.assertFalse((args.output / "latest.pt").exists())

    def test_invalid_parent_or_overlapping_panel_never_scores(self):
        for defect in (
            "unfinished",
            "truncated",
            "wrong_counters",
            "source_changed",
            "overlap",
            "payload_config",
            "payload_boundary",
        ):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                args, payload, rows = self.fixture(Path(directory))
                if defect == "unfinished":
                    rows[-1]["reason"] = "time_cap"
                elif defect == "truncated":
                    rows[-3]["truncated_by_budget"] = True
                elif defect == "wrong_counters":
                    rows[-1]["updates"] = 48999
                elif defect == "source_changed":
                    (args.source / "utils.py").write_text("# changed\n")
                elif defect == "overlap":
                    args.seed_start = 10009
                else:
                    if defect == "payload_config":
                        payload["config"] = {**payload["config"], "seed": 456}
                    else:
                        payload["counters"]["boundary_kind"] = "budget_truncation"
                    torch.save(payload, args.parent_run / "latest.pt", pickle_protocol=4)
                    rows[-2]["bytes"] = (args.parent_run / "latest.pt").stat().st_size
                self.write_rows(args.parent_run, rows)
                with self.assertRaises(ValueError):
                    self.run_main(args)
                result = json.loads((args.output / "result.json").read_text())
                self.assertFalse(result["complete"])
                self.assertNotIn("mean_return", result)
                self.assertFalse(
                    any(
                        row.get("type") == "eval_episode"
                        for row in [
                            json.loads(line)
                            for line in (args.output / "metrics.jsonl").read_text().splitlines()
                        ]
                    )
                )

    def test_failure_retains_streamed_episode_and_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                self.run_main(args, fail_seed=20001)
            result = json.loads((args.output / "result.json").read_text())
            self.assertFalse(result["complete"])
            self.assertEqual(result["episodes_completed"], 1)
            self.assertNotIn("mean_return", result)
            original = (args.output / "metrics.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                self.run_main(args)
            self.assertEqual((args.output / "metrics.jsonl").read_bytes(), original)

    def test_time_cap_is_incomplete_without_fixed_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(Path(directory))
            self.run_main(args, expired=True)
            result = json.loads((args.output / "result.json").read_text())
            self.assertFalse(result["complete"])
            self.assertEqual(result["reason"], "time_cap")
            self.assertNotIn("mean_return", result)


if __name__ == "__main__":
    unittest.main()
