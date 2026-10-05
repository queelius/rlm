"""Focused evaluation isolation checks without rendering or GPU use."""

import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import checkpoint
import numpy as np
import torch
import train_reference


class EvaluationTest(unittest.TestCase):
    def test_raw_reward_sums_fixed_seeds_and_training_rng_survive_evaluation(self):
        # Sharing RNG or averaging action-repeat rewards breaks this result.
        class Env:
            def seed(self, seed):
                self.current_seed = seed

            def reset(self):
                self.steps = 0
                return np.zeros((9, 100, 100), dtype=np.uint8)

            def step(self, action):
                self.steps += 1
                random.random()
                np.random.random()
                torch.rand(())
                return (
                    np.zeros((9, 100, 100)),
                    self.current_seed - 9990,
                    self.steps == 2,
                    {"env_steps": 8},
                )

        class Agent:
            training = True

            def train(self, value=True):
                self.training = value

            def select_action(self, observation):
                assert observation.shape == (9, 84, 84)
                return np.zeros(1)

        agent = Agent()
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        before = checkpoint.capture_rng()
        records = train_reference.evaluate(
            Env(), agent, step=500, env_steps=4000, seeds=[10000, 10001], image_size=84
        )
        actual = (random.random(), np.random.random(), torch.rand(()))
        checkpoint.restore_rng(before)
        expected = (random.random(), np.random.random(), torch.rand(()))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        self.assertTrue(agent.training)
        self.assertEqual([record["return"] for record in records], [20.0, 22.0])
        self.assertEqual([record["eval_env_steps"] for record in records], [16, 16])
        self.assertEqual([record["eval_seed"] for record in records], [10000, 10001])


class ResumeConfigTest(unittest.TestCase):
    def run_resume(
        self,
        output,
        source,
        checkpoint_seed=123,
        arm="curl",
        private_state=None,
        previous_arm=None,
        captured=None,
    ):
        # Replace expensive external learning/simulation; execute the real main/file/logger path.
        args = SimpleNamespace(
            source=source,
            output=output,
            seed=123,
            domain="cartpole",
            task="swingup",
            action_repeat=8,
            steps=2,
            eval_every=500,
            eval_episodes=1,
            arm=arm,
            resume=source / "parent.pt",
            max_seconds=10,
            checkpoint_seconds=900,
            device="cpu",
        )
        shape = (9, 100, 100)
        env = SimpleNamespace(
            action_space=SimpleNamespace(shape=(1,), sample=lambda: np.zeros(1)),
            observation_space=SimpleNamespace(shape=shape),
            _max_episode_steps=1,
            reset=lambda: np.zeros(shape, dtype=np.uint8),
            step=lambda action: (np.zeros(shape, dtype=np.uint8), 1, True, {"env_steps": 8}),
            set_rng_state=lambda state: None,
            get_rng_state=lambda: {},
            close=lambda: None,
        )

        class TinyCURL:
            def compute_logits(self, query, keys):
                return query @ keys.T

        class TinyReplay:
            batch_size, capacity, idx, full = 2, 3, 1, False

            def sample_cpc(self):
                return None

            def add(self, *args):
                pass

        class TinyAgent:
            CURL = TinyCURL()

            def update_cpc(self, *args):
                return "enabled"

        agent = TinyAgent()
        replay = TinyReplay()
        utils = SimpleNamespace(
            set_seed_everywhere=lambda seed: None,
            ReplayBuffer=lambda **kwargs: replay,
        )
        upstream = SimpleNamespace(CurlSacAgent=object)
        counters = {
            "step": 1,
            "env_steps": 8,
            "episode": 1,
            "updates": 0,
            "last_eval_step": 1,
            "episode_boundary": True,
            "env_rng": {},
            "eval_env_steps": 0,
            "elapsed_seconds": 0,
        }
        if private_state is not None:
            counters["contrastive_permutation_rng"] = private_state

        def imports(name):
            return utils if name == "utils" else upstream

        def saved(path, saved_agent, saved_replay, saved_counters, saved_config):
            path.write_bytes(b"test-checkpoint")
            if captured is not None:
                captured.update(
                    counters=copy.deepcopy(saved_counters),
                    config=copy.deepcopy(saved_config),
                    agent=saved_agent,
                    replay=saved_replay,
                )

        with (
            patch.object(train_reference, "parse_args", return_value=args),
            patch("subprocess.check_output", return_value=train_reference.UPSTREAM_COMMIT + "\n"),
            patch.object(train_reference.importlib, "import_module", side_effect=imports),
            patch.dict(sys.modules, {"env_adapter": SimpleNamespace(make_env=lambda *args: env)}),
            patch.object(sys, "path", list(sys.path)),
            patch.object(train_reference, "build_agent", return_value=agent),
            patch.object(
                train_reference,
                "load_checkpoint",
                return_value=(
                    counters,
                    {"seed": checkpoint_seed, "num_train_steps": 1, "arm": previous_arm or arm},
                ),
            ),
            patch.object(
                train_reference, "evaluate", return_value=[{"return": 1, "eval_env_steps": 8}]
            ),
            patch.object(train_reference, "save_checkpoint", side_effect=saved),
            patch.object(train_reference.signal, "signal"),
        ):
            train_reference.main()

    def test_new_output_resume_writes_extended_config_and_logs_parent(self):
        # Putting config persistence only in the fresh-run branch causes this regression.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            for name in ("curl_sac.py", "utils.py", "encoder.py", "train.py"):
                (source / name).write_text("# immutable source fixture\n")
            output = base / "new-run"
            self.run_resume(output, source)
            self.assertTrue((output / "config.json").exists(), "Resume output needs its config")
            config = json.loads((output / "config.json").read_text())
            self.assertEqual((config["num_train_steps"], config["seed"]), (2, 123))
            resume = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual((resume["type"], resume["config"]["num_train_steps"]), ("resume", 2))
            self.assertEqual(resume["resume_path"], str(source / "parent.pt"))

    def test_existing_output_resume_keeps_original_config_and_rejects_changed_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("curl_sac.py", "utils.py", "encoder.py", "train.py"):
                (base / name).write_text("# immutable source fixture\n")
            output = base / "run"
            output.mkdir()
            original = '{"num_train_steps": 1, "seed": 123}\n'
            (output / "config.json").write_text(original)
            self.run_resume(output, base)
            self.assertEqual((output / "config.json").read_text(), original)
            resume = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(resume["config"]["num_train_steps"], 2)
            rejected = base / "rejected"
            with self.assertRaisesRegex(ValueError, "scientific inputs.*seed"):
                self.run_resume(rejected, base, checkpoint_seed=456)
            self.assertFalse((rejected / "config.json").exists())

    def test_shuffled_resume_requires_private_state_and_preserves_arm_rejection(self):
        from contrastive_control import ShuffledKeys

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("curl_sac.py", "utils.py", "encoder.py", "train.py"):
                (base / name).write_text("# frozen\n")
            for state, previous_arm, expected in (
                (None, None, "private permutation"),
                (None, "curl", "scientific inputs.*arm"),
            ):
                with (
                    self.subTest(previous_arm=previous_arm),
                    self.assertRaisesRegex(ValueError, expected),
                ):
                    self.run_resume(
                        base / (previous_arm or "missing"),
                        base,
                        arm="shuffled_curl",
                        private_state=state,
                        previous_arm=previous_arm,
                    )
            dummy = SimpleNamespace(CURL=SimpleNamespace(compute_logits=lambda q, p: q @ p.T))
            replay = SimpleNamespace(batch_size=2, sample_cpc=lambda: None)
            control = ShuffledKeys(dummy, replay, 123)
            control.rng.permutation(128)
            state = control.state_dict()
            captured = {}
            self.run_resume(
                base / "valid", base, arm="shuffled_curl", private_state=state, captured=captured
            )
            self.assertEqual(captured["counters"]["contrastive_permutation_rng"], state)
            self.assertEqual(captured["config"]["contrastive_control"]["seed"], 1000123)

    def test_main_reference_arms_have_no_shuffling_hooks_or_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("curl_sac.py", "utils.py", "encoder.py", "train.py"):
                (base / name).write_text("# frozen\n")
            for arm in ("curl", "no_curl"):
                with self.subTest(arm=arm):
                    captured = {}
                    self.run_resume(base / arm, base, arm=arm, captured=captured)
                    self.assertNotIn("contrastive_control", captured["config"])
                    self.assertNotIn("contrastive_permutation_rng", captured["counters"])
                    self.assertNotIn("compute_logits", captured["agent"].CURL.__dict__)
                    self.assertNotIn("sample_cpc", captured["replay"].__dict__)
                    self.assertEqual(
                        captured["agent"].update_cpc(), "enabled" if arm == "curl" else None
                    )


if __name__ == "__main__":
    unittest.main()
