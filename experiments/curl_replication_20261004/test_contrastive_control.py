"""Small pinned-agent CPU checks for the deliberately wrong key correspondence."""

import copy
import os
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

try:
    from contrastive_control import ShuffledKeys, derangement
except ModuleNotFoundError:
    ShuffledKeys = derangement = None


class Logger:
    def __init__(self):
        self.values = {}

    def log(self, key, value, step):
        self.values[key] = float(value.detach()) if torch.is_tensor(value) else float(value)

    def __getattr__(self, name):
        return lambda *args: None


class DerangementTest(unittest.TestCase):
    def test_rejects_singleton_and_retries_fixed_points_with_private_rng(self):
        self.assertIsNotNone(derangement, "Wrong-matching controller must exist")
        with self.assertRaises(ValueError):
            derangement(np.random.default_rng(1), 1)
        draws = iter([np.arange(4), np.array([1, 0, 3, 2])])
        result = derangement(SimpleNamespace(permutation=lambda size: next(draws)), 4)
        np.testing.assert_array_equal(result, [1, 0, 3, 2])
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        before = checkpoint.capture_rng()
        generator = np.random.Generator(np.random.PCG64(1000123))
        for _ in range(50):
            permutation = derangement(generator, 128)
            self.assertEqual(sorted(permutation.tolist()), list(range(128)))
            self.assertFalse(np.any(permutation == np.arange(128)))
        after = checkpoint.capture_rng()
        self.assertEqual(before["python"], after["python"])
        np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
        self.assertEqual(before["numpy"][2:], after["numpy"][2:])
        torch.testing.assert_close(before["torch"], after["torch"], rtol=0, atol=0)

    def test_duplicate_replay_records_survive_a_positional_derangement(self):
        self.assertIsNotNone(ShuffledKeys)
        consumed = []
        replay = SimpleNamespace(capacity=1, idx=1, full=False, batch_size=4)

        def sample():
            consumed.append(np.random.randint(0, 1, size=4))
            np.random.randint(0, 17, size=3)  # Stand in for the unchanged crop RNG draws.
            return "original_batch"

        replay.sample_cpc = sample
        agent = SimpleNamespace(CURL=SimpleNamespace(compute_logits=lambda q, p: q @ p.T))
        np.random.seed(19)
        initialization_rng = checkpoint.capture_rng()
        control = ShuffledKeys(agent, replay, 123)
        np.testing.assert_array_equal(initialization_rng["numpy"][1], np.random.get_state()[1])
        self.assertEqual(initialization_rng["numpy"][2:], np.random.get_state()[2:])
        self.assertEqual(initialization_rng["python"], random.getstate())
        torch.testing.assert_close(
            initialization_rng["torch"], torch.get_rng_state(), rtol=0, atol=0
        )
        before = np.random.get_state()
        self.assertEqual(replay.sample_cpc(), "original_batch")
        actual = np.random.get_state()
        np.random.set_state(before)
        sample()
        expected = np.random.get_state()
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2:], expected[2:])
        agent.CURL.compute_logits(torch.eye(4), torch.eye(4))
        self.assertEqual(control.diagnostics["fixed_positions"], 0)
        self.assertEqual(control.diagnostics["duplicate_indices"], 3)
        self.assertEqual(control.diagnostics["residual_same_record_matches"], 4)
        np.testing.assert_array_equal(control.replay_indices, consumed[0])


@unittest.skipUnless(
    os.environ.get("CURL_SOURCE"), "Set CURL_SOURCE for actual pinned-agent checks"
)
class UpstreamControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.environ["CURL_SOURCE"])
        import utils
        from curl_sac import CurlSacAgent

        cls.utils, cls.Agent = utils, CurlSacAgent
        torch.set_num_threads(1)

    def make_agent(self):
        config = {
            "hidden_dim": 16,
            "encoder_feature_dim": 8,
            "num_layers": 2,
            "num_filters": 2,
            "log_interval": 100,
        }
        return train_reference.build_agent(self.Agent, config, (1,), torch.device("cpu"))

    def replay(self, batch_size=4):
        replay = self.utils.ReplayBuffer((9, 100, 100), (1,), 8, batch_size, torch.device("cpu"))
        pixels = np.arange(9 * 100 * 100).reshape(9, 100, 100)
        for i in range(4):
            replay.add(
                ((pixels + i * 47) % 256).astype(np.uint8),
                [i / 10],
                i,
                ((pixels + i * 47 + 11) % 256).astype(np.uint8),
                False,
            )
        return replay

    def assert_tree_equal(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right, strict=True):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def learning_state(self, agent):
        return copy.deepcopy(
            {
                "modules": {
                    name: getattr(agent, name).state_dict() for name in checkpoint.MODULE_NAMES
                },
                "optimizers": {
                    name: getattr(agent, name).state_dict() for name in checkpoint.OPTIMIZER_NAMES
                },
                "alpha": agent.log_alpha,
            }
        )

    def test_reference_update_matches_unwrapped_authors_agent_exactly(self):
        for arm in ("curl", "no_curl"):
            with self.subTest(arm=arm):
                reference = self.make_agent()
                adapted = copy.deepcopy(reference)
                original_batch, adapted_batch = self.replay(), self.replay()
                if arm == "no_curl":
                    reference.update_cpc = lambda *args: None
                config = {"arm": arm, "seed": 123}
                self.assertIsNone(train_reference.configure_arm(adapted, adapted_batch, config))
                self.assertEqual(config, {"arm": arm, "seed": 123})
                rng = checkpoint.capture_rng()
                reference.update(original_batch, Logger(), 1000)
                expected_random = (random.random(), np.random.random(), torch.rand(1))
                checkpoint.restore_rng(rng)
                adapted.update(adapted_batch, Logger(), 1000)
                self.assert_tree_equal(self.learning_state(reference), self.learning_state(adapted))
                actual_random = (random.random(), np.random.random(), torch.rand(1))
                self.assert_tree_equal(expected_random, actual_random)
                self.assertNotIn("compute_logits", adapted.CURL.__dict__)
                self.assertNotIn("sample_cpc", adapted_batch.__dict__)

    def test_known_swap_uses_original_loss_labels_and_both_optimizer_steps(self):
        self.assertIsNotNone(ShuffledKeys)
        agent = self.make_agent()
        oracle = copy.deepcopy(agent)
        replay, oracle_replay = self.replay(2), self.replay(2)
        control = train_reference.configure_arm(
            agent, replay, {"arm": "shuffled_curl", "seed": 123}
        )
        original_control_logits = control._logits
        losses = {}

        def observe_logits(query, permuted_keys):
            shuffled = original_control_logits(query, permuted_keys)
            matched = original_control_logits(query, permuted_keys[[1, 0]])
            labels = torch.arange(2)
            losses.update(
                shuffled=float(torch.nn.functional.cross_entropy(shuffled, labels).detach()),
                matched=float(torch.nn.functional.cross_entropy(matched, labels).detach()),
            )
            return shuffled

        control._logits = observe_logits
        original_logits = oracle.CURL.compute_logits
        oracle.CURL.compute_logits = lambda q, p: original_logits(q, p[[1, 0]])
        np.random.seed(0)  # The first pinned replay draw is distinct indices [0, 3].
        rng = checkpoint.capture_rng()
        oracle_logger = Logger()
        oracle.update(oracle_replay, oracle_logger, 1000)
        expected_random = (random.random(), np.random.random(), torch.rand(1))
        checkpoint.restore_rng(rng)
        logger = Logger()
        with (
            patch.object(
                agent.encoder_optimizer, "step", wraps=agent.encoder_optimizer.step
            ) as enc,
            patch.object(agent.cpc_optimizer, "step", wraps=agent.cpc_optimizer.step) as cpc,
            patch.object(
                agent.cross_entropy_loss, "forward", wraps=agent.cross_entropy_loss.forward
            ) as loss,
        ):
            agent.update(replay, logger, 1000)
        self.assertEqual(enc.call_count, 1)
        self.assertEqual(cpc.call_count, 1)
        torch.testing.assert_close(loss.call_args.args[1], torch.arange(2), rtol=0, atol=0)
        np.testing.assert_array_equal(control.permutation, [1, 0])
        np.testing.assert_array_equal(control.replay_indices, [0, 3])
        self.assertEqual(control.diagnostics["duplicate_indices"], 0)
        self.assertEqual(control.diagnostics["residual_same_record_matches"], 0)
        self.assertEqual(logger.values["train/curl_loss"], oracle_logger.values["train/curl_loss"])
        self.assertEqual(logger.values["train/curl_loss"], losses["shuffled"])
        self.assertNotEqual(losses["matched"], losses["shuffled"])
        self.assert_tree_equal(self.learning_state(agent), self.learning_state(oracle))
        self.assert_tree_equal(
            expected_random, (random.random(), np.random.random(), torch.rand(1))
        )
        encoder_weight = agent.critic.encoder.convs[0].weight
        self.assertTrue(
            any(
                p is encoder_weight
                for group in agent.encoder_optimizer.param_groups
                for p in group["params"]
            )
        )
        self.assertTrue(
            any(
                p is encoder_weight
                for group in agent.cpc_optimizer.param_groups
                for p in group["params"]
            )
        )

    def test_full_checkpoint_restores_next_permutation_loss_parameters_and_optimizers(self):
        self.assertIsNotNone(ShuffledKeys)
        agent, replay = self.make_agent(), self.replay()
        control = train_reference.configure_arm(
            agent, replay, {"arm": "shuffled_curl", "seed": 456}
        )
        agent.update(replay, Logger(), 1000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            config = {"arm": "shuffled_curl", "seed": 456}
            counters = {"step": 1001, "contrastive_permutation_rng": control.state_dict()}
            checkpoint.save_checkpoint(path, agent, replay, counters, config)
            expected_logger = Logger()
            agent.update(replay, expected_logger, 1100)
            expected_permutation = control.permutation.copy()
            expected_private_rng = control.state_dict()
            expected_state = self.learning_state(agent)
            expected_random = (random.random(), np.random.random(), torch.rand(1))
            restored, restored_replay = self.make_agent(), self.replay()
            restored_control = train_reference.configure_arm(
                restored, restored_replay, {"arm": "shuffled_curl", "seed": 456}
            )
            saved, previous = checkpoint.load_checkpoint(path, restored, restored_replay)
            restored_control.load_state_dict(saved["contrastive_permutation_rng"])
            logger = Logger()
            restored.update(restored_replay, logger, 1100)
            self.assertEqual(previous, config)
            np.testing.assert_array_equal(restored_control.permutation, expected_permutation)
            self.assertEqual(restored_control.state_dict(), expected_private_rng)
            self.assertEqual(
                logger.values["train/curl_loss"], expected_logger.values["train/curl_loss"]
            )
            self.assert_tree_equal(self.learning_state(restored), expected_state)
            self.assert_tree_equal(
                expected_random, (random.random(), np.random.random(), torch.rand(1))
            )
            self.assertIs(restored.CURL.encoder, restored.critic.encoder)
            self.assertIs(
                restored.actor.encoder.convs[0].weight, restored.critic.encoder.convs[0].weight
            )
            with self.assertRaises(ValueError):
                restored_control.load_state_dict(None)


if __name__ == "__main__":
    unittest.main()
