"""Focused checkpoint regression; CPU only, no simulator required."""

import hashlib
import json
import os
import random
import resource
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import checkpoint
import numpy as np
import torch


def small_agent():
    agent = SimpleNamespace(training=True)
    for name in checkpoint.MODULE_NAMES:
        setattr(agent, name, torch.nn.Linear(3, 3))
    agent.log_alpha = torch.tensor(-2.0, requires_grad=True)
    for name, module in zip(checkpoint.OPTIMIZER_NAMES[:4], checkpoint.MODULE_NAMES, strict=True):
        setattr(agent, name, torch.optim.Adam(getattr(agent, module).parameters()))
    agent.log_alpha_optimizer = torch.optim.Adam([agent.log_alpha])
    agent.train = lambda training=True: setattr(agent, "training", training)
    return agent


class CheckpointTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("CURL_LARGE_CHECKPOINT_TEST_DIR"), "Opt in to the >4 GiB CPU/disk regression"
    )
    def test_numpy_array_larger_than_four_gib_round_trips_with_atomic_save(self):
        # Default pickle protocol 2 cannot encode an individual replay array past 4 GiB.
        torch.set_num_threads(1)
        start = time.monotonic()
        obs = np.full((1, 2**32 + 1), 17, dtype=np.uint8)
        obs[0, 0], obs[0, 2**31], obs[0, -1] = 11, 23, 37
        replay = SimpleNamespace(
            capacity=1, batch_size=1, image_size=84, idx=0, full=True, last_save=0
        )
        for name in checkpoint.REPLAY_ARRAY_NAMES:
            setattr(replay, name, obs if name == "obses" else np.ones((1, 1), dtype=np.float32))
        agent = small_agent()
        with tempfile.TemporaryDirectory(
            dir=os.environ["CURL_LARGE_CHECKPOINT_TEST_DIR"]
        ) as directory:
            path = Path(directory) / "latest.pt"
            with self.assertRaisesRegex(OverflowError, "4 GiB.*protocol 4"):
                torch.save({"obs": obs}, Path(directory) / "protocol2.pt", pickle_protocol=2)
            print(
                json.dumps({"protocol2": "observed_OverflowError", "array_bytes": obs.nbytes}),
                flush=True,
            )
            checkpoint.save_checkpoint(path, agent, replay, {"step": 51125}, {"seed": 123})
            file_bytes = path.stat().st_size
            target = SimpleNamespace(
                capacity=1, batch_size=1, image_size=84, idx=0, full=False, last_save=0
            )
            for name in checkpoint.REPLAY_ARRAY_NAMES:
                setattr(target, name, np.empty_like(getattr(replay, name)))
            state, config = checkpoint.load_checkpoint(path, small_agent(), target)
            self.assertEqual(state, {"step": 51125})
            self.assertEqual(config, {"seed": 123})
            self.assertTrue(target.full)
            self.assertEqual(target.obses.nbytes, 2**32 + 1)
            self.assertEqual(
                hashlib.sha256(memoryview(target.obses)).digest(),
                hashlib.sha256(memoryview(obs)).digest(),
            )
            for name in checkpoint.REPLAY_ARRAY_NAMES[1:]:
                np.testing.assert_array_equal(getattr(target, name), getattr(replay, name))
            self.assertFalse(list(Path(directory).glob(".checkpoint-*")))
        print(
            json.dumps(
                {
                    "large_roundtrip": "passed",
                    "array_bytes": obs.nbytes,
                    "checkpoint_bytes": file_bytes,
                    "seconds": time.monotonic() - start,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            ),
            flush=True,
        )

    def test_restore_reproduces_next_update_and_wrapped_replay(self):
        # Missing optimizer moments, alpha, replay wrap state or RNG breaks continuation.
        torch.set_num_threads(1)

        make_agent = small_agent

        def update(agent):
            for module, optimizer in zip(
                checkpoint.MODULE_NAMES, checkpoint.OPTIMIZER_NAMES[:4], strict=True
            ):
                opt = getattr(agent, optimizer)
                opt.zero_grad()
                getattr(agent, module)(torch.randn(2, 3)).square().mean().backward()
                opt.step()
            agent.log_alpha_optimizer.zero_grad()
            (agent.log_alpha.exp() * torch.rand(())).backward()
            agent.log_alpha_optimizer.step()
            return random.random(), np.random.random(), torch.rand(2)

        replay = SimpleNamespace(
            capacity=3, batch_size=2, image_size=84, idx=1, full=True, last_save=2
        )
        for name in checkpoint.REPLAY_ARRAY_NAMES:
            setattr(replay, name, np.arange(6, dtype=np.float32).reshape(3, 2))
        agent = make_agent()
        update(agent)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            checkpoint.save_checkpoint(
                path, agent, replay, {"step": 7, "episode": 2}, {"seed": 123}
            )
            expected_random = update(agent)
            expected = {
                name: {
                    key: value.clone() for key, value in getattr(agent, name).state_dict().items()
                }
                for name in checkpoint.MODULE_NAMES
            }
            expected_alpha = agent.log_alpha.detach().clone()
            restored = make_agent()
            restored_replay = SimpleNamespace(
                capacity=3, batch_size=2, image_size=84, idx=0, full=False, last_save=0
            )
            for name in checkpoint.REPLAY_ARRAY_NAMES:
                setattr(restored_replay, name, np.zeros((3, 2), dtype=np.float32))
            state, config = checkpoint.load_checkpoint(path, restored, restored_replay)
            actual_random = update(restored)
        self.assertEqual(state, {"step": 7, "episode": 2})
        self.assertEqual(config, {"seed": 123})
        self.assertEqual(
            (restored_replay.idx, restored_replay.full, restored_replay.last_save), (1, True, 2)
        )
        self.assertEqual(actual_random[:2], expected_random[:2])
        torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)
        torch.testing.assert_close(restored.log_alpha, expected_alpha, rtol=0, atol=0)
        for name in checkpoint.MODULE_NAMES:
            for key, value in getattr(restored, name).state_dict().items():
                torch.testing.assert_close(value, expected[name][key], rtol=0, atol=0)
        for name in checkpoint.REPLAY_ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(restored_replay, name), getattr(replay, name))

    @unittest.skipUnless(os.environ.get("CURL_SOURCE"), "Set CURL_SOURCE for the pinned agent test")
    def test_authors_pixel_agent_next_fixed_batch_update_and_parameter_ties(self):
        # A weights-only saver or restored objects instead of tensors breaks this check.
        sys.path.insert(0, os.environ["CURL_SOURCE"])
        import utils
        from curl_sac import CurlSacAgent
        from train_reference import build_agent

        torch.set_num_threads(1)
        device = torch.device("cpu")

        def make_agent():
            config = {
                "hidden_dim": 16,
                "encoder_feature_dim": 8,
                "num_layers": 2,
                "num_filters": 2,
                "device": "cpu",
                "obs_shape": "ignored",
                "action_shape": "ignored",
                "upstream_commit": "metadata",
            }
            return build_agent(CurlSacAgent, config, (1,), device)

        class Logger:
            def __getattr__(self, name):
                return lambda *args: None

        class Batch:
            def sample_cpc(self):
                return batch

        pixels = torch.arange(2 * 9 * 84 * 84, dtype=torch.float32).reshape(2, 9, 84, 84) % 256
        batch = (
            pixels,
            torch.zeros(2, 1),
            torch.ones(2, 1),
            pixels.flip(-1),
            torch.ones(2, 1),
            {"obs_anchor": pixels, "obs_pos": pixels.flip(-2)},
        )
        agent = make_agent()
        agent.update(Batch(), Logger(), 1000)
        replay = utils.ReplayBuffer((9, 100, 100), (1,), 3, 2, device)
        for index in range(2):
            replay.add(
                np.full((9, 100, 100), index, dtype=np.uint8),
                [0],
                1,
                np.full((9, 100, 100), index + 1, dtype=np.uint8),
                False,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            checkpoint.save_checkpoint(path, agent, replay, {"step": 1001}, {})
            agent.update(Batch(), Logger(), 1002)
            expected = {
                name: {
                    key: value.clone() for key, value in getattr(agent, name).state_dict().items()
                }
                for name in checkpoint.MODULE_NAMES
            }
            alpha = agent.log_alpha.detach().clone()
            restored = make_agent()
            restored_replay = utils.ReplayBuffer((9, 100, 100), (1,), 3, 2, device)
            checkpoint.load_checkpoint(path, restored, restored_replay)
            self.assertIs(
                restored.actor.encoder.convs[0].weight, restored.critic.encoder.convs[0].weight
            )
            self.assertIs(restored.CURL.encoder, restored.critic.encoder)
            self.assertIs(restored.CURL.encoder_target, restored.critic_target.encoder)
            restored.update(Batch(), Logger(), 1002)
            torch.testing.assert_close(restored.log_alpha, alpha, rtol=0, atol=0)
            for name in checkpoint.MODULE_NAMES:
                for key, value in getattr(restored, name).state_dict().items():
                    torch.testing.assert_close(value, expected[name][key], rtol=0, atol=0)
            np.testing.assert_array_equal(restored_replay.obses[:2], replay.obses[:2])


if __name__ == "__main__":
    unittest.main()
