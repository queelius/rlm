"""Focused evaluation isolation checks without rendering or GPU use."""

import random
import unittest

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


if __name__ == "__main__":
    unittest.main()
