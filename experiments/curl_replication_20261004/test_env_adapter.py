"""Focused simulator contracts: wrong repeats, reseeding or frame order must fail."""

import numpy as np


def test_reward_sum_and_frame_stack_match_raw_simulator():
    from dm_control import suite
    from env_adapter import make_env

    env = make_env("cartpole", "swingup", 123, 8)
    raw = suite.load("cartpole", "swingup", task_kwargs={"random": 123},
                     visualize_reward=False)
    try:
        first = env.reset()
        raw.reset()
        assert first.shape == (9, 100, 100)
        assert first.dtype == np.uint8
        assert first.max() > first.min()
        np.testing.assert_array_equal(first[:3], first[3:6])
        action = np.array([0.25], dtype=np.float32)
        expected_reward = sum(raw.step(action).reward for _ in range(8))
        obs, reward, done, info = env.step(action)
        assert reward == expected_reward
        assert not done
        assert info["env_steps"] == 8
        np.testing.assert_array_equal(obs[:6], first[3:])
    finally:
        env.close()
        raw.close()


def test_episode_end_counts_simulator_steps_without_extra_transition():
    from env_adapter import make_env

    env = make_env("cartpole", "swingup", 123, 8)
    try:
        env.reset()
        total = 0
        for decision in range(125):
            _, _, done, info = env.step(np.zeros(1, dtype=np.float32))
            total += info["env_steps"]
            assert done == (decision == 124)
        assert total == 1000
        assert info["TimeLimit.truncated"]
        assert env._max_episode_steps == 125
    finally:
        env.close()


def test_task_seed_and_rng_restore_reproduce_next_reset():
    from env_adapter import make_env

    env = make_env("cartpole", "swingup", 1, 8)
    try:
        env.seed(10000)
        first = env.reset()
        env.step(np.zeros(1, dtype=np.float32))
        env.seed(10000)
        np.testing.assert_array_equal(first, env.reset())
        state = env.get_rng_state()
        next_reset = env.reset()
        action = env.action_space.sample()
        env.set_rng_state(state)
        np.testing.assert_array_equal(next_reset, env.reset())
        np.testing.assert_array_equal(action, env.action_space.sample())
    finally:
        env.close()
