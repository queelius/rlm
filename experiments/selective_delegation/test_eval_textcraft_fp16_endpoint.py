import copy

import pytest


def test_only_one_new_committed_finite_step_two_endpoint_is_eligible():
    import eval_textcraft_fp16_endpoint as e

    terminal = dict(
        complete=True,
        failure=None,
        stopped=False,
        endpoint_usable=True,
        actual_optimizer_steps=2,
        new_optimizer_steps=1,
        cumulative_optimizer_steps=2,
        committed_optimizer_steps=2,
        committed_sampled_batches=2,
        new_sampled_batches=1,
    )
    summary = dict(terminal)
    state = dict(
        step=2,
        sample_cursor=2,
        update=dict(
            optimizer_called=True,
            nonzero_action_calls=10,
            gradient_norm=1.0,
            adapter_l2_delta=0.01,
            objective=-1.0,
            train_eval_replay_max_abs=0.01,
            train_eval_replay_mean_abs=0.001,
            train_eval_replay_token_count=100,
        ),
    )
    e.validate_terminal(terminal, summary, state, dict(step=2), [2])
    for key, value in [
        ("new_optimizer_steps", 0),
        ("complete", False),
        ("failure", "error"),
        ("committed_optimizer_steps", 1),
    ]:
        wrong = dict(terminal, **{key: value})
        with pytest.raises(ValueError):
            e.validate_terminal(wrong, summary, state, dict(step=2), [2])
    wrong = copy.deepcopy(state)
    wrong["update"]["gradient_norm"] = float("nan")
    with pytest.raises(ValueError):
        e.validate_terminal(terminal, summary, wrong, dict(step=2), [2])
    with pytest.raises(ValueError):
        e.validate_terminal(terminal, summary, state, dict(step=2), [1])
