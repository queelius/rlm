import copy

import pytest


def test_only_exact_failed_one_step_boundary_can_be_amended():
    import textcraft_stopped_amendment as a

    terminal = dict(
        complete=False,
        endpoint_usable=False,
        failure="ValueError: generation/replay discrepancy exceeded declared tolerance",
        actual_optimizer_steps=1,
        committed_optimizer_steps=1,
        committed_sampled_batches=1,
    )
    summary = {k: v for k, v in terminal.items() if k != "endpoint_usable"}
    state, commit = dict(step=1, sample_cursor=1), dict(step=1)
    a.validate_stopped(terminal, summary, state, commit)
    for key, value in [("complete", True), ("failure", "other"), ("actual_optimizer_steps", 2)]:
        wrong = copy.deepcopy(terminal)
        wrong[key] = value
        with pytest.raises(ValueError):
            a.validate_stopped(wrong, summary, state, commit)


def test_original_success_only_policy_remains_strict():
    import train_textcraft_matched_sft as control

    with pytest.raises(ValueError):
        control.endpoint_steps(dict(complete=False, endpoint_usable=False), dict(complete=False))
