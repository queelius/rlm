import pytest


def test_fixed_commands_and_zero_update_resolution(tmp_path):
    import launch_sufficiency_rl as launch

    commands = launch.commands(tmp_path)
    assert len(commands) == 2
    assert commands[0][commands[0].index("--hours") + 1] == "0.75"
    assert commands[1][commands[1].index("--hours") + 1] == "0.5"
    assert commands[1][commands[1].index("--rl-output") + 1].endswith("sufficiency-rl-001")
    summary = {"actual_optimizer_steps": 0, "committed_sampled_blocks": 4, "failure": None}
    assert launch.endpoint_action(summary) == "skip_zero_update_control_and_readout"
    summary["actual_optimizer_steps"] = 1
    assert launch.endpoint_action(summary) == "run_matched_control"
    summary["failure"] = "transport unknown"
    with pytest.raises(ValueError):
        launch.endpoint_action(summary)
