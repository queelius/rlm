"""The queued control must use its actual predecessor and PEFT environment."""

import launch_training_direct_control as launcher
from launch_rl_diagnostics import TRAIN_PYTHON, released


def test_launch_contract_uses_shared_release_and_training_interpreter(tmp_path):
    assert launcher.PREDECESSOR == "alfworld-closed-loop-001"
    assert launcher.released is released
    assert launcher.command(tmp_path)[0] == TRAIN_PYTHON
    assert str(tmp_path / "training-direct-control-002") in launcher.command(tmp_path)


def test_predecessor_requires_all_original_episodes():
    assert not launcher.complete_predecessor({"groups": {}})
    assert not launcher.complete_predecessor(
        {"groups": {"flat": {"observed": 16}, "manager_worker": {"observed": 15}}}
    )
    assert launcher.complete_predecessor(
        {"groups": {"flat": {"observed": 16}, "manager_worker": {"observed": 16}}}
    )
