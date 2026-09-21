"""The fresh paired readout must retain two seeds for all three fixed arms."""

from pathlib import Path


def test_commands_preserve_two_seed_64_variant_three_arm_contract():
    import launch_sufficiency_canonical_readout as launcher

    commands = launcher.commands(Path("/study"))
    assert len(commands) == 3
    assert launcher.PLANNED_NEW_CALLS == 384
    assert commands[0][1] == "/study/source-024-sufficiency/sufficiency_probe.py"
    assert commands[0][commands[0].index("--cases") + 1].endswith(
        "sufficiency-canonical-inputs-003/cases.jsonl"
    )
    for arm, command in zip(("joint", "positive_only"), commands[1:], strict=True):
        assert command[1] == "/study/source-031-sufficiency-readout/eval_sufficiency.py"
        assert command[command.index("--arm") + 1] == arm
        assert command[command.index("--adapter") + 1].endswith(
            f"sufficiency-sft-{arm}-001/checkpoint-0032"
        )
