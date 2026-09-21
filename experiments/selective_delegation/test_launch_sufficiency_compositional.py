from pathlib import Path

import launch_sufficiency_compositional as launcher


def test_044_completion_and_fixed_readout_command():
    summary = {
        "planned_episodes": 32,
        "recorded_episodes": 32,
        "groups": {p: {"observed": 16, "missing_or_unknown": 0} for p in ("flat", "recursive")},
        "unresolved_starts": [],
    }
    assert launcher.predecessor_complete(summary)
    assert not launcher.predecessor_complete({**summary, "recorded_episodes": 31})
    assert not launcher.predecessor_complete({**summary, "unresolved_starts": ["x"]})
    command = launcher.command(Path("/runs"))
    assert (
        command[1]
        == "/runs/source-046-sufficiency-compositional-readout/eval_sufficiency_compositional.py"
    )
    assert command[command.index("--cases") + 1].endswith("compositional-inputs-001/cases.jsonl")
    assert command[command.index("--warm-adapter") + 1].endswith("joint-001/checkpoint-0032")
