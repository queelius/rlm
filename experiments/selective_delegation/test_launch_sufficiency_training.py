import json

import pytest


def test_actual_qampari_summary_shape_requires_all_calls_and_observed_episodes(tmp_path):
    import launch_sufficiency_training as launcher
    import qampari_probe as qampari

    # Generate the actual SUMMARY schema rather than a hand-invented 'groups' shape.
    plan = {
        "jobs": [
            {
                "condition": condition,
                "episode_id": f"{condition}-{i}",
                "case_id": str(i),
                "repeat": 0,
            }
            for condition in qampari.CONDITIONS
            for i in range(32)
        ],
        "planned_calls": 160,
        "planned_episodes": 64,
    }
    (tmp_path / "PLAN.json").write_text(json.dumps(plan))
    summary = qampari.summarize(tmp_path, plan)
    summary["recorded_episodes"] = 64
    summary["physical_cost"]["calls"] = 160
    for group in summary["conditions"].values():
        group.update(observed=32, missing_or_unavailable=0)
    launcher.require_predecessor_complete(summary)
    summary["conditions"]["map50"]["observed"] = 31
    with pytest.raises(RuntimeError, match="incomplete"):
        launcher.require_predecessor_complete(summary)


def test_fixed_two_arm_commands_and_no_checkpoint_substitution(tmp_path, monkeypatch):
    import launch_sufficiency_training as launcher

    for arm in launcher.ARMS:
        train, evaluate = launcher.commands(tmp_path, arm)
        assert train[train.index("--prepared") + 1].endswith("/" + arm)
        assert evaluate[evaluate.index("--adapter") + 1].endswith("checkpoint-0032")
        assert evaluate[evaluate.index("--arm") + 1] == arm
        assert evaluate[evaluate.index("--baseline") + 1].endswith("sufficiency-001")
    monkeypatch.setattr(launcher, "released", lambda _: True)
    (tmp_path / "TERMINAL-fixture.json").write_text(json.dumps({"step": 31, "complete": False}))
    with pytest.raises(RuntimeError, match="exact committed step32"):
        launcher.require_endpoint(tmp_path)
    (tmp_path / "TERMINAL-fixture.json").write_text(json.dumps({"step": 32, "complete": True}))
    ckpt = tmp_path / "checkpoint-0032"
    ckpt.mkdir()
    for name in ("COMMIT.json", "STATE.json"):
        (ckpt / name).write_text(json.dumps({"step": 32}))
    launcher.require_endpoint(tmp_path)
