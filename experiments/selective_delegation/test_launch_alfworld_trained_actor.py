import json

import launch_alfworld_trained_actor as launcher


def test_complete_requires_every_trained_actor_slot():
    group = {"observed": 24, "missing_or_unobserved": 0}
    summary = {
        "planned_episodes": 48,
        "recorded_episodes": 48,
        "groups": {"flat": dict(group), "manager_worker": dict(group)},
    }
    assert launcher.complete(summary)
    summary["groups"]["flat"]["missing_or_unobserved"] = 1
    assert not launcher.complete(summary)


def test_proposed_launcher_refuses_before_any_subprocess(tmp_path, monkeypatch):
    (tmp_path / launcher.DECISION).write_text(
        json.dumps(
            {
                "schema": "alfworld-trained-actor-decision-v1",
                "source": launcher.SOURCE,
                "status": "proposed",
                "sha256": {},
            }
        )
    )
    monkeypatch.setattr(launcher, "validate", lambda *_: None)
    try:
        launcher.main(type("Args", (), {"root": tmp_path, "validate_only": False})())
    except ValueError as exc:
        assert "acceptance" in str(exc)
    else:
        raise AssertionError("a proposed run reached a subprocess")
