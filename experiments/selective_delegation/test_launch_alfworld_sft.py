import json

import launch_alfworld_sft as launcher


def test_predecessor_resolution_accepts_only_a_resolved_zero_update_skip(tmp_path, monkeypatch):
    root = tmp_path
    (root / launcher.RESOLVED).write_text(
        json.dumps(
            {
                "chain_resolved": True,
                "scientific_readout": False,
                "output": launcher.PREDECESSOR,
                "reason": "zero_real_updates",
            }
        )
    )
    output = root / launcher.PREDECESSOR
    output.mkdir()
    (output / "SKIPPED.json").write_text("{}")
    (root / "SUFFICIENCY-ZERO-UPDATE-SKIP-001.json").write_text(
        json.dumps(
            {
                "reason": "zero_real_updates",
                "chain_resolved": True,
                "scientific_readout": False,
                "rl_endpoint": {"endpoint": "fixed", "state": {"step": 0, "sample_cursor": 4}},
            }
        )
    )
    rl = root / "sufficiency-rl-001"
    rl.mkdir()
    (rl / "SUMMARY.json").write_text(
        json.dumps(
            {"actual_optimizer_steps": 0, "committed_sampled_blocks": 4, "endpoint": "fixed"}
        )
    )
    monkeypatch.setattr(launcher, "released", lambda path: path == rl)
    assert launcher.predecessor_resolved(root)
    (root / launcher.RESOLVED).write_text(json.dumps({"chain_resolved": False}))
    assert not launcher.predecessor_resolved(root)


def test_proposed_decision_is_never_launchable(tmp_path, monkeypatch):
    decision = {
        "schema": "alfworld-action-sft-decision-v1",
        "source": launcher.SOURCE,
        "predecessor": launcher.RESOLVED,
        "status": "proposed",
        "sha256": {},
    }
    (tmp_path / launcher.DECISION).write_text(json.dumps(decision))
    monkeypatch.setattr(launcher, "validate", lambda *_: None)
    try:
        launcher.main(type("Args", (), {"root": tmp_path, "validate_only": False})())
    except ValueError as exc:
        assert "acceptance" in str(exc)
    else:
        raise AssertionError("an incomplete proposed decision was launchable")
