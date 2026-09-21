import json

import pytest


def stopped_fixture(tmp_path):
    from compare_rl_stopped import probe

    ancestor = tmp_path / "original/checkpoint-0016"
    ancestor.mkdir(parents=True)
    (ancestor / "COMMIT.json").write_text('{"step":16}')
    root = tmp_path / "continuation"
    checkpoint = root / "checkpoint-0021"
    checkpoint.mkdir(parents=True)
    source = root / "source.py"
    source.write_text("pass")
    continuation = {
        "ancestor_checkpoint": str(ancestor),
        "ancestor_commit_sha256": probe.campaign.sha(ancestor / "COMMIT.json"),
    }
    plan = {
        "updates": 24,
        "continuation": continuation,
        "helper_contract": {},
        "parent_schedule": "consecutive",
        "case_ids_by_update": [[]] * 24,
        "dependencies": {str(source): probe.campaign.sha(source)},
    }
    (root / "PLAN.json").write_text(json.dumps(plan))
    state = {
        "step": 21,
        "cursor": 0,
        "next_update": 22,
        "continuation_identity": probe.runtime.digest(continuation),
    }
    (checkpoint / "STATE.json").write_text(json.dumps(state))
    (checkpoint / "adapter_model.safetensors").write_text("fixture weights")
    (checkpoint / "COMMIT.json").write_text(
        json.dumps(
            {"step": 21, "files": {p.name: probe.campaign.sha(p) for p in checkpoint.iterdir()}}
        )
    )
    (root / "OWNER-a.json").write_text("{}")
    (root / "TERMINAL-a.json").write_text(
        json.dumps(
            {
                "state": "admission_failed_no_update",
                "optimizer_steps": 21,
                "additional_optimizer_steps": 5,
                "failure": None,
                "unresolved_started_attempts": 0,
            }
        )
    )
    batch = root / "batch-0022"
    batch.mkdir()
    (batch / "BATCH.json").write_text(
        json.dumps(
            {
                "admitted": False,
                "qualifying_groups": 0,
                "episodes": 64,
                "groups": [{"rewards": [1, 1, 1, 1]}] * 16,
            }
        )
    )
    return checkpoint, ancestor


def test_stopped_identity_requires_last_committed21_and_exact_ancestry_hash(tmp_path):
    from compare_rl_stopped import stopped_identity

    checkpoint, ancestor = stopped_fixture(tmp_path)
    result = stopped_identity(checkpoint, ancestor)
    assert (
        result["selected_step"] == 21
        and result["selection_rule"] == "last committed before admission stop"
    )
    source = checkpoint.parent / "source.py"
    source.write_text("changed source")
    with pytest.raises(ValueError, match="source changed"):
        stopped_identity(checkpoint, ancestor)
    source.write_text("pass")
    (ancestor / "COMMIT.json").write_text('{"step":15}')
    with pytest.raises(ValueError, match="ancestor"):
        stopped_identity(checkpoint, ancestor)


def test_failed_terminal_and_changed_weights_are_rejected(tmp_path):
    from compare_rl_stopped import stopped_identity

    checkpoint, ancestor = stopped_fixture(tmp_path)
    terminal = checkpoint.parent / "TERMINAL-a.json"
    value = json.loads(terminal.read_text())
    value["failure"] = "inference error"
    terminal.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="admission stop"):
        stopped_identity(checkpoint, ancestor)
    value["failure"] = None
    terminal.write_text(json.dumps(value))
    (checkpoint / "adapter_model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="checkpoint hash"):
        stopped_identity(checkpoint, ancestor)


def test_stopped_comparison_keeps_native_missing_protocol_and_rl21_labels():
    from compare_rl_stopped import paired, score_episode

    valid = dict(em=1.0, f1=1.0, valid=True, observed=True, status="scored")
    missing = score_episode(None, [], {})
    protocol = score_episode(
        {"status": "invalid_helper"}, [{"role": "helper", "available": True}], {}
    )
    result = paired(
        {("p", 0): protocol, ("p", 1): valid},
        {("p", 0): valid, ("p", 1): missing},
        [{"id": "p"}],
        2,
        draws=10,
    )
    assert result["right"] == "rl21"
    assert result["wins"]["categories"] == {"protocol_involved": 1}
    assert result["losses"]["categories"] == {"unobserved_involved": 1}
    assert result["lower_bound_difference_not_effect_estimate"]
