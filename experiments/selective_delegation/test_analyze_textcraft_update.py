import analyze_textcraft_update as a
import pytest


def test_known_token_logps_keep_signed_movement_and_reject_missing_after():
    calls = {"x": {"output_token_ids": [1, 2]}, "y": {"output_token_ids": [3]}}
    before = {"x": [-2.0, -1.0], "y": [-3.0]}
    after = {"x": [-1.5, -1.5], "y": [-2.0]}
    delta = a.likelihood_rows(calls, before, after, {"x": 1 / 3, "y": -1.0})
    assert delta["x"]["sequence_logp_delta"] == 0
    assert delta["y"]["advantage_weighted_delta"] == -1
    assert delta["y"]["improved_in_advantage_direction"] is False
    with pytest.raises(ValueError, match="coverage"):
        a.likelihood_rows(calls, before, {"x": after["x"]}, {"x": 1 / 3, "y": -1.0})
    with pytest.raises(ValueError, match="length"):
        a.likelihood_rows(calls, before, {"x": [-1.0], "y": [-2.0]}, {"x": 1 / 3, "y": -1.0})


def test_actual063_credit_inventory_and_cpu_adam_step(tmp_path):
    import torch

    root = a.Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
    source = root / "textcraft-train-readiness-001"
    episodes = [a.read(p) for p in (source / "episodes").glob("*.json")]
    calls = {p.stem: a.read(p) for p in (source / "calls").glob("*.json")}
    active = [c for c in a.census.loss_math.batch_credits(episodes, calls) if c.advantage]
    assert len(active) == 388
    assert sum(len(calls[c.call_id]["output_token_ids"]) for c in active) == 12074
    path = tmp_path / "optimizer.pt"
    torch.save({"state": {0: {"step": torch.tensor(2.0), "exp_avg": torch.zeros(3)}}}, path)
    assert a.adam_steps(path) == [2]


def test_uncommitted_or_missing_after_boundary_is_explicitly_incomplete(tmp_path):
    import json

    (tmp_path / "PLAN.json").write_text(
        json.dumps({"schema": "textcraft-terminal-rloo-two-update-v1"})
    )
    result = a.analyze(tmp_path, 1)
    assert result["status"] == "incomplete"
    assert len(result["missing"]) == 4
    assert "credited" not in result
