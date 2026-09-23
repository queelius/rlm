import copy

import pytest
import textcraft_lr_replay as run
import torch


def test_lr_change_preserves_adam_moments_and_rejects_wrong_ancestor():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([p], lr=2e-5, weight_decay=0)
    p.grad = torch.tensor([0.4])
    optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    run.change_lr(optimizer)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert torch.equal(optimizer.state[p]["exp_avg"], before["state"][0]["exp_avg"])
    assert torch.equal(optimizer.state[p]["exp_avg_sq"], before["state"][0]["exp_avg_sq"])
    p.grad = torch.tensor([0.2])
    optimizer.step()
    assert int(optimizer.state[p]["step"]) == 2
    with pytest.raises(ValueError):
        run.change_lr(optimizer)


def test_actual_source_inventory_is_complete_same_sampled_cp1():
    identity = run.source_identity()
    assert identity["episodes"] == 32
    assert identity["calls"] == 909
    assert identity["sample_cursor"] == 2
    assert identity["source_update"]["nonzero_action_calls"] == 819


def test_endpoint_rejects_failed_or_uncommitted_replay():
    row = {"complete": True, "failure": None, "stopped": False, "actual_optimizer_steps": 2}
    run.valid_terminal(row)
    for changed in ({"complete": False}, {"failure": "bad"}, {"actual_optimizer_steps": 1}):
        with pytest.raises(ValueError):
            run.valid_terminal({**row, **changed})
