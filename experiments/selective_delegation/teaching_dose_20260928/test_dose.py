"""Scientific seams of cumulative SFT dose, without loading the research model."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("dose_common.py")
    assert path.exists(), "matched dose implementation has not been written"
    spec = importlib.util.spec_from_file_location("dose_fixture", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_continuation_visits_every_row_twice_under_new_epoch_shuffles():
    d = module()
    batches = list(d.schedule(dict(step=23, epoch=1, cursor=0)))
    assert len(batches) == 46
    assert [len(indices) for _, _, indices in batches] == ([16] * 22 + [14]) * 2
    first = [i for epoch, _, ids in batches for i in ids if epoch == 1]
    second = [i for epoch, _, ids in batches for i in ids if epoch == 2]
    assert sorted(first) == sorted(second) == list(range(366))
    assert first != second
    assert first == d.recipe().epoch_order(366, 2026092208, 1)
    assert first != d.recipe().epoch_order(366, 2026092208, 0)


def test_cumulative_state_uses_rows_not_padded_effective_batch():
    d = module()
    state = dict(
        step=45,
        epoch=1,
        cursor=352,
        training_seconds=200.0,
        cumulative_microbatches=718,
        cumulative_target_tokens=17300,
    )
    result = d.advance(state, rows=14, tokens=340, seconds=2.0)
    assert {k: result[k] for k in ("step", "epoch", "cursor")} == dict(step=46, epoch=2, cursor=0)
    assert result["cumulative_microbatches"] == 732
    assert result["cumulative_target_tokens"] == 17640
    assert result["training_seconds"] == 202.0
    assert len(list(d.schedule(result))) == 23
    with pytest.raises(ValueError, match="state"):
        list(d.schedule(dict(step=46, epoch=1, cursor=0)))


def test_endpoint_requires_all_rows_and_updates_at_fixed_boundary():
    d = module()
    steps = [
        dict(step=i, rows=14 if i % 23 == 0 else 16, target_tokens=460 if i % 23 == 0 else 380)
        for i in range(24, 47)
    ]
    state = dict(
        step=46, epoch=2, cursor=0, cumulative_microbatches=732, cumulative_target_tokens=17640
    )
    d.validate_dose(state, steps, 46)
    with pytest.raises(ValueError, match="receipts"):
        d.validate_dose(state, steps[:-1], 46)
    with pytest.raises(ValueError, match="fixed"):
        d.validate_dose(state, steps, 45)
    with pytest.raises(ValueError, match="state"):
        d.validate_dose(dict(state, cumulative_microbatches=736), steps, 46)


def test_optimizer_and_rng_roundtrip_reproduces_next_update(tmp_path):
    d = module()
    import random

    import torch

    random.seed(1)
    torch.manual_seed(1)
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.0)
    for _ in range(23):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    torch.save(
        dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=[]), tmp_path / "rng.pt"
    )
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored_optimizer = torch.optim.AdamW([restored_parameter], lr=1e-4, weight_decay=0.0)
    d.restore_optimizer_rng(restored_optimizer, tmp_path, 23, device="cpu")
    for candidate, opt in ((parameter, optimizer), (restored_parameter, restored_optimizer)):
        opt.zero_grad()
        candidate.square().sum().backward()
        opt.step()
    assert torch.equal(parameter, restored_parameter)
    assert {int(v["step"]) for v in restored_optimizer.state.values()} == {24}


def test_readout_changes_teacher_and_checkpoint_without_changing_paired_jobs():
    path = Path(__file__).with_name("readout.py")
    assert path.exists(), "dose readout implementation has not been written"
    import sys

    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("dose_readout_fixture", path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    template = dict(
        teacher="discovery",
        jobs=[dict(task_id="goal", repeat=0, seed=7, condition="old")],
        conditions=["old"],
        adapter=dict(path="old"),
    )
    result = reader.replace_endpoint(
        template,
        dict(path="known/checkpoint-0046"),
        teacher="known_recipe",
        step=46,
        world=50,
        execution="raw",
    )
    assert result["teacher"] == "known_recipe"
    assert result["cumulative_training_updates"] == 46
    assert result["adapter"]["path"] == "known/checkpoint-0046"
    assert result["jobs"] == [
        dict(task_id="goal", repeat=0, seed=7, condition="dose_known_recipe_cp46_p00_w50_raw")
    ]
    assert template["jobs"][0]["condition"] == "old"
