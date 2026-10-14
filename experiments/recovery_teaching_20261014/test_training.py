"""Catch causal-shift, label-mask and one-step accounting errors before GPU fitting."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def module():
    path = Path(__file__).with_name("train.py")
    assert path.exists(), "recovery trainer missing"
    spec = importlib.util.spec_from_file_location("recovery_training", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def row():
    return dict(
        input_ids=[1, 2, 3, 4],
        labels=[-100, -100, 3, 4],
        target_ids=[3, 4],
        prompt_tokens=2,
        target_tokens=2,
    )


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(5, 5))

    @property
    def device(self):
        return self.table.device

    def forward(self, input_ids, use_cache, logits_to_keep):
        assert use_cache is False
        return SimpleNamespace(logits=self.table[input_ids][:, -logits_to_keep:])


def test_saved_full_sequence_is_shifted_exactly_once_and_only_targets_receive_loss():
    m = module()
    model = Tiny()
    with torch.no_grad():
        model.table[2, 3] = 12
        model.table[3, 4] = 12
    example = m.example(row())
    assert example == {"input_ids": [1, 2, 3], "target_ids": [3, 4]}
    loss = m.objective(model, example, 2)
    assert float(loss.detach()) < 0.0001
    loss.backward()
    assert model.table.grad[1].abs().sum() == 0
    assert model.table.grad[4].abs().sum() == 0
    assert model.table.grad[2].abs().sum() > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("labels", [-100, 2, 3, 4]),
        ("target_ids", [2, 4]),
        ("target_tokens", 3),
        ("prompt_tokens", 0),
    ],
)
def test_corrupt_masks_or_token_boundaries_are_rejected(field, value):
    data = row()
    data[field] = value
    with pytest.raises(ValueError):
        module().example(data)


def test_one_step_uses_all_rows_and_commits_one_finite_parameter_change():
    m = module()
    model = Tiny()
    examples = [m.example(row()), m.example(row())]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    result = m.one_update(model, optimizer, examples, lambda: None)
    assert result["rows"] == 2
    assert result["target_tokens"] == 4
    assert result["actual_new_optimizer_steps"] == 1
    assert result["loss_before"] == pytest.approx(math.log(5), abs=1e-6)
    assert result["adapter_l2_delta"] > 0
    assert result["loss_after"] < result["loss_before"]
    assert {int(s["step"]) for s in optimizer.state.values()} == {1}


def test_prepare_reencodes_both_real_frozen_arms_without_loading_gpu(tmp_path):
    m = module()
    assert hasattr(m, "prepare"), "preparation seam missing"
    results = []
    for arm in ("recovery", "ordinary"):
        args = SimpleNamespace(arm=arm, output=tmp_path / arm)
        plan, examples, _ = m.prepare(args)
        assert len(examples) == 32
        assert sum(len(r["target_ids"]) for r in examples) == 823
        assert len(plan["task_ids"]) == 8
        assert not list(args.output.glob("OWNER*"))
        assert plan["adapter"]["sha256"] == (
            "f029d36967eb1cef30e102fe82e52f22f90d5eacdaf91e82937b60218ba5b00f"
        )
        results.append(plan)
    assert results[0]["prompt_tokens"] == 80120
    assert results[1]["prompt_tokens"] == 65044
    assert results[0]["adapter"] == results[1]["adapter"]
    with pytest.raises(ValueError, match="immutable"):
        m.prepare(SimpleNamespace(arm="recovery", output=tmp_path / "ordinary"))
