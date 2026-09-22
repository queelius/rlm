import json
import random

import pytest


def test_restore_preserves_adam_step_moments_and_rng(tmp_path):
    import textcraft_fp16_continuation as f
    import torch

    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.AdamW([p], lr=2e-5, weight_decay=0)
    p.square().sum().backward()
    opt.step()
    torch.save(opt.state_dict(), tmp_path / "optimizer.pt")
    random.seed(13)
    torch.manual_seed(19)
    torch.save(
        dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=[]), tmp_path / "rng.pt"
    )
    wanted = (random.random(), torch.rand(1))
    restored = f.restore_optimizer_rng([p], tmp_path)
    assert float(restored.state[p]["step"]) == 1
    assert torch.equal(restored.state[p]["exp_avg"], opt.state[p]["exp_avg"])
    assert torch.equal(restored.state[p]["exp_avg_sq"], opt.state[p]["exp_avg_sq"])
    assert random.random() == wanted[0] and torch.equal(torch.rand(1), wanted[1])
    restored.zero_grad(set_to_none=True)
    p.square().sum().backward()
    restored.step()
    assert float(restored.state[p]["step"]) == 2


def test_failed_numeric_check_is_persisted_before_rejection(tmp_path):
    import textcraft_fp16_continuation as f

    with pytest.raises(ValueError, match="generation/replay"):
        f.record_before(tmp_path, {"a": [-1.0]}, {"a": [-1.5]})
    receipt = json.loads((tmp_path / "BEFORE_LOGPS.json").read_text())
    assert receipt["generation_replay_max_abs"] == 0.5
    assert receipt["logps"] == {"a": [-1.0]}
    assert receipt["captured_generation_logps"] == {"a": [-1.5]}


def test_only_two_new_seed_blocks_not_bf16_reuse():
    import textcraft_fp16_continuation as f

    jobs = [dict(repeat=i, seed=1) for i in range(4)]
    fresh = f.fresh_jobs(jobs)
    assert set(fresh) == {"2", "3"}
    assert [x["seed"] for x in fresh["2"]] == [2026092273, 2026092274, 2026092275, 2026092276]
    assert all(x["seed"] != 1 for rows in fresh.values() for x in rows)


def test_nan_scores_reject_after_saving_receipt(tmp_path):
    import textcraft_fp16_continuation as f

    with pytest.raises(ValueError, match="generation/replay"):
        f.record_before(tmp_path, {"a": [float("nan")]}, {"a": [-1.0]})
    assert json.loads((tmp_path / "BEFORE_LOGPS.json").read_text())["all_finite"] is False


def test_nonfinite_backward_preflight_persists_and_clears_without_optimizer(tmp_path):
    from types import SimpleNamespace

    import textcraft_fp16_continuation as f
    import torch

    class NonfiniteActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.tensor(float("nan")))

        @property
        def device(self):
            return self.lora_weight.device

        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.lora_weight.expand(1, 1, 2))

    call = tmp_path / "call.json"
    call.write_text(json.dumps(dict(call_id="fixed", input_token_ids=[1], output_token_ids=[0])))
    torch.save(
        dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=[]), tmp_path / "rng.pt"
    )
    model = NonfiniteActor()
    with pytest.raises(ValueError, match="finite backward qualification"):
        f.pre_rollout_probe(
            model,
            list(model.parameters()),
            dict(numeric_probe_call=str(call), restore_checkpoint=str(tmp_path)),
            tmp_path,
        )
    row = json.loads((tmp_path / "FP16-BACKWARD-PREFLIGHT.json").read_text())
    assert row["finite_loss"] is False and row["optimizer_called"] is False
    assert all(p.grad is None for p in model.parameters())
