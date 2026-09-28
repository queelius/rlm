"""Guard the estimator and stage boundary without loading the 4B model."""

import importlib.util

import pytest


def trainer():
    assert importlib.util.find_spec("train") is not None
    import train

    return train


def test_signed_objective_sums_every_emitted_token_with_fixed_episode_denominator():
    t = trainer()
    import torch

    logps = torch.tensor([-2.0, -3.0, -1.0], requires_grad=True)
    loss = t.signed_objective(logps, -2 / 3)
    assert float(loss.detach()) == pytest.approx(-0.125)
    loss.backward()
    assert logps.grad.tolist() == pytest.approx([1 / 48] * 3)


def test_probability_diagnostic_does_not_relabel_selected_tokens_as_kl():
    t = trainer()
    from types import SimpleNamespace

    credits = [
        SimpleNamespace(call_id="a", advantage=1 / 3),
        SimpleNamespace(call_id="b", advantage=-1.0),
    ]
    out = t.probability_changes(
        credits,
        {"a": [-2, -3], "b": [-1]},
        {"a": [-1, -3], "b": [-2]},
    )
    assert out["positive"]["tokens"] == 2
    assert out["positive"]["mean_sampled_token_logp_delta"] == 0.5
    assert out["negative"]["mean_sampled_token_logp_delta"] == -1.0
    assert "not KL" in out["scope"]
