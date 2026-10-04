"""Independent numeric expectations for the bilinear objective, not RL outcomes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_module():
    path = Path(__file__).with_name("compare_objective.py")
    assert path.exists(), "Independent contrastive objective check is not implemented"
    spec = importlib.util.spec_from_file_location("curl_objective", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_loss_and_gradients_match_hand_calculated_two_candidate_example():
    result = load_module().manual(np.eye(2), np.eye(2), np.eye(2))
    assert result["loss"] == pytest.approx(0.313261687518, abs=1e-12)
    np.testing.assert_allclose(result["logits"], [[0, -1], [-1, 0]], atol=1e-12)
    expected = [[-0.134470710685, 0.134470710685], [0.134470710685, -0.134470710685]]
    for name in ("query_gradient", "weight_gradient", "positive_gradient"):
        np.testing.assert_allclose(result[name], expected, atol=1e-12)


def test_official_torch_logits_loss_gradients_and_sgd_update_agree():
    source = Path("/project/alex_phd/research-cache/repos/curl-8416d6e")
    if not source.exists():
        pytest.skip("Pinned external CURL source is not present")
    result = load_module().compare(source)
    assert result["evidence_type"] == "CPU objective unit check; not an RL result"
    assert max(result["absolute_errors"].values()) < 1e-10
    assert result["correct_positive_loss"] < result["permuted_positive_loss"]
