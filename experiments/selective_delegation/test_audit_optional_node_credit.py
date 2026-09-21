import pytest
from audit_optional_node_credit import enumerate_gradients


@pytest.mark.parametrize("n,q,p", [(2, 0.5, 0.5), (4, 0.2, 0.5), (4, 0.05, 0.3)])
def test_optional_singleton_credit_matches_exact_derivative(n, q, p):
    result = enumerate_gradients(n, q, p)
    derivative = q * p * (1 - p)
    assert result["probability_mass"] == pytest.approx(1)
    assert result["singleton_zero_gradient"] == pytest.approx(derivative * (1 - (1 - q) ** (n - 1)))
    assert result["singleton_zero_baseline_gradient"] == pytest.approx(derivative)
    assert result["full_group_terminal_rloo_gradient"] == pytest.approx(derivative)


def test_always_reached_group_has_no_singleton_bias():
    result = enumerate_gradients(4, 1, 0.3)
    assert result["singleton_zero_gradient"] == pytest.approx(result["true_gradient"])
