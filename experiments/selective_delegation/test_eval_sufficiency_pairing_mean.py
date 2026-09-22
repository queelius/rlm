import eval_sufficiency_heldout as shared
import pytest


def _identity(estimator):
    return {
        "step": 8,
        "sample_cursor": 8,
        "training_plan": {
            "mode": "rl",
            "reward_objective": "product",
            "estimator": estimator,
            **{key: "same" for key in shared.PAIRING_COMMON},
        },
    }


def test_pairing_mean_profile_accepts_only_same_product_training_contract():
    product, pairing = _identity("diagonal"), _identity("pairing_mean")
    shared.validate_pairing_mean_control(product, pairing)
    pairing["training_plan"]["reward_objective"] = "additive"
    with pytest.raises(ValueError, match="product objectives"):
        shared.validate_pairing_mean_control(product, pairing)


def test_pairing_mean_profile_rejects_reduced_dose_instead_of_substituting_endpoint():
    product, pairing = _identity("diagonal"), _identity("pairing_mean")
    pairing["step"] = 7
    pairing["sample_cursor"] = 7
    with pytest.raises(ValueError, match="eight committed"):
        shared.validate_pairing_mean_control(product, pairing)
