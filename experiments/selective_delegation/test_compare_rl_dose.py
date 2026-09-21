import copy

import pytest


def test_contract_allows_checkpoint_change_not_scientific_contract_change():
    from compare_rl_dose import validate_contract

    first = {
        "conditions": ["rl"],
        "case_ids": ["p"],
        "repeats": 2,
        "cases_sha256": "cases",
        "seed": 12,
        "source_sha256": "eval",
        "execution": "isolated",
        "model": "base",
        "model_manifest_sha256": "basehash",
        "split": "development",
        "temperature": 0.5,
        "top_p": 1.0,
        "top_k": 0,
        "caps": {"root": 128, "helper": 384, "final": 128},
        "helper_contract": {"mode": "trained_helper", "adapter": "helper36"},
        "adapter": "checkpoint16",
    }
    second = {**first, "adapter": "checkpoint24"}
    validate_contract(first, second)
    for key, value in (
        ("seed", 13),
        ("source_sha256", "other"),
        ("helper_contract", {"mode": "base"}),
        ("conditions", ["sft"]),
    ):
        with pytest.raises(ValueError):
            validate_contract(first, {**second, key: value})


def test_paired_changes_separate_missing_from_protocol_and_cluster_parent_means():
    from compare_rl_dose import paired

    cases = [{"id": p, "metadata": {"component_ids": ["shared"]}} for p in ("a", "b")]
    old, new = {}, {}
    for p in ("a", "b"):
        for r in range(2):
            old[p, r] = dict(em=0.0, f1=0.0, valid=True, observed=True, status="scored")
    new = copy.deepcopy(old)
    new["a", 0].update(em=1.0, f1=1.0)
    old["a", 0].update(valid=False, status="invalid_helper")
    new["a", 1].update(em=1.0, f1=1.0)
    old["b", 0].update(em=1.0, f1=1.0)
    new["b", 0].update(observed=False, valid=False, status="missing_episode")
    result = paired(old, new, cases, 2, draws=20, seed=7)
    assert result["wins"]["categories"] == {"protocol_involved": 1, "both_valid": 1}
    assert result["losses"]["categories"] == {"unobserved_involved": 1}
    assert result["em"] == {"estimate": 0.25, "ci95": [0.25, 0.25]}
    assert result["lower_bound_difference_not_effect_estimate"]


def test_returned_invalid_is_observed_but_unavailable_call_is_missing():
    from compare_rl_dose import score_episode

    case = {"answer": "yes", "answer_aliases": []}
    episode = {"status": "invalid_helper"}
    record = {"role": "helper", "available": True}
    assert score_episode(episode, [record], case)["observed"]
    record["available"] = False
    assert not score_episode(episode, [record], case)["observed"]
    assert not score_episode(None, [], case)["observed"]
    assert not score_episode(episode, [], case)["observed"]
