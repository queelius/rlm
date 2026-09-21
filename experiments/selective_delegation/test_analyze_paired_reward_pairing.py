"""Regression checks for the exact finite-pairing Rao--Blackwell calculation."""

import json
from itertools import permutations

import analyze_paired_reward_pairing as pairing
import pytest


def test_all_permutations_match_closed_form_for_vector_scores():
    positive = [1, 0, 1, 0]
    negative = [0, 1, 1, 0]
    pos_scores = [1.5, -2.0, 0.25, 4.0]
    neg_scores = [-1.0, 3.0, 2.5, -0.5]

    enumerated = pairing.enumerated_gradient(positive, negative, pos_scores, neg_scores)
    closed = pairing.rao_blackwell_gradient(positive, negative, pos_scores, neg_scores)

    assert len(list(permutations(range(4)))) == 24
    assert enumerated == pytest.approx(closed)


def test_both_uniform_marginals_have_zero_credit():
    positive = [1, 1, 1, 1]
    negative = [0, 0, 0, 0]
    assert pairing.coefficients(positive, negative) == ([0.0] * 4, [0.0] * 4)


def test_uncommitted_update_is_excluded_from_boundary_inventory(tmp_path):
    batch = tmp_path / "batches" / "sample-0001"
    batch.mkdir(parents=True)
    (batch / "UPDATE.json").write_text(json.dumps({"updated": True}))
    (batch / "ROLLOUT.json").write_text(json.dumps({"parents": []}))

    assert pairing.committed_block_directories(tmp_path) == []
