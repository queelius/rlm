"""Exact two-rollout RAO weighting and child-distribution diagnostic; no model calls."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


def g2_weighting(p: float) -> dict[str, float]:
    """All quantities are sums over two roots, not per-root means.

    Each root has A~Bernoulli(p), R_root=A, and spawns one deterministic child iff A=1.
    Eq.4 gives root weights 1, .75, 1 for total child counts 0, 1, 2 respectively.
    """
    baseline_unweighted = baseline_weighted = 0.0
    reward_unweighted = reward_weighted = 0.0
    for first, second in product((0, 1), repeat=2):
        probability = (p if first else 1 - p) * (p if second else 1 - p)
        root_weight = 1.0 if first + second in (0, 2) else 0.75
        for action, other_reward in ((first, second), (second, first)):
            score = action - p
            baseline_unweighted += probability * other_reward * score
            baseline_weighted += probability * root_weight * other_reward * score
            reward_unweighted += probability * action * score
            reward_weighted += probability * root_weight * action * score
    return {
        "p": p,
        "loo_baseline_unweighted_sum_g2": baseline_unweighted,
        "loo_baseline_weighted_sum_g2": baseline_weighted,
        "weighted_baseline_closed_form_sum_g2": 0.5 * p * p * (1 - p),
        "reward_score_unweighted_sum_g2": reward_unweighted,
        "reward_score_weighted_sum_g2": reward_weighted,
        "weighted_advantage_sum_g2": reward_weighted - baseline_weighted,
    }


def child_distribution_semigradient(p: float, q: float) -> dict[str, float]:
    """A separate-parameter toy for the omitted ancestor distribution score term.

    Root delegates iff A~Bernoulli(p), with root reward A. A delegated child succeeds
    C~Bernoulli(q); its local reward is C. For J=E[A]+E[1{A=1}C]=p+p*q,
    a root-local score using only A has d/dp expectation 1, while full dJ/dp=1+q.
    """
    return {
        "p": p,
        "q": q,
        "full_dJ_dp": 1 + q,
        "root_local_score_term_dp": 1.0,
        "omitted_child_distribution_term_dp": q,
        "child_score_term_dq": p,
        "full_dJ_dq": p,
    }


def report() -> dict:
    return {
        "schema": "rao-weighting-toy-v1",
        "status": "cpu_diagnostic_not_novelty",
        "g2": [g2_weighting(p) for p in (0.3, 0.5)],
        "child_distribution_semigradient": child_distribution_semigradient(0.3, 0.4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report(), indent=2) + "\n")
