"""Exact CPU toy: optional-node credit lost by discarding singleton reached groups."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def enumerate_gradients(n, reach_probability, success_probability):
    """Derivative wrt a child-success logit, holding the reach probability fixed.

    A trajectory terminates successfully before the child with probability1-q.
    Otherwise the child succeeds with probabilityp. No history dependence,
    shared-parent-parameter derivative, clipping, Adam or GRPO normalization.
    """
    if not 2 <= n <= 8 or not 0 <= reach_probability <= 1 or not 0 < success_probability < 1:
        raise ValueError("bounded exact enumeration: N2..8, q0..1, p strictly0..1")
    q, p = reach_probability, success_probability
    probabilities = (1 - q, q * (1 - p), q * p)
    mass = zero = fallback = terminal = 0.0
    for outcomes in itertools.product(range(3), repeat=n):
        weight = 1.0
        for outcome in outcomes:
            weight *= probabilities[outcome]
        mass += weight
        reached = [outcome - 1 for outcome in outcomes if outcome]
        count = len(reached)
        if count > 1:
            gradient = sum((y - (sum(reached) - y) / (count - 1)) * (y - p) for y in reached) / n
            zero += weight * gradient
            fallback += weight * gradient
        elif count == 1:
            y = reached[0]
            fallback += weight * y * (y - p) / n
        rewards = [int(outcome != 1) for outcome in outcomes]
        terminal += (
            weight
            * sum(
                (rewards[i] - (sum(rewards) - rewards[i]) / (n - 1)) * (outcome - 1 - p)
                for i, outcome in enumerate(outcomes)
                if outcome
            )
            / n
        )
    derivative = q * p * (1 - p)
    return {
        "root_trajectories": n,
        "reach_probability": q,
        "child_success_probability": p,
        "probability_mass": mass,
        "true_gradient": derivative,
        "singleton_zero_gradient": zero,
        "singleton_zero_baseline_gradient": fallback,
        "full_group_terminal_rloo_gradient": terminal,
        "analytic_singleton_zero_gradient": derivative * (1 - (1 - q) ** (n - 1)),
        "expected_retained_fraction": 1 - (1 - q) ** (n - 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "optional-node-singleton-credit-toy-v1",
        "question": "Does zeroing singleton reached groups suppress optional-node credit?",
        "status": "exact_cpu_toy_not_model_experiment",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [
            enumerate_gradients(n, q, p)
            for n, q, p in [(2, 0.5, 0.5), (4, 0.2, 0.5), (4, 0.05, 0.5), (8, 0.2, 0.5)]
        ],
        "scope": "child-logit coordinate with fixed reach probability; independent roots; "
        "exact expectations before clipping/Adam; no variance or learning-gain claim",
        "prior_art": "https://arxiv.org/html/2604.17912v1",
        "prior_art_caveat": "Appendix C explicitly assumes reached-group size>=2 almost surely; "
        "the optional-singleton toy falls outside that assumption, not a refutation of it.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
