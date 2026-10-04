"""CPU bilinear contrastive objective agreement; this is not an RL result."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def manual(query, positive, weight):
    logits = query @ weight @ positive.T
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    gradient = (probabilities - np.eye(len(query))) / len(query)
    return {
        "logits": logits,
        "loss": -np.log(probabilities.diagonal()).mean(),
        "query_gradient": gradient @ positive @ weight.T,
        "weight_gradient": query.T @ gradient @ positive,
        "positive_gradient": gradient.T @ query @ weight,
    }


def compare(source: Path) -> dict:
    import torch

    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != "8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc":
        raise ValueError("Expected pinned official CURL commit")
    sys.path.insert(0, str(source))
    from curl_sac import CURL

    query = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 1.0]])
    positive, weight = query.copy(), np.eye(3) * 2
    reference = manual(query, positive, weight)
    q, p, w = [
        torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (query, positive, weight)
    ]
    logits = CURL.compute_logits(SimpleNamespace(W=w), q, p)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(3))
    loss.backward()
    actual = {
        "logits": logits.detach().numpy(),
        "loss": loss.item(),
        "query_gradient": q.grad.numpy(),
        "positive_gradient": p.grad.numpy(),
        "weight_gradient": w.grad.numpy(),
    }
    optimizer = torch.optim.SGD([w], lr=0.01)
    optimizer.step()
    errors = {
        name: float(np.max(np.abs(actual[name] - expected))) for name, expected in reference.items()
    }
    errors["sgd_weight_update"] = float(
        np.max(np.abs(w.detach().numpy() - (weight - 0.01 * reference["weight_gradient"])))
    )
    if max(errors.values()) > 1e-10:
        raise AssertionError(errors)
    return {
        "evidence_type": "CPU objective unit check; not an RL result",
        "commit": commit,
        "source_sha256": hashlib.sha256((source / "curl_sac.py").read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tolerance": 1e-10,
        "absolute_errors": errors,
        "correct_positive_loss": float(reference["loss"]),
        "permuted_positive_loss": float(manual(query, positive[[1, 2, 0]], weight)["loss"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    print(json.dumps(compare(parser.parse_args().source.resolve()), indent=2))
