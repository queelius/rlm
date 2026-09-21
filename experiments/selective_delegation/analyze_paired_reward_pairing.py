"""Read-only finite-permutation audit of paired positive/negative RLOO rewards."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def rloo(values):
    if len(values) != 4:
        raise ValueError("four pair rewards required")
    return [value - (sum(values) - value) / 3 for value in values]


def coefficients(positive, negative):
    """Per-trajectory coefficients for the *mean-over-four* paired objective."""
    if len(positive) != 4 or len(negative) != 4:
        raise ValueError("four positive and four negative marginal successes required")
    positive = [float(value) for value in positive]
    negative = [float(value) for value in negative]
    pmean, nmean = sum(positive) / 4, sum(negative) / 4
    pos = [nmean * (value - (sum(positive) - value) / 3) / 4 for value in positive]
    neg = [pmean * (value - (sum(negative) - value) / 3) / 4 for value in negative]
    return pos, neg


def enumerated_gradient(positive, negative, positive_scores, negative_scores):
    """Average the current diagonal-pairing RLOO score estimator over all 24 matchings."""
    total = 0.0
    for permutation in itertools.permutations(range(4)):
        rewards = [positive[index] * negative[permutation[index]] for index in range(4)]
        advantage = rloo(rewards)
        total += (
            sum(
                advantage[index] * (positive_scores[index] + negative_scores[permutation[index]])
                for index in range(4)
            )
            / 4
        )
    return total / 24


def rao_blackwell_gradient(positive, negative, positive_scores, negative_scores):
    """Closed form of ``enumerated_gradient`` conditional on eight saved draws."""
    pos, neg = coefficients(positive, negative)
    return sum(weight * score for weight, score in zip(pos, positive_scores, strict=True)) + sum(
        weight * score for weight, score in zip(neg, negative_scores, strict=True)
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def committed_block_directories(run_root):
    """Return batch directories admitted by their sealed boundary, never UPDATE alone."""
    run_root = Path(run_root)
    result = []
    for boundary_path in sorted((run_root / "boundaries").glob("sample-*/BOUNDARY.json")):
        boundary = load_json(boundary_path)
        name = boundary_path.parent.name
        try:
            cursor = int(name.removeprefix("sample-"))
        except ValueError as error:
            raise ValueError("boundary sample directory is malformed") from error
        if cursor == 0:
            continue
        checkpoint = boundary_path.parent / f"checkpoint-{cursor:04d}"
        commit_path = checkpoint / "COMMIT.json"
        state_path = checkpoint / "STATE.json"
        if (
            not commit_path.exists()
            or not state_path.exists()
            or Path(boundary.get("checkpoint", "")) != checkpoint
            or boundary.get("commit_sha256") != sha256(commit_path)
        ):
            raise ValueError("boundary lacks its authenticated checkpoint commit")
        commit = load_json(commit_path)
        state = load_json(state_path)
        if (
            boundary["state"] != state
            or state.get("sample_cursor") != cursor
            or commit.get("step") != state.get("step")
            or commit.get("files", {}).get("STATE.json") != sha256(state_path)
        ):
            raise ValueError("boundary state/commit identity differs")
        batch = run_root / "batches" / name
        if not (batch / "ROLLOUT.json").exists() or not (batch / "UPDATE.json").exists():
            raise ValueError("committed boundary lacks complete rollout/update receipt")
        result.append(batch)
    return result


def positive_success(case, prediction, baseline):
    if prediction is None:
        return 0
    known_good_negative = {"answerable": False, "answer": ""}
    return int(baseline.group_score(case, [prediction, known_good_negative])["em"] == 1.0)


def negative_success(prediction):
    # The official sufficiency score only requires the negative label; saved product
    # receipts below verify this reconstruction against every recorded pair reward.
    return int(prediction is not None and prediction["answerable"] is False)


def group_summary(positive, negative, diagonal_rewards):
    pos_weights, neg_weights = coefficients(positive, negative)
    current = rloo(diagonal_rewards)
    return {
        "positive_success": positive,
        "negative_success": negative,
        "positive_marginal_mixed": len(set(positive)) == 2,
        "negative_marginal_mixed": len(set(negative)) == 2,
        "diagonal_rewards": diagonal_rewards,
        "diagonal_effective": any(current),
        "rb_positive_coefficients": pos_weights,
        "rb_negative_coefficients": neg_weights,
        "rb_effective": any(pos_weights + neg_weights),
        "diagonal_absolute_credit": 2 * sum(abs(value) for value in current) / 4,
        "rb_absolute_credit": sum(abs(value) for value in pos_weights + neg_weights),
    }


def analyze(run_root, source_root):
    import sys

    source_root = Path(source_root)
    sys.path.insert(0, str(source_root))
    import sufficiency_probe as baseline

    run_root = Path(run_root)
    plan = load_json(run_root / "PLAN.json")
    cases = [json.loads(line) for line in Path(plan["cases"]).read_text().splitlines()]
    case_by_parent = {}
    for case in cases:
        case_by_parent.setdefault(case["parent_id"], {})[case["answerable"]] = case
    blocks = []
    boundary_inventory = committed_block_directories(run_root)
    for directory in boundary_inventory:
        rollout = load_json(directory / "ROLLOUT.json")
        groups = []
        for index, parent in enumerate(rollout["parents"]):
            pair_rows = [
                load_json(directory / "pairs" / f"p{index:02d}-k{candidate}.json")
                for candidate in range(4)
            ]
            if any(
                row["parent_id"] != parent or row["candidate"] != candidate
                for candidate, row in enumerate(pair_rows)
            ):
                raise ValueError("saved parent/candidate identity differs")
            positive_case = case_by_parent[parent][True]
            positive = [
                positive_success(positive_case, row["predictions"][0], baseline)
                for row in pair_rows
            ]
            negative = [negative_success(row["predictions"][1]) for row in pair_rows]
            diagonal = [row["reward"] for row in pair_rows]
            if diagonal != [p * n for p, n in zip(positive, negative, strict=True)]:
                raise ValueError(
                    "marginal reconstruction differs from saved official product reward"
                )
            groups.append({"parent_id": parent, **group_summary(positive, negative, diagonal)})
        blocks.append(
            {
                "block": directory.name,
                "groups": groups,
                "diagonal_effective_groups": sum(group["diagonal_effective"] for group in groups),
                "rb_effective_groups": sum(group["rb_effective"] for group in groups),
                "restored_groups": sum(
                    group["rb_effective"] and not group["diagonal_effective"] for group in groups
                ),
                "positive_marginal_mixed_groups": sum(
                    group["positive_marginal_mixed"] for group in groups
                ),
                "negative_marginal_mixed_groups": sum(
                    group["negative_marginal_mixed"] for group in groups
                ),
                "diagonal_absolute_credit": sum(
                    group["diagonal_absolute_credit"] for group in groups
                ),
                "rb_absolute_credit": sum(group["rb_absolute_credit"] for group in groups),
            }
        )
    return {
        "schema": "paired-reward-pairing-variance-audit-v1",
        "run_root": str(run_root),
        "completed_blocks": len(blocks),
        "boundary_inventory": {
            "committed_batch_directories": [str(path) for path in boundary_inventory],
            "boundary_sha256": {
                str(path.parent.parent / "boundaries" / path.name / "BOUNDARY.json"): sha256(
                    path.parent.parent / "boundaries" / path.name / "BOUNDARY.json"
                )
                for path in boundary_inventory
            },
        },
        "blocks": blocks,
        "totals": {
            key: sum(block[key] for block in blocks)
            for key in (
                "diagonal_effective_groups",
                "rb_effective_groups",
                "restored_groups",
                "positive_marginal_mixed_groups",
                "negative_marginal_mixed_groups",
                "diagonal_absolute_credit",
                "rb_absolute_credit",
            )
        },
        "source_sha256": {
            "script": sha256(__file__),
            "plan": sha256(run_root / "PLAN.json"),
            "cases": sha256(plan["cases"]),
            "source_rl_sufficiency": sha256(source_root / "rl_sufficiency.py"),
        },
        "scope": (
            "conditional expectation over all 24 pairings of eight fixed independent "
            "variant draws; before gradient clipping and Adam"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_root, args.source_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
