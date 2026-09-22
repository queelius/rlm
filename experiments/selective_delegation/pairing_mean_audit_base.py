"""Terminal-only audit of source058 product reward with pairing-mean response credit."""

import argparse
import json
import math
from collections import Counter
from itertools import chain
from pathlib import Path

import analyze_sufficiency_rl as shared


def marginal_credit(positive, negative):
    """Byte-for-byte formula semantics from sealed source058's rl_sufficiency.py."""
    if len(positive) != 4 or len(negative) != 4:
        raise ValueError("four marginal successes per side required")
    pmean, nmean = sum(positive) / 4, sum(negative) / 4
    return [
        (
            nmean * (p - (sum(positive) - p) / 3),
            pmean * (n - (sum(negative) - n) / 3),
        )
        for p, n in zip(positive, negative, strict=True)
    ]


def grade_pair(case, predictions, observed):
    if not observed:
        return None
    baseline = shared.paired.baseline
    positive = int(
        baseline.group_score(case, [predictions[0], {"answerable": False, "answer": ""}])["em"]
    )
    negative = int(predictions[1] is not None and predictions[1]["answerable"] is False)
    reward = int(baseline.group_score(case, predictions)["em"])
    if reward != positive * negative:
        raise ValueError("official product reward does not factor into marginals")
    return {
        "positive_success": positive,
        "negative_success": negative,
        "reward": reward,
        "training_reward": reward,
        "all_valid": all(p is not None for p in predictions),
    }


def coefficient_map(groups):
    if len(groups) != 16 or any(len(g) != 4 or any(p is None for p in g) for g in groups):
        raise ValueError("complete observed16x4 product groups required")
    advantages = [
        marginal_credit(
            [pair["positive_success"] for pair in group],
            [pair["negative_success"] for pair in group],
        )
        for group in groups
    ]
    return {
        f"p{i:02d}-k{k}-v{v}": coefficient
        for i, group in enumerate(advantages)
        for k, pair in enumerate(group)
        for v, coefficient in enumerate(pair)
    }, advantages


def validate_likelihood(update, credits, native):
    rows = update.get("likelihoods", [])
    selected = {key: value for key, value in credits.items() if value != 0}
    if {row["call_id"] for row in rows} != set(selected):
        raise ValueError("credited response inventory differs from pairing-mean coefficients")
    loss = 0.0
    for row in rows:
        call = native[row["call_id"]]
        if (
            row["advantage"] != selected[row["call_id"]]
            or row["generation"] != call["generation_logps"]
            or len(row["before"]) != len(call["output_token_ids"])
            or len(row["after"]) != len(row["before"])
            or not all(
                math.isfinite(x) for key in ("before", "after", "generation") for x in row[key]
            )
        ):
            raise ValueError("per-response likelihood receipt differs")
        loss -= row["advantage"] * sum(row["before"]) / 64
    if rows and (
        not math.isclose(loss, update["loss"], rel_tol=1e-5, abs_tol=1e-4)
        or update["credited_responses"] != len(rows)
    ):
        raise ValueError("saved pairing-mean loss differs")
    return {"credited_responses": len(rows), "loss_recomputed": loss}


def analyze(output, tokenizer):
    audit, output = shared.Audit(), output.resolve()
    terminals = audit.terminal(output)
    plan, summary = audit.read(output / "PLAN.json"), audit.read(output / "SUMMARY.json")
    if (plan["mode"], plan.get("reward_objective"), plan.get("estimator")) != (
        "rl",
        "product",
        "pairing_mean",
    ):
        raise ValueError("requires source058 product pairing-mean RL")
    if (plan["max_sampled_blocks"], plan["pair_denominator"], plan["max_calls"]) != (8, 64, 1024):
        raise ValueError("fixed finite TRAIN contract differs")
    for mapping in (plan["source_sha256"], plan["official_metric_sha256"]):
        for path, digest in mapping.items():
            audit.hash(path, digest)
    boundaries = shared.training.boundary_inventory(output)
    commits = [audit.checkpoint(b) for b in boundaries]
    cases = audit.cases(plan["cases"], plan["cases_sha256"])
    lookup = shared.training.paired_cases(cases, list(chain.from_iterable(plan["parent_blocks"])))
    batches, statuses, physical = [], Counter(), []
    for directory in sorted((output / "batches").glob("sample-*")):
        cursor = int(directory.name[-4:])
        prior, commit = boundaries[cursor - 1], commits[cursor - 1]
        groups, native, missing = [], {}, 0
        for i, parent in enumerate(plan["parent_blocks"][cursor - 1]):
            group = []
            for k in range(4):
                predictions, observed, ids = [], True, []
                for v, case in enumerate(lookup[parent]):
                    identity = f"p{i:02d}-k{k}-v{v}"
                    ids.append(identity)
                    path = directory / "calls" / f"{identity}.json"
                    status, prediction = "missing", None
                    if path.exists():
                        row = audit.read(path)
                        native[identity], physical = row, physical + [row]
                        status, prediction = shared.audit_training_call(
                            row,
                            case,
                            plan["seed"] + cursor * 100000 + i * 1000 + k * 10 + v,
                            plan,
                            prior,
                            commit,
                            tokenizer,
                        )
                    statuses[status] += 1
                    predictions.append(prediction)
                    observed &= status in ("valid", "protocol_invalid")
                pair = grade_pair(lookup[parent][0], predictions, observed)
                receipt = directory / "pairs" / f"p{i:02d}-k{k}.json"
                if not receipt.exists():
                    missing += 1
                elif pair is None or any(
                    audit.read(receipt)[key] != value for key, value in pair.items()
                ):
                    raise ValueError("saved product pair receipt differs")
                group.append(pair)
            groups.append(group)
        committed = cursor < len(boundaries)
        batch = {
            "sample_cursor": cursor,
            "committed": committed,
            "unknown_pairs": sum(p is None for g in groups for p in g),
        }
        if not batch["unknown_pairs"]:
            credits, advantages = coefficient_map(groups)
            batch["pairing_mean"] = {
                "effective_groups": sum(
                    any(x != 0 for pair in g for x in pair) for g in advantages
                ),
                "nonzero_response_coefficients": sum(x != 0 for x in credits.values()),
                "absolute_response_coefficient_mass": sum(abs(x) for x in credits.values()),
                "product_reward_sum": sum(p["reward"] for g in groups for p in g),
            }
        if committed:
            if batch["unknown_pairs"] or missing:
                raise ValueError("committed batch is incomplete")
            update, rollout = (
                audit.read(directory / "UPDATE.json"),
                audit.read(directory / "ROLLOUT.json"),
            )
            rewards = [[p["reward"] for p in g] for g in groups]
            if (
                update["response_advantages"] != advantages
                or rollout["response_advantages"] != advantages
                or update["rewards"] != rewards
                or rollout["rewards"] != rewards
                or update["estimator"] != "pairing_mean"
                or update["reward_objective"] != "product"
            ):
                raise ValueError("saved estimator/reward diagnostics differ")
            state = boundaries[cursor]["state"]
            updated = batch["pairing_mean"]["effective_groups"] > 0
            if (
                update["updated"] != updated
                or state["step"] != prior["state"]["step"] + int(updated)
                or state["zero_streak"] != (0 if updated else prior["state"]["zero_streak"] + 1)
            ):
                raise ValueError("actual Adam/zero-skip boundary differs")
            if updated and (
                not update["gradient_norm"] > 0 or not update["parameter_delta_l2"] > 0
            ):
                raise ValueError("claimed real Adam update lacks finite movement")
            if not updated and (
                update.get("optimizer_called") is not False or update["parameter_delta_l2"] != 0
            ):
                raise ValueError("zero-credit block changed Adam")
            batch["likelihood"] = validate_likelihood(update, credits, native)
            batch["updated"] = updated
        batches.append(batch)
    endpoint = boundaries[-1]["state"]
    if (summary["actual_optimizer_steps"], summary["committed_sampled_blocks"]) != (
        endpoint["step"],
        endpoint["sample_cursor"],
    ):
        raise ValueError("summary differs from committed endpoint")
    return {
        "schema": "sufficiency-pairing-mean-training-audit-v1",
        "plan": plan,
        "summary": summary,
        "terminals": terminals,
        "batches": batches,
        "native_status_counts": dict(statuses),
        "physical_cost": shared.paired.analyze_helper.measured(physical),
        "source_sha256": audit.hashes,
        "caveat": (
            "Product rewards remain separate from pairing-mean response credit. "
            "TRAIN audit only; no held claim."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        shared.training.BASE, local_files_only=True, trust_remote_code=False
    )
    report = analyze(args.output, tokenizer)
    shared.paired.baseline.native.save(args.report, report)
    print(json.dumps({"report": str(args.report)}))
