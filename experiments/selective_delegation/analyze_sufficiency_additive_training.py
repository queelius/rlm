"""Terminal-only CPU audit of additive diagonal RLOO; product EM stays separate."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import analyze_sufficiency_rl as shared


def grade_pair(case, predictions, *, observed):
    if not observed:
        return None
    baseline = shared.paired.baseline
    p = int(baseline.group_score(case, [predictions[0], {"answerable": False, "answer": ""}])["em"])
    n = int(predictions[1] is not None and predictions[1]["answerable"] is False)
    product = int(baseline.group_score(case, predictions)["em"])
    if product != p * n:
        raise ValueError("official paired reward does not factor into declared marginals")
    return {
        "positive_success": p,
        "negative_success": n,
        "reward": product,
        "training_reward": (p + n) / 2,
        "all_valid": all(p is not None for p in predictions),
    }


def rloo(values):
    return [x - (sum(values) - x) / 3 for x in values]


def credit_statistics(groups):
    if len(groups) != 16 or any(len(g) != 4 or any(p is None for p in g) for g in groups):
        raise ValueError("complete16x4 observed pairs required; unknown is not zero")
    product = [[p["reward"] for p in g] for g in groups]
    additive = [[p["training_reward"] for p in g] for g in groups]
    credits = {
        f"p{i:02d}-k{k}-v{v}": a
        for i, g in enumerate(additive)
        for k, a in enumerate(rloo(g))
        for v in (0, 1)
        if a != 0
    }
    return {
        "paired_denominator": 64,
        "valid_pairs": sum(p["all_valid"] for g in groups for p in g),
        "product_reward_sum": sum(map(sum, product)),
        "additive_training_reward_sum": sum(map(sum, additive)),
        "product_effective_groups": sum(any(rloo(g)) for g in product),
        "additive_effective_groups": sum(any(rloo(g)) for g in additive),
        "nonzero_response_coefficients": len(credits),
        "absolute_response_coefficient_mass": sum(map(abs, credits.values())),
        "positive_successes": sum(p["positive_success"] for g in groups for p in g),
        "negative_successes": sum(p["negative_success"] for g in groups for p in g),
        "active_groups_without_positive_success": sum(
            any(rloo(a)) and not any(p["positive_success"] for p in g)
            for g, a in zip(groups, additive, strict=True)
        ),
    }, credits


def likelihood_statistics(update, credits, native):
    rows = update.get("likelihoods", [])
    if len(rows) != len(credits) or {r["call_id"] for r in rows} != set(credits):
        raise ValueError("credited response inventory differs")
    gaps, sequence_gaps = [], []
    for row in rows:
        call = native[row["call_id"]]
        if (
            row["advantage"] != credits[row["call_id"]]
            or row["generation"] != call["generation_logps"]
            or len(row["before"]) != len(call["output_token_ids"])
            or len(row["after"]) != len(row["before"])
            or not all(
                math.isfinite(x) for key in ("before", "after", "generation") for x in row[key]
            )
        ):
            raise ValueError("native emitted-token/advantage/likelihood alignment differs")
        gaps.extend(abs(x - y) for x, y in zip(row["before"], row["generation"], strict=True))
        sequence_gaps.append(
            {
                "call_id": row["call_id"],
                "advantage": row["advantage"],
                "signed_gap": sum(row["before"]) - sum(row["generation"]),
            }
        )
    loss = sum(-r["advantage"] * sum(r["before"]) / 64 for r in rows)
    if rows and (
        not math.isclose(loss, update["loss"], rel_tol=1e-5, abs_tol=1e-4)
        or update["credited_responses"] != len(rows)
        or not math.isclose(max(gaps), update["generation_replay_max_abs_gap"], abs_tol=1e-7)
    ):
        raise ValueError("recorded loss/credited count/replay gap differs")
    gaps.sort()
    return {
        "loss_recomputed": loss,
        "credited_responses": len(rows),
        "credited_tokens": sum(len(r["before"]) for r in rows),
        "advantage_weighted_logp_movement": sum(
            r["advantage"] * (sum(r["after"]) - sum(r["before"])) for r in rows
        ),
        "token_abs_replay_gap": {
            str(q): gaps[round(q * (len(gaps) - 1))] if gaps else None for q in (0.5, 0.95, 1.0)
        },
        "sequence_sum_replay_gaps": sequence_gaps,
    }


def adapter_delta(before, after):
    from safetensors import safe_open

    total = 0.0
    scalars = 0
    with (
        safe_open(before, framework="pt", device="cpu") as a,
        safe_open(after, framework="pt", device="cpu") as b,
    ):
        keys = list(a.keys())
        if keys != list(b.keys()) or not keys or any("lora_" not in k for k in keys):
            raise ValueError("saved adapter tensor inventory is not matching LoRA-only state")
        for key in keys:
            left, right = a.get_tensor(key), b.get_tensor(key)
            if left.shape != right.shape:
                raise ValueError("adapter tensor shape changed")
            total += float((right.double() - left.double()).square().sum())
            scalars += left.numel()
    return {"l2": math.sqrt(total), "tensors": len(keys), "scalars": scalars}


def optimizer_steps(path, expected):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    values = {int(s["step"]) for s in state["state"].values()}
    if values != ({expected} if expected else set()) or any(
        group["lr"] != 2e-5 or group["weight_decay"] != 0 for group in state["param_groups"]
    ):
        raise ValueError("actual Adam state differs from recorded real updates/configuration")
    return {"parameter_states": len(state["state"]), "step_values": sorted(values)}


def analyze(output, tokenizer):
    output = output.resolve()
    audit = shared.Audit()
    terminals = audit.terminal(output)
    plan = audit.read(output / "PLAN.json")
    summary = audit.read(output / "SUMMARY.json")
    if (plan["mode"], plan.get("reward_objective"), plan.get("estimator")) != (
        "rl",
        "additive",
        "diagonal",
    ):
        raise ValueError("source051 additive diagonal RL only")
    if (
        plan["max_sampled_blocks"] != 8
        or plan["pair_denominator"] != 64
        or plan["max_calls"] != 1024
    ):
        raise ValueError("finite8block/64pair/1024call contract differs")
    for mapping in (plan["source_sha256"], plan["official_metric_sha256"]):
        for path, digest in mapping.items():
            audit.hash(path, digest)
    for owner in output.glob("OWNER-*.json"):
        audit.hash(audit.read(owner)["source"], plan["runner_sha256"])
    boundaries = shared.training.boundary_inventory(output)
    commits = [audit.checkpoint(b) for b in boundaries]
    if not boundaries or boundaries[-1]["checkpoint"] != summary["endpoint"]:
        raise ValueError("summary does not identify last committed endpoint")
    for b in boundaries:
        if b["state"]["plan_sha256"] != audit.hashes[str(output / "PLAN.json")]:
            raise ValueError("boundary PLAN identity differs")
    cases = audit.cases(plan["cases"], plan["cases_sha256"])
    lookup = shared.training.paired_cases(cases, sum(plan["parent_blocks"], []))
    batches, calls, unresolved, statuses = [], [], [], Counter()
    recorded_pairs = Counter()
    directories = sorted((output / "batches").glob("sample-*"))
    for directory in directories:
        cursor = int(directory.name.split("-")[-1])
        if not 1 <= cursor <= 8 or cursor > len(boundaries):
            raise ValueError("sample cursor has no preceding committed policy")
        prior, commit = boundaries[cursor - 1], commits[cursor - 1]
        groups, native, expected_ids = [], {}, set()
        missing_pair_receipts = 0
        for i, parent in enumerate(plan["parent_blocks"][cursor - 1]):
            group = []
            for k in range(4):
                predictions, observed, ids = [], True, []
                for v, case in enumerate(lookup[parent]):
                    identity = f"p{i:02d}-k{k}-v{v}"
                    ids.append(identity)
                    expected_ids.add(identity)
                    path = directory / "calls" / f"{identity}.json"
                    status, prediction = "missing", None
                    if path.exists():
                        row = audit.read(path)
                        native[identity] = row
                        calls.append(row)
                        if row["call_id"] != identity:
                            raise ValueError("native call identity differs")
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
                graded = grade_pair(lookup[parent][0], predictions, observed=observed)
                path = directory / "pairs" / f"p{i:02d}-k{k}.json"
                if path.exists():
                    pair = audit.read(path)
                    if (
                        graded is None
                        or any(pair[key] != value for key, value in graded.items())
                        or (
                            pair["predictions"] != predictions
                            or pair["call_ids"] != ids
                            or pair["parent_id"] != parent
                            or pair["candidate"] != k
                        )
                    ):
                        raise ValueError(
                            "saved marginal/product/additive reward differs from native grade"
                        )
                    recorded_pairs.update(
                        recorded_pairs=1,
                        valid_pairs=int(graded["all_valid"]),
                        paired_reward_sum=graded["reward"],
                    )
                else:
                    missing_pair_receipts += 1
                group.append(graded)
            groups.append(group)
        if any(p.stem not in expected_ids for p in (directory / "calls").glob("*.json")):
            raise ValueError("unplanned native call")
        unresolved.extend(
            audit.read(p) for p in (directory / "starts").glob("*.json") if p.stem not in native
        )
        committed = cursor < len(boundaries)
        batch = {
            "sample_cursor": cursor,
            "committed": committed,
            "unknown_pairs": sum(p is None for g in groups for p in g),
            "missing_pair_receipts": missing_pair_receipts,
        }
        if not batch["unknown_pairs"]:
            stats, credits = credit_statistics(groups)
            batch["recomputed"] = stats
        if committed:
            if batch["unknown_pairs"] or missing_pair_receipts:
                raise ValueError("committed batch contains unknown/missing pair")
            update = audit.read(directory / "UPDATE.json")
            rollout = audit.read(directory / "ROLLOUT.json")
            product = [[p["reward"] for p in g] for g in groups]
            additive = [[p["training_reward"] for p in g] for g in groups]
            advantages = [[[a, a] for a in rloo(g)] for g in additive]
            for receipt in (update, rollout):
                if (
                    receipt["rewards"] != product
                    or receipt["training_rewards"] != additive
                    or receipt["response_advantages"] != advantages
                    or receipt["effective_groups"] != stats["additive_effective_groups"]
                    or receipt["parents"] != plan["parent_blocks"][cursor - 1]
                    or receipt["reward_objective"] != "additive"
                    or receipt["estimator"] != "diagonal"
                ):
                    raise ValueError("rollout/update selected additive credit differs")
            state = boundaries[cursor]["state"]
            updated = bool(credits)
            if (
                update["updated"] != updated
                or rollout["updated"] != updated
                or state["step"] != prior["state"]["step"] + int(updated)
                or state["zero_streak"] != (0 if updated else prior["state"]["zero_streak"] + 1)
                or state["effective_groups"] != stats["additive_effective_groups"]
            ):
                raise ValueError("sample/real-update/zero-Adam cursor differs")
            before, after = Path(prior["checkpoint"]), Path(boundaries[cursor]["checkpoint"])
            for cp, cmt in ((before, commit), (after, commits[cursor])):
                for name in ("adapter_model.safetensors", "optimizer.pt"):
                    if str(cp / name) not in audit.hashes:
                        audit.hash(cp / name, cmt["files"][name])
            delta = adapter_delta(
                before / "adapter_model.safetensors", after / "adapter_model.safetensors"
            )
            adam = optimizer_steps(after / "optimizer.pt", state["step"])
            if updated:
                if (
                    not update["gradient_norm"] > 0
                    or not delta["l2"] > 0
                    or not math.isclose(
                        delta["l2"], update["parameter_delta_l2"], rel_tol=1e-6, abs_tol=1e-8
                    )
                ):
                    raise ValueError("claimed gradient/adapter movement differs from saved tensors")
            elif (
                update.get("optimizer_called") is not False
                or delta["l2"] != 0
                or commit["files"]["optimizer.pt"] != commits[cursor]["files"]["optimizer.pt"]
            ):
                raise ValueError("zero-credit batch changed Adam or adapter")
            batch.update(
                updated=updated,
                gradient_norm=update.get("gradient_norm"),
                adapter_delta=delta,
                actual_adam=adam,
                likelihood=likelihood_statistics(update, credits, native),
            )
        batches.append(batch)
    last = boundaries[-1]["state"]
    if any(
        summary[key] != recorded_pairs[key]
        for key in ("recorded_pairs", "valid_pairs", "paired_reward_sum")
    ):
        raise ValueError("summary paired product counts differ from native saved pairs")
    if (summary["reward_objective"], summary["estimator"]) != ("additive", "diagonal"):
        raise ValueError("summary objective differs")
    if (summary["actual_optimizer_steps"], summary["committed_sampled_blocks"]) != (
        last["step"],
        last["sample_cursor"],
    ):
        raise ValueError("summary update/cursor differs from committed endpoint")
    terminal = max(terminals, key=lambda t: t["sample_cursor"])
    if (
        terminal["endpoint"] != summary["endpoint"]
        or (terminal["step"], terminal["sample_cursor"]) != (last["step"], last["sample_cursor"])
        or terminal["endpoint_selection"] != "last committed boundary, never held score"
    ):
        raise ValueError("terminal differs from committed endpoint")
    return {
        "schema": "additive-sufficiency-training-audit-v1",
        "output": str(output),
        "plan": plan,
        "summary": summary,
        "terminals": terminals,
        "boundaries": boundaries,
        "batches": batches,
        "native_status_counts": dict(statuses),
        "unresolved_started_calls": len(unresolved),
        "max_planned_calls": 1024,
        "not_sampled_after_terminal_slots": 128 * (8 - len(directories)),
        "physical_cost": shared.paired.analyze_helper.measured(calls + unresolved),
        "source_sha256": audit.hashes,
        "caveat": "Changing TRAIN blocks are not a learning curve. Additive reward is not "
        "paired EM. BF16 cached/full replay is approximate, not numerically identical. "
        "No held inference or improvement claim follows from this audit.",
    }


def markdown(report):
    lines = [
        "# Additive sufficiency TRAIN audit",
        "",
        report["caveat"],
        "",
        f"Native statuses: {report['native_status_counts']}; "
        f"unresolved starts: {report['unresolved_started_calls']}.",
        f"Physical cost: {report['physical_cost']}",
        "",
        "| Block | Committed | Product successes /64 | Additive reward sum /64 | "
        "Active groups | Gradient | Actual adapter delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for b in report["batches"]:
        s = b.get("recomputed", {})
        lines.append(
            f"| {b['sample_cursor']} | {b['committed']} | {s.get('product_reward_sum')} | "
            f"{s.get('additive_training_reward_sum')} | {s.get('additive_effective_groups')} | "
            f"{b.get('gradient_norm')} | {b.get('adapter_delta', {}).get('l2')} |"
        )
    lines += [
        "",
        f"Actual optimizer updates: {report['summary']['actual_optimizer_steps']}; "
        f"committed sampled blocks: {report['summary']['committed_sampled_blocks']}; "
        f"not-sampled future slots: {report['not_sampled_after_terminal_slots']}. "
        "Missing/inference-unavailable is unknown, never a fabricated reward-zero sample.",
    ]
    return "\n".join(lines) + "\n"


def main(args):
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        shared.training.BASE, local_files_only=True, trust_remote_code=False
    )
    report = analyze(args.output, tokenizer)
    report["analyzer_sha256"] = {
        str(Path(m.__file__).resolve()): shared.paired.baseline.panel.sha256(Path(m.__file__))
        for m in (shared, shared.paired)
    }
    report["analyzer_sha256"][str(Path(__file__).resolve())] = shared.paired.baseline.panel.sha256(
        Path(__file__)
    )
    shared.paired.baseline.native.save(args.report, report)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(report))
    print(json.dumps({"report": str(args.report)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    main(parser.parse_args())
