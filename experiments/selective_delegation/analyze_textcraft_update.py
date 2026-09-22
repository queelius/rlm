"""CPU same-prefix likelihood movement after a committed TextCraft RLOO update."""

import argparse
import json
import math
from pathlib import Path

import analyze_textcraft_credit as census

read, sha = census.read, census.sha


def likelihood_rows(calls, before, after, advantages):
    active = {cid for cid, value in advantages.items() if value != 0}
    if set(before) != set(calls) or set(after) != active:
        raise ValueError("exact before/all and after/nonzero call coverage required")
    result = {}
    for cid in sorted(active):
        left, right = before[cid], after[cid]
        n = len(calls[cid]["output_token_ids"])
        if len(left) != n or len(right) != n or not all(map(math.isfinite, left + right)):
            raise ValueError("finite native emitted-token logp length required")
        delta = sum(b - a for a, b in zip(left, right, strict=True))
        result[cid] = dict(
            tokens=n,
            before_sequence_logp=sum(left),
            after_sequence_logp=sum(right),
            sequence_logp_delta=delta,
            mean_token_logp_delta=delta / n,
            advantage_weighted_delta=advantages[cid] * delta,
            improved_in_advantage_direction=advantages[cid] * delta > 0,
            unchanged=delta == 0,
        )
    return result


def adam_steps(path):
    import torch

    # Map optimizer storage read-only; touch scalar step tensors only, not model/Adam moments.
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return sorted({int(row["step"].item()) for row in state["state"].values() if "step" in row})


def aggregate(rows):
    tokens = sum(row["tokens"] for row in rows)
    return dict(
        calls=len(rows),
        episodes=len({r["episode_id"] for r in rows}),
        tasks=len({r["task_id"] for r in rows}),
        tokens=tokens,
        sequence_logp_delta_sum=sum(r["sequence_logp_delta"] for r in rows),
        token_weighted_mean_delta=(
            sum(r["sequence_logp_delta"] for r in rows) / tokens if tokens else None
        ),
        advantage_weighted_delta_sum=sum(r["advantage_weighted_delta"] for r in rows),
        loss_objective_improvement=sum(r["advantage_weighted_delta"] for r in rows) / 32,
        calls_improved_in_advantage_direction=sum(
            r["improved_in_advantage_direction"] for r in rows
        ),
        calls_unchanged=sum(r["unchanged"] for r in rows),
    )


def analyze(output, sample):
    plan_path = output / "PLAN.json"
    plan = read(plan_path)
    if plan.get("schema") != "textcraft-terminal-rloo-two-update-v1" or not 1 <= sample <= 4:
        raise ValueError("bounded terminal065 sample required")
    batch_dir = output / "batches" / f"sample-{sample:04d}"
    boundary_path = output / "boundaries" / f"sample-{sample:04d}" / "BOUNDARY.json"
    required = [
        boundary_path,
        batch_dir / "BATCH.json",
        batch_dir / "BEFORE_LOGPS.json",
        batch_dir / "AFTER_LOGPS.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return dict(
            status="incomplete",
            sample=sample,
            missing=missing,
            message="No zero substitution or partial likelihood estimates.",
            training_plan_sha256=sha(plan_path),
            analyzer_sha256=sha(Path(__file__)),
        )
    boundary = read(boundary_path)
    endpoint = Path(boundary["checkpoint"])
    if endpoint.parent.resolve() != boundary_path.parent.resolve():
        raise ValueError("checkpoint outside selected committed boundary")
    commit_path, state_path = endpoint / "COMMIT.json", endpoint / "STATE.json"
    commit, state = read(commit_path), read(state_path)
    if (
        sha(commit_path) != boundary["commit_sha256"]
        or sha(state_path) != commit["files"]["STATE.json"]
        or state != boundary["state"]
        or state["sample_cursor"] != sample
        or state["plan_sha256"] != sha(plan_path)
        or commit["step"] != state["step"]
        or state["batch_sha256"] != sha(batch_dir / "BATCH.json")
    ):
        raise ValueError("committed PLAN/batch/state identity differs")
    batch = read(batch_dir / "BATCH.json")
    data = Path(batch["data"])
    calls, histories = {}, {}
    for filename, digest in batch["native_receipt_sha256"].items():
        path = Path(filename)
        if path.parent.name in ("calls", "nodes", "episodes"):
            if sha(path) != digest:
                raise ValueError("native batch receipt changed")
            row = read(path)
            if path.parent.name == "calls":
                calls[path.stem] = row
            elif path.parent.name == "nodes":
                if row["depth"] != 0:
                    raise ValueError("flat actor only")
                for cid, history in zip(row["call_ids"], row["public_history"], strict=True):
                    if cid in histories:
                        raise ValueError("duplicate native call history")
                    histories[cid] = history
    episodes = batch["episodes"]
    credits = census.loss_math.batch_credits(episodes, calls)
    advantages = {item.call_id: item.advantage for item in credits}
    if set(histories) != set(calls):
        raise ValueError("native history inventory differs")
    before, after = read(batch_dir / "BEFORE_LOGPS.json"), read(batch_dir / "AFTER_LOGPS.json")
    if Path(after["endpoint"]).resolve() != endpoint.resolve():
        raise ValueError("after likelihoods belong to a different endpoint")
    movement = likelihood_rows(calls, before["logps"], after["logps"], advantages)
    update = state["update"]
    tokens = sum(len(calls[cid]["output_token_ids"]) for cid in movement)
    if (
        update["optimizer_called"] is not bool(movement)
        or update["nonzero_action_calls"] != len(movement)
        or (movement and update["train_eval_replay_token_count"] != tokens)
    ):
        raise ValueError("committed update/credit token count mismatch")
    actual_steps = adam_steps(endpoint / "optimizer.pt")
    if (actual_steps or [0]) != [state["step"]]:
        raise ValueError("actual saved Adam scalar step differs from committed STATE")
    if movement:
        marker = read(batch_dir / "OPTIMIZER-STEP-STARTED.json")
        if marker["previous_step"] + 1 != state["step"]:
            raise ValueError("actual optimizer update counter differs")
    episode_map = {row["episode_id"]: row for row in episodes}
    rows = []
    for cid, value in movement.items():
        call, history = calls[cid], histories[cid]
        if "action" in history:
            action = census.bridge.parse_action(call["text"])
            if action != history["action"]:
                raise ValueError("native response/action mismatch")
            name = action["action"]
        else:
            if call["text"] != history["response"]:
                raise ValueError("invalid response/history mismatch")
            name = "unparsed"
        episode = episode_map[call["episode_id"]]
        rows.append(
            dict(
                **value,
                call_id=cid,
                episode_id=call["episode_id"],
                task_id=episode["task_id"],
                reward=episode["native_score"],
                advantage=advantages[cid],
                action=name,
                error=census.error_type(history),
                sign="positive" if advantages[cid] > 0 else "negative",
            )
        )
    norm, threshold = update.get("gradient_norm"), plan["clip_grad_norm"]
    groups = {
        task: dict(
            aggregate=aggregate([r for r in rows if r["task_id"] == task]),
            episode_rewards=[e["native_score"] for e in episodes if e["task_id"] == task],
            post_update_omitted_zero_advantage=not any(r["task_id"] == task for r in rows),
        )
        for task in sorted({e["task_id"] for e in episodes})
    }
    return dict(
        status="complete",
        sample=sample,
        step=state["step"],
        endpoint=str(endpoint),
        actual_adam_steps=actual_steps,
        optimizer_update=update,
        gradient_clip_threshold=threshold,
        preclip_norm_exceeded_threshold=None if norm is None else norm > threshold,
        expected_global_clip_coefficient=None
        if norm is None
        else min(1.0, threshold / (norm + 1e-6)),
        all_planned=dict(
            tasks=8,
            episodes=32,
            calls=len(calls),
            tokens=sum(len(c["output_token_ids"]) for c in calls.values()),
        ),
        credited=aggregate(rows),
        groups=groups,
        by_action_error_sign={
            f"{action}/{error}/{sign}": aggregate(
                [r for r in rows if (r["action"], r["error"], r["sign"]) == (action, error, sign)]
            )
            for action, error, sign in sorted({(r["action"], r["error"], r["sign"]) for r in rows})
        },
        call_rows=rows,
        before_generation_replay={k: v for k, v in before.items() if k != "logps"},
        zero_credit_calls_without_after=len(calls) - len(rows),
        sha256={str(path): sha(path) for path in [plan_path, *required, commit_path, state_path]},
        optimizer_sha256_from_commit_not_rehashed=commit["files"]["optimizer.pt"],
        source_sha256={
            str(Path(m.__file__)): sha(Path(m.__file__))
            for m in (census, census.bridge, census.loss_math)
        },
        analyzer_sha256=sha(Path(__file__)),
        native_data=str(data),
        caveat="Same saved native prefixes/emitted tokens atT0.5 before/after, not new "
        "on-policy rollouts or held improvement. Flat-group after logps not recorded "
        "and not imputed. Per-task aggregates retain all8; active tasks are clusters, "
        "not independent calls. Correlation is not causal action value. Clip "
        "coefficient is inferred from recorded norm/threshold, not separately logged.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    result = analyze(args.output.resolve(), args.sample)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write("# TextCraft committed update likelihood audit\n\n")
        if result["status"] != "complete":
            stream.write(json.dumps(result) + "\n")
        else:
            stream.write(result["caveat"] + "\n\n")
            for key in (
                "sample",
                "step",
                "actual_adam_steps",
                "optimizer_update",
                "credited",
                "groups",
                "by_action_error_sign",
            ):
                stream.write(key + ": " + json.dumps(result[key]) + "\n\n")
