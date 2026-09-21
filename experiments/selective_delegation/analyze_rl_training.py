"""Read-only diagnostics for a frozen prefix of completed planner-RLOO batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import analyze_helper
import eval_planner
import probe


def group_summary(rows):
    rewards = [r["reward"] for r in rows]
    if len(rows) != 4 or any(r not in (0, 1) for r in rewards):
        raise ValueError("expected four binary-reward trajectories")
    valid_rewards = {r["reward"] for r in rows if r["plan_valid"]}
    scored_rewards = {r["reward"] for r in rows if r["status"] == "scored"}
    distinct = {
        json.dumps(r["plan"]["subquestions"], ensure_ascii=False) for r in rows if r["plan_valid"]
    }
    mixed = len(set(rewards)) == 2
    return {
        "rewards": rewards,
        "advantages": [(4 * r - sum(rewards)) / 3 for r in rewards],
        "mean_reward": mean(rewards),
        "reward_variance": mean((r - mean(rewards)) ** 2 for r in rewards),
        "valid_plans": sum(r["plan_valid"] for r in rows),
        "distinct_valid_plans": len(distinct),
        "mixed_rewards": mixed,
        "mixed_valid_plan_rewards": len(valid_rewards) == 2,
        "qualifies": len(distinct) >= 2 and len(valid_rewards) == 2,
        "fully_scored_reward_variation": len(scored_rewards) == 2,
        "variation_includes_protocol_zeros": mixed and any(r["status"] != "scored" for r in rows),
        "protocol_only_reward_variation": mixed and len(scored_rewards) < 2,
    }


def likelihood_movement(before, after, advantages):
    if not len(before) == len(after) == len(advantages):
        raise ValueError("likelihood trajectory count differs")
    if any(len(a) != len(b) for a, b in zip(before, after, strict=True)):
        raise ValueError("likelihood emitted-token alignment differs")
    changes = [sum(b) - sum(a) for a, b in zip(before, after, strict=True)]
    positive = [d for a, d in zip(advantages, changes, strict=True) if a > 0]
    negative = [d for a, d in zip(advantages, changes, strict=True) if a < 0]
    return {
        "advantage_weighted_before_per_64": sum(
            a * sum(p) for a, p in zip(advantages, before, strict=True)
        )
        / 64,
        "advantage_weighted_after_per_64": sum(
            a * sum(p) for a, p in zip(advantages, after, strict=True)
        )
        / 64,
        "advantage_weighted_delta_per_64": sum(
            a * d for a, d in zip(advantages, changes, strict=True)
        )
        / 64,
        "positive_advantage_mean_sum_logp_change": mean(positive) if positive else None,
        "negative_advantage_mean_sum_logp_change": mean(negative) if negative else None,
        "positive_advantage_logp_increased": sum(d > 0 for d in positive),
        "negative_advantage_logp_decreased": sum(d < 0 for d in negative),
    }


def analyze(output, cases_path, *, comparison_output=None):
    output, cases_path = Path(output).resolve(), Path(cases_path).resolve()
    cutoff = datetime.now(timezone.utc).isoformat()
    hashes, cache = {}, {}

    def track(path, expected=None):
        path = str(Path(path).resolve())
        if path not in cache:
            cache[path] = Path(path).read_bytes()
            hashes[path] = hashlib.sha256(cache[path]).hexdigest()
        if expected is not None and hashes[path] != expected:
            raise ValueError("immutable input hash differs: " + path)
        return cache[path]

    def read(path):
        return json.loads(track(path))

    plan = read(output / "PLAN.json")
    for path, expected in plan.get("dependencies", {}).items():
        if Path(path).suffix in (".py", ".json"):
            track(path, expected)
    cases = {
        c["id"]: c
        for c in (
            json.loads(line)
            for line in track(cases_path, plan["cases_sha256"]).splitlines()
            if line.strip()
        )
    }
    # Freeze membership before analysis: an actively completing batch cannot enter later.
    ready, excluded = [], []
    for batch in sorted(output.glob("batch-*")):
        update = int(batch.name.split("-")[1])
        checkpoint = output / f"checkpoint-{update:04d}"
        if not (batch / "BATCH.json").exists():
            excluded.append(batch.name)
            continue
        manifest = read(batch / "BATCH.json")
        if manifest["admitted"] and not all(
            p.exists()
            for p in (
                checkpoint / "COMMIT.json",
                checkpoint / "STATE.json",
                batch / "BEFORE_LOGPS.json",
                batch / "AFTER_LOGPS.json",
            )
        ):
            excluded.append(batch.name)
            continue
        ready.append((update, batch, checkpoint, manifest))
    if not ready or [u for u, *_ in ready] != list(range(1, len(ready) + 1)):
        raise ValueError("no contiguous completed batch prefix")

    def load_batch(source, source_plan, update, batch, checkpoint, manifest):
        parents = (
            source_plan["case_ids_by_update"][update - 1]
            if "case_ids_by_update" in source_plan
            else source_plan["case_ids"]
        )
        if len(parents) != 16 or any(cases[p]["split"] != "train" for p in parents):
            raise ValueError("expected 16 TRAIN parents")
        contract = source_plan.get("helper_contract", {"mode": "base"})
        if contract["mode"] == "trained_helper" and not contract["weights_frozen"]:
            raise ValueError("helper contract not frozen")
        root_binding = source_plan["adapter_binding"]["adapter_model.safetensors"]
        if update > 1:
            previous = read(source / f"checkpoint-{update - 1:04d}/COMMIT.json")
            root_binding = previous["files"]["adapter_model.safetensors"]
        calls, episodes = {}, {}
        for path in sorted((batch / "calls").glob("*.json")):
            call = read(path)
            if call["call_id"] != path.stem or call["call_id"] in calls:
                raise ValueError("unexpected native call identity")
            request, role = call["request"], call["role"]
            if probe.runtime.digest(request) != call["request_digest"]:
                raise ValueError("native request digest differs")
            enabled = role == "root" or (role == "helper" and contract["mode"] == "trained_helper")
            binding = (
                root_binding
                if role == "root"
                else (contract["adapter_binding"]["adapter_model.safetensors"] if enabled else None)
            )
            model = contract["model"] if role == "helper" and enabled else source_plan["model"]
            if (
                request["role"] != role
                or request["adapter_enabled"] != enabled
                or call["adapter_enabled"] != enabled
                or request["adapter_sha256"] != binding
                or request["model"] != model
            ):
                raise ValueError("native model/adapter role differs from frozen contract")
            if "model_instance" in request and request["model_instance"] != (
                "helper" if role == "helper" and enabled else "root"
            ):
                raise ValueError("model instance routing differs")
            if "helper_contract" in request and request["helper_contract"] != contract:
                raise ValueError("native helper contract differs")
            for field, ids in (
                ("prompt_tokens", "input_token_ids"),
                ("completion_tokens", "output_token_ids"),
            ):
                if (
                    ids in call
                    and field in call.get("usage", {})
                    and call["usage"][field] != len(call[ids])
                ):
                    raise ValueError("native token usage differs")
            calls[call["call_id"]] = call
        for path in sorted((batch / "episodes").glob("*.json")):
            row = read(path)
            key = row["case_id"], row["candidate"]
            if key in episodes or row["update"] != update:
                raise ValueError("duplicate/mismatched rollout")
            episodes[key] = row
        if set(episodes) != {(p, candidate) for p in parents for candidate in range(4)}:
            raise ValueError("batch rollout inventory incomplete")
        roots, ordered, groups, linked = [], [], [], set()
        for parent_index, p in enumerate(parents):
            group = []
            for candidate in range(4):
                row = dict(episodes[p, candidate])
                seed_base = source_plan["seed"] + update * 100000 + parent_index * 1000
                if row["seeds"] != {"root": seed_base + candidate, "downstream": seed_base + 100}:
                    raise ValueError("rollout seed schedule differs")
                records = [calls[cid] for cid in row["call_ids"]]
                for call in records:
                    role, request = call["role"], call["request"]
                    expected_seed = (
                        row["seeds"]["root"]
                        if role == "root"
                        else (
                            row["seeds"]["downstream"]
                            + (2 if role == "final" else int(call["call_id"].rsplit("-", 1)[1]))
                        )
                    )
                    temperature = (
                        source_plan["root_temperature"]
                        if role == "root"
                        else source_plan["helper_final_temperature"]
                    )
                    cap = 384 // len(row["plan"]["subquestions"]) if role == "helper" else 128
                    if (
                        request["seed"] != expected_seed
                        or request["temperature"] != temperature
                        or request["max_new_tokens"] != cap
                        or request["top_p"] != 1.0
                        or request["top_k"] != 0
                    ):
                        raise ValueError("native rollout sampling differs from fixed contract")
                if linked.intersection(row["call_ids"]):
                    raise ValueError("reused physical call")
                linked.update(row["call_ids"])
                root_calls = [c for c in records if c["role"] == "root"]
                finals = [c for c in records if c["role"] == "final"]
                if len(root_calls) != 1 or len(finals) > 1:
                    raise ValueError("root/final inventory differs")
                root = root_calls[0]
                if root["request"]["prompt"] != eval_planner.planner_prompt(cases[p]):
                    raise ValueError("root public prompt differs")
                try:
                    parsed = eval_planner.parse_plan(root["text"]) if root["available"] else None
                except (ValueError, TypeError):
                    parsed = None
                if bool(parsed) != row["plan_valid"] or (parsed and parsed != row["plan"]):
                    raise ValueError("native root parse disagrees with rollout")
                final = finals[0] if finals else None
                grade = probe.grade(final["text"] if final and final["available"] else "", cases[p])
                if row["reward"] != int(grade["correct"]):
                    raise ValueError("native EM differs from training reward")
                row["regraded_f1"] = grade["f1"]
                if row["status"] == "scored" and not grade["valid"]:
                    raise ValueError("scored rollout lacks valid native final")
                roots.append(root)
                ordered.append(row)
                group.append(row)
            summary = group_summary(group)
            stored = manifest["groups"][parent_index]
            for key in ("rewards", "valid_plans", "distinct_valid_plans", "qualifies"):
                if summary[key] != stored[key]:
                    raise ValueError("regraded group differs from batch diagnostics: " + key)
            if any(
                abs(a - b) > 1e-12
                for a, b in zip(summary["advantages"], stored["advantages"], strict=True)
            ):
                raise ValueError("RLOO advantages differ")
            groups.append({"case_id": p, "hops": cases[p]["metadata"]["hops"], **summary})
        for path in sorted((batch / "starts").glob("*.json")):
            if read(path)["call_id"] not in calls:
                raise ValueError("unresolved native start in completed batch")
        advantages = [a for g in groups for a in g["advantages"]]
        state, commit, movement = None, None, None
        if manifest["admitted"]:
            commit = read(checkpoint / "COMMIT.json")
            state = json.loads(track(checkpoint / "STATE.json", commit["files"]["STATE.json"]))
            if commit["step"] != update or state["step"] != update:
                raise ValueError("committed optimizer step differs")
            track(batch / "BATCH.json", state["batch_sha256"])
            before, after = read(batch / "BEFORE_LOGPS.json"), read(batch / "AFTER_LOGPS.json")
            if before["call_ids"] != after["call_ids"] or before["call_ids"] != [
                r["call_id"] for r in roots
            ]:
                raise ValueError("likelihood/root trajectory alignment differs")
            if after["adapter_sha256"] != commit["files"]["adapter_model.safetensors"]:
                raise ValueError("post-update likelihood checkpoint binding differs")
            if any(
                len(v) != len(r["output_token_ids"])
                for v, r in zip(before["logps"], roots, strict=True)
            ):
                raise ValueError("likelihood does not cover all emitted root tokens")
            movement = likelihood_movement(before["logps"], after["logps"], advantages)
            movement["generation_replay_max_abs_difference"] = before[
                "generation_full_forward_max_abs_difference"
            ]
        reward = mean(r["reward"] for r in ordered)
        if abs(reward - manifest["mean_reward"]) > 1e-12:
            raise ValueError("batch reward mean differs")
        report = {
            "update": update,
            "parent_ids": parents,
            "groups": groups,
            "parent_hops": dict(Counter(str(g["hops"]) for g in groups)),
            "pre_update_mean_reward": reward,
            "reward_distribution": dict(Counter(str(r["reward"]) for r in ordered)),
            "status_counts": dict(Counter(r["status"] for r in ordered)),
            "valid_plans": sum(r["plan_valid"] for r in ordered),
            "distinct_valid_lists_per_parent_distribution": dict(
                Counter(str(g["distinct_valid_plans"]) for g in groups)
            ),
            "plan_length_counts": dict(
                Counter(str(len(r["plan"]["subquestions"])) for r in ordered if r["plan_valid"])
            ),
            "group_counts": {
                k: sum(g[k] for g in groups)
                for k in (
                    "qualifies",
                    "mixed_rewards",
                    "mixed_valid_plan_rewards",
                    "fully_scored_reward_variation",
                    "variation_includes_protocol_zeros",
                    "protocol_only_reward_variation",
                )
            },
            "nonzero_advantages_per_64": sum(a != 0 for a in advantages),
            "native_cost": analyze_helper.measured(list(calls.values())),
            "role_counts": dict(Counter(c["role"] for c in calls.values())),
            "unlinked_call_ids": sorted(set(calls) - linked),
            "committed_optimizer_state": state,
            "checkpoint_commit": commit,
            "likelihood_movement": movement,
            "role_contract_verified": True,
        }
        return report, ordered, roots

    batches, first = [], None
    for update, batch, checkpoint, manifest in ready:
        report, ordered, roots = load_batch(output, plan, update, batch, checkpoint, manifest)
        batches.append(report)
        if first is None:
            first = ordered, roots
    comparison = None
    if comparison_output is not None:
        old = Path(comparison_output).resolve()
        old_plan = read(old / "PLAN.json")
        comparison = {"source": str(old), "eligible": False}
        if old_plan["cases_sha256"] != plan["cases_sha256"]:
            comparison["reason"] = "case hashes differ"
        else:
            old_report, old_rows, old_roots = load_batch(
                old,
                old_plan,
                1,
                old / "batch-0001",
                old / "checkpoint-0001",
                read(old / "batch-0001/BATCH.json"),
            )
            fields = (
                "prompt",
                "input_token_ids",
                "role",
                "model",
                "adapter_enabled",
                "adapter_sha256",
                "seed",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "max_new_tokens",
                "max_time",
            )
            matched = all(
                (a["case_id"], a["candidate"], a["seeds"])
                == (b["case_id"], b["candidate"], b["seeds"])
                and all(x["request"][k] == y["request"][k] for k in fields)
                and x["text"] == y["text"]
                and x["output_token_ids"] == y["output_token_ids"]
                for a, b, x, y in zip(first[0], old_rows, first[1], old_roots, strict=True)
            )
            comparison.update(
                eligible=matched,
                matched_root_requests_and_outputs=sum(
                    all(x["request"][k] == y["request"][k] for k in fields)
                    and x["output_token_ids"] == y["output_token_ids"]
                    for x, y in zip(first[1], old_roots, strict=True)
                ),
            )
            if matched:
                changes = Counter()
                for a, b in zip(first[0], old_rows, strict=True):
                    delta = a["reward"] - b["reward"]
                    if delta:
                        category = (
                            "both_scored"
                            if a["status"] == b["status"] == "scored"
                            else "protocol_involved"
                        )
                        changes[("wins_" if delta > 0 else "losses_") + category] += 1
                comparison.update(
                    old_reward=old_report["pre_update_mean_reward"],
                    new_reward=batches[0]["pre_update_mean_reward"],
                    changes=dict(changes),
                    old_statuses=old_report["status_counts"],
                    new_statuses=batches[0]["status_counts"],
                    old_helper_contract=old_plan.get("helper_contract", {"mode": "base"}),
                    new_helper_contract=plan.get("helper_contract", {"mode": "base"}),
                    interpretation="Matched frozen-root TRAIN helper intervention before either "
                    "first update; not an RL gain.",
                )
            else:
                comparison["reason"] = (
                    "root/case/candidate/seed identity failed; no paired helper-effect estimate"
                )
    for path in (
        Path(__file__),
        Path(analyze_helper.__file__),
        Path(eval_planner.__file__),
        Path(probe.__file__),
        probe.MUSIQUE / "metrics/answer.py",
    ):
        track(path)
    return {
        "source": str(output),
        "cutoff_utc": cutoff,
        "source_plan": plan,
        "completed_prefix_updates": len(batches),
        "excluded_incomplete_batches": excluded,
        "batches": batches,
        "old_first_batch_comparison": comparison,
        "input_source_receipt_sha256": hashes,
        "cautions": [
            "Rewards are sampled before each update. Consecutive parent blocks change task "
            "composition; their reward sequence is not a learning curve or held-out improvement.",
            "Mixed scored rewards show end-to-end variation among protocol-valid episodes, "
            "not proof of semantic plan quality. Full-source finals can bypass or repair helpers.",
            "Likelihood movement is on the same sampled training trajectories: sum of all "
            "emitted-token log probabilities at temperature0.8, advantage weighted, divided by64. "
            "It verifies update direction, not generalization; exact-list diversity is not "
            "semantic diversity.",
            "Role checks verify request/model/adapter bindings against frozen contracts and "
            "committed manifests. Multi-GB weights and optimizer states are not reread or "
            "executed.",
            "Only committed batches with after-update receipts, or complete no-admission "
            "batches, enter this frozen prefix. Incomplete live batches are not scored "
            "as failures.",
        ],
    }


def markdown(report):
    lines = [
        "# Planner RL: completed training-prefix diagnostics",
        "",
        f"Source: `{report['source']}`; cutoff {report['cutoff_utc']}.",
        "",
        "Pre-update training rewards below are not a learning curve or a held-out result.",
        "",
        "| Batch | Pre-update EM | Mixed groups | Fully-scored variation | "
        "Protocol-only variation | "
        "Nonzero advantages /64 | Calls | Weighted logp movement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["batches"]:
        counts, movement = row["group_counts"], row["likelihood_movement"]
        delta = f"{movement['advantage_weighted_delta_per_64']:+.6f}" if movement else "no update"
        lines.append(
            f"| {row['update']} | {row['pre_update_mean_reward']:.2%} | "
            f"{counts['mixed_rewards']} | {counts['fully_scored_reward_variation']} | "
            f"{counts['protocol_only_reward_variation']} | {row['nonzero_advantages_per_64']} | "
            f"{row['native_cost']['calls']} | {delta} |"
        )
    lines += ["", "Per-batch execution and update checks:", ""]
    for row in report["batches"]:
        state = row["committed_optimizer_state"]
        detail = (
            (
                f"step {state['step']}, pre-clip gradient norm {state['gradient_norm']:.4f}, "
                f"adapter L2 change {state['adapter_l2_delta']:.6f}"
            )
            if state
            else "no optimizer update"
        )
        lines.append(
            f"- Batch {row['update']}: parent hops {row['parent_hops']}; "
            f"statuses {row['status_counts']}; plan lengths {row['plan_length_counts']}; "
            f"{row['native_cost']['total_tokens']} tokens; {detail}."
        )
    if report["old_first_batch_comparison"]:
        comparison = report["old_first_batch_comparison"]
        lines += ["", "Old first-batch comparison:", "", json.dumps(comparison, sort_keys=True), ""]
    lines += [
        "",
        f"Excluded incomplete batches: {report['excluded_incomplete_batches']}.",
        "",
        *["- " + caution for caution in report["cautions"]],
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "cases", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path)
    args = parser.parse_args()
    sibling = args.report.with_suffix(".md")
    if args.report.exists() or sibling.exists() or args.report == sibling:
        parser.error("choose unused distinct JSON/Markdown report paths")
    report = analyze(args.output, args.cases, comparison_output=args.comparison_output)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with sibling.open("x") as stream:
        stream.write(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
