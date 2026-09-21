"""Native receipt audit and exploratory game-paired ALFWorld screen readout."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import alfworld_probe as collector
import analyze_helper

probe = collector.probe
SEED = 2026092180


def summarize(plan, rows, calls, *, draws=20000):
    jobs = {j["episode_id"]: j for j in plan["cases"]}
    indexed = {r["episode_id"]: r for r in rows}
    if len(indexed) != len(rows) or not indexed.keys() <= jobs.keys():
        raise ValueError("duplicate/unplanned episodes")
    if any(r["won"] and not r["observed"] for r in rows):
        raise ValueError("unobserved episode cannot be a verified win")
    policies = {}
    for policy in plan["policies"]:
        selected = [r for r in rows if r["policy"] == policy]
        known = [r for r in selected if r["observed"]]
        count = sum(j["policy"] == policy for j in jobs.values())
        policy_calls = [r for r in calls if r["request"]["condition"] == policy]
        policies[policy] = dict(
            planned=count,
            recorded=len(selected),
            observed=len(known),
            missing=count - len(selected),
            unavailable=sum(not r["observed"] for r in selected),
            unknown=count - len(known),
            won=sum(r["won"] for r in known),
            observed_nonwins=sum(not r["won"] for r in known),
            success_lower_bound=sum(r["won"] for r in known) / count,
            success_upper_bound=(sum(r["won"] for r in known) + count - len(known)) / count,
            terminations=dict(Counter(r["termination"] for r in selected)),
            invalid_policy_outputs=sum(r["invalid_outputs"] for r in selected),
            environment_actions=sum(r["actions"] for r in selected),
            native_cost=analyze_helper.measured(policy_calls),
            by_role={
                role: analyze_helper.measured(
                    [c for c in policy_calls if c["request"]["role"] == role]
                )
                for role in ("flat", "manager", "worker")
            },
        )
    lookup = {(r["game_index"], r["seed"], r["policy"]): r for r in rows}
    pairs, changes, game_differences = [], Counter(), {}
    for game in sorted({j["game_index"] for j in jobs.values()}):
        differences = []
        for seed in plan["seeds"]:
            flat, manager = [lookup.get((game, seed, p)) for p in plan["policies"]]
            if not flat or not manager or not flat["observed"] or not manager["observed"]:
                continue
            delta = int(manager["won"]) - int(flat["won"])
            changes["manager_win" if delta > 0 else "manager_loss" if delta < 0 else "tie"] += 1
            pairs.append(dict(game_index=game, seed=seed, manager_minus_flat=delta))
            differences.append(delta)
        if len(differences) == len(plan["seeds"]):
            game_differences[str(game)] = mean(differences)
    all_games = len({j["game_index"] for j in jobs.values()})
    interval = (
        analyze_helper.clustered_interval(
            game_differences, [[p] for p in game_differences], draws, SEED
        )
        if len(game_differences) == all_games
        else None
    )
    return dict(
        policies=policies,
        paired=dict(
            complete_pairs=len(pairs),
            complete_games=len(game_differences),
            planned_games=all_games,
            changes=dict(changes),
            pairs=pairs,
            game_bootstrap=interval,
        ),
        method=dict(
            bootstrap_draws=draws,
            bootstrap_seed=SEED,
            unit="game, with both seeds together",
            caveat="Eight exposed seen-development games; no verified scene/component "
            "independence. Descriptive exploratory uncertainty, not benchmark superiority.",
        ),
        physical_cost=analyze_helper.measured(calls),
    )


def audit_episode(job, row, calls, read, output, plan):
    """Reconstruct public prompts, native actions, refresh/seed policy; no environment run."""
    indexed_interface = plan.get("schema") == "alfworld-closed-loop-indexed-v1"
    if indexed_interface:
        import alfworld_closed_loop as policy_module
    else:
        policy_module = collector
    identity = job["episode_id"]
    if any(row[k] != job[k] for k in job):
        raise ValueError("episode differs from planned identity")
    initial_path = output / "observations" / (identity + "-000.json")
    if not initial_path.exists():
        if row["observed"] or row["actions"] or row["call_ids"]:
            raise ValueError("missing native reset")
        return dict(audited_calls=0, actions=0, invalid=0)
    event = read(initial_path)
    if (
        set(event["public"]) != {"feedback", "admissible_commands"}
        or event["public"] != job["game"]["public"]
    ):
        raise ValueError("reset public projection/readiness differs")
    initial, current = event["public"]["feedback"], event["public"]
    history, goal, goal_at, actions, tokens, invalid, streak = [], "", -1, 0, 0, 0, 0
    action_attempt = manager_attempt = 0
    invalid_reasons = Counter()
    for index, cid in enumerate(row["call_ids"]):
        call = calls[cid]
        role = (
            "manager"
            if job["policy"] == "manager_worker" and (not goal or actions - goal_at >= 4)
            else "flat"
            if job["policy"] == "flat"
            else "worker"
        )
        seed = job["seed"] + (100000 + manager_attempt if role == "manager" else action_attempt)
        manager_attempt += role == "manager"
        action_attempt += role != "manager"
        cap = min(plan["request_cap"], plan["token_limit"] - tokens)
        trim, request = call["trimming"], call["request"]
        dropped = trim["dropped_history"]
        if not 0 <= dropped <= len(history) or trim["original_history"] != len(history):
            raise ValueError("history trimming receipt differs")
        prompt = policy_module.prompts(initial, current, history[dropped:], goal)[role]
        sampling = dict(
            temperature=0.5, top_p=1.0, top_k=0, max_new_tokens=cap, max_time=90.0, do_sample=True
        )
        if (
            cid != identity + f"-call-{index:03d}-{role}"
            or request["prompt"] != prompt
            or request["condition"] != job["policy"]
            or request["role"] != role
            or request["seed"] != seed
            or request["sampling"] != sampling
            or request["model"] != plan["model"]
            or request["adapter_enabled"]
            or request["adapter_sha256"] is not None
            or probe.runtime.digest(request) != call["request_digest"]
            or request["input_token_ids"] != call["input_token_ids"]
        ):
            raise ValueError("native public prompt/role/seed/model receipt differs")
        if not call["available"]:
            if row["observed"] or index != len(row["call_ids"]) - 1:
                raise ValueError("unavailable call masquerades as observed outcome")
            break
        usage = call["usage"]
        if (
            usage["prompt_tokens"] != len(call["input_token_ids"])
            or usage["completion_tokens"] != len(call["output_token_ids"])
            or not 1 <= usage["completion_tokens"] <= cap
        ):
            raise ValueError("native token usage differs")
        tokens += usage["completion_tokens"]
        path = output / "decisions" / (cid + ".json")
        if not path.exists() and not row["observed"]:
            # A bridge exception can occur between native inference and decision persistence.
            break
        decision = read(path)
        try:
            value = policy_module.parse_output(call["text"], role, current["admissible_commands"])
        except (ValueError, TypeError) as error:
            if decision["valid"]:
                raise ValueError("invalid native response recorded as valid") from None
            invalid += 1
            streak += 1
            category = (
                "invalid_index"
                if str(error).startswith("action_index must")
                else "inadmissible_command"
                if str(error).startswith("action is not")
                else "json_or_schema"
            )
            invalid_reasons[category] += 1
            if indexed_interface:
                if not decision.get("public_rejection_feedback") or decision["error"] != str(error):
                    raise ValueError("indexed rejection feedback receipt differs") from error
                history.append(
                    dict(
                        event="controller_rejection",
                        action=call["text"],
                        feedback="Controller rejected this policy response: "
                        + str(error)
                        + ". Environment state unchanged. "
                        "Use the current response schema and numbered list.",
                    )
                )
            continue
        if not decision["valid"] or decision["parsed"] != value:
            raise ValueError("saved decision differs from strict parser")
        streak = 0
        if role == "manager":
            goal, goal_at = value, actions
        else:
            actions += 1
            event = read(output / "observations" / (identity + f"-{actions:03d}.json"))
            if event["action"] != value or set(event["public"]) != {
                "feedback",
                "admissible_commands",
            }:
                raise ValueError("executed command/public projection differs")
            current = event["public"]
            history.append(dict(action=value, feedback=current["feedback"]))
    if row["observed"]:
        if (row["actions"], row["generated_tokens"], row["invalid_outputs"]) != (
            actions,
            tokens,
            invalid,
        ):
            raise ValueError("episode counters differ from native receipts")
        reason = row["termination"]
        valid_stop = {
            "native_done": bool(event["host"]["done"] or event["host"]["won"]),
            "token_budget": tokens >= plan["token_limit"],
            "action_budget": actions >= plan["action_limit"],
            "three_consecutive_invalid": streak >= 3,
        }
        if not valid_stop.get(reason, False) or row["won"] != bool(event["host"]["won"]):
            raise ValueError("observed outcome is not backed by native stop/won")
    return dict(
        audited_calls=len(row["call_ids"]),
        actions=actions,
        invalid=invalid,
        invalid_reasons=dict(invalid_reasons),
        final_public_feedback=current["feedback"],
        public_actions=[h["action"] for h in history],
    )


def analyze(output):
    output = Path(output).resolve()
    hashes = {}

    def read(path):
        hashes[str(path)] = probe.campaign.sha(path)
        return json.loads(path.read_text())

    plan = read(output / "PLAN.json")
    owners = list(output.glob("OWNER-*.json"))
    if not owners or {p.stem[6:] for p in owners} != {
        p.stem[9:] for p in output.glob("TERMINAL-*.json")
    }:
        raise ValueError("terminal owner required; no partial policy ranking")
    terminals = [read(p) for p in output.glob("TERMINAL-*.json")]
    for path, digest in plan["source_sha256"].items():
        if probe.campaign.sha(path) != digest:
            raise ValueError("sealed source changed")
        hashes[path] = digest
    if plan["policies"] != list(collector.POLICIES) or len(plan["cases"]) != 32:
        raise ValueError("expected fixed32-slot two-policy panel")
    inventory = {(j["game_index"], j["seed"], j["policy"]) for j in plan["cases"]}
    if inventory != {(g, s, p) for g in range(8) for s in plan["seeds"] for p in plan["policies"]}:
        raise ValueError("planned game/seed pairing differs")
    if probe.campaign.sha(plan["readiness_manifest"]) != plan["readiness_manifest_sha256"]:
        raise ValueError("readiness manifest changed")
    hashes[plan["readiness_manifest"]] = plan["readiness_manifest_sha256"]
    calls = {p.stem: read(p) for p in (output / "calls").glob("*.json")}
    rows = [read(p) for p in (output / "episodes").glob("*.json")]
    indexed = {r["episode_id"]: r for r in rows}
    audits = {
        j["episode_id"]: audit_episode(j, indexed[j["episode_id"]], calls, read, output, plan)
        for j in plan["cases"]
        if j["episode_id"] in indexed
    }
    starts = [read(p) for p in (output / "starts").glob("*.json")]
    unresolved = [s for s in starts if s["call_id"] not in calls]
    result = summarize(plan, rows, list(calls.values()) + unresolved)
    smoke_path = output / "SMOKE.json"
    smoke = read(smoke_path) if smoke_path.exists() else None
    if smoke and smoke["passed"] != all(
        indexed[j["episode_id"]]["actions"] > 0 for j in plan["cases"][:2]
    ):
        raise ValueError("smoke must use actions, not win")
    result.update(
        output=str(output),
        source_plan=plan,
        source_sha256=hashes,
        terminals=terminals,
        smoke=smoke,
        receipt_audits=audits,
        unresolved_starts=len(unresolved),
        unlinked_calls=sorted(calls.keys() - {c for r in rows for c in r["call_ids"]}),
        task_types_host_only={
            str(j["game_index"]): Path(j["game"]["game"]).parent.parent.name.split("-")[0]
            for j in plan["cases"]
        },
        public_admissible_assistance="Both policies see native admissible commands: affordance "
        "assistance, not syntax-only. Manager adds calls/goals under the same2048 token cap. "
        + (
            "Combined indexed-command and explicit rejection-feedback qualification; "
            "not an isolated mechanism claim."
            if plan.get("schema") == "alfworld-closed-loop-indexed-v1"
            else "Rejected outputs are not added to public history: "
            "no corrective rejection feedback."
        ),
        analyzer_sha256=probe.campaign.sha(__file__),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise ValueError("immutable report exists")
    result = analyze(args.output)
    probe.runtime.save(args.report, result)
    lines = [
        "# Exploratory ALFWorld screen",
        "",
        result["method"]["caveat"],
        "",
        result["public_admissible_assistance"],
        "",
    ]
    for policy, row in result["policies"].items():
        lines.append(
            f"{policy}: {row['won']}/{row['planned']} wins; {row['unknown']} unknown; "
            f"{row['environment_actions']} actions; "
            f"{row['invalid_policy_outputs']} invalid responses. "
            f"Stops: {row['terminations']}. Native cost (including managers): {row['native_cost']}."
        )
    lines.extend(["", f"Paired readout: {result['paired']}", "", f"Smoke: {result['smoke']}"])
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"report": str(args.report)}))


if __name__ == "__main__":
    main()
