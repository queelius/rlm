"""Source032 versus audited source026: exposed-game local deliberation comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import alfworld_local_reason as collector
import analyze_helper
import numpy as np

SEED = 2026092191
POLICIES = ("flat", "manager_worker", "local_reason")


def pair(rows, left, right, seeds, draws=20000):
    lookup = {(r["game_index"], r["seed"], r["policy"]): r for r in rows}
    pairs, values, unknown = [], [], 0
    for game in range(8):
        differences = []
        for seed in seeds:
            a, b = lookup.get((game, seed, left)), lookup.get((game, seed, right))
            if not a or not b or not a["observed"] or not b["observed"]:
                unknown += 1
                continue
            delta = int(a["won"]) - int(b["won"])
            differences.append(delta)
            pairs.append(
                {
                    "game_index": game,
                    "seed": seed,
                    "delta": delta,
                    "left_won": a["won"],
                    "right_won": b["won"],
                }
            )
        if len(differences) == 2:
            values.append(mean(differences))
    interval = None
    if len(values) == 8:
        samples = np.array(values)[np.random.default_rng(SEED).integers(0, 8, (draws, 8))].mean(1)
        interval = {"estimate": mean(values), "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}
    return {
        "left": left,
        "right": right,
        "planned_pairs": 16,
        "unobserved_pairs": unknown,
        "wins": sum(p["delta"] > 0 for p in pairs),
        "losses": sum(p["delta"] < 0 for p in pairs),
        "ties": sum(p["delta"] == 0 for p in pairs),
        "pairs": pairs,
        "game_bootstrap": interval,
    }


def audit_episode(job, row, calls, read, output, plan, tokenizer):
    if any(row[k] != v for k, v in job.items()):
        raise ValueError("episode planned identity changed")
    identity = job["episode_id"]
    reset = output / "observations" / (identity + "-000.json")
    if not reset.exists():
        if row["observed"] or row["call_ids"] or row["actions"]:
            raise ValueError("missing reset for observed episode")
        return {"actions": 0, "generated_tokens": 0, "reason_characters": [], "incomplete": True}
    event = read(reset)
    if event["public"] != job["game"]["public"] or set(event["public"]) != {
        "feedback",
        "admissible_commands",
    }:
        raise ValueError("public reset differs from frozen game")
    initial, current = event["public"]["feedback"], event["public"]
    history, reasons, stops, trim_counts, output_lengths, error_counts = (
        [],
        [],
        Counter(),
        [],
        [],
        Counter(),
    )
    actions = tokens = invalid = streak = 0
    client = collector.base.BaseClient(None, tokenizer, output, 0)
    for i, cid in enumerate(row["call_ids"]):
        call = calls[cid]
        cap = min(128, 2048 - tokens)
        prompt, trim = collector.bounded_prompt(client, initial, current, history, cap)
        request = {
            "prompt": prompt,
            "input_token_ids": client.ids(prompt),
            "condition": "local_reason",
            "role": "local_reason",
            "model": plan["model"],
            "adapter_enabled": False,
            "adapter_sha256": None,
            "seed": job["seed"] + i,
            "sampling": {
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": cap,
                "max_time": 90.0,
                "do_sample": True,
            },
        }
        if (
            cid != identity + f"-call-{i:03d}-local_reason"
            or call["request"] != request
            or call["request_digest"] != collector.probe.runtime.digest(request)
            or call["input_token_ids"] != request["input_token_ids"]
            or call["trimming"] != trim
        ):
            raise ValueError("native request/seed/prompt/retained-history differs")
        trim_counts.append(trim["dropped_history"])
        if not call["available"]:
            if row["observed"] or i != len(row["call_ids"]) - 1:
                raise ValueError("unavailable call cannot become an observed outcome")
            break
        ids = call["output_token_ids"]
        if (
            call["usage"]
            != {"prompt_tokens": len(request["input_token_ids"]), "completion_tokens": len(ids)}
            or not 1 <= len(ids) <= cap
            or tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            != call["text"]
        ):
            raise ValueError("native decoding/token accounting differs")
        tokens += len(ids)
        output_lengths.append(len(ids))
        eos = ids[-1] == tokenizer.eos_token_id
        if call["finish_reason"] != ("eos" if eos else "length_or_time"):
            raise ValueError("native EOS receipt differs")
        stops["eos" if eos else "at_output_cap" if len(ids) == cap else "non_eos_before_cap"] += 1
        path = output / "decisions" / (cid + ".json")
        if not path.exists() and not row["observed"]:
            break
        decision = read(path)
        try:
            reason, action = collector.parse_output(call["text"], current["admissible_commands"])
        except (ValueError, TypeError) as error:
            if (
                decision["valid"]
                or decision["error"] != str(error)
                or not decision["public_rejection_feedback"]
            ):
                raise ValueError("native protocol rejection differs") from error
            invalid += 1
            streak += 1
            error_counts[str(error)] += 1
            history.append(
                {
                    "event": "controller_rejection",
                    "action": call["text"],
                    "feedback": "Controller rejected this policy response: "
                    + str(error)
                    + ". Environment state unchanged. "
                    "Use the current response schema and numbered list.",
                }
            )
        else:
            if (
                not decision["valid"]
                or decision["parsed"] != action
                or decision["reason"] != reason
                or decision["reason_retained_in_history"] is not False
            ):
                raise ValueError("saved action/reason differs from native parser")
            reasons.append(len(reason))
            streak = 0
            actions += 1
            event = read(output / "observations" / (identity + f"-{actions:03d}.json"))
            if event["action"] != action or set(event["public"]) != {
                "feedback",
                "admissible_commands",
            }:
                raise ValueError("native action/public projection differs")
            current = event["public"]
            history.append({"action": action, "feedback": current["feedback"]})
        if i + 1 < len(row["call_ids"]) and (
            event["host"]["done"]
            or event["host"]["won"]
            or tokens >= 2048
            or actions >= 50
            or streak >= 3
        ):
            raise ValueError("calls after terminal budget/environment state")
    if row["observed"]:
        if (row["actions"], row["generated_tokens"], row["invalid_outputs"]) != (
            actions,
            tokens,
            invalid,
        ):
            raise ValueError("episode counters differ")
        valid_stop = {
            "native_done": event["host"]["done"] or event["host"]["won"],
            "token_budget": tokens >= 2048,
            "action_budget": actions >= 50,
            "three_consecutive_invalid": streak >= 3,
        }
        if not valid_stop.get(row["termination"], False) or row["won"] != bool(
            event["host"]["won"]
        ):
            raise ValueError("observed native won/stop differs")
    elif row["won"]:
        raise ValueError("unobserved cannot claim native success")
    return {
        "actions": actions,
        "generated_tokens": tokens,
        "invalid_outputs": invalid,
        "reason_characters": reasons,
        "native_stop_counts": dict(stops),
        "output_token_lengths": output_lengths,
        "history_dropped_per_call": trim_counts,
        "invalid_reasons": dict(error_counts),
        "public_actions": [h["action"] for h in history],
        "final_public_feedback": current["feedback"],
    }


def analyze(root, output, baseline_report, draws=20000):
    root, output, baseline_report = map(
        lambda p: Path(p).resolve(), (root, output, baseline_report)
    )
    hashes, cache = {}, {}

    def read(path):
        path = Path(path).resolve()
        if str(path) not in cache:
            data = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(data).hexdigest()
            cache[str(path)] = json.loads(data)
        return cache[str(path)]

    def terminal(directory):
        import psutil

        owners = list(directory.glob("OWNER-*.json"))
        if len(owners) != 1:
            raise ValueError("one completed owner required")
        owner = read(owners[0])
        result = read(owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-")))
        try:
            p = psutil.Process(owner["pid"])
            if (
                abs(p.create_time() - owner["create_time"]) < 0.01
                and p.status() != psutil.STATUS_ZOMBIE
            ):
                raise ValueError("owner still active; no partial ranking")
        except psutil.NoSuchProcess:
            pass
        return result

    new_terminal = terminal(output)
    old_output = root / "alfworld-closed-loop-001"
    old_terminal = terminal(old_output)
    old = read(baseline_report)
    if (
        Path(old["output"]) != old_output
        or old["source_plan"]["schema"] != "alfworld-closed-loop-indexed-v1"
    ):
        raise ValueError("requires source026 independently audited baseline")
    for path, expected in old["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("audited baseline input changed: " + path)
        hashes[path] = expected
    plan, original = read(output / "PLAN.json"), read(old_output / "PLAN.json")
    if (
        plan["schema"] != "alfworld-local-reason-v1"
        or plan["source_plan_sha256"] != collector.SOURCE_PLAN_SHA
        or hashes[str(old_output / "PLAN.json")] != collector.SOURCE_PLAN_SHA
        or plan["seeds"] != original["seeds"]
        or len(plan["cases"]) != 16
    ):
        raise ValueError("planned source/pairing differs")
    for key in (
        "action_limit",
        "token_limit",
        "request_cap",
        "context_limit",
        "goal_reserve",
        "model",
        "model_manifest_sha256",
        "readiness_manifest_sha256",
        "adapters",
        "training",
    ):
        if plan[key] != original[key]:
            raise ValueError("scientific budget/model/environment mismatch: " + key)
    flat = {(j["game_index"], j["seed"]): j for j in original["cases"] if j["policy"] == "flat"}
    if {(j["game_index"], j["seed"]) for j in plan["cases"]} != set(flat):
        raise ValueError("not all sixteen original slots")
    for job in plan["cases"]:
        if job["game"] != flat[job["game_index"], job["seed"]]["game"]:
            raise ValueError("game changed")
    for path, expected in plan["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("source032 dependency changed")
        hashes[path] = expected
    source_match = [
        h for p, h in plan["source_sha256"].items() if Path(p).name == Path(collector.__file__).name
    ]
    if source_match != [collector.probe.campaign.sha(collector.__file__)]:
        raise ValueError("analysis collector differs from frozen source032")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = Path(plan["model"]) / name
        hashes[str(path)] = collector.probe.campaign.sha(path)
    new_calls = {p.stem: read(p) for p in (output / "calls").glob("*.json")}
    starts = {p.stem: read(p) for p in (output / "starts").glob("*.json")}
    for cid, call in new_calls.items():
        if cid not in starts or any(
            call[k] != starts[cid][k] for k in ("request", "request_digest", "started")
        ):
            raise ValueError("native start/return differs")
    new_rows = [read(p) for p in (output / "episodes").glob("*.json")]
    jobs = {j["episode_id"]: j for j in plan["cases"]}
    if len({e["episode_id"] for e in new_rows}) != len(new_rows) or any(
        e["episode_id"] not in jobs for e in new_rows
    ):
        raise ValueError("unexpected/duplicate episode")
    audits = {
        e["episode_id"]: audit_episode(
            jobs[e["episode_id"]], e, new_calls, read, output, plan, tokenizer
        )
        for e in new_rows
    }
    old_rows = [read(p) for p in (old_output / "episodes").glob("*.json")]
    old_calls = [read(p) for p in (old_output / "calls").glob("*.json")]
    if {e["episode_id"] for e in old_rows} != set(old["receipt_audits"]):
        raise ValueError("baseline episode inventory differs from consumed audit")
    audited_call_ids = {
        Path(p).stem for p in old["source_sha256"] if Path(p).parent == old_output / "calls"
    }
    if {c["call_id"] for c in old_calls} != audited_call_ids:
        raise ValueError("baseline call inventory differs from consumed audit")
    unresolved = [s for cid, s in starts.items() if cid not in new_calls]
    rows = old_rows + new_rows
    all_calls = old_calls + list(new_calls.values()) + unresolved
    policies = {}
    for policy in POLICIES:
        episodes = [e for e in rows if e["policy"] == policy]
        calls = [c for c in all_calls if c["request"]["condition"] == policy]
        lengths = [c.get("usage", {}).get("completion_tokens") for c in calls]
        policies[policy] = {
            "planned": 16,
            "recorded": len(episodes),
            "observed": sum(e["observed"] for e in episodes),
            "unknown": 16 - sum(e["observed"] for e in episodes),
            "won": sum(e["won"] and e["observed"] for e in episodes),
            "actions": sum(e["actions"] for e in episodes),
            "generated_tokens": sum(e["generated_tokens"] for e in episodes),
            "invalid_outputs": sum(e["invalid_outputs"] for e in episodes),
            "terminations": dict(Counter(e["termination"] for e in episodes)),
            "physical_cost": analyze_helper.measured(calls),
            "max_prompt_tokens": max((len(c["input_token_ids"]) for c in calls), default=0),
            "output_token_lengths": lengths,
            "calls_at_output_cap": sum(
                c.get("usage", {}).get("completion_tokens")
                == c["request"]["sampling"]["max_new_tokens"]
                for c in calls
            ),
            "history_trimmed_calls": sum(c["trimming"]["dropped_history"] > 0 for c in calls),
            "max_history_entries_dropped": max(
                (c["trimming"]["dropped_history"] for c in calls), default=0
            ),
        }
    for path in (Path(__file__), Path(collector.__file__), Path(analyze_helper.__file__)):
        hashes[str(path)] = collector.probe.campaign.sha(path)
    return {
        "source_output": str(output),
        "baseline_report": str(baseline_report),
        "policies": policies,
        "comparisons": [
            pair(rows, a, b, plan["seeds"], draws)
            for a, b in (
                ("local_reason", "flat"),
                ("local_reason", "manager_worker"),
                ("manager_worker", "flat"),
            )
        ],
        "method": {
            "bootstrap_draws": draws,
            "seed": SEED,
            "unit": "eight games with both seeds",
            "warning": "Exposed development games; no scene independence or fresh confirmation.",
        },
        "native_audits": audits,
        "accepted_reason_character_lengths": [
            n for a in audits.values() for n in a["reason_characters"]
        ],
        "rows": rows,
        "new_physical_cost": analyze_helper.measured(list(new_calls.values()) + unresolved),
        "unresolved_new_starts": sorted(set(starts) - set(new_calls)),
        "unlinked_new_calls": sorted(
            set(new_calls) - {cid for e in new_rows for cid in e["call_ids"]}
        ),
        "terminals": {"local_reason": new_terminal, "source026": old_terminal},
        "source_and_receipt_sha256": hashes,
        "limitations": [
            "All emitted reason tokens are charged, including invalid outputs.",
            "Accepted reasons are not persistent; rejected raw output remains public feedback.",
            "Affordance-assisted exposed games; not a benchmark-level or novelty claim.",
            "Prompt/schema, token allocation, and sequential reasoning differ together.",
            "Source026 audit is reused by receipt hashes; source032 is independently replayed.",
            "Missing outcomes are unknown; returned invalid/budget outcomes are observed nonwins.",
        ],
    }


def markdown(report):
    lines = [
        "# ALFWorld: local deliberation control",
        "",
        report["method"]["warning"],
        "",
        "| Policy | Won / 16 | Unknown | Actions | Invalid | Calls | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, p in report["policies"].items():
        cost = p["physical_cost"]
        lines.append(
            f"| {policy} | {p['won']} | {p['unknown']} | {p['actions']} | "
            f"{p['invalid_outputs']} | {cost['calls']} | "
            f"{cost['prompt_tokens'] + cost['completion_tokens']} |"
        )
    lines += ["", "Game-paired comparisons (both seeds stay together):", ""]
    for c in report["comparisons"]:
        lines.append(
            f"- {c['left']} minus {c['right']}: {c['wins']} wins / {c['losses']} losses; "
            f"{c['unobserved_pairs']} unknown pairs; interval {c['game_bootstrap']}."
        )
    lines += ["", "Limitations:", "", *["- " + x for x in report["limitations"]]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    report = analyze(args.root, args.output, args.baseline_report)
    collector.save(args.report, report)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(report))
