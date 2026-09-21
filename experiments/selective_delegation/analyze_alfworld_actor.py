"""Fixed action-SFT versus base: paired flat/manager interaction on twelve games."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import alfworld_trained_actor as collector
import analyze_alfworld_unseen as unseen

runtime = collector.probe.runtime
POLICIES = ("base_flat", "base_manager", "trained_flat", "trained_manager")


def adapter_neutral_projection(row, checkpoint, adapter_sha):
    """Authenticate native role metadata, then project only for the base-state auditor."""
    req = row["request"]
    expected = collector.role_identity(req["role"], adapter_sha)
    if (
        req["adapter_checkpoint"] != checkpoint
        or any(req[k] != v or row[k] != v for k, v in expected.items())
        or runtime.digest(req) != row["request_digest"]
    ):
        raise ValueError("native adapter role/checkpoint/digest differs")
    projected = {k: v for k, v in req.items() if k != "adapter_checkpoint"}
    projected.update(adapter_enabled=False, adapter_sha256=None)
    return {**row, "request": projected, "request_digest": runtime.digest(projected)}


def interaction(rows, games, seeds, draws=20000):
    lookup = {(r["game_index"], r["seed"], r["policy"]): r for r in rows}
    pairs, values, scenes, unknown = [], {}, defaultdict(list), 0
    for game, meta in games.items():
        deltas = []
        for seed in seeds:
            cells = {p: lookup.get((game, seed, p)) for p in POLICIES}
            if any(r is None or not r["observed"] for r in cells.values()):
                unknown += 1
                continue
            wins = {p: int(r["won"]) for p, r in cells.items()}
            delta = (
                wins["trained_manager"]
                - wins["trained_flat"]
                - wins["base_manager"]
                + wins["base_flat"]
            )
            pairs.append({"game_index": game, "seed": seed, "cells": wins, "delta": delta})
            deltas.append(delta)
        if len(deltas) == len(seeds):
            values[str(game)] = mean(deltas)
        scenes[meta["scene"]].append(str(game))
    complete = len(values) == len(games)
    clusters = [[str(g)] for g in games]
    return {
        "definition": "(trained_manager-trained_flat)-(base_manager-base_flat)",
        "planned_pairs": len(games) * len(seeds),
        "unknown_pairs": unknown,
        "pairs": pairs,
        "game_mean_interactions": values,
        "game_bootstrap": unseen.analyze_helper.clustered_interval(
            values, clusters, draws, unseen.SEED
        )
        if complete
        else None,
        "scene_bootstrap": unseen.analyze_helper.clustered_interval(
            values, list(scenes.values()), draws, unseen.SEED
        )
        if complete
        else None,
        "game_signflip": unseen.signflip(values, clusters) if complete else None,
        "scene_signflip": unseen.signflip(values, list(scenes.values())) if complete else None,
    }


def analyze(output, baseline_report):
    output, baseline_report = Path(output).resolve(), Path(baseline_report).resolve()
    hashes, cache = {}, {}

    def read(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in cache:
            data = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(data).hexdigest()
            cache[str(path)] = json.loads(data)
        if expected is not None and hashes[str(path)] != expected:
            raise ValueError("consumed receipt changed: " + str(path))
        return cache[str(path)]

    import psutil

    owners = list(output.glob("OWNER-*.json"))
    if len(owners) != 1:
        raise ValueError("one terminal owner required")
    owner = read(owners[0])
    terminal_path = owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-"))
    terminal = read(terminal_path)
    try:
        process = psutil.Process(owner["pid"])
        if (
            abs(process.create_time() - owner["create_time"]) < 0.01
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            raise ValueError("owner still live; no partial ranking")
    except psutil.NoSuchProcess:
        pass
    plan, old = read(output / "PLAN.json"), read(baseline_report)
    base_output = Path(plan["baseline_output"])
    bp = read(base_output / "PLAN.json", plan["baseline_plan_sha256"])
    if (
        old["output"] != str(base_output)
        or old["plan"] != bp
        or plan["schema"] != "alfworld-fixed-action-adapter-readout-v1"
        or plan["cases"] != [j for j in bp["cases"] if j["policy"] in collector.POLICIES]
        or plan["policies"] != list(collector.POLICIES)
        or plan["planned_episodes"] != 48
        or plan["role_adapter"]
        != {"flat": "enabled", "worker": "enabled", "manager": "disabled_base"}
    ):
        raise ValueError("fixed two-by-two inventory differs")
    for key in (
        "seeds",
        "action_limit",
        "token_limit",
        "request_cap",
        "context_limit",
        "alfworld_python",
        "data_root",
    ):
        if plan[key] != bp[key]:
            raise ValueError("source035 matched contract differs: " + key)
    source = Path(owner["source"]).parent
    seal = read(source / "SOURCE.json")
    for name, expected in seal["files"].items():
        path = source / name
        if collector.probe.campaign.sha(path) != expected:
            raise ValueError("sealed source042 changed")
        hashes[str(path)] = expected
    if (
        collector.probe.campaign.sha(Path(collector.__file__))
        != seal["files"]["alfworld_trained_actor.py"]
    ):
        raise ValueError("analyzer role contract differs from source042")
    checkpoint = Path(plan["checkpoint"])
    if checkpoint != collector.CHECKPOINT or checkpoint.name != "checkpoint-0033":
        raise ValueError("fixed checkpoint0033 required")
    commit = read(checkpoint / "COMMIT.json", plan["checkpoint_identity"]["commit_sha256"])
    state = read(checkpoint / "STATE.json", commit["files"]["STATE.json"])
    read(checkpoint / "adapter_config.json", commit["files"]["adapter_config.json"])
    training = read(checkpoint.parent / "PLAN.json")
    if (
        (commit["step"], state["step"], state["epoch"], state["cursor"]) != (33, 33, 1, 0)
        or training["schema"] != "alfworld-public-action-sft-v1"
        or training["planned_updates"] != 33
        or training["epochs"] != 1
        or training["base_manifest_sha256"] != bp["model_manifest_sha256"]
        or commit["files"]["adapter_model.safetensors"]
        != plan["checkpoint_identity"]["adapter_sha256"]
    ):
        raise ValueError("fixed complete action-SFT endpoint differs")
    # The collector authenticated weight bytes once. Reuse the immutable COMMIT binding here.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        bp["model"], local_files_only=True, trust_remote_code=False
    )
    client = collector.unseen.local.base.BaseClient(None, tokenizer, output, 0)
    jobs = {j["episode_id"]: j for j in plan["cases"]}
    calls = {p.stem: read(p) for p in (output / "calls").glob("*.json")}
    starts = {p.stem: read(p) for p in (output / "starts").glob("*.json")}
    projected = {}
    expected_keys = {
        "prompt",
        "input_token_ids",
        "condition",
        "role",
        "model",
        "adapter_checkpoint",
        "adapter_enabled",
        "adapter_sha256",
        "seed",
        "sampling",
    }
    for cid, call in calls.items():
        if cid not in starts or any(
            call[k] != starts[cid][k] for k in ("request", "request_digest", "started")
        ):
            raise ValueError("native start/return differs")
        if set(call["request"]) != expected_keys or call["input_token_ids"] != client.ids(
            call["request"]["prompt"]
        ):
            raise ValueError("native request fields/tokenization differ")
        if (
            call["role"] != call["request"]["role"]
            or call["condition"] != call["request"]["condition"]
        ):
            raise ValueError("native role/condition differs")
        projected[cid] = adapter_neutral_projection(
            call, str(checkpoint), plan["checkpoint_identity"]["adapter_sha256"]
        )
        adapter_neutral_projection(
            starts[cid], str(checkpoint), plan["checkpoint_identity"]["adapter_sha256"]
        )
        if call["available"]:
            ids = call["output_token_ids"]
            if (
                tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                != call["text"]
            ):
                raise ValueError("native output decode differs")
            if call["finish_reason"] != (
                "eos" if ids[-1] == tokenizer.eos_token_id else "length_or_time"
            ):
                raise ValueError("native EOS differs")
    new_rows = [read(p) for p in (output / "episodes").glob("*.json")]
    if len({r["episode_id"] for r in new_rows}) != len(new_rows) or any(
        r["episode_id"] not in jobs for r in new_rows
    ):
        raise ValueError("unplanned/duplicate trained episode")
    audits = {
        r["episode_id"]: unseen.indexed_audit.audit_episode(
            jobs[r["episode_id"]],
            r,
            projected,
            read,
            output,
            {**bp, "schema": "alfworld-closed-loop-indexed-v1"},
        )
        for r in new_rows
    }
    base_rows, base_calls = [], []
    for row in old["rows"]:
        if row["policy"] not in collector.POLICIES:
            continue
        path = base_output / "episodes" / (row["episode_id"] + ".json")
        if read(path, old["source_and_receipt_sha256"][str(path)]) != row:
            raise ValueError("baseline episode differs from independently audited row")
        base_rows.append(row)
        for cid in row["call_ids"]:
            path = base_output / "calls" / (cid + ".json")
            base_calls.append(read(path, old["source_and_receipt_sha256"][str(path)]))
    rows = [
        {**r, "policy": prefix + ("flat" if r["policy"] == "flat" else "manager")}
        for prefix, records in (("base_", base_rows), ("trained_", new_rows))
        for r in records
    ]
    games = {j["game_index"]: j["game"] for j in jobs.values()}
    unresolved = [r for cid, r in starts.items() if cid not in calls]
    for row in unresolved:
        adapter_neutral_projection(
            row, str(checkpoint), plan["checkpoint_identity"]["adapter_sha256"]
        )
    groups = {}
    for condition in POLICIES:
        trained = condition.startswith("trained_")
        policy = "flat" if condition.endswith("flat") else "manager_worker"
        records = [
            c
            for c in (list(calls.values()) + unresolved if trained else base_calls)
            if c["request"]["condition"] == policy
        ]
        selected = [r for r in rows if r["policy"] == condition]
        groups[condition] = {
            "planned": 24,
            "recorded": len(selected),
            "observed": sum(r["observed"] for r in selected),
            "unknown": 24 - sum(r["observed"] for r in selected),
            "won": sum(r["won"] and r["observed"] for r in selected),
            "actions": sum(r["actions"] for r in selected),
            "invalid_outputs": sum(r["invalid_outputs"] for r in selected),
            "terminations": dict(Counter(r["termination"] for r in selected)),
            "cost": unseen.analyze_helper.measured(records),
            "by_role": {
                role: unseen.analyze_helper.measured([c for c in records if c["role"] == role])
                for role in sorted({c["role"] for c in records})
            },
            "history_trimmed_calls": sum(c["trimming"]["dropped_history"] > 0 for c in records),
        }
    for module in (collector, unseen, unseen.indexed_audit, unseen.analyze_helper):
        hashes[str(Path(module.__file__).resolve())] = collector.probe.campaign.sha(
            Path(module.__file__)
        )
    hashes[str(Path(__file__).resolve())] = collector.probe.campaign.sha(Path(__file__))
    return {
        "output": str(output),
        "baseline_report": str(baseline_report),
        "terminal": terminal,
        "terminal_sha256": hashes[str(terminal_path)],
        "base_terminal": old["terminal"],
        "base_terminal_sha256": {
            p: h
            for p, h in old["source_and_receipt_sha256"].items()
            if Path(p).name.startswith("TERMINAL-")
        },
        "plan": plan,
        "checkpoint_state": state,
        "policies": groups,
        "rows": rows,
        "native_audits": audits,
        "comparisons": [
            unseen.paired(rows, games, plan["seeds"], left, right)
            for left, right in (
                ("trained_flat", "base_flat"),
                ("trained_manager", "base_manager"),
                ("trained_manager", "trained_flat"),
                ("base_manager", "base_flat"),
            )
        ],
        "interaction": interaction(rows, games, plan["seeds"]),
        "new_physical_cost": unseen.analyze_helper.measured(list(calls.values()) + unresolved),
        "unresolved_starts": sorted(set(starts) - set(calls)),
        "unlinked_calls": sorted(set(calls) - {cid for r in new_rows for cid in r["call_ids"]}),
        "source_and_receipt_sha256": hashes,
        "method": {"draws": 20000, "seed": unseen.SEED, "game_units": 12, "scene_units": 4},
        "limitations": [
            "Same exposed035 games; fixed checkpoint, no held-score selection "
            "or fresh confirmation.",
            "Action adapter enabled for flat/worker only; manager remains frozen base, "
            "on changed trajectories.",
            "Interaction is exploratory; preserve two repeats/game and only four scene clusters.",
            "Native original role identities/digests checked before host-only "
            "adapter-neutral state-machine replay.",
            "Weight identity uses collector-authenticated COMMIT; no repeated weight-file "
            "rehash or per-call weight measurement claim.",
            "Missing outcomes unknown; observed protocol/budget failures retained, "
            "all24 slots per condition.",
        ],
    }


def markdown(report):
    lines = [
        "# ALFWorld fixed actor-SFT × flat/manager comparison",
        "",
        "Twelve exposed games × two repeats; only four scenes. Fixed checkpoint0033.",
        "",
        "| Condition | Wins / 24 | Unknown | Calls | Tokens | Native seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["policies"].items():
        c = row["cost"]
        lines.append(
            f"| {name} | {row['won']} | {row['unknown']} | {c['calls']} | "
            f"{c['total_tokens']} | {c['known_latency_seconds']:.1f} |"
        )
    for c in report["comparisons"]:
        lines += [
            "",
            f"{c['left']}−{c['right']}: {c['wins']} wins/{c['losses']} losses; "
            f"game {c['game_bootstrap']}; scene {c['scene_bootstrap']}; "
            f"game sign-flip {c['game_signflip']}; scene sign-flip {c['scene_signflip']}.",
        ]
    lines += [
        "",
        "Interaction: " + json.dumps(report["interaction"]),
        "",
        *["- " + s for s in report["limitations"]],
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "baseline-report", "report"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    result = analyze(args.output, args.baseline_report)
    collector.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(result))
