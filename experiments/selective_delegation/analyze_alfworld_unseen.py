"""Native three-policy unseen ALFWorld audit; twelve games, only four scenes."""

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import alfworld_unseen as collector
import analyze_alfworld_local_reason as local_audit
import analyze_alfworld_screen as indexed_audit
import analyze_helper

SEED = 2026092200


def signflip(values, clusters):
    observed = abs(sum(values.values()))
    sums = [sum(values[g] for g in cluster) for cluster in clusters]
    extreme = sum(
        abs(sum(s * v for s, v in zip(signs, sums, strict=True))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(clusters))
    )
    return {
        "p_two_sided": extreme / (2 ** len(clusters)),
        "sign_vectors": 2 ** len(clusters),
        "null": "Exchangeable cluster signs; exploratory sensitivity, not confirmation",
    }


def paired(rows, games, seeds, left, right, draws=20000):
    lookup = {(r["game_index"], r["seed"], r["policy"]): r for r in rows}
    pairs, values, scenes, unknown = [], {}, defaultdict(list), 0
    for game, metadata in games.items():
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
        if len(differences) == len(seeds):
            values[str(game)] = mean(differences)
        scenes[metadata["scene"]].append(str(game))
    complete = len(values) == len(games)
    clusters = [[str(g)] for g in games]
    scene_clusters = list(scenes.values())
    return {
        "left": left,
        "right": right,
        "planned_pairs": len(games) * len(seeds),
        "unknown_pairs": unknown,
        "wins": sum(p["delta"] > 0 for p in pairs),
        "losses": sum(p["delta"] < 0 for p in pairs),
        "ties": sum(p["delta"] == 0 for p in pairs),
        "pairs": pairs,
        "game_mean_differences": values,
        "game_bootstrap": analyze_helper.clustered_interval(values, clusters, draws, SEED)
        if complete
        else None,
        "scene_bootstrap": analyze_helper.clustered_interval(values, scene_clusters, draws, SEED)
        if complete
        else None,
        "game_signflip": signflip(values, clusters) if complete else None,
        "scene_signflip": signflip(values, scene_clusters) if complete else None,
        "scene_clusters": dict(scenes),
    }


def analyze(output, draws=20000):
    output = Path(output).resolve()
    hashes, cache = {}, {}

    def read(path):
        path = Path(path).resolve()
        if str(path) not in cache:
            raw = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            cache[str(path)] = json.loads(raw)
        return cache[str(path)]

    import psutil

    owners = list(output.glob("OWNER-*.json"))
    if len(owners) != 1:
        raise ValueError("one completed owner required")
    owner = read(owners[0])
    terminal = read(owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-")))
    try:
        process = psutil.Process(owner["pid"])
        if (
            abs(process.create_time() - owner["create_time"]) < 0.01
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            raise ValueError("owner live; never rank partial outcomes")
    except psutil.NoSuchProcess:
        pass
    plan = read(output / "PLAN.json")
    if hashes[str(output / "PLAN.json")] != (
        "ca766c1a54a972b2b785cfd5b9cad995600d38d4134e2b23aba75b1056f73a49"
    ):
        raise ValueError("frozen source035 PLAN differs")
    manifest = read(Path(plan["input_manifest"]))
    if (
        plan["schema"] != "alfworld-unseen-frozen-policies-v1"
        or plan["input_manifest_sha256"] != hashes[str(Path(plan["input_manifest"]).resolve())]
        or plan["input_manifest_sha256"] != collector.INPUT_SHA
        or plan["policies"] != list(collector.POLICIES)
        or plan["seeds"] != list(collector.SEEDS)
        or plan["split"] != "valid_unseen"
        or plan["planned_episodes"] != 72
    ):
        raise ValueError("fixed unseen inventory contract differs")
    for name, expected in plan["source_sha256"].items():
        path = Path(name)
        if collector.probe.campaign.sha(path) != expected:
            raise ValueError("frozen source changed")
        local = Path(collector.__file__).with_name(path.name)
        if collector.probe.campaign.sha(local) != expected:
            raise ValueError("analyzer runtime dependency differs from frozen collector")
        hashes[str(path)] = expected
    jobs = {j["episode_id"]: j for j in plan["cases"]}
    if len(jobs) != 72 or {(j["game_index"], j["seed"], j["policy"]) for j in jobs.values()} != {
        (g, s, p) for g in range(12) for s in collector.SEEDS for p in collector.POLICIES
    }:
        raise ValueError("not exactly all72 planned slots")
    games = {j["game_index"]: j["game"] for j in jobs.values()}
    if len({g["scene"] for g in games.values()}) != 4:
        raise ValueError("scene inventory differs")
    for job in jobs.values():
        if job["game"] != games[job["game_index"]]:
            raise ValueError("within-game identity differs")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    client = collector.local.base.BaseClient(None, tokenizer, output, 0)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = Path(plan["model"]) / name
        hashes[str(path)] = collector.probe.campaign.sha(path)
    calls = {p.stem: read(p) for p in (output / "calls").glob("*.json")}
    starts = {p.stem: read(p) for p in (output / "starts").glob("*.json")}
    for cid, call in calls.items():
        if cid not in starts or any(
            call[k] != starts[cid][k] for k in ("request", "request_digest", "started")
        ):
            raise ValueError("native start/return mismatch")
        request = call["request"]
        if client.ids(request["prompt"]) != call["input_token_ids"]:
            raise ValueError("native chat tokenization differs")
        if (
            call["available"]
            and tokenizer.decode(
                call["output_token_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            != call["text"]
        ):
            raise ValueError("native output decode differs")
        if call["available"]:
            eos = call["output_token_ids"][-1] == tokenizer.eos_token_id
            if call["finish_reason"] != ("eos" if eos else "length_or_time"):
                raise ValueError("native finish reason differs")
    rows = [read(p) for p in (output / "episodes").glob("*.json")]
    if len({r["episode_id"] for r in rows}) != len(rows) or any(
        r["episode_id"] not in jobs for r in rows
    ):
        raise ValueError("duplicate/unplanned episodes")
    audits = {}
    for row in rows:
        job = jobs[row["episode_id"]]
        if row["policy"] == "local_reason":
            audit = local_audit.audit_episode(job, row, calls, read, output, plan, tokenizer)
        else:
            # Same immutable source026 policy, selected explicitly for the existing auditor.
            audit = indexed_audit.audit_episode(
                job, row, calls, read, output, {**plan, "schema": "alfworld-closed-loop-indexed-v1"}
            )
        audits[row["episode_id"]] = audit
    unresolved = [start for cid, start in starts.items() if cid not in calls]
    all_calls = list(calls.values()) + unresolved
    groups = {}
    for policy in collector.POLICIES:
        selected = [r for r in rows if r["policy"] == policy]
        records = [c for c in all_calls if c["request"]["condition"] == policy]
        groups[policy] = {
            "planned": 24,
            "recorded": len(selected),
            "observed": sum(r["observed"] for r in selected),
            "unknown": 24 - sum(r["observed"] for r in selected),
            "won": sum(r["won"] and r["observed"] for r in selected),
            "actions": sum(r["actions"] for r in selected),
            "invalid_outputs": sum(r["invalid_outputs"] for r in selected),
            "terminations": dict(Counter(r["termination"] for r in selected)),
            "physical_cost": analyze_helper.measured(records),
            "by_role": {
                role: analyze_helper.measured([c for c in records if c["role"] == role])
                for role in sorted({c["role"] for c in records})
            },
            "max_prompt_tokens": max((len(c["input_token_ids"]) for c in records), default=0),
            "history_trimmed_calls": sum(c["trimming"]["dropped_history"] > 0 for c in records),
            "output_lengths": [c.get("usage", {}).get("completion_tokens") for c in records],
            "native_stops": dict(
                Counter(
                    "eos"
                    if c["finish_reason"] == "eos"
                    else "at_output_cap"
                    if len(c["output_token_ids"]) == c["request"]["sampling"]["max_new_tokens"]
                    else "non_eos_before_cap"
                    for c in records
                    if c.get("available")
                )
            ),
            "by_family": {
                family: {
                    "planned": 4,
                    "won": sum(
                        r["won"] and r["observed"]
                        for r in selected
                        if r["game"]["family"] == family
                    ),
                    "observed": sum(
                        r["observed"] for r in selected if r["game"]["family"] == family
                    ),
                }
                for family in sorted({g["family"] for g in games.values()})
            },
        }
    for module in (collector, indexed_audit, local_audit, analyze_helper):
        hashes[str(Path(module.__file__).resolve())] = collector.probe.campaign.sha(
            Path(module.__file__)
        )
    hashes[str(Path(__file__).resolve())] = collector.probe.campaign.sha(Path(__file__))
    return {
        "output": str(output),
        "plan": plan,
        "manifest": manifest,
        "terminal": terminal,
        "policies": groups,
        "comparisons": [
            paired(rows, games, plan["seeds"], a, b, draws)
            for a, b in (
                ("manager_worker", "flat"),
                ("local_reason", "flat"),
                ("manager_worker", "local_reason"),
            )
        ],
        "rows": rows,
        "native_audits": audits,
        "reason_character_lengths": [
            n for a in audits.values() for n in a.get("reason_characters", [])
        ],
        "physical_cost": analyze_helper.measured(all_calls),
        "unresolved_starts": sorted(set(starts) - set(calls)),
        "unlinked_calls": sorted(set(calls) - {cid for r in rows for cid in r["call_ids"]}),
        "source_and_receipt_sha256": hashes,
        "method": {
            "draws": draws,
            "seed": SEED,
            "game_units": 12,
            "scene_units": 4,
            "scene_bootstrap": "Resample scenes, retaining all games/repeats and game weighting",
            "signflip": "Exhaustive game and scene cluster signs; small-n sensitivity",
        },
        "limitations": [
            "Balanced two games per six families, not natural benchmark mixture.",
            "Only four scenes; 72 episodes are not independent samples.",
            "Shared admissible commands are affordance assistance, not syntax-only.",
            "Missing/inference-unavailable outcomes remain unknown; "
            "returned protocol failures are observed.",
            "Frozen scaffolding comparison, not novelty or learned-hierarchy evidence.",
        ],
    }


def markdown(report):
    lines = [
        "# ALFWorld unseen-game three-policy screen",
        "",
        "Twelve fixed games, two seeds each, six balanced task families, only four scenes.",
        "",
        "| Policy | Won / 24 | Unknown | Calls | Tokens | Native seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, r in report["policies"].items():
        cost = r["physical_cost"]
        lines.append(
            f"| {policy} | {r['won']} | {r['unknown']} | {cost['calls']} | "
            f"{cost['total_tokens']} | {cost['known_latency_seconds']:.1f} |"
        )
    lines += ["", "Paired sensitivity (both repeats retained):", ""]
    for c in report["comparisons"]:
        lines.append(
            f"- {c['left']} minus {c['right']}: {c['wins']} wins/{c['losses']} losses, "
            f"{c['unknown_pairs']} unknown; game CI {c['game_bootstrap']}; "
            f"scene CI {c['scene_bootstrap']}; game sign-flip {c['game_signflip']}; "
            f"scene sign-flip {c['scene_signflip']}."
        )
    return "\n".join(lines + ["", *["- " + s for s in report["limitations"]]]) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    result = analyze(args.output)
    collector.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(result))
