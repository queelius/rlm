"""Paired official-Hotpot architecture comparison; never relax score_hotpot comparison mode."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import eval_planner as evaluation
import probe
import score_hotpot

SEED = 2026092176


def outcome(episode, final, case):
    status = "missing_episode"
    valid, em, f1 = False, 0.0, 0.0
    if episode is not None:
        if not final or not final.get("available"):
            source_status = episode.get("status", "missing_final")
            status = (
                source_status
                if source_status in ("invalid_plan", "invalid_dependency", "invalid_helper")
                else "unavailable_final"
                if final
                else "missing_final"
            )
        else:
            try:
                answer = score_hotpot.parse_answer(final.get("text"))
                em, f1 = score_hotpot.official_score(answer, case["answer"])
                valid, status = True, "scored"
            except (ValueError, TypeError):
                status = "invalid_final"
    return {"em": em, "f1": f1, "valid": valid, "status": status}


def paired(values, parents, clusters, *, draws=20000):
    result = {}
    for metric in ("em", "f1"):
        differences = {
            p: mean(
                values[p, r, "planner"][metric] - values[p, r, "direct"][metric] for r in range(2)
            )
            for p in parents
        }
        result[metric] = analyze_helper.clustered_interval(differences, clusters, draws, SEED)
    for sign, name in ((1, "wins"), (-1, "losses")):
        changes = []
        for p in parents:
            for r in range(2):
                direct, planner = values[p, r, "direct"], values[p, r, "planner"]
                if (planner["em"] - direct["em"]) * sign > 0:
                    category = (
                        "both_valid"
                        if direct["valid"] and planner["valid"]
                        else (
                            "protocol_involved"
                            if any(v["status"].startswith("invalid_") for v in (direct, planner))
                            else "unavailable_or_missing_involved"
                        )
                    )
                    changes.append({"case_id": p, "repeat": r, "category": category})
        result[name] = {
            "episodes": len(changes),
            "parents": sorted({c["case_id"] for c in changes}),
            **dict(Counter(c["category"] for c in changes)),
            "rows": changes,
        }
    return result


def validate_call(call, plan, condition, role, prompt, seed, cap):
    helper = role == "helper"
    enabled = condition == "sft" and role != "final"
    adapter = (
        (
            plan["helper_contract"]["adapter_binding"]["adapter_model.safetensors"]
            if helper
            else plan["adapter_files_sha256"]["adapter_model.safetensors"]
        )
        if enabled
        else None
    )
    expected = {
        "condition": condition,
        "role": role,
        "model": plan["model"],
        "model_instance": "helper" if helper else "root",
        "adapter_enabled": enabled,
        "adapter_sha256": adapter,
        "helper_contract": plan["helper_contract"],
    }
    request = call["request"]
    scientific = {
        **expected,
        "prompt": prompt,
        "seed": seed,
        "sampling": {
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": cap,
            "max_time": 90.0,
        },
    }
    if (
        probe.runtime.digest(request) != call["request_digest"]
        or any(request.get(k) != v for k, v in scientific.items())
        or any(call.get(k) != v for k, v in expected.items())
    ):
        raise ValueError("native request/role/model/adapter/seed differs")
    if call["input_token_ids"] != request["input_token_ids"]:
        raise ValueError("native input token binding differs")
    for field, ids in (
        ("prompt_tokens", "input_token_ids"),
        ("completion_tokens", "output_token_ids"),
    ):
        value = call.get("usage", {}).get(field)
        if (value is not None and value != len(call.get(ids, []))) or (
            call["available"] and (value is None or not call.get(ids))
        ):
            raise ValueError("native token usage differs")


def validate_episode(row, calls, case, plan, arm):
    """Replay only public prompt construction and dependency binding; never generate or write."""
    condition = "sft" if arm == "planner" else "base"
    seed = plan["seed"] + int(probe.runtime.digest(case["id"])[:6], 16) + row["repeat"] * 100
    if row["seed"] != seed:
        raise ValueError("episode seed differs")
    records = [calls[cid] for cid in row["call_ids"]]
    if len(set(row["call_ids"])) != len(records) or any(
        not c["call_id"].startswith(row["episode_id"] + "-") for c in records
    ):
        raise ValueError("cross-episode/reused call reference")
    expected_ids, checks = [], []

    def check(suffix, role, prompt, call_seed, cap):
        cid = row["episode_id"] + suffix
        if cid not in calls:
            return None
        expected_ids.append(cid)
        call = calls[cid]
        checks.append((call, plan, condition, role, prompt, call_seed, cap))
        return call

    if arm == "direct":
        check("-final", "final", evaluation.direct_prompt(case), seed + 2, 128)
    else:
        root = check("-root", "root", evaluation.planner_prompt(case), seed, 128)
        try:
            if root is None or not root["available"]:
                raise ValueError("root unavailable")
            parsed = evaluation.parse_plan(root["text"])
            answers, trace = [], []
            cap = 384 // len(parsed["subquestions"])
            for index, question in enumerate(parsed["subquestions"]):
                resolved = evaluation.bind_question(question, answers)
                helper = check(
                    f"-helper-{index + 1}",
                    "helper",
                    evaluation.contracted_helper_prompt(case, resolved, plan["helper_contract"]),
                    seed + index + 1,
                    cap,
                )
                if helper is None or not helper["available"]:
                    raise ValueError("helper unavailable")
                answer = evaluation.parse_helper_answer(helper["text"])
                answers.append(answer)
                trace.append(
                    {
                        "step": index + 1,
                        "question": question,
                        "resolved_question": resolved,
                        "answer": answer,
                    }
                )
        except (ValueError, TypeError):
            # A protocol/unavailable prefix correctly has no final call. Native request
            # validation happens outside this scientific protocol-error boundary.
            if any(c["role"] == "final" for c in records):
                raise ValueError(
                    "final exists after invalid or unavailable execution prefix"
                ) from None
        else:
            check(
                "-final",
                "final",
                evaluation.final_prompt(case, parsed, {"execution": "isolated", "steps": trace}),
                seed + 2,
                128,
            )
    if expected_ids != row["call_ids"]:
        raise ValueError("native trajectory call order/coverage differs")
    for arguments in checks:
        validate_call(*arguments)
    return records


def analyze(planner_output, direct_output, cases_path, *, draws=20000):
    cases_path = Path(cases_path).resolve()
    cases = score_hotpot._load_cases(cases_path)
    if len(cases) != 128 or draws < 1:
        raise ValueError("fixed128-parent panel and positive bootstrap draws required")
    hashes = {}

    def track(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in hashes:
            hashes[str(path)] = score_hotpot.sha256(path)
        actual = hashes[str(path)]
        if expected is not None and actual != expected:
            raise ValueError("source/receipt hash differs: " + str(path))
        return actual

    def read(path):
        track(path)
        return json.loads(Path(path).read_text())

    values, groups, plans, seeds, coverage = {}, {}, {}, {}, {}
    for arm, source in (("planner", planner_output), ("direct", direct_output)):
        source = Path(source).resolve()
        plan = read(source / "PLAN.json")
        condition = "sft" if arm == "planner" else "base"
        if (
            plan["case_ids"] != list(cases)
            or plan["repeats"] != 2
            or plan["conditions"] != [condition]
            or plan["split"] != "transfer"
            or plan["temperature"] != 0.5
            or plan["top_p"] != 1
            or plan["top_k"] != 0
            or plan["seed"] != evaluation.SEED
            or plan.get("mode", "planner") != ("planner" if arm == "planner" else "direct")
            or plan["execution"] != ("isolated" if arm == "planner" else "direct")
            or plan["helper_contract"]["mode"] != ("trained_helper" if arm == "planner" else "base")
        ):
            raise ValueError("fixed architecture/panel/sampling contract differs")
        track(cases_path, plan["cases_sha256"])
        report_path = source.parent / ("analysis-" + source.name) / "REPORT.json"
        official = read(report_path)
        for field, actual in (
            ("cases_sha256", track(cases_path)),
            ("evaluation_plan_sha256", track(source / "PLAN.json")),
            ("episodes_sha256", score_hotpot.tree_sha256(source / "episodes")),
            ("calls_sha256", score_hotpot.tree_sha256(source / "calls")),
        ):
            if official["sources"][field] != actual:
                raise ValueError("official regrade receipt binding differs: " + field)
        track(score_hotpot.EVALUATOR, official["scoring"]["evaluator_sha256"])
        for name, expected in official["code_sha256"].items():
            track(Path(score_hotpot.__file__).with_name(name), expected)
        for binding in official["sources"]["evaluations"]:
            for path, expected in binding["verified_bindings"].items():
                track(path, expected)
        for path, expected in plan["dependencies"].items():
            track(path, expected)
        collector_paths = [
            Path(p).with_name("eval_planner.py")
            for p in plan["dependencies"]
            if Path(p).name == "rl_planner.py"
        ]
        if len(collector_paths) != 1:
            raise ValueError("collector source path not uniquely bound")
        track(collector_paths[0], plan["source_sha256"])
        root_state = read(Path(plan["adapter"]) / "STATE.json")
        if root_state["step"] != 48:
            raise ValueError("fixed root SFT48 required")
        if arm == "planner":
            helper = plan["helper_contract"]
            for name, expected in helper["adapter_binding"].items():
                track(Path(helper["adapter"]) / name, expected)
            track(Path(helper["adapter"]).parent / "PLAN.json", helper["training_plan_sha256"])
            if read(Path(helper["adapter"]) / "STATE.json")["step"] != 36:
                raise ValueError("fixed helper36 required")
        plans[arm] = plan
        calls = {p.stem: read(p) for p in sorted((source / "calls").glob("*.json"))}
        episodes = {}
        for path in sorted((source / "episodes").glob("*.json")):
            row = read(path)
            key = row["case_id"], row["repeat"]
            if (
                key in episodes
                or key[0] not in cases
                or key[1] not in (0, 1)
                or row["condition"] != condition
                or path.stem != row["episode_id"]
            ):
                raise ValueError("episode outside planned inventory or duplicate")
            episodes[key] = row
        linked, deployed, final_seeds = set(), [], {}
        for p in cases:
            for repeat in range(2):
                row = episodes.get((p, repeat))
                records = validate_episode(row, calls, cases[p], plan, arm) if row else []
                linked.update(c["call_id"] for c in records)
                deployed.extend(records)
                finals = [c for c in records if c["role"] == "final"]
                final = finals[0] if finals else None
                if final:
                    final_seeds[p, repeat] = final["request"]["seed"]
                values[p, repeat, arm] = outcome(row, final, cases[p])
        starts = [read(p) for p in sorted((source / "starts").glob("*.json"))]
        if any(probe.runtime.digest(s["request"]) != s["request_digest"] for s in starts):
            raise ValueError("native start request digest differs")
        seen_starts, unresolved = set(), []
        for start in starts:
            if start["call_id"] not in calls or start["call_id"] in seen_starts:
                unresolved.append(start)
            seen_starts.add(start["call_id"])
        if len({c["call_id"] for c in calls.values()}) != len(calls) or any(
            cid != c["call_id"] for cid, c in calls.items()
        ):
            raise ValueError("native call filename/identity differs")
        if any(
            not any(
                s["call_id"] == cid and s["request_digest"] == c["request_digest"] for s in starts
            )
            for cid, c in calls.items()
        ):
            raise ValueError("native call lacks matching start")
        subset = [values[p, r, arm] for p in cases for r in range(2)]
        statuses = dict(Counter(v["status"] for v in subset))
        em, f1 = mean(v["em"] for v in subset), mean(v["f1"] for v in subset)
        summary = official["conditions"][condition]
        if (
            summary["planned"] != 256
            or summary["status_counts"] != statuses
            or abs(summary["em"] - em) > 1e-12
            or abs(summary["f1"] - f1) > 1e-12
        ):
            raise ValueError("paired per-attempt scores differ from official regrade")
        physical = analyze_helper.measured(list(calls.values()) + unresolved)
        groups[arm] = {
            "planned": 256,
            "recorded": len(episodes),
            "correct": sum(v["em"] for v in subset),
            "em": em,
            "f1": f1,
            "status_counts": statuses,
            "official_summary": summary,
            "physical_cost": physical,
            "deployed_known_cost": analyze_helper.measured(deployed),
            "per_planned_attempt_physical": {
                k: physical[k] / 256 for k in ("calls", "total_tokens")
            },
            "cost_by_role": {
                role: analyze_helper.measured([c for c in calls.values() if c["role"] == role])
                for role in ("root", "helper", "final")
            },
        }
        coverage[arm] = {
            "unlinked_call_ids": sorted(set(calls) - linked),
            "unresolved_start_ids": [s["call_id"] for s in unresolved],
            "extra_start_attempts": len(starts) - len({s["call_id"] for s in starts}),
        }
        seeds[arm] = final_seeds
    for field in (
        "cases_sha256",
        "case_ids",
        "repeats",
        "seed",
        "model",
        "model_manifest_sha256",
        "adapter_files_sha256",
        "source_sha256",
        "environment",
    ):
        if plans["planner"][field] != plans["direct"][field]:
            raise ValueError("paired source/model/panel contract differs: " + field)
    common = set(seeds["planner"]) & set(seeds["direct"])
    if any(seeds["planner"][k] != seeds["direct"][k] for k in common):
        raise ValueError("saved final/direct seeds are not matched")
    clusters = analyze_helper.component_clusters(list(cases.values()))
    for path in (
        Path(__file__),
        Path(score_hotpot.__file__),
        Path(analyze_helper.__file__),
        Path(evaluation.__file__),
    ):
        track(path)
    return {
        "source_plans": plans,
        "groups": groups,
        "coverage": coverage,
        "all_planned_episodes_present": all(g["recorded"] == 256 for g in groups.values()),
        "paired_planner_minus_direct": paired(values, list(cases), clusters, draws=draws),
        "method": {
            "parents": 128,
            "repeats": 2,
            "planned_attempts_per_arm": 256,
            "metric": "official_hotpotqa_em_f1",
            "bootstrap_draws": draws,
            "bootstrap_seed": SEED,
            "clusters": clusters,
            "parents_missing_component_ids": sum(
                not c.get("metadata", {}).get("component_ids") for c in cases.values()
            ),
            "bootstrap": "Parent-weighted paired repeat means; resample public-document "
            "components; percentile95.",
            "seeds": {
                "seed_base": plans["planner"]["seed"],
                "formula": "eval_planner.SEED + int(digest(case_id)[:6],16) + repeat*100 +2",
                "paired_observed_final_requests": len(common),
                "all_observed_pairs_equal": True,
                "no_final_planner_slots": 256 - len(seeds["planner"]),
                "no_final_direct_slots": 256 - len(seeds["direct"]),
            },
        },
        "input_source_receipt_sha256": hashes,
        "cautions": [
            "128 parents, not256 independent observations. Missing outcomes remain "
            "planned zero/lower bounds.",
            "Public-document components are not verified independent atomic reasoning components.",
            "Both final roles use base weights and matched numeric seeds, but different prompts; "
            "helper2/final seed collision is inherited.",
            "Planner architecture includes SFT root and trained helpers; contrast is not pure "
            "decomposition or equal-compute evidence.",
            "Both-valid gains do not establish intermediate correctness; full-source final "
            "may bypass or repair the plan.",
            "Native MuSiQue scores are diagnostic only; separate official Hotpot regrades "
            "are authoritative.",
            "Unknown token usage is not zero; incomplete collection costs are lower bounds. "
            "Extra starts/unlinked calls are explicitly reported, not hypothetical "
            "deployment savings.",
        ],
    }


def markdown(report):
    lines = [
        "# Paired Hotpot architecture comparison",
        "",
        "128 parents ×two repeats;256 planned attempts per arm. Official Hotpot EM/F1.",
        "",
        "| Architecture | Correct/planned | EM | F1 | Physical calls | Native tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, row in report["groups"].items():
        lines.append(
            f"| {arm} | {row['correct']:g}/256 | {row['em']:.2%} | {row['f1']:.2%} | "
            f"{row['physical_cost']['calls']} | {row['physical_cost']['total_tokens']} |"
        )
    pair, method = report["paired_planner_minus_direct"], report["method"]
    lines += [
        "",
        f"Paired bootstrap:{method['bootstrap_draws']} draws, seed{method['bootstrap_seed']}; "
        f"{len(method['clusters'])} public-document clusters (not verified atomic independence).",
    ]
    for metric in ("em", "f1"):
        item = pair[metric]
        lines.append(
            f"Planner−direct {metric.upper()}: {100 * item['estimate']:+.2f}pp "
            f"[{100 * item['ci95'][0]:+.2f}, {100 * item['ci95'][1]:+.2f}]."
        )
    lines += [
        "",
        "Wins/losses: "
        + json.dumps(
            {k: {a: b for a, b in pair[k].items() if a != "rows"} for k in ("wins", "losses")}
        ),
        "",
        "Statuses: " + json.dumps({k: v["status_counts"] for k, v in report["groups"].items()}),
        "",
        "Native seed audit: " + json.dumps(method["seeds"]),
        "",
        "Receipt coverage: " + json.dumps(report["coverage"]),
        "",
        *report["cautions"],
    ]
    return "\n".join(lines) + "\n"


def write_report(report, path):
    path = Path(path)
    sibling = path.with_suffix(".md")
    if path.exists() or sibling.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with sibling.open("x") as stream:
        stream.write(markdown(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-output", type=Path, required=True)
    parser.add_argument("--direct-output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    write_report(analyze(args.planner_output, args.direct_output, args.cases), args.report)
