"""Read-only paired analysis of the frozen-prefix next-question feedback screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from contextlib import suppress
from pathlib import Path
from statistics import mean

import analyze_helper
import next_question_probe as collector
import probe

SEED = 2026092171


class BindingError(AssertionError):
    """Receipt audit failure must escape the collector's scientific-error handlers."""


def expected_request(prompt, role, seed, cap, plan):
    contract = plan["helper_contract"]
    return dict(
        prompt=prompt,
        condition="sft",
        role=role,
        model=contract["model"] if role == "helper" else str(collector.evaluation.planner.BASE),
        model_instance="helper" if role == "helper" else "root",
        helper_contract=contract,
        adapter_enabled=role != "final",
        adapter_sha256=plan["root_adapter_binding"]["adapter_model.safetensors"]
        if role == "root"
        else contract["adapter_binding"]["adapter_model.safetensors"]
        if role == "helper"
        else None,
        seed=seed,
        sampling=dict(
            do_sample=True, temperature=0.5, top_p=1.0, top_k=0, max_new_tokens=cap, max_time=90.0
        ),
    )


def validate_call(call, prompt, role, seed, cap, plan):
    expected = expected_request(prompt, role, seed, cap, plan)
    request = call["request"]
    if (
        probe.runtime.digest(request) != call["request_digest"]
        or any(request.get(k) != v for k, v in expected.items())
        or any(
            call.get(k) != expected[k]
            for k in (
                "condition",
                "role",
                "model",
                "model_instance",
                "helper_contract",
                "adapter_enabled",
                "adapter_sha256",
            )
        )
        or request["input_token_ids"] != call["input_token_ids"]
    ):
        raise BindingError("native request/prompt/seed/role/adapter differs")
    for field, token_ids in (
        ("prompt_tokens", "input_token_ids"),
        ("completion_tokens", "output_token_ids"),
    ):
        count = call.get("usage", {}).get(field)
        if count is not None and count != len(call.get(token_ids, [])):
            raise BindingError("native usage differs")
    if call["available"] and (
        not call.get("output_token_ids")
        or any(
            field not in call.get("usage", {}) for field in ("prompt_tokens", "completion_tokens")
        )
    ):
        raise BindingError("available call lacks actual native tokens")


def replay_episode(row, case, prefix, plan, calls):
    """Reconstruct exact prompts, prediction binding, protocol outcomes and official final grade."""
    used = []
    with tempfile.TemporaryDirectory(prefix="next-question-replay-") as temporary:

        class Replay:
            output = Path(temporary)

            def call(self, identity, prompt, condition, role, seed, *, max_new_tokens):
                if identity not in calls or identity in used or condition != "sft":
                    raise BindingError("missing/reused native request")
                call = calls[identity]
                validate_call(call, prompt, role, seed, max_new_tokens, plan)
                used.append(identity)
                return call

        try:
            rebuilt = collector.collect(Replay(), case, prefix, row["arm"], row["repeat"])
        except RuntimeError:
            rebuilt = collector.read(Path(temporary) / "episodes" / (row["episode_id"] + ".json"))
    saved = {k: v for k, v in row.items() if k not in ("started", "ended")}
    actual = {k: v for k, v in rebuilt.items() if k not in ("started", "ended")}
    if saved != actual or used != row["call_ids"]:
        raise BindingError("saved episode differs from native replay/official regrade")
    return rebuilt


def panel_summary(values, parents, cases, *, draws=20000, seed=SEED, selected_clusters=None):
    if not parents:
        return None
    full_clusters = (
        analyze_helper.component_clusters(cases) if selected_clusters is None else selected_clusters
    )
    parent_set = set(parents)
    clusters = [[p for p in cluster if p in parent_set] for cluster in full_clusters]
    clusters = [cluster for cluster in clusters if cluster]
    result = dict(
        parents=len(parents),
        repeats=2,
        arms={},
        paired={},
        contrasts={},
        component_clusters=clusters,
        parents_missing_component_ids=sum(
            not c.get("metadata", {}).get("component_ids") for c in cases
        ),
    )
    for arm in collector.ARMS:
        rows = [values[p, r, arm] for p in parents for r in range(2)]
        valid = [v for v in rows if v["valid"]]
        missing = sum(not v["observed"] for v in rows)
        stats = dict(
            planned=len(rows),
            recorded=sum(v["present"] for v in rows),
            missing=missing,
            observed=len(rows) - missing,
            valid=len(valid),
            protocol_failures=sum(v["observed"] and not v["valid"] for v in rows),
            status_counts=dict(Counter(v["status"] for v in rows)),
        )
        for metric in ("em", "f1"):
            score = sum(v[metric] for v in rows)
            stats.update(
                {
                    metric + "_lower": score / len(rows),
                    metric + "_upper": (score + missing) / len(rows),
                    metric + "_valid_only": mean(v[metric] for v in valid) if valid else None,
                }
            )
        result["arms"][arm] = stats
    counts = Counter()
    changes = []
    for p in parents:
        for repeat in range(2):
            a, b = [values[p, repeat, arm] for arm in collector.ARMS]
            if a.get("next_question") is not None and b.get("next_question") is not None:
                counts["both_parsed_next_questions"] += 1
                counts["changed_raw_next_question"] += a["next_question"] != b["next_question"]
                if (
                    a.get("resolved_question") is not None
                    and b.get("resolved_question") is not None
                ):
                    counts["both_bound_next_questions"] += 1
                    counts["changed_resolved_next_question"] += (
                        a["resolved_question"] != b["resolved_question"]
                    )
            if not a["observed"] or not b["observed"]:
                counts["unobserved"] += 1
                continue
            counts["observed"] += 1
            if a["em"] != b["em"]:
                direction = "wins" if b["em"] > a["em"] else "losses"
                kind = "both_valid" if a["valid"] and b["valid"] else "protocol"
                counts[direction + "_" + kind] += 1
                changes.append(dict(case_id=p, repeat=repeat, direction=direction, kind=kind))
    result["paired"] = dict(counts)
    result["paired_changes"] = changes
    for metric in ("em", "f1"):
        deltas = {
            p: mean(
                values[p, r, "feedback"][metric] - values[p, r, "hidden"][metric] for r in range(2)
            )
            for p in parents
        }
        a, b = [result["arms"][arm] for arm in collector.ARMS]
        result["contrasts"][metric] = dict(
            parent=analyze_helper.clustered_interval(deltas, [[p] for p in parents], draws, seed),
            component=analyze_helper.clustered_interval(deltas, clusters, draws, seed),
            missing_identification_bounds=[
                b[metric + "_lower"] - a[metric + "_upper"],
                b[metric + "_upper"] - a[metric + "_lower"],
            ],
        )
    return result


def analyze(output, cases_path, *, draws=20000, seed=SEED):
    output, cases_path = Path(output).resolve(), Path(cases_path).resolve()
    hashes = {}

    def track(path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise BindingError("immutable source/case hash differs: " + str(path))
        hashes[str(path)] = digest
        return data

    plan = json.loads(track(output / "PLAN.json"))
    selected = json.loads(track(output / "SELECTION.json"))
    rows = [json.loads(line) for line in track(cases_path, selected["cases_sha256"]).splitlines()]
    cases = {c["id"]: c for c in rows if c["split"] == "transfer"}
    if (
        plan["schema"] != "frozen-prefix-next-question-v1"
        or plan["selection"] != selected
        or selected["case_ids"] != collector.select_parents(list(cases.values()))
        or selected["repeats"] != 2
        or selected["source_repeat"] != 0
        or selected["arms"] != list(collector.ARMS)
        or selected["seed"] != collector.SEED
        or selected["planned_slots"] != 64
        or selected["max_new_calls"] != 192
        or plan["caps"] != collector.CAPS
        or plan["instruction"] != collector.INSTRUCTION
        or plan["temperature"] != 0.5
        or plan["top_p"] != 1.0
        or plan["top_k"] != 0
        or plan["helper_contract"]["mode"] != "trained_helper"
        or draws < 1
    ):
        raise BindingError("unexpected next-question screen contract")
    for path, digest in plan["dependencies"].items():
        track(path, digest)
    prefixes, source_hashes = collector.prepare_prefixes(
        Path(selected["source_output"]),
        selected,
        cases,
        plan["root_adapter_binding"],
        plan["helper_contract"]["adapter_binding"],
    )
    if source_hashes != plan["source_hashes"] or plan["eligible"] != {
        cid: p["eligible"] for cid, p in prefixes.items()
    }:
        raise BindingError("frozen prefixes changed")
    hashes.update(source_hashes)
    parents = selected["case_ids"]
    expected = {
        f"{p}-r{r}-{arm}": (p, r, arm) for p in parents for r in range(2) for arm in collector.ARMS
    }
    expected_calls = {eid + "-" + role for eid in expected for role in collector.CAPS}
    calls, episodes, starts = {}, {}, {}
    for path in output.glob("calls/*.json"):
        row = json.loads(track(path))
        cid = row["call_id"]
        if cid != path.stem or cid not in expected_calls or cid in calls:
            raise BindingError("native call outside planned inventory")
        calls[cid] = row
    for path in output.glob("starts/*.json"):
        row = json.loads(track(path))
        cid = row["call_id"]
        if cid not in expected_calls or cid in starts or not path.stem.startswith(cid + "-"):
            raise BindingError("duplicate/out-of-inventory native attempt")
        if probe.runtime.digest(row["request"]) != row["request_digest"]:
            raise BindingError("native start digest differs")
        starts[cid] = row
    for cid, call in calls.items():
        if cid not in starts or starts[cid]["request_digest"] != call["request_digest"]:
            raise BindingError("native return lacks matching start")
    # Unlinked attempts still count as cost, never as observed episode outcomes.
    for call in [*calls.values(), *starts.values()]:
        eid, role = call["call_id"].rsplit("-", 1)
        cid, repeat, arm = expected[eid]
        if not prefixes[cid]["eligible"]:
            raise BindingError("native call exists for unavailable prefix")
        trace = [dict(t) for t in prefixes[cid]["trace"]]
        if role == "root":
            prompt = collector.next_prompt(cases[cid], trace, arm)
        else:
            root = calls.get(eid + "-root")
            if not root or not root["available"]:
                raise BindingError("downstream request lacks available root")
            question = collector.parse_next(root["text"])
            resolved = collector.evaluation.bind_question(question, [t["answer"] for t in trace])
            if role == "helper":
                prompt = collector.evaluation.isolated_helper_prompt(cases[cid], resolved)
            else:
                helper = calls.get(eid + "-helper")
                if not helper or not helper["available"]:
                    raise BindingError("final request lacks available helper")
                trace.append(
                    dict(
                        step=3,
                        question=question,
                        resolved_question=resolved,
                        answer=collector.evaluation.parse_helper_answer(helper["text"]),
                    )
                )
                prompt = collector.evaluation.final_prompt(
                    cases[cid],
                    {"subquestions": [t["question"] for t in trace]},
                    {"execution": "isolated", "steps": trace},
                )
        validate_call(
            call, prompt, role, collector.seeds(cid, repeat)[role], collector.CAPS[role], plan
        )
    for path in output.glob("episodes/*.json"):
        row = json.loads(track(path))
        eid = row["episode_id"]
        if (
            eid != path.stem
            or eid not in expected
            or eid in episodes
            or expected[eid] != (row["case_id"], row["repeat"], row["arm"])
        ):
            raise BindingError("episode outside planned inventory")
        episodes[eid] = row
    values, linked, gold_discordance = {}, set(), Counter()
    for eid, (cid, repeat, arm) in expected.items():
        row, prefix = episodes.get(eid), prefixes[cid]
        entry = dict(
            present=row is not None,
            observed=False,
            valid=False,
            em=0.0,
            f1=0.0,
            eligible=prefix["eligible"],
            status="missing_episode" if prefix["eligible"] else "source_prefix_unavailable",
        )
        if row is not None:
            rebuilt = replay_episode(row, cases[cid], prefix, plan, calls)
            if linked.intersection(row["call_ids"]):
                raise BindingError("physical call reused by multiple episodes")
            linked.update(row["call_ids"])
            entry.update(
                observed=rebuilt["outcome_observed"],
                valid=rebuilt["valid"],
                em=float(rebuilt["correct"]),
                f1=rebuilt["f1"],
                status=rebuilt["status"],
                next_question=rebuilt.get("next_question"),
            )
            if rebuilt.get("next_question") is not None:
                with suppress(ValueError):
                    entry["resolved_question"] = collector.evaluation.bind_question(
                        rebuilt["next_question"], [t["answer"] for t in prefix["trace"]]
                    )
            if len(rebuilt["helper_trace"]) == 3 and rebuilt["valid"]:
                helper_answer = rebuilt["helper_trace"][-1]["answer"]
                helper_grade = probe.grade(json.dumps({"answer": helper_answer}), cases[cid])
                key = (
                    "last_helper_matches_gold_final_does_not"
                    if helper_grade["correct"] and not rebuilt["correct"]
                    else "final_matches_gold_last_helper_does_not"
                    if not helper_grade["correct"] and rebuilt["correct"]
                    else "same_gold_match_status"
                )
                gold_discordance[arm + ":" + key] += 1
        values[cid, repeat, arm] = entry
    eligible = [p for p in parents if prefixes[p]["eligible"]]
    selected_clusters = analyze_helper.component_clusters([cases[p] for p in parents])
    unresolved = [row for cid, row in starts.items() if cid not in calls]
    physical = list(calls.values()) + unresolved
    reused = {row["call_id"]: row for prefix in prefixes.values() for row in prefix["dependencies"]}
    acquisition = dict(reused)
    source = Path(selected["source_output"])
    for cid in parents:
        historical_id = f"{cid}-r0-trained_helper"
        historical = collector.read(source / "episodes" / (historical_id + ".json"))
        for step in (1, 2):
            call_id = historical_id + f"-helper-{step}"
            if call_id in historical["call_ids"] and call_id not in acquisition:
                call = json.loads(track(source / "calls" / (call_id + ".json")))
                if (
                    call["call_id"] != call_id
                    or probe.runtime.digest(call["request"]) != call["request_digest"]
                ):
                    raise BindingError("historical attempted-prefix receipt differs")
                acquisition[call_id] = call
    for path in (Path(__file__), Path(analyze_helper.__file__), Path(collector.__file__)):
        track(path)
    return dict(
        output=str(output),
        plan=plan,
        selected_parent_ids=parents,
        eligible_parent_ids=eligible,
        unavailable_prefix_parent_ids=sorted(set(parents) - set(eligible)),
        selected=panel_summary(
            values,
            parents,
            [cases[p] for p in parents],
            draws=draws,
            seed=seed,
            selected_clusters=selected_clusters,
        ),
        eligible=panel_summary(
            values,
            eligible,
            [cases[p] for p in eligible],
            draws=draws,
            seed=seed,
            selected_clusters=selected_clusters,
        ),
        per_slot=[dict(case_id=p, repeat=r, arm=a, **v) for (p, r, a), v in values.items()],
        new_physical_cost=analyze_helper.measured(physical),
        per_arm_role_cost={
            arm: {
                role: analyze_helper.measured(
                    [row for row in physical if row["call_id"].endswith("-" + arm + "-" + role)]
                )
                for role in collector.CAPS
            }
            for arm in collector.ARMS
        },
        historical_selected_prefix_acquisition_once=analyze_helper.measured(
            list(acquisition.values())
        ),
        historical_reused_dependencies_once=analyze_helper.measured(list(reused.values())),
        hypothetical_deployed_cost_sum_by_arm={
            arm: collector.evaluation.cost(
                [
                    call
                    for eid, (cid, _, a) in expected.items()
                    if a == arm and eid in episodes and prefixes[cid]["eligible"]
                    for call in prefixes[cid]["dependencies"]
                    + [calls[i] for i in episodes[eid]["call_ids"]]
                ]
            )
            for arm in collector.ARMS
        },
        last_helper_final_gold_match_discordance=dict(gold_discordance),
        unlinked_call_ids=sorted(set(calls) - linked),
        unresolved_start_ids=[r["call_id"] for r in unresolved],
        all_selected_slots_recorded=len(episodes) == 64,
        all_eligible_outcomes_observed=bool(eligible)
        and all(v["observed"] for v in values.values() if v["eligible"]),
        input_source_receipt_sha256=hashes,
        method=dict(
            draws=draws,
            seed=seed,
            contrast="feedback minus hidden; average two repeats per parent",
            missing="Prefix unavailability, uncollected slots and native failures remain missing, "
            "not scientific errors. Planned metrics are lower bounds; missing bounds are separate.",
            uncertainty="Exploratory parent and component-cluster bootstrap of paired lower-bound "
            "scores. Two downstream seeds reuse one prefix; they are not independent parents. "
            "Eligible clusters restrict full selected-panel connectivity, preserving bridges "
            "through ineligible parents. Missing component IDs fall back to singleton parents.",
            costs="Historical selected roots and first-two-helper attempts counted once, including "
            "failed attempts. Dependency-only cost also shown. Deployment sums reuse dependencies "
            "per recorded eligible attempt, not physical cost or a128-root-token policy.",
            helper="Mechanical last-helper/final gold-match discordance among valid finals, "
            "not helper correctness or causal overrides. The next question may not answer "
            "the original question; useful intermediate answers need not match its gold.",
            scope="Exposed four-hop frozen-state screen; no stopping or full incremental policy, "
            "new learning, faithful reasoning or generalization claim.",
        ),
    )


def write_report(report, path):
    path = Path(path)
    if path.exists() or path.with_suffix(".md").exists():
        raise FileExistsError(path)
    lines = [
        "# Next-question feedback screen",
        "",
        f"Source: `{report['output']}`",
        "",
        f"16 selected parents; {len(report['eligible_parent_ids'])} have eligible frozen prefixes. "
        "Two repeats are not32 independent parents.",
        "",
    ]
    for name in ("selected", "eligible"):
        panel = report[name]
        if panel is None:
            continue
        lines += [
            f"## {name.title()} panel",
            "",
            "| Arm | Planned | Observed | Protocol failures | EM bounds | F1 bounds |",
            "|---|---:|---:|---:|---|---|",
        ]
        for arm, row in panel["arms"].items():
            lines.append(
                f"| {arm} | {row['planned']} | {row['observed']} | {row['protocol_failures']} "
                f"| {row['em_lower']:.3f}–{row['em_upper']:.3f} "
                f"| {row['f1_lower']:.3f}–{row['f1_upper']:.3f} |"
            )
        lines += [
            "",
            f"Paired outcomes/questions: `{json.dumps(panel['paired'], sort_keys=True)}`.",
            f"Feedback–hidden intervals: `{json.dumps(panel['contrasts'], sort_keys=True)}`.",
            "",
        ]
    lines += [
        f"New physical cost: `{json.dumps(report['new_physical_cost'], sort_keys=True)}`.",
        "",
        "Historical selected-prefix acquisition: "
        f"`{json.dumps(report['historical_selected_prefix_acquisition_once'], sort_keys=True)}`.",
        "",
        *report["method"].values(),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    with path.with_suffix(".md").open("x") as stream:
        stream.write("\n".join(map(str, lines)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    write_report(analyze(args.output, args.cases), args.report)
