"""Native-audited fixed-endpoint sufficiency comparison; parent-clustered exploratory CIs."""

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

import analyze_helper
import sufficiency_probe as baseline

SEED = 2026092190
SAMPLING = {
    "temperature": 0.5,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 128,
    "max_time": 90.0,
    "do_sample": True,
}
METRICS = (
    "joint_em",
    "joint_f1",
    "both_labels",
    "positive_em",
    "positive_f1",
    "positive_abstention",
    "negative_overanswer",
)


def audit_call(call, job, case, plan, arm, tokenizer):
    req = call["request"]
    condition = "full_context" if arm == "base" else arm
    if (
        call["call_id"] != job["episode_id"]
        or req["prompt"] != baseline.prompt(case)
        or req["seed"] != job["seed"]
        or req["sampling"] != SAMPLING
        or req["model"] != plan["model"]
        or req["condition"] != condition
        or call["condition"] != condition
        or req["role"] != "sufficiency"
        or call["role"] != "sufficiency"
        or call["request_digest"] != baseline.native.probe.runtime.digest(req)
    ):
        raise ValueError("native request differs from planned public contract")
    adapter = plan["adapters"]
    if (
        req["adapter_enabled"] != (arm != "base")
        or req.get("adapter_sha256")
        != (adapter["files"]["adapter_model.safetensors"] if adapter else None)
        or (arm != "base" and req.get("adapter") != adapter)
    ):
        raise ValueError("native adapter binding differs")
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": baseline.prompt(case)}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if req["input_token_ids"] != ids or call["input_token_ids"] != ids:
        raise ValueError("native input token IDs differ")
    if not call["available"]:
        return {"status": "unavailable", "prediction": None}
    output_ids = call["output_token_ids"]
    if not 1 <= len(output_ids) <= 128 or call["usage"] != {
        "prompt_tokens": len(ids),
        "completion_tokens": len(output_ids),
    }:
        raise ValueError("native output/cost differs")
    if (
        tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        != call["text"]
    ):
        raise ValueError("native response decode differs")
    try:
        prediction = baseline.parse_output(call["text"])
    except (ValueError, TypeError):
        return {"status": "protocol_invalid", "prediction": None}
    return {"status": "valid", "prediction": prediction}


def pair_metrics(cases, records):
    answer_class, _ = baseline.official_metrics()
    result = []
    for parent in sorted({c["parent_id"] for c in cases}):
        pair = sorted(
            [c for c in cases if c["parent_id"] == parent], key=lambda c: not c["answerable"]
        )
        if len(pair) != 2 or [c["answerable"] for c in pair] != [True, False]:
            raise ValueError("complete positive/negative pair required")
        for seed in baseline.SEEDS:
            rows = [records[c["id"], seed] for c in pair]
            predictions = [r["prediction"] for r in rows]
            score = baseline.group_score(pair[0], predictions)
            metric = answer_class()
            if predictions[0] is not None:
                metric(predictions[0]["answer"], [pair[0]["answer"], *pair[0]["answer_aliases"]])
            positive_em, positive_f1 = metric.get_metric(reset=True)
            result.append(
                {
                    "parent_id": parent,
                    "seed": seed,
                    "joint_em": score["em"],
                    "joint_f1": score["f1"],
                    "both_labels": score["suff"],
                    "positive_em": positive_em,
                    "positive_f1": positive_f1,
                    "positive_abstention": int(
                        predictions[0] is not None and not predictions[0]["answerable"]
                    ),
                    "negative_overanswer": int(
                        predictions[1] is not None and predictions[1]["answerable"]
                    ),
                    "both_valid": all(r["status"] == "valid" for r in rows),
                    "both_observed": all(
                        r["status"] in ("valid", "protocol_invalid") for r in rows
                    ),
                    "statuses": [r["status"] for r in rows],
                    "predictions": predictions,
                }
            )
    return result


def parent_values(rows, metric):
    return {
        p: mean(r[metric] for r in rows if r["parent_id"] == p)
        for p in sorted({r["parent_id"] for r in rows})
    }


def interval(values, draws=20000, seed=SEED):
    return analyze_helper.clustered_interval(values, [[p] for p in sorted(values)], draws, seed)


def analyze(outputs, cases_path, *, draws=20000):
    hashes = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("source/receipt hash mismatch: " + str(path))
        hashes[str(path)] = digest
        return json.loads(data)

    cases_path = Path(cases_path).resolve()
    cases_bytes = cases_path.read_bytes()
    cases_sha = hashlib.sha256(cases_bytes).hexdigest()
    hashes[str(cases_path)] = cases_sha
    cases = [json.loads(line) for line in cases_bytes.splitlines() if line.strip()]
    if len(cases) != 64 or len({c["parent_id"] for c in cases}) != 32:
        raise ValueError("fixed32parent/64variant panel required")
    by_id = {c["id"]: c for c in cases}
    manifest = read(cases_path.with_name("MANIFEST.json"))
    if manifest["cases_sha256"] != cases_sha:
        raise ValueError("cases manifest differs")
    for path, sha in manifest["official_metric_sha256"].items():
        if baseline.panel.sha256(Path(path)) != sha:
            raise ValueError("official metric source differs")
        hashes[path] = sha
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        baseline.native.evaluation.planner.BASE, local_files_only=True, trust_remote_code=False
    )
    jobs = [
        {
            "episode_id": f"{c['id']}-{seed}",
            "case_id": c["id"],
            "parent_id": c["parent_id"],
            "seed": seed,
        }
        for c in cases
        for seed in baseline.SEEDS
    ]
    plans, groups, all_pairs = {}, {}, {}
    for arm, directory in outputs.items():
        output = Path(directory).resolve()
        owners = list(output.glob("OWNER-*.json"))
        if not owners or any(
            not p.with_name(p.name.replace("OWNER-", "TERMINAL-")).exists() for p in owners
        ):
            raise ValueError("terminal completed/capped owner required; no partial readout")
        plan = read(output / "PLAN.json")
        plans[arm] = plan
        for path, sha in plan["source_sha256"].items():
            if baseline.panel.sha256(Path(path)) != sha:
                raise ValueError("sealed collector dependency changed")
            hashes[path] = sha
        model_manifest = Path(plan["model"]) / "local-research-manifest.json"
        read(model_manifest, plan["model_manifest_sha256"])
        if (
            plan["cases_sha256"] != cases_sha
            or plan["jobs"] != jobs
            or plan["seeds"] != list(baseline.SEEDS)
            or plan["model"] != str(baseline.native.evaluation.planner.BASE)
            or plan["prompt_instruction"] != baseline.INSTRUCTION
        ):
            raise ValueError("paired observation/sampling inventory differs")
        if arm == "base":
            if plan["adapters"] is not None:
                raise ValueError("base unexpectedly adapted")
        else:
            adapter = plan["adapters"]
            if plan["comparison_arm"] != arm or adapter["arm"] != arm or adapter["step"] != 32:
                raise ValueError("fixed endpoint arm differs")
            path = Path(adapter["path"])
            commit = read(path / "COMMIT.json", adapter["commit_sha256"])
            state = read(path / "STATE.json", adapter["files"]["STATE.json"])
            training = read(path.parent / "PLAN.json", adapter["training_plan_sha256"])
            read(path / "adapter_config.json", adapter["files"]["adapter_config.json"])
            if (
                commit["step"] != 32
                or state["step"] != 32
                or state["epoch"] != 1
                or state["cursor"] != 0
                or training["comparison_arm"] != arm
                or any(commit["files"][k] != sha for k, sha in adapter["files"].items())
            ):
                raise ValueError("checkpoint/training identity differs")
        records, calls = {}, []
        expected_ids = {j["episode_id"] for j in jobs}
        if any(
            p.stem not in expected_ids
            for folder in ("calls", "episodes")
            for p in (output / folder).glob("*.json")
        ):
            raise ValueError("unplanned receipt")
        for job in jobs:
            call_path = output / "calls" / (job["episode_id"] + ".json")
            episode_path = output / "episodes" / (job["episode_id"] + ".json")
            current = {"status": "missing", "prediction": None}
            if call_path.exists():
                call = read(call_path)
                calls.append(call)
                current = audit_call(call, job, by_id[job["case_id"]], plan, arm, tokenizer)
                if not episode_path.exists():
                    current = {"status": "missing_episode", "prediction": None}
                else:
                    episode = read(episode_path)
                    if (
                        any(episode[k] != v for k, v in job.items())
                        or episode["request_digest"] != call["request_digest"]
                        or episode["available"] != call["available"]
                        or episode["prediction"] != current["prediction"]
                        or episode["call_id"] != job["episode_id"]
                    ):
                        raise ValueError("saved episode differs from native response")
            elif episode_path.exists():
                raise ValueError("episode without native receipt")
            records[job["case_id"], job["seed"]] = current
        rows = pair_metrics(cases, records)
        all_pairs[arm] = rows
        returned_ids = {c["call_id"] for c in calls}
        unresolved = [
            read(p) for p in (output / "starts").glob("*.json") if p.stem not in returned_ids
        ]
        groups[arm] = {
            "output": str(output),
            "planned_variant_attempts": 128,
            "planned_pair_attempts": 64,
            "parent_count": 32,
            "status_counts": dict(Counter(r["status"] for r in records.values())),
            "metrics": {
                key: {
                    **interval(parent_values(rows, key), draws),
                    "numerator": sum(r[key] for r in rows),
                    "denominator": 64,
                }
                for key in METRICS
            },
            "unresolved_started_calls": len(unresolved),
            "physical_cost": analyze_helper.measured(calls + unresolved),
            "pairs": rows,
        }
        for owner in owners:
            read(owner)
            read(owner.with_name(owner.name.replace("OWNER-", "TERMINAL-")))
    for arm in ("joint", "positive_only"):
        if arm in plans:
            if (
                plans[arm]["baseline_plan_sha256"]
                != hashes[str(Path(outputs["base"]).resolve() / "PLAN.json")]
            ):
                raise ValueError("reused baseline PLAN binding differs")
            for name, sha in plans[arm]["baseline_calls_sha256"].items():
                read(Path(outputs["base"]) / "calls" / name, sha)
    comparisons = {}
    for left, right in combinations(outputs, 2):
        pairs = list(zip(all_pairs[left], all_pairs[right], strict=True))
        deltas = {}
        for metric in METRICS:
            a, b = parent_values(all_pairs[left], metric), parent_values(all_pairs[right], metric)
            deltas[metric] = interval({p: b[p] - a[p] for p in a}, draws)
        changes = Counter()
        for a, b in pairs:
            if not a["both_observed"] or not b["both_observed"]:
                changes["unknown_pair_comparison"] += 1
            elif a["joint_em"] != b["joint_em"]:
                direction = "win" if b["joint_em"] > a["joint_em"] else "loss"
                changes[
                    direction
                    + (
                        "_both_valid"
                        if a["both_valid"] and b["both_valid"]
                        else "_protocol_involved"
                    )
                ] += 1
            else:
                changes["tie"] += 1
        comparisons[right + "_minus_" + left] = {"metrics": deltas, "changes": dict(changes)}
    return {
        "method": {
            "parents": 32,
            "repeats": 2,
            "variants_per_parent": 2,
            "bootstrap_unit": "parent, retaining both variants and both seeds",
            "draws": draws,
            "seed": SEED,
            "ci": "percentile95",
            "caveat": "Exploratory exposed DEV; parent clusters do not assert atomic "
            "independence. Missing metrics use planned-denominator lower bounds; "
            "abstention/overanswer counts are observed-only, not missing correctness.",
            "adapter_audit": "Native request identity and committed small receipts; "
            "adapter/base weight hashes reused, no multi-GB ancestry rehash.",
        },
        "groups": groups,
        "comparisons": comparisons,
        "plans": plans,
        "source_sha256": hashes,
        "analyzer_sha256": baseline.panel.sha256(Path(__file__)),
    }


def markdown(report):
    lines = [
        "# Fixed-endpoint answer/sufficiency readout",
        "",
        "32 exposed DEV parents, each with both variants and two seeds; exploratory.",
        "",
        "| Arm | Pair EM /64 | Pair F1 | Both labels /64 | Positive EM /64 | "
        "Positive abstain /64 | Negative overanswer /64 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, group in report["groups"].items():
        m = group["metrics"]
        lines.append(
            f"|{arm}|{m['joint_em']['numerator']:.0f}|{m['joint_f1']['estimate']:.4f}|"
            f"{m['both_labels']['numerator']:.0f}|{m['positive_em']['numerator']:.0f}|"
            f"{m['positive_abstention']['numerator']:.0f}|"
            f"{m['negative_overanswer']['numerator']:.0f}|"
        )
    for name, comparison in report["comparisons"].items():
        lines += [
            "",
            f"{name}: paired EM {comparison['metrics']['joint_em']}; "
            f"F1 {comparison['metrics']['joint_f1']}; {comparison['changes']}.",
        ]
    for arm, group in report["groups"].items():
        lines += [
            "",
            f"{arm}: status {group['status_counts']}; native cost {group['physical_cost']}.",
        ]
    lines += [
        "",
        report["method"]["caveat"],
        "",
        "Official paired EM/F1 requires both answerability decisions correct; "
        "supported-answer EM/F1 is reported separately. Gains from abstention alone "
        "do not establish improved reading. Equal training examples/updates, not "
        "equal target tokens or information. No selected DEV checkpoint.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--positive-only", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report already exists")
    report = analyze(
        {"base": args.baseline, "positive_only": args.positive_only, "joint": args.joint},
        args.cases,
    )
    baseline.native.save(args.report, report)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(report))
    print(json.dumps({"report": str(args.report)}))
