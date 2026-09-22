"""Independent finite paired-RL audit and optional frozen-component held readout."""

import argparse
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import analyze_sufficiency as paired
import rl_sufficiency as training

SEED = 2026092200


def reward_statistics(groups):
    if len(groups) != 16 or any(len(g) != 4 for g in groups):
        raise ValueError("complete16x4 paired candidates required")
    rewards = [[p["reward"] for p in g] for g in groups]
    advantages = [training.rl.rloo(g) for g in rewards]
    return {
        "paired_candidates": 64,
        "paired_reward_sum": sum(map(sum, rewards)),
        "valid_paired_candidates": sum(p["all_valid"] for g in groups for p in g),
        "effective_groups": sum(any(a != 0 for a in g) for g in advantages),
        "valid_mixed_groups": sum(
            len({p["reward"] for p in g if p["all_valid"]}) == 2 for g in groups
        ),
        "fully_protocol_valid_groups": sum(all(p["all_valid"] for p in g) for g in groups),
        "success_count_histogram": dict(Counter(map(sum, rewards))),
        "absolute_advantage_mass": sum(abs(a) for g in advantages for a in g),
    }


def cluster_contrast(left, right, metric, clusters, draws=20000):
    a, b = paired.parent_values(left, metric), paired.parent_values(right, metric)
    flattened = [p for cluster in clusters for p in cluster]
    if set(a) != set(b) or set(flattened) != set(a) or len(flattened) != len(a):
        raise ValueError("frozen complete component partition required")
    delta = {p: b[p] - a[p] for p in a}
    return {
        **paired.analyze_helper.clustered_interval(delta, clusters, draws, SEED),
        "parent_bootstrap_ci95": paired.interval(delta, draws, SEED)["ci95"],
    }


class Audit:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("immutable receipt changed: " + str(path))
        self.hashes[str(path)] = digest
        return json.loads(data)

    def hash(self, path, expected):
        path = Path(path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("source dependency changed: " + str(path))
        self.hashes[str(path)] = digest

    def cases(self, path, expected):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError("case inventory changed")
        self.hashes[str(path)] = digest
        return [json.loads(line) for line in data.splitlines() if line.strip()]

    def terminal(self, output):
        owners = sorted(output.glob("OWNER-*.json"))
        if not owners:
            raise ValueError("no terminal owner; do not analyze ongoing outcomes")
        terminals = []
        for owner in owners:
            self.read(owner)
            terminals.append(self.read(owner.with_name(owner.name.replace("OWNER-", "TERMINAL-"))))
        return terminals

    def checkpoint(self, boundary):
        path = Path(boundary["checkpoint"])
        commit = self.read(path / "COMMIT.json", boundary["commit_sha256"])
        state = self.read(path / "STATE.json", commit["files"]["STATE.json"])
        self.read(path / "adapter_config.json", commit["files"]["adapter_config.json"])
        if state != boundary["state"] or commit["step"] != state["step"]:
            raise ValueError("checkpoint STATE/COMMIT differs")
        return commit


def audit_training_call(call, case, seed, plan, boundary, commit, tokenizer):
    req = call["request"]
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": paired.baseline.prompt(case)}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    expected_adapter = {
        "checkpoint": boundary["checkpoint"],
        "adapter_sha256": commit["files"]["adapter_model.safetensors"],
        "optimizer_step": boundary["state"]["step"],
        "sample_cursor": boundary["state"]["sample_cursor"],
    }
    if (
        req["prompt"] != paired.baseline.prompt(case)
        or req["input_token_ids"] != ids
        or call["input_token_ids"] != ids
        or req["seed"] != seed
        or req["model"] != plan["model"]
        or req["adapter"] != expected_adapter
        or req["adapter_enabled"] is not True
        or req["role"] != "answer_sufficiency"
        or call["request_digest"] != training.probe.runtime.digest(req)
        or req["sampling"]
        != {
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 128,
            "max_time": 90.0,
        }
    ):
        raise ValueError("native TRAIN request/weights/seed differs")
    if not call["available"]:
        return "unavailable", None
    emitted = call["output_token_ids"]
    if (
        not 1 <= len(emitted) <= 128
        or len(call["generation_logps"]) != len(emitted)
        or call["usage"] != {"prompt_tokens": len(ids), "completion_tokens": len(emitted)}
        or tokenizer.decode(emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        != call["text"]
    ):
        raise ValueError("native TRAIN decode/token/logp receipt differs")
    try:
        return "valid", paired.baseline.parse_output(call["text"])
    except (ValueError, TypeError):
        return "protocol_invalid", None


def training_report(output, audit, tokenizer):
    output = output.resolve()
    terminals = audit.terminal(output)
    plan, summary = audit.read(output / "PLAN.json"), audit.read(output / "SUMMARY.json")
    for mapping in (plan["source_sha256"], plan["official_metric_sha256"]):
        for path, digest in mapping.items():
            audit.hash(path, digest)
    for owner_path in output.glob("OWNER-*.json"):
        audit.hash(audit.read(owner_path)["source"], plan["runner_sha256"])
    boundaries = training.boundary_inventory(output)
    commits = [audit.checkpoint(b) for b in boundaries]
    if not boundaries or boundaries[-1]["checkpoint"] != summary["endpoint"]:
        raise ValueError("summary endpoint is not final committed boundary")
    if any(
        b["state"]["plan_sha256"] != audit.hashes[str(output / "PLAN.json")] for b in boundaries
    ):
        raise ValueError("boundary PLAN identity differs")
    cases = audit.cases(plan["cases"], plan["cases_sha256"])
    lookup = training.paired_cases(cases, sum(plan["parent_blocks"], []))
    calls, unresolved, batches, statuses = [], [], [], Counter()
    for directory in sorted((output / "batches").glob("sample-*")):
        cursor = int(directory.name.split("-")[-1])
        prior, commit = boundaries[cursor - 1], commits[cursor - 1]
        groups = []
        if plan["mode"] == "rl":
            expected_ids = set()
            native_records = {}
            for index, parent in enumerate(plan["parent_blocks"][cursor - 1]):
                group = []
                for k in range(4):
                    predictions, observed = [], True
                    for variant, case in enumerate(lookup[parent]):
                        identity = f"p{index:02d}-k{k}-v{variant}"
                        expected_ids.add(identity)
                        path = directory / "calls" / (identity + ".json")
                        status, prediction = "missing", None
                        if path.exists():
                            call = audit.read(path)
                            if call["call_id"] != identity:
                                raise ValueError("native call ID differs")
                            calls.append(call)
                            native_records[identity] = call
                            seed = plan["seed"] + cursor * 100000 + index * 1000 + k * 10 + variant
                            status, prediction = audit_training_call(
                                call, case, seed, plan, prior, commit, tokenizer
                            )
                        statuses[status] += 1
                        predictions.append(prediction)
                        observed &= status in ("valid", "protocol_invalid")
                    pair_path = directory / "pairs" / f"p{index:02d}-k{k}.json"
                    if pair_path.exists():
                        pair = audit.read(pair_path)
                        if (
                            not observed
                            or pair["predictions"] != predictions
                            or pair["reward"]
                            != training.pair_reward(lookup[parent][0], predictions)
                            or pair["all_valid"] != all(p is not None for p in predictions)
                        ):
                            raise ValueError(
                                "saved paired reward differs from native official grade"
                            )
                        group.append(pair)
                groups.append(group)
            if any(p.stem not in expected_ids for p in (directory / "calls").glob("*.json")):
                raise ValueError("unplanned TRAIN receipt")
            unresolved.extend(
                audit.read(p)
                for p in (directory / "starts").glob("*.json")
                if p.stem not in native_records
            )
        committed = cursor < len(boundaries)
        batch = {"sample_cursor": cursor, "committed": committed}
        if committed:
            update = audit.read(directory / "UPDATE.json")
            batch.update(update)
            updated = boundaries[cursor]["state"]["step"] > prior["state"]["step"]
            if update["updated"] != updated or (
                not updated and update.get("optimizer_called") is not False
            ):
                raise ValueError("Adam/update boundary differs")
            if updated and (update["gradient_norm"] <= 0 or update["parameter_delta_l2"] <= 0):
                raise ValueError("claimed update lacks actual gradient/parameter movement")
            if plan["mode"] == "rl":
                stats = reward_statistics(groups)
                actual_rewards = [[pair["reward"] for pair in group] for group in groups]
                if update["rewards"] != actual_rewards:
                    raise ValueError("saved rewards differ from official native grades")
                if stats["effective_groups"] != update["effective_groups"] or updated != bool(
                    stats["effective_groups"]
                ):
                    raise ValueError("RLOO admission/cursor differs")
                batch["recomputed"] = stats
                likelihoods = batch.pop("likelihoods", [])
                if likelihoods:
                    expected_advantages = {
                        f"p{i:02d}-k{k}-v{v}": advantage
                        for i, rewards in enumerate(actual_rewards)
                        for k, advantage in enumerate(training.rl.rloo(rewards))
                        if advantage != 0
                        for v in (0, 1)
                    }
                    if {r["call_id"] for r in likelihoods} != set(expected_advantages):
                        raise ValueError("credited likelihood response inventory differs")
                    for row in likelihoods:
                        native = native_records[row["call_id"]]
                        if (
                            row["advantage"] != expected_advantages[row["call_id"]]
                            or row["generation"] != native["generation_logps"]
                            or len(row["before"]) != len(native["output_token_ids"])
                            or len(row["after"]) != len(row["before"])
                        ):
                            raise ValueError("likelihood/advantage/native token alignment differs")
                    loss = sum(-r["advantage"] * sum(r["before"]) / 64 for r in likelihoods)
                    if not math.isclose(loss, update["loss"], rel_tol=1e-5, abs_tol=1e-4):
                        raise ValueError(
                            "saved loss differs from all64-pair emitted-token objective"
                        )
                    batch["credited_token_count"] = sum(len(r["before"]) for r in likelihoods)
                    batch["advantage_weighted_logp_movement"] = sum(
                        r["advantage"] * (sum(r["after"]) - sum(r["before"])) for r in likelihoods
                    )
                    batch["generation_replay_max_abs_gap_recomputed"] = max(
                        abs(a - b)
                        for row in likelihoods
                        for a, b in zip(row["before"], row["generation"], strict=True)
                    )
        batches.append(batch)
    return {
        "output": str(output),
        "mode": plan["mode"],
        "plan": plan,
        "terminals": terminals,
        "summary": summary,
        "boundaries": boundaries,
        "batches": batches,
        "native_status_counts": dict(statuses),
        "physical_cost": paired.analyze_helper.measured(calls + unresolved),
        "unresolved_started_calls": len(unresolved),
        "native_service_seconds": sum(c["ended"] - c["started"] for c in calls),
        "max_planned_native_calls": plan["max_calls"],
        "unattempted_or_missing_native_slots": plan["max_calls"] - len(calls),
        "caveat": "TRAIN fit/admission, not generalization; "
        "skipped/capped slots are not reward zero. "
        "Extra SFT matches response/update dose, not target tokens, information or FLOPs.",
    }


def frozen_clusters(cases, manifest, planned_calls, *, expected_calls=384):
    clusters = manifest["component_clusters"]
    parents = Counter(c["parent_id"] for c in cases)
    flattened = [parent for cluster in clusters for parent in cluster]
    if (
        len(cases) != 64
        or len(parents) != 32
        or set(parents.values()) != {2}
        or planned_calls != expected_calls
        or not all(clusters)
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(parents)
    ):
        raise ValueError("frozen held32 component partition/call inventory differs")
    return clusters


def held_metric_sources(plan, manifest, audit):
    if "official_metric_sha256" in manifest:
        mapping = manifest["official_metric_sha256"]
    else:
        profile = plan["panel_profile"]
        roots = {Path(p).parent for p in plan["source_sha256"]}
        if len(roots) != 1:
            raise ValueError("held profile source root differs")
        root = roots.pop()
        frozen = audit.read(root / "COMPOSITIONAL-PROFILE.json", profile["profile_sha256"])
        audit.hash(root / "eval_sufficiency_compositional.py", profile["entrypoint_sha256"])
        if (
            frozen
            != {
                k: v for k, v in profile.items() if k not in ("profile_sha256", "entrypoint_sha256")
            }
            or frozen["cases_sha256"] != plan["cases_sha256"]
            or frozen["manifest_sha256"] != plan["manifest_sha256"]
            or frozen["readout_schema"] != plan["schema"]
            or frozen["component_cluster_count"] != len(manifest["component_clusters"])
        ):
            raise ValueError("held frozen profile differs")
        mapping = frozen["metric_sha256"]
    actual = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (paired.baseline.panel.OFFICIAL / "metrics").glob("*.py")
    }
    if mapping != actual:
        raise ValueError("held metric identity differs from actual official grader")
    return mapping


def held_report(output, cases_path, audit, tokenizer, draws=20000, *, expected_calls=384):
    output = output.resolve()
    audit.terminal(output)
    plan = audit.read(output / "PLAN.json")
    manifest = audit.read(cases_path.with_name("MANIFEST.json"), plan["manifest_sha256"])
    cases = audit.cases(cases_path, plan["cases_sha256"])
    for mapping in (plan["source_sha256"], held_metric_sources(plan, manifest, audit)):
        for path, digest in mapping.items():
            audit.hash(path, digest)
    expected_jobs = [
        {
            "episode_id": f"{c['id']}-{seed}",
            "case_id": c["id"],
            "parent_id": c["parent_id"],
            "seed": seed,
        }
        for c in cases
        for seed in paired.baseline.SEEDS
    ]
    if plan["jobs"] != expected_jobs or plan["seeds"] != list(paired.baseline.SEEDS):
        raise ValueError("held full paired inventory/seeds differ")
    if (
        plan["model"] != str(training.BASE)
        or plan["prompt_instruction"] != paired.baseline.INSTRUCTION
    ):
        raise ValueError("held model/public instruction differs")
    clusters = frozen_clusters(
        cases, manifest, plan["planned_calls"], expected_calls=expected_calls
    )
    lookup, groups, all_rows = {c["id"]: c for c in cases}, {}, {}
    for condition in plan["conditions"]:
        subplan = audit.read(output / condition / "PLAN.json")
        if subplan["adapters"] != plan["adapters"][condition]:
            raise ValueError("held adapter identity differs")
        adapter = subplan["adapters"]
        checkpoint = Path(adapter["path"])
        commit = audit.read(checkpoint / "COMMIT.json", adapter["commit_sha256"])
        for name in ("STATE.json", "adapter_config.json"):
            audit.read(checkpoint / name, adapter["files"][name])
        if any(commit["files"][name] != digest for name, digest in adapter["files"].items()):
            raise ValueError("held checkpoint identity differs")
        records, calls = {}, []
        expected_ids = {j["episode_id"] for j in plan["jobs"]}
        if any(
            p.stem not in expected_ids
            for folder in ("calls", "episodes")
            for p in (output / condition / folder).glob("*.json")
        ):
            raise ValueError("unplanned held receipt")
        for job in plan["jobs"]:
            path = output / condition / "calls" / (job["episode_id"] + ".json")
            row = {"status": "missing", "prediction": None}
            if path.exists():
                call = audit.read(path)
                calls.append(call)
                row = paired.audit_call(
                    call, job, lookup[job["case_id"]], subplan, condition, tokenizer
                )
                episode_path = output / condition / "episodes" / path.name
                if not episode_path.exists():
                    row = {"status": "missing_episode", "prediction": None}
                else:
                    episode = audit.read(episode_path)
                    if any(episode[k] != v for k, v in job.items()) or (
                        episode["request_digest"] != call["request_digest"]
                        or episode["prediction"] != row["prediction"]
                        or episode["available"] != call["available"]
                    ):
                        raise ValueError("held episode/native response differs")
            records[job["case_id"], job["seed"]] = row
        rows = paired.pair_metrics(cases, records)
        all_rows[condition] = rows
        returned_ids = {c["call_id"] for c in calls}
        unresolved = [
            audit.read(p)
            for p in (output / condition / "starts").glob("*.json")
            if p.stem not in returned_ids
        ]
        groups[condition] = {
            "planned_variant_attempts": 128,
            "planned_pair_attempts": 64,
            "status_counts": dict(Counter(r["status"] for r in records.values())),
            "metrics": {
                m: {"numerator": sum(r[m] for r in rows), "denominator": 64} for m in paired.METRICS
            },
            "physical_cost": paired.analyze_helper.measured(calls + unresolved),
            "unresolved_started_calls": len(unresolved),
            "pairs": rows,
        }
    comparisons = {}
    for left, right in combinations(plan["conditions"], 2):
        changes = Counter()
        for a, b in zip(all_rows[left], all_rows[right], strict=True):
            if not a["both_observed"] or not b["both_observed"]:
                changes["unknown_pair_comparison"] += 1
            elif a["joint_em"] == b["joint_em"]:
                changes["tie"] += 1
            else:
                direction = "win" if b["joint_em"] > a["joint_em"] else "loss"
                changes[
                    direction
                    + (
                        "_both_valid"
                        if a["both_valid"] and b["both_valid"]
                        else "_protocol_involved"
                    )
                ] += 1
        comparisons[right + "_minus_" + left] = {
            "metrics": {
                m: cluster_contrast(all_rows[left], all_rows[right], m, clusters, draws)
                for m in paired.METRICS
            },
            "changes": dict(changes),
        }
    return {
        "groups": groups,
        "comparisons": comparisons,
        "plan": plan,
        "method": {
            "parent_count": 32,
            "component_count": len(clusters),
            "clusters": clusters,
            "unit": "frozen atomic-component clusters, both variants and repeats retained",
            "draws": draws,
            "seed": SEED,
            "ci": "percentile95; parent-only companion",
            "caveat": "32 parents, not64 independent pairs. Missing score lower bounds "
            "retain planned denominators; unknown is not observed failure.",
        },
    }


def markdown(report):
    lines = ["# Paired sufficiency RL/control readout", ""]
    for arm, data in report["training"].items():
        summary = data["summary"]
        lines += [
            f"{arm}: {summary['actual_optimizer_steps']} real updates from "
            f"{summary['committed_sampled_blocks']} sampled blocks; "
            f"stop={summary['stop_reason']}, failure={summary['failure']}.",
            f"Native status {data['native_status_counts']}; cost {data['physical_cost']}.",
            "",
        ]
        for batch in data["batches"]:
            lines.append(
                f"- Block {batch['sample_cursor']}: committed={batch['committed']}; "
                f"reward={batch.get('recomputed', {})}; gradient={batch.get('gradient_norm')}; "
                f"delta={batch.get('parameter_delta_l2')}; replay gap="
                f"{batch.get('generation_replay_max_abs_gap_recomputed')}."
            )
        lines += ["", data["caveat"], ""]
    if report.get("held"):
        held = report["held"]
        lines += [
            "## Frozen held readout",
            "",
            f"32 parents /{held['method']['component_count']} frozen component clusters; "
            "two seeds retain paired positive/negative dependence.",
            "",
        ]
        for arm, group in held["groups"].items():
            metrics = {k: v["numerator"] for k, v in group["metrics"].items()}
            lines.append(
                f"{arm}: pair EM {metrics['joint_em']:.0f}/64, "
                f"pair F1 {metrics['joint_f1'] / 64:.4f}, "
                f"both labels {metrics['both_labels']:.0f}/64, "
                f"positive EM {metrics['positive_em']:.0f}/64; "
                f"positive abstain {metrics['positive_abstention']:.0f}/64, "
                f"negative overanswer {metrics['negative_overanswer']:.0f}/64. "
                f"Status {group['status_counts']}; native cost {group['physical_cost']}."
            )
        for name, comparison in held["comparisons"].items():
            lines += [
                "",
                f"{name}: paired EM {comparison['metrics']['joint_em']}; "
                f"paired F1 {comparison['metrics']['joint_f1']}; "
                f"changes {comparison['changes']}.",
            ]
        lines += ["", held["method"]["caveat"]]
    lines += [
        "",
        "Competency/objective diagnostic, not a novelty claim. More abstention alone "
        "does not establish better supported-answer accuracy.",
    ]
    return "\n".join(lines) + "\n"


def main(args):
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        training.BASE, local_files_only=True, trust_remote_code=False
    )
    audit = Audit()
    report = {"training": {"rl": training_report(args.rl_output, audit, tokenizer)}}
    if args.sft_output:
        report["training"]["extra_sft"] = training_report(args.sft_output, audit, tokenizer)
        rl, sft = report["training"]["rl"], report["training"]["extra_sft"]
        if (
            sft["plan"]["matched_rl_boundaries"] != rl["boundaries"]
            or sft["plan"]["warmstart"] != rl["plan"]["warmstart"]
        ):
            raise ValueError("control does not match actual committed RL schedule")
    if args.held_output:
        if not args.held_cases:
            raise ValueError("held cases required")
        report["held"] = held_report(args.held_output, args.held_cases, audit, tokenizer)
        for condition, arm in (("rl_terminal", "rl"), ("matched_sft_terminal", "extra_sft")):
            if arm not in report["training"]:
                raise ValueError("held readout requires both authenticated training endpoints")
            identity = report["held"]["plan"]["adapters"][condition]
            data = report["training"][arm]
            if (
                any(t.get("failure") or t.get("stopped") for t in data["terminals"])
                or identity["path"] != data["summary"]["endpoint"]
                or identity["boundaries"] != data["boundaries"]
                or identity["training_plan_sha256"]
                != audit.hashes[str(Path(data["output"]) / "PLAN.json")]
            ):
                raise ValueError("held policy not actual declared training endpoint")
    report["source_sha256"] = audit.hashes
    report["analyzer_sha256"] = paired.baseline.panel.sha256(Path(__file__))
    paired.baseline.native.save(args.report, report)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(report))
    print(json.dumps({"report": str(args.report)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-output", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path)
    parser.add_argument("--held-output", type=Path)
    parser.add_argument("--held-cases", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    main(parser.parse_args())
