"""Native-audited public056 minus privileged048 TextCraft transfer comparison."""

import argparse
import json
from pathlib import Path

import analyze_textcraft_profiles as profiles
import eval_textcraft_public as public

audit = profiles.audit


def contrasts(jobs, privileged_rows, public_rows):
    result = {}
    for profile in sorted({j["prompt_profile"] for j in jobs}):
        selected = [j for j in jobs if j["prompt_profile"] == profile]
        lookup = []
        for rows in (privileged_rows, public_rows):
            lookup.append(
                {
                    (j["task_id"], j["repeat"]): rows[j["episode_id"]]
                    for j in selected
                    if j["episode_id"] in rows
                }
            )
        result[profile] = profiles.compare(selected, *lookup)
    return result


def analyze(output, privileged_report, expected_collector_sha):
    from transformers import AutoTokenizer

    plan = audit.read(output / "PLAN.json")
    audit.require(plan["schema"] == "textcraft-public-discovery-sft23-readout-v1", "public057 only")
    binding = public.shared.endpoint(
        Path(plan["adapter"]["path"]),
        training_plan_sha256=public.PLAN_SHA,
        rows_sha256=public.ROWS_SHA,
    )
    audit.require(binding["sha256"] == plan["adapter"]["sha256"], "public endpoint changed")
    audit.require(plan["teacher_contract_sha256"] == public.CONTRACT_SHA, "teacher differs")
    old_output = audit.collector.ROOT / "textcraft-trained-readout-001"
    old_plan_path = old_output / "PLAN.json"
    old_plan, old_report = audit.read(old_plan_path), audit.read(privileged_report)
    audit.require(
        audit.collector.inputs.sha(old_plan_path)
        == public.REFERENCE_PLAN_SHA
        == old_report["sha256"][str(old_plan_path)]
        == plan["privileged_readout_plan_sha256"],
        "privileged052 PLAN differs from independent audit",
    )
    for key in (
        "jobs",
        "tasks_sha256",
        "manifest_sha256",
        "seeds",
        "sampling",
        "model_manifest_sha256",
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
        "seed_rule",
        "budget_seconds",
    ):
        audit.require(plan[key] == old_plan[key], "unmatched teacher readout: " + key)
    for path, digest in old_report["sha256"].items():
        audit.require(audit.collector.inputs.sha(Path(path)) == digest, "old audit input changed")
    old_rows = old_report["episode_rows"]
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    result = audit.analyze(output, tokenizer, expected_collector_sha256=expected_collector_sha)
    rows = {p.stem: audit.read(p) for p in (output / "episodes").glob("*.json")}
    result.pop("paired", None)
    result.pop("depth_strata", None)
    result.update(
        public_minus_privileged=contrasts(plan["jobs"], old_rows, rows),
        privileged_groups=old_report["groups"],
        privileged_physical_cost=old_report["physical_cost"],
        privileged_report=str(privileged_report),
        privileged_report_sha256=audit.collector.inputs.sha(privileged_report),
        primary_profile="original",
        secondary_profile="instruction_control",
        episode_rows=rows,
    )
    result["method"].update(
        bootstrap_draws=20000,
        bootstrap_seed=2026092206,
        teacher_caveat="Eight exposed task parents, two correlated seeds, shared recipe world. "
        "Same prompts/dose of366 training rows and23 updates, different teacher histories/token "
        "dose. Original prompt is primary. Missing044 is unknown;049 failures remain failures.",
    )
    for module in (profiles, public):
        path = Path(module.__file__)
        result["sha256"][str(path)] = audit.collector.inputs.sha(path)
    result["sha256"][str(Path(__file__))] = audit.collector.inputs.sha(Path(__file__))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--privileged-report", type=Path, required=True)
    parser.add_argument("--expected-collector-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise ValueError("immutable report exists")
    result = analyze(args.output, args.privileged_report, args.expected_collector_sha256)
    audit.collector.save(args.report, result)
    lines = [
        "# Public versus privileged TextCraft teacher",
        "",
        result["method"]["teacher_caveat"],
        "",
    ]
    for profile, row in result["public_minus_privileged"].items():
        lines += [profile + ": " + json.dumps({k: v for k, v in row.items() if k != "rows"}), ""]
    lines += [
        "Public groups/costs: " + json.dumps(result["groups"]),
        "",
        "Privileged groups/costs: " + json.dumps(result["privileged_groups"]),
    ]
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps(result["public_minus_privileged"]))
