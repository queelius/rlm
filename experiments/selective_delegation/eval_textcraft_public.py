"""Matched052 readout for the fixed public-discovery056 checkpoint23."""

import argparse
import json
from pathlib import Path

import eval_textcraft_trained as shared

collector = shared.collector
PLAN_SHA = "5a34080562a39dc92dee2a813078e730b883fb60944dad96b613598c31f9624c"
CONTRACT_SHA = "d9c0b3afb48a41b79ce7840a2e804e8c0f2f3e56a47fdec48ad419ce735a85c0"
ROWS_SHA = "dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a"
REFERENCE_PLAN_SHA = "c060faeb0efb5fb78d07767870d847af51000d25822060916d1eff801f1dcc2f"


def validate_teacher(plan, contract):
    if (
        contract.get("schema") != "textcraft-public-discovery-training-contract-v1"
        or contract.get("teaching_policy") != "public_observation_prerequisite_discovery"
        or contract.get("rows_sha256") != ROWS_SHA
        or plan.get("rows_sha256") != ROWS_SHA
        or contract.get("manifest_sha256") != plan.get("prepared_manifest_sha256")
        or contract.get("prepared") != plan.get("prepared")
        or contract.get("fixed_checkpoint") != 23
        or contract.get("rows") != 366
    ):
        raise ValueError("fixed public-discovery teacher identity mismatch")


def prepare(args, require_endpoint=True):
    directory = args.adapter.resolve().parent
    plan_path, contract_path = directory / "PLAN.json", directory / "TEACHER-CONTRACT.json"
    if collector.inputs.sha(plan_path) != PLAN_SHA:
        raise ValueError("fixed056 training PLAN differs")
    if collector.inputs.sha(contract_path) != CONTRACT_SHA:
        raise ValueError("fixed056 teacher contract differs")
    contract = json.loads(contract_path.read_text())
    validate_teacher(json.loads(plan_path.read_text()), contract)
    plan, tasks, _ = shared.prepare(args, require_endpoint=False)
    reference_path = collector.ROOT / "textcraft-trained-readout-001/PLAN.json"
    if collector.inputs.sha(reference_path) != REFERENCE_PLAN_SHA:
        raise ValueError("fixed052 comparison PLAN differs")
    reference = json.loads(reference_path.read_text())
    for key in (
        "jobs",
        "sampling",
        "seeds",
        "seed_rule",
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
        "budget_seconds",
        "initial_token_audit_by_profile",
        "tasks_sha256",
    ):
        if plan[key] != reference[key]:
            raise ValueError(f"matched052 scientific schedule differs: {key}")
    plan.update(
        schema="textcraft-public-discovery-sft23-readout-v1",
        teaching_policy=contract["teaching_policy"],
        teacher_contract=contract,
        teacher_contract_sha256=CONTRACT_SHA,
        training_plan_sha256=PLAN_SHA,
        privileged_readout_plan_sha256=collector.inputs.sha(reference_path),
        primary_comparison="public_vs_privileged_original_prompt",
        secondary_comparison="public_vs_privileged_instruction_control",
        prompt_difference="Exact052 jobs/prompts/sampling/budgets; only fixed trained adapter "
        "changes. Public discovery versus privileged teacher histories are not token/FLOP matched. "
        "Base044 missing outcomes remain unknown; base049 observed failures remain failures.",
    )
    plan["source_sha256"][str(Path(__file__).resolve())] = collector.inputs.sha(Path(__file__))
    if not require_endpoint:
        return plan, tasks, None
    binding = shared.endpoint(args.adapter, training_plan_sha256=PLAN_SHA, rows_sha256=ROWS_SHA)
    binding.update(
        teacher_contract_sha256=CONTRACT_SHA, teaching_policy=contract["teaching_policy"]
    )
    plan["adapter"] = binding
    path = args.output / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable public-discovery readout PLAN changed")
    else:
        collector.save(path, plan)
    return plan, tasks, binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args()
    plan, tasks, binding = prepare(args, require_endpoint=not args.validate_inputs_only)
    if args.validate_inputs_only:
        print(
            json.dumps(
                dict(
                    planned_episodes=len(plan["jobs"]),
                    endpoint_pending=True,
                    teaching_policy=plan["teaching_policy"],
                    conditions=plan["conditions"],
                    token_audit=plan["initial_token_audit_by_profile"],
                )
            )
        )
    else:
        collector.run(args, prepared_run=(plan, tasks), adapter=binding)
