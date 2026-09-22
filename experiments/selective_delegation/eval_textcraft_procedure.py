"""Frozen048 checkpoint23 plus one procedural prompt, sixteen fixed flat slots."""

import argparse
import json
from pathlib import Path

import eval_textcraft_trained as shared

collector = shared.collector
REFERENCE_PLAN_SHA = "c060faeb0efb5fb78d07767870d847af51000d25822060916d1eff801f1dcc2f"
REFERENCE_REPORT_SHA = "f84ea2aa2a86c1190fe4db0cb069242f0c1e88e35849e6a1a63eca819f8e9887"


def jobs(base_jobs):
    return [
        dict(
            j,
            episode_id=j["episode_id"].removesuffix("-original") + "-procedure_control",
            condition="trained_procedure_control",
            prompt_profile="procedure_control",
        )
        for j in base_jobs
    ]


def prepare(args):
    from transformers import AutoTokenizer

    if not 0 < args.hours <= 0.75:
        raise ValueError("procedure control capped at45minutes")
    plan, tasks, _ = shared.prepare(args, require_endpoint=False)
    reference_path = collector.ROOT / "textcraft-trained-readout-001/PLAN.json"
    report = collector.ROOT / "analysis-textcraft-trained-readout-001.json"
    if (
        collector.inputs.sha(reference_path) != REFERENCE_PLAN_SHA
        or collector.inputs.sha(report) != REFERENCE_REPORT_SHA
    ):
        raise ValueError("fixed052 comparison identity changed")
    reference = json.loads(reference_path.read_text())
    original_jobs = [j for j in reference["jobs"] if j["prompt_profile"] == "original"]
    if [j for j in plan["jobs"] if j["prompt_profile"] == "original"] != original_jobs:
        raise ValueError("fixed task/seed schedule differs")
    binding = shared.endpoint(args.adapter)
    if any(binding[k] != reference["adapter"][k] for k in ("path", "sha256", "commit_sha256")):
        raise ValueError("unchanged048 checkpoint23 required")
    tokenizer = AutoTokenizer.from_pretrained(
        collector.BASE, local_files_only=True, trust_remote_code=False
    )
    lengths = [
        len(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": collector.bridge.initial_prompt(t, "flat")
                        + collector.PROCEDURAL_INSTRUCTION,
                    }
                ],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        for t in tasks
    ]
    if max(lengths) + 256 > 8192:
        raise ValueError("initial procedure context exceeds8192; no truncation")
    plan.update(
        schema="textcraft-procedure-control-v1",
        profile="procedure_control",
        jobs=jobs(original_jobs),
        planned_episodes=16,
        max_native_calls=1536,
        conditions=["trained_procedure_control"],
        adapter=binding,
        procedural_instruction=collector.PROCEDURAL_INSTRUCTION,
        reference_plan_sha256=REFERENCE_PLAN_SHA,
        reference_report_sha256=REFERENCE_REPORT_SHA,
        initial_token_audit={"prompt_tokens": lengths, "max_prompt_plus_cap": max(lengths) + 256},
        prompt_difference="Exact original052 public frame plus one frozen procedural instruction; "
        "same048 checkpoint23, task/seed inventory and per-episode budgets. Packaged prompt "
        "baseline, not isolated teacher training effect. Owner cap45min versus original90min.",
    )
    plan.pop("instruction_reminder", None)
    plan.pop("initial_token_audit_by_profile", None)
    plan["source_sha256"][str(Path(__file__).resolve())] = collector.inputs.sha(Path(__file__))
    path = args.output / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable procedure PLAN changed")
    else:
        collector.save(path, plan)
    return plan, tasks, binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.75)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan, tasks, binding = prepare(args)
    collector.run(args, prepared_run=(plan, tasks), adapter=binding)
