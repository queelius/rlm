"""Fixed TextCraft action-SFT23 readout, original and instruction-control flat prompts."""

import argparse
import json
from pathlib import Path

import eval_textcraft as collector

TRAINING_PLAN_SHA256 = "722a031355e0c09601e94e926d3831456d072e89acca2797c3376f2d2756046a"


def paired_jobs(base_jobs):
    return [
        {
            **j,
            "episode_id": j["episode_id"] + "-" + profile,
            "condition": "trained_" + profile,
            "prompt_profile": profile,
        }
        for j in base_jobs
        for profile in ("original", "instruction_control")
    ]


def validate_dose(plan, state, steps):
    expected = dict(
        schema="textcraft-public-flat-action-sft-v1",
        model=str(collector.BASE),
        rows=366,
        tasks=32,
        planned_updates=23,
        epochs=1,
        effective_batch=16,
        last_update_rows=14,
        target_only_json_eos=True,
    )
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("training input/role/model/dose mismatch")
    if any(state.get(k) != v for k, v in dict(step=23, epoch=1, cursor=0).items()):
        raise ValueError("fixed complete checkpoint23 required; no partial or selected endpoint")
    if [s.get("step") for s in steps] != list(range(1, 24)) or [s.get("rows") for s in steps] != [
        16
    ] * 22 + [14]:
        raise ValueError("all23 optimizer receipts/366 consumed rows required")


def endpoint(
    adapter,
    *,
    training_plan_sha256=TRAINING_PLAN_SHA256,
    rows_sha256="caa78390f9d4ac28e600674b26e56375203b72d8cdad1c3f9471da3fb25776a9",
):
    import psutil

    adapter = adapter.resolve()
    if adapter.name != "checkpoint-0023":
        raise ValueError("fixed checkpoint0023 only")
    output = adapter.parent
    plan_path = output / "PLAN.json"
    if collector.inputs.sha(plan_path) != training_plan_sha256:
        raise ValueError("not accepted fixed training PLAN")
    plan = json.loads(plan_path.read_text())
    state = json.loads((adapter / "STATE.json").read_text())
    commit = json.loads((adapter / "COMMIT.json").read_text())
    step_paths = sorted((output / "steps").glob("*.json"))
    validate_dose(plan, state, [json.loads(p.read_text()) for p in step_paths])
    owners = sorted(output.glob("OWNER-*.json"))
    if not owners:
        raise ValueError("authenticated completed training owner required")
    for owner_path in owners:
        owner = json.loads(owner_path.read_text())
        if (
            owner.get("source_sha256") != plan["source_sha256"]
            or collector.inputs.sha(Path(owner["source"])) != plan["source_sha256"]
        ):
            raise ValueError("training owner/source identity mismatch")
        terminal_path = owner_path.with_name(owner_path.name.replace("OWNER-", "TERMINAL-"))
        terminal = json.loads(terminal_path.read_text())
        if terminal.get("failure") or terminal.get("stopped"):
            raise ValueError("failed/interrupted training owner is not accepted")
        try:
            process = psutil.Process(owner["pid"])
            if (
                abs(process.create_time() - owner["create_time"]) < 0.01
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                raise ValueError("training owner still live")
        except psutil.NoSuchProcess:
            pass
    if not terminal.get("complete") or terminal.get("step") != 23:
        raise ValueError("complete terminal23 required")
    prepared = Path(plan["prepared"])
    for p, expected in [
        (prepared / "rows.jsonl", plan["rows_sha256"]),
        (prepared / "MANIFEST.json", plan["prepared_manifest_sha256"]),
    ]:
        if collector.inputs.sha(p) != expected:
            raise ValueError("training public-input identity changed")
    if plan["rows_sha256"] != rows_sha256:
        raise ValueError("not frozen training rows")
    required = {"STATE.json", "adapter_config.json", "adapter_model.safetensors"}
    if commit.get("step") != 23 or not required <= set(commit["files"]):
        raise ValueError("checkpoint commit incomplete")
    for name in required:
        if collector.inputs.sha(adapter / name) != commit["files"][name]:
            raise ValueError("checkpoint content differs from COMMIT")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if (
        config.get("r") != 8
        or config.get("lora_alpha") != 16
        or config.get("base_model_name_or_path") != str(collector.BASE)
    ):
        raise ValueError("adapter role/base/rank configuration differs")
    return dict(
        path=str(adapter),
        sha256=commit["files"]["adapter_model.safetensors"],
        commit_sha256=collector.inputs.sha(adapter / "COMMIT.json"),
        state=state,
        training_plan_sha256=collector.inputs.sha(plan_path),
        training_rows_sha256=plan["rows_sha256"],
        terminal_sha256=collector.inputs.sha(terminal_path),
        step_receipts_sha256={str(p): collector.inputs.sha(p) for p in step_paths},
    )


def prepare(args, require_endpoint=True):
    if not 0 < args.hours <= 1.5:
        raise ValueError("readout cap must be at most90minutes")
    plan, tasks = collector.prepare(
        args.prepared, args.output, 0.75, profile="instruction_control", persist=False
    )
    plan.update(
        schema="textcraft-fixed-action-sft23-readout-v1",
        profile="paired_flat_profiles",
        jobs=paired_jobs(plan["jobs"]),
        planned_episodes=32,
        planned_per_policy=None,
        planned_per_condition=16,
        max_native_calls=3072,
        budget_seconds=args.hours * 3600,
        conditions=["trained_original", "trained_instruction_control"],
        fixed_adapter=str(args.adapter.resolve()),
        adapters="textcraft_action, enabled for every flat call; all parameters frozen",
        training_plan_sha256=TRAINING_PLAN_SHA256,
        prompt_difference="Same fixed trained weights; original or fixed trailing reminder. "
        "Compare corresponding base044/base049, missing base outcomes remain unknown.",
        base_original_plan_sha256=collector.inputs.sha(
            collector.ROOT / "textcraft-pilot-001/PLAN.json"
        ),
        base_reminder_plan_sha256=collector.inputs.sha(
            collector.ROOT / "textcraft-instruction-control-001/PLAN.json"
        ),
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        collector.BASE, local_files_only=True, trust_remote_code=False
    )
    original_lengths = [
        len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": collector.bridge.initial_prompt(task, "flat")}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        for task in tasks
    ]
    plan["initial_token_audit_by_profile"] = {
        "original": {
            "prompt_tokens": original_lengths,
            "max_prompt_plus_cap": max(original_lengths) + 256,
        },
        "instruction_control": plan["initial_token_audit"],
    }
    plan["source_sha256"][str(Path(__file__).resolve())] = collector.inputs.sha(Path(__file__))
    if not require_endpoint:
        return plan, tasks, None
    binding = endpoint(args.adapter)
    plan["adapter"] = binding
    path = args.output / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable readout PLAN changed")
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
                    conditions=plan["conditions"],
                    token_audit=plan["initial_token_audit"],
                )
            )
        )
    else:
        collector.run(args, prepared_run=(plan, tasks), adapter=binding)
