"""Fixed BF16 readout of one successful additional FP16-trained update only."""

import argparse
import json
import math
from pathlib import Path

import analyze_textcraft_endpoint_fresh as comparison
import eval_textcraft_endpoint_fresh as reader

c, read, sha = reader.c, reader.control.read, reader.c.inputs.sha
TRAINING = c.ROOT / "textcraft-terminal-fp16-continuation-002"
TRAIN_PLAN_SHA = "8de589b04f52907c4565bb0e7ac6f2cb7483e55deea50d3e3652d7d151c343b4"
CP1_OUTPUT = c.ROOT / "textcraft-fresh-stopped-rl1-001"
AMENDMENT = c.ROOT / "TEXTCRAFT-STOPPED-STEP1-AMENDMENT-001.json"


def validate_terminal(terminal, summary, state, commit, adam_steps):
    for row in (terminal, summary):
        if (
            row.get("complete") is not True
            or row.get("failure")
            or row.get("actual_optimizer_steps") != 2
            or row.get("new_optimizer_steps") != 1
            or row.get("cumulative_optimizer_steps") != 2
            or row.get("committed_optimizer_steps") != 2
            or row.get("committed_sampled_batches") not in (2, 3)
            or row.get("new_sampled_batches") != row["committed_sampled_batches"] - 1
        ):
            raise ValueError("normal exact one-new/two-cumulative update endpoint required")
    if (
        terminal.get("stopped")
        or terminal.get("endpoint_usable") is not True
        or terminal["committed_sampled_batches"] != summary["committed_sampled_batches"]
        or state.get("step") != 2
        or commit.get("step") != 2
        or adam_steps != [2]
        or state.get("sample_cursor") != terminal["committed_sampled_batches"]
    ):
        raise ValueError("actual committed endpoint/Adam step2 required, not initial cp1")
    update = state.get("update", {})
    fields = (
        "objective",
        "gradient_norm",
        "adapter_l2_delta",
        "train_eval_replay_max_abs",
        "train_eval_replay_mean_abs",
    )
    if (
        update.get("optimizer_called") is not True
        or update.get("nonzero_action_calls", 0) <= 0
        or update.get("train_eval_replay_token_count", 0) <= 0
        or any(not math.isfinite(update.get(k, float("nan"))) for k in fields)
        or update["gradient_norm"] <= 0
        or update["adapter_l2_delta"] <= 0
        or update["train_eval_replay_max_abs"] > 0.25
        or update["train_eval_replay_mean_abs"] > 0.025
    ):
        raise ValueError("finite nonzero qualified update required")


def inputs():
    template, tasks = reader.template_inputs()
    if sha(TRAINING / "PLAN.json") != TRAIN_PLAN_SHA:
        raise ValueError("fixed continuation PLAN changed; new acceptance required")
    plan = read(TRAINING / "PLAN.json")
    if (
        plan["schema"] != "textcraft-explicit-fp16-one-update-continuation-v1"
        or plan["model"] != template["model"]
        or plan["model_manifest_sha256"] != template["model_manifest_sha256"]
        or plan["base_dtype"] != "torch.float16"
        or plan["lora_dtype"] != "torch.float32"
        or plan["ancestor_actual_steps"] != 1
        or plan["maximum_new_optimizer_updates"] != 1
    ):
        raise ValueError("accepted continuation/base contract differs")
    return template, tasks, plan


def endpoint():
    import torch

    _, _, plan = inputs()
    terminal, pins = reader.released_terminal(TRAINING)
    summary = read(TRAINING / "SUMMARY.json")
    adapter = Path(terminal["endpoint"]).resolve()
    cursor = terminal["committed_sampled_batches"]
    expected = TRAINING / "boundaries" / f"sample-{cursor:04d}" / "checkpoint-0002"
    if adapter != expected.resolve():
        raise ValueError("terminal-selected checkpoint must be actual fixed cp2 boundary")
    state, commit = read(adapter / "STATE.json"), read(adapter / "COMMIT.json")
    for name in ("STATE.json", "adapter_config.json", "adapter_model.safetensors", "optimizer.pt"):
        if sha(adapter / name) != commit["files"][name]:
            raise ValueError("committed endpoint changed: " + name)
    optimizer = torch.load(
        adapter / "optimizer.pt", map_location="cpu", weights_only=True, mmap=True
    )
    adam_steps = sorted({int(x["step"].item()) for x in optimizer["state"].values()})
    validate_terminal(terminal, summary, state, commit, adam_steps)
    ancestor = Path(plan["restore_checkpoint"])
    if (
        state["plan_sha256"] != TRAIN_PLAN_SHA
        or state["ancestor_commit_sha256"]
        != plan["ancestor_receipt_sha256"][str(ancestor / "COMMIT.json")]
    ):
        raise ValueError("restored ancestry/state plan differs")
    cfg = read(adapter / "adapter_config.json")
    prior_cfg = read(ancestor / "adapter_config.json")
    for key in ("r", "lora_alpha", "lora_dropout", "base_model_name_or_path", "target_modules"):
        current = sorted(cfg[key]) if key == "target_modules" else cfg[key]
        prior = sorted(prior_cfg[key]) if key == "target_modules" else prior_cfg[key]
        if current != prior:
            raise ValueError("LoRA/base endpoint configuration differs")
    boundary_path = adapter.parent / "BOUNDARY.json"
    boundary = read(boundary_path)
    batch_dir = TRAINING / "batches" / f"sample-{cursor:04d}"
    if (
        boundary["state"] != state
        or boundary["commit_sha256"] != sha(adapter / "COMMIT.json")
        or sha(batch_dir / "BATCH.json") != state["batch_sha256"]
        or read(batch_dir / "OPTIMIZER-STEP-STARTED.json")["previous_step"] != 1
        or read(batch_dir / "BEFORE_LOGPS.json")["all_finite"] is not True
        or read(batch_dir / "TRAIN-EVAL-REPLAY.json")["passed"] is not True
    ):
        raise ValueError("actual update boundary/replay identity differs")
    for path in (
        TRAINING / "PLAN.json",
        TRAINING / "SUMMARY.json",
        adapter / "COMMIT.json",
        adapter / "STATE.json",
        boundary_path,
        batch_dir / "BATCH.json",
        batch_dir / "BEFORE_LOGPS.json",
        batch_dir / "TRAIN-EVAL-REPLAY.json",
    ):
        pins[str(path)] = sha(path)
    return dict(
        path=str(adapter),
        sha256=commit["files"]["adapter_model.safetensors"],
        commit_sha256=sha(adapter / "COMMIT.json"),
        state=state,
        endpoint_kind="fp16_continuation_cp2",
        training_plan_sha256=TRAIN_PLAN_SHA,
        actual_optimizer_steps=2,
        new_optimizer_steps=1,
        actual_adam_steps=adam_steps,
        training_receipt_sha256=pins,
        ancestry=plan["ancestor_receipt_sha256"],
        inference_dtype="torch.bfloat16",
    )


def prepare(output):
    template, tasks, _ = inputs()
    binding = endpoint()
    plan = reader.bound_plan(template, "fp16_continuation_cp2", binding, TRAINING)
    plan.update(
        schema="textcraft-fresh16-fp16-trained-cp2-bf16-readout-v1",
        training_dtype="torch.float16",
        inference_dtype="torch.bfloat16",
        endpoint_selection="Only normal completed additional-step endpoint cp2; "
        "initial/flat/failure not eligible. No quality selection.",
        caveat="Additional RL update with FP16 training; BF16 test arithmetic and fixed32slots "
        "constant against cp1. Not an SFT-matched causal claim; fixed panel already exposed.",
    )
    for module in (reader, comparison):
        plan["source_sha256"][str(Path(module.__file__).resolve())] = sha(Path(module.__file__))
    plan["source_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__))
    destination = output / "PLAN.json"
    if destination.exists():
        if read(destination) != plan:
            raise ValueError("immutable fixed cp2 readout changed")
    else:
        c.save(destination, plan)
    return plan, tasks, binding


def analyze(output, report):
    from transformers import AutoTokenizer

    if report.exists() or report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    first, second = read(CP1_OUTPUT / "PLAN.json"), read(output / "PLAN.json")
    if first["adapter"] != reader.endpoint(
        "rl", Path(first["training_output"]), stopped_amendment=AMENDMENT
    ):
        raise ValueError("fixed cp1 amendment binding changed")
    if second["adapter"] != endpoint():
        raise ValueError("actual cp2 binding changed")
    comparison.match_slots([first, second])
    tokenizer = AutoTokenizer.from_pretrained(
        first["model"], local_files_only=True, trust_remote_code=False
    )
    audit, profiles = comparison.profiles.audit, comparison.profiles
    arms = [audit.analyze(p, tokenizer, expected_task_count=16) for p in (CP1_OUTPUT, output)]
    rows = [
        {
            (x["task_id"], x["repeat"]): x
            for x in (read(p) for p in (directory / "episodes").glob("*.json"))
        }
        for directory in (CP1_OUTPUT, output)
    ]
    difference = profiles.compare(first["jobs"], rows[0], rows[1])
    result = dict(
        schema="textcraft-cp2-minus-cp1-fixed-bf16-comparison-v1",
        arms=arms,
        cp2_minus_cp1=difference,
        parent_count=16,
        repeats=2,
        planned_per_arm=32,
        bootstrap_draws=20000,
        bootstrap_seed=2026092206,
        unknown="Never imputedzero; sharedrecipes do not imply independentworlds",
        caveat=second["caveat"],
        source_sha256=sha(Path(__file__)),
        plans={str(p): sha(p) for p in (CP1_OUTPUT / "PLAN.json", output / "PLAN.json")},
    )
    c.save(report, result)
    with report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Additional update, fixed BF16 readout\n\n"
            + result["caveat"]
            + "\n\n16parents ×2seeds perpolicy; unknowns are notzero.\n\n```json\n"
            + json.dumps(difference, indent=2)
            + "\n```\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.validate_inputs_only:
        template, tasks, _ = inputs()
        print(
            json.dumps(
                dict(
                    planned_episodes=len(template["jobs"]),
                    parents=len(tasks),
                    endpoint_pending=True,
                    GPU_loaded=False,
                )
            )
        )
    elif args.report:
        analyze(args.output, args.report)
    else:
        plan, tasks, binding = prepare(args.output)
        c.run(args, prepared_run=(plan, tasks), adapter=binding)
