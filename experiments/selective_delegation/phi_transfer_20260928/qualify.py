"""Meta-only architecture/LoRA and exact-text native-tokenizer qualification."""

import hashlib
import inspect
import json
import platform
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import acquire  # noqa: E402
import textcraft_bridge as bridge  # noqa: E402
import train_textcraft_sft as recipe  # noqa: E402

TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
STOP_IDS = [200020, 199999]
SOURCES = {
    "discovery": {
        "path": acquire.ROOT / "textcraft-public-discovery-prototype-001",
        "manifest_sha256": "c69ef258a07f4c4f9b45b5bc044880f1cc9aa884e510fe590f3ddf77f19441da",
        "rows_sha256": "dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a",
    },
    "known": {
        "path": acquire.ROOT / "textcraft-quantity-matched-inputs-004",
        "manifest_sha256": "25ed7f91209365533d1d93222d76ea55b0b777c84461d1a7e3ddbf3563d89c48",
        "rows_sha256": "24ea72cb1242f2e0d819d8fb115737864de03fb750e064f145f9ec48a245e6d6",
    },
}


def tokenize_rows(rows, tokenizer):
    examples = []
    for row in rows:
        recipe.strict_action(row["target"])
        messages = [{"role": "user", "content": row["prompt"]}]
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row["target"]}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        target = full[len(prefix) :]
        if full[: len(prefix)] != prefix or target[-2:] != STOP_IDS:
            raise ValueError("native Phi assistant template/prefix does not match pinned contract")
        if (
            tokenizer.decode(
                target[:-2], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            != row["target"]
        ):
            raise ValueError("action target changed under native tokenization")
        if len(full) > 8192 or len(target) > 256:
            raise ValueError("native context/target cap exceeded; no truncation or row removal")
        examples.append(
            dict(
                id=f"{row['task_id']}:{row['step']}",
                input_ids=full[:-1],
                target_ids=target,
                prompt_tokens=len(prefix),
                target_tokens=len(target),
            )
        )
    return examples


def qualify():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    asset = acquire.read(acquire.MODEL / "SMALL-ASSET-ACQUISITION.json")
    for row in asset["assets"]:
        if acquire.sha(acquire.MODEL / row["name"]) != row["sha256"]:
            raise ValueError("small official asset changed")
    config = AutoConfig.from_pretrained(
        acquire.MODEL, local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        acquire.MODEL, local_files_only=True, trust_remote_code=False
    )
    if acquire.read(acquire.MODEL / "generation_config.json")["eos_token_id"] != STOP_IDS:
        raise ValueError("pinned native stop IDs changed")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    if not type(model).__module__.startswith("transformers.models.phi3.") or (
        {str(p.device) for p in model.parameters()} != {"meta"}
    ):
        raise ValueError("built-in meta-only Phi model required")
    model_class = type(model)
    base_parameters = sum(p.numel() for p in model.parameters())
    modules = [
        dict(name=name, input=module.in_features, output=module.out_features)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_MODULES
    ]
    if len(modules) != 128 or "logits_to_keep" not in inspect.signature(model.forward).parameters:
        raise ValueError("expected32 layers/four projections and target-only logits API")
    checkpoint_names = set(
        acquire.read(acquire.MODEL / "model.safetensors.index.json")["weight_map"]
    )
    state_names = set(model.state_dict())
    if checkpoint_names - state_names or state_names - checkpoint_names not in (
        set(),
        {"lm_head.weight"},
    ):
        raise ValueError("built-in parameter names differ from official safetensors index")
    with torch.device("meta"):
        adapted = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=TARGET_MODULES,
            ),
        )
    trainable = [(name, p) for name, p in adapted.named_parameters() if p.requires_grad]
    if not trainable or any(
        "lora_" not in name or str(p.device) != "meta" for name, p in trainable
    ):
        raise ValueError("only meta LoRA parameters may be trainable")
    doses, identities = {}, {}
    for teacher, source in SOURCES.items():
        prepared = source["path"]
        for name, key in (("MANIFEST.json", "manifest_sha256"), ("rows.jsonl", "rows_sha256")):
            if acquire.sha(prepared / name) != source[key]:
                raise ValueError("accepted teacher input changed: " + teacher)
        manifest = acquire.read(prepared / "MANIFEST.json")
        rows = [json.loads(line) for line in (prepared / "rows.jsonl").read_text().splitlines()]
        recipe.validate_rows(rows, manifest)
        tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
        if any(not t["id"].startswith("textcraft_synth.train.") for t in tasks):
            raise ValueError("teacher source contains non-TRAIN task")
        identities[teacher] = sorted((t["id"], t["misc"]["target_items"]) for t in tasks)
        examples = tokenize_rows(rows, tokenizer)
        directory = acquire.OUTPUT / teacher
        directory.mkdir(parents=True)
        # Drop only stale Qwen token caches; preserve each prompt/target and source row order.
        with (directory / "rows.jsonl").open("x") as stream:
            for row in rows:
                record = {k: row[k] for k in ("task_id", "step", "prompt", "target", "feedback")}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (directory / "tasks.jsonl").open("xb") as stream:
            stream.write((prepared / "tasks.jsonl").read_bytes())
        per_row = [
            dict(
                id=e["id"],
                prompt_tokens=e["prompt_tokens"],
                target_tokens=e["target_tokens"],
                input_sha256=hashlib.sha256(json.dumps(e["input_ids"]).encode()).hexdigest(),
                target_sha256=hashlib.sha256(json.dumps(e["target_ids"]).encode()).hexdigest(),
            )
            for e in examples
        ]
        dose = dict(
            source=str(prepared),
            source_manifest_sha256=source["manifest_sha256"],
            source_rows_sha256=source["rows_sha256"],
            source_tasks_sha256=acquire.sha(prepared / "tasks.jsonl"),
            rows=366,
            tasks=32,
            prompt_tokens=sum(e["prompt_tokens"] for e in examples),
            supervised_tokens=sum(e["target_tokens"] for e in examples),
            max_full_tokens=max(e["prompt_tokens"] + e["target_tokens"] for e in examples),
            max_target_tokens=max(e["target_tokens"] for e in examples),
            target_action_counts=dict(Counter(json.loads(row["target"])["action"] for row in rows)),
            exact_prompt_target_text_and_row_order_preserved=True,
            output_rows_sha256=acquire.sha(directory / "rows.jsonl"),
            per_row=per_row,
            target_suffix=STOP_IDS,
            truncated_or_removed_rows=0,
        )
        acquire.save(directory / "TOKEN-AUDIT.json", dose)
        doses[teacher] = {k: v for k, v in dose.items() if k != "per_row"}
    if identities["discovery"] != identities["known"]:
        raise ValueError(
            "corrected teacher packages must have identical TRAIN IDs and goal quantities"
        )
    panels = {}
    for seed in (42, 50):
        baseline = acquire.ROOT / f"textcraft-breadth-p00-w{seed}-soriginal-binder-001"
        plan = acquire.read(baseline / "PLAN.json")
        prepared = Path(plan["prepared"])
        if acquire.sha(prepared / "tasks.jsonl") != plan["tasks_sha256"]:
            raise ValueError("fixed panel00 source changed")
        tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
        lengths = [
            len(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": bridge.initial_prompt(task, "flat")}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                )
            )
            for task in tasks
        ]
        if max(lengths) + 256 > 8192 or len(tasks) != 8:
            raise ValueError("fixed panel native initial context exceeded")
        panels[str(seed)] = dict(
            prepared=str(prepared),
            baseline_plan_sha256=acquire.sha(baseline / "PLAN.json"),
            tasks_sha256=plan["tasks_sha256"],
            manifest_sha256=plan["manifest_sha256"],
            task_ids=[t["id"] for t in tasks],
            world_sha256=plan["world_sha256"],
            initial_prompt_tokens=lengths,
            initial_prompt_plus_cap=max(lengths) + 256,
        )
    report = dict(
        schema="phi-textcraft-cpu-qualification-20260928-v1",
        ready_for_weights=True,
        gpu_used=False,
        scientific_model_calls=0,
        model_revision=acquire.REVISION,
        model_class=model_class.__module__ + "." + model_class.__name__,
        model_source_path=inspect.getfile(model_class),
        model_source_sha256=acquire.sha(inspect.getfile(model_class)),
        config_class=type(config).__module__ + "." + type(config).__name__,
        tokenizer_class=type(tokenizer).__module__ + "." + type(tokenizer).__name__,
        base_parameters=base_parameters,
        native_builtin=True,
        trust_remote_code=False,
        base_dtype="bfloat16",
        devices=["meta"],
        lora_target_modules=TARGET_MODULES,
        lora_module_shapes=modules,
        lora_trainable_parameters=sum(p.numel() for _, p in trainable),
        lora_trainable_tensors=len(trainable),
        lora_dtype_counts=dict(Counter(str(p.dtype) for _, p in trainable)),
        checkpoint_names_not_in_builtin=sorted(checkpoint_names - state_names),
        builtin_names_not_in_checkpoint=sorted(state_names - checkpoint_names),
        tied_input_output_embeddings=config.tie_word_embeddings,
        official_generation_stop_ids=STOP_IDS,
        tokenizer_eos_token_id=tokenizer.eos_token_id,
        teaching_doses=doses,
        fixed_panels=panels,
        environment=dict(
            python=platform.python_version(),
            executable=sys.executable,
            **{name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        ),
        source_sha256={
            str(p.resolve()): acquire.sha(p)
            for p in (
                Path(__file__),
                Path(acquire.__file__),
                Path(recipe.__file__),
                Path(bridge.__file__),
            )
        },
        small_asset_manifest_sha256=acquire.sha(acquire.MODEL / "SMALL-ASSET-ACQUISITION.json"),
        limitations="Meta inspection does not qualify real weight loading or GPU forward/backward. "
        "Original training prompt text includes original Qwen-era budget counters, retained "
        "unchanged as part of the teaching package; live Phi rollouts must use actual Phi costs. "
        "Equal23 updates within family does not equalize tokens, LoRA parameters or compute "
        "across families.",
    )
    acquire.save(acquire.OUTPUT / "QUALIFICATION.json", report)
    return report


if __name__ == "__main__":
    report = qualify()
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "ready_for_weights",
                    "model_class",
                    "base_parameters",
                    "lora_trainable_parameters",
                    "lora_dtype_counts",
                    "teaching_doses",
                    "fixed_panels",
                )
            },
            indent=2,
        )
    )
