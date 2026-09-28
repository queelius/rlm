"""Thin Phi-family binding of the accepted23-update optimizer/checkpoint recipe."""

import argparse
from pathlib import Path
from unittest.mock import patch

import acquire
import qualify

recipe = qualify.recipe
QUALIFICATION_SHA = "006cb29676dd92e0fe4ce56e531c34e2a10be88a6653fcf10998ab4f61bea537"
RECIPE_SHA = "274daede3a32aab96f3ad7914ae22b00cb9c4301efb39f457eeb8c786e6ec439"
QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def phi_lora_kwargs(arguments):
    if arguments.get("target_modules") != QWEN_TARGETS or (
        arguments.get("r") != 8 or arguments.get("lora_alpha") != 16
    ):
        raise ValueError("undeclared change to accepted LoRA recipe")
    return {**arguments, "target_modules": qualify.TARGET_MODULES}


def prepare(teacher, output):
    if acquire.sha(acquire.OUTPUT / "QUALIFICATION.json") != QUALIFICATION_SHA:
        raise ValueError("Phi qualification changed")
    qualification = acquire.read(acquire.OUTPUT / "QUALIFICATION.json")
    for path, digest in qualification["source_sha256"].items():
        if acquire.sha(path) != digest:
            raise ValueError("qualified Phi source changed")
    if acquire.sha(recipe.__file__) != RECIPE_SHA:
        raise ValueError("accepted optimizer/checkpoint recipe changed")
    model_manifest = acquire.MODEL / "local-research-manifest.json"
    acquired = acquire.read(model_manifest)
    if acquired["revision"] != acquire.REVISION or acquired["qualification_sha256"] != (
        QUALIFICATION_SHA
    ):
        raise ValueError("pinned Phi weight qualification differs")
    prepared = acquire.OUTPUT / teacher
    dose = qualification["teaching_doses"][teacher]
    if acquire.sha(prepared / "rows.jsonl") != dose["output_rows_sha256"] or (
        acquire.sha(prepared / "tasks.jsonl") != dose["source_tasks_sha256"]
    ):
        raise ValueError("retokenized teacher text/task source changed")
    manifest = dict(
        schema="textcraft-phi-native-template-sft-20260928-v1",
        teacher=teacher,
        rows=366,
        tasks=32,
        eligible_task_count=32,
        native_successful_tasks=32,
        rows_sha256=dose["output_rows_sha256"],
        tasks_sha256=dose["source_tasks_sha256"],
        model=str(acquire.MODEL),
        model_manifest_sha256=acquire.sha(model_manifest),
        qualification_sha256=QUALIFICATION_SHA,
        source_teaching_dose=dose,
        exact_prompt_target_text_preserved=True,
        stale_qwen_token_caches_removed=True,
    )
    path = prepared / "MANIFEST.json"
    if path.exists():
        if acquire.read(path) != manifest:
            raise ValueError("immutable Phi teacher manifest changed")
    else:
        acquire.save(path, manifest)
    contract = dict(
        schema="textcraft-phi-training-binding-20260928-v1",
        teacher=teacher,
        model=str(acquire.MODEL),
        model_revision=acquire.REVISION,
        model_manifest_sha256=acquire.sha(model_manifest),
        prepared=str(prepared),
        prepared_manifest_sha256=acquire.sha(path),
        rows_sha256=manifest["rows_sha256"],
        qualification_sha256=QUALIFICATION_SHA,
        lora=dict(
            rank=8,
            alpha=16,
            dropout=0,
            target_modules=qualify.TARGET_MODULES,
            trainable_parameters=qualification["lora_trainable_parameters"],
        ),
        seed=recipe.SEED,
        updates=23,
        rows=366,
        tasks=32,
        endpoint="checkpoint-0023",
        target_suffix=qualify.STOP_IDS,
        native_chat_template=True,
        supervised_tokens=dose["supervised_tokens"],
        prompt_tokens=dose["prompt_tokens"],
        overrides="Process-local base model binding, LoRA fused projection names, and "
        "native chat-template target tokenization. Optimizer, target-only loss, update order, "
        "gradient accumulation, checkpoint cadence and owner/caps unchanged.",
        legacy_plan_interpretation="Underlying target_only_json_eos=True includes Phi native "
        "assistant turn-end AND EOS, as explicitly pinned here; no prompt tokens supervised.",
        source_sha256={
            str(p.resolve()): acquire.sha(p)
            for p in (
                Path(__file__),
                Path(qualify.__file__),
                Path(acquire.__file__),
                Path(recipe.__file__),
                Path(recipe.target_loss.__code__.co_filename),
                Path(recipe.probe.__file__),
            )
        },
        dose_caveat="Equal23 updates and366 rows within Phi; neither input-token dose nor "
        "LoRA parameter count nor FLOPs are matched across model families.",
    )
    path = output / "PHI-CONTRACT.json"
    if path.exists():
        if acquire.read(path) != contract:
            raise ValueError("immutable Phi training contract changed")
    else:
        acquire.save(path, contract)
    return prepared, contract


def run(args):
    import peft
    from transformers import AutoTokenizer

    prepared, _ = prepare(args.teacher, args.output.resolve())
    original_config = peft.LoraConfig
    original_tokenizer = AutoTokenizer.from_pretrained

    def config(**kwargs):
        return original_config(**phi_lora_kwargs(kwargs))

    def tokenizer(*args, **kwargs):
        if kwargs.get("trust_remote_code"):
            raise ValueError("remote tokenizer code forbidden")
        return original_tokenizer(*args, **kwargs, trust_remote_code=False)

    # Isolated process only; no edit to any accepted recipe or unrelated environment.
    recipe.probe.campaign.MODELS["4b"] = str(acquire.MODEL)
    recipe.tokenize_rows = qualify.tokenize_rows
    with (
        patch.object(peft, "LoraConfig", config),
        patch.object(AutoTokenizer, "from_pretrained", tokenizer),
    ):
        recipe.run(
            argparse.Namespace(
                prepared=prepared,
                output=args.output,
                epochs=1,
                learning_rate=1e-4,
                hours=args.hours,
                prepare_only=args.prepare_only,
                resume=args.resume,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("discovery", "known"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        recipe.ACTIVE_FAILURE = f"{type(exc).__name__}: {exc}"
        raise
