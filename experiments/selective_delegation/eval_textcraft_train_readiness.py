"""Proposed TRAIN-only native rollout-diversity screen; no optimizer or RL update."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import eval_textcraft_public as public
import eval_textcraft_trained as shared

c = shared.collector
SOURCE_TASKS = "textcraft-public-discovery-prototype-001/tasks.jsonl"
SOURCE_TASKS_SHA = "390dff9bb19d0fe71c7bec0505c97608013c66c65aea90a821839f01b615ab30"
SOURCE_MANIFEST_SHA = "c69ef258a07f4c4f9b45b5bc044880f1cc9aa884e510fe590f3ddf77f19441da"
SELECT_SEED = 2026092212
SEEDS = (2026092213, 2026092214, 2026092215, 2026092216)
QUOTAS = {2: 2, 3: 2, 4: 4}


def select_tasks(tasks):
    if len({t["id"] for t in tasks}) != len(tasks) or any(
        not t["id"].startswith("textcraft_synth.train.") for t in tasks
    ):
        raise ValueError("unique official TRAIN task identities required")
    selected = []
    for depth, count in QUOTAS.items():
        candidates = [t for t in tasks if t["misc"]["max_depth"] == depth]
        if len(candidates) < count:
            raise ValueError("declared depth quota unavailable; no implicit replacement")
        selected.extend(
            sorted(
                candidates,
                key=lambda t: hashlib.sha256(f"{SELECT_SEED}:{t['id']}".encode()).hexdigest(),
            )[:count]
        )
    return selected


def jobs(tasks):
    return [
        dict(
            episode_id=f"t{i:02d}-r{repeat}-flat",
            task_id=task["id"],
            repeat=repeat,
            seed=seed,
            policy="flat",
            condition="train_public056",
            prompt_profile="original",
        )
        for repeat, seed in enumerate(SEEDS)
        for i, task in enumerate(tasks)
    ]


def freeze_inputs(prepared):
    source = c.ROOT / SOURCE_TASKS
    if (
        c.inputs.sha(source) != SOURCE_TASKS_SHA
        or c.inputs.sha(source.parent / "MANIFEST.json") != SOURCE_MANIFEST_SHA
    ):
        raise ValueError("fixed32 public SFT TRAIN source changed")
    tasks = list(map(json.loads, source.read_text().splitlines()))
    if len(tasks) != 32:
        raise ValueError("exact32 existing SFT TRAIN tasks required")
    selected = select_tasks(tasks)
    selection = dict(
        schema="textcraft-train-readiness-selection-v1",
        split="train",
        selection_seed=SELECT_SEED,
        rule="First SHA256(seed:official_id) within each declared depth; no outcome fields read",
        declared_depth_quotas=QUOTAS,
        source_depth_counts=dict(Counter(t["misc"]["max_depth"] for t in tasks)),
        selected_task_ids=[t["id"] for t in selected],
        source_tasks_sha256=SOURCE_TASKS_SHA,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        prospective_adjustment=None,
        no_replacement=True,
    )
    selection_path = prepared / "SELECTION.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != json.loads(json.dumps(selection)):
            raise ValueError("immutable TRAIN selection changed")
    else:
        c.save(selection_path, selection)
    serialized = "".join(json.dumps(t, sort_keys=True) + "\n" for t in selected)
    task_path = prepared / "tasks.jsonl"
    if task_path.exists():
        if task_path.read_text() != serialized:
            raise ValueError("immutable selected TRAIN tasks changed")
    else:
        with task_path.open("x") as stream:
            stream.write(serialized)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    lengths = [
        len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": c.bridge.initial_prompt(t, "flat")}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        for t in selected
    ]
    if max(lengths) + 256 > 8192:
        raise ValueError("initial TRAIN prompt exceeds unchanged context budget")
    manifest = dict(
        **selection,
        tasks_sha256=c.inputs.sha(task_path),
        selection_sha256=c.inputs.sha(selection_path),
        episode_seeds=list(SEEDS),
        model=str(c.BASE),
        model_manifest_sha256=c.inputs.sha(c.BASE / "local-research-manifest.json"),
        initial_prompt_tokens=lengths,
        max_initial_prompt_plus_cap=max(lengths) + 256,
        source_sha256={str(Path(__file__).resolve()): c.inputs.sha(Path(__file__))},
        caveat="Exact SFT training tasks and same seed42 world. Declared recipe depth is not "
        "remaining execution depth. No VAL task selection or held-out generalization claim. "
        "All sampled native failures/caps and unavailable episodes retained; no success filtering.",
    )
    path = prepared / "MANIFEST.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("immutable TRAIN manifest changed")
    else:
        c.save(path, manifest)
    return manifest, selected


def prepare(args):
    if not 0 < args.hours <= 1:
        raise ValueError("TRAIN readiness is capped at60minutes")
    manifest, tasks = freeze_inputs(args.prepared)
    plan, _ = c.prepare(
        c.ROOT / "textcraft-inputs-001", args.output, args.hours, profile="original", persist=False
    )
    adapter = c.ROOT / "textcraft-public-discovery-sft-001/checkpoint-0023"
    contract_path = adapter.parent / "TEACHER-CONTRACT.json"
    if c.inputs.sha(contract_path) != public.CONTRACT_SHA:
        raise ValueError("fixed056 public teacher contract changed")
    contract = json.loads(contract_path.read_text())
    public.validate_teacher(json.loads((adapter.parent / "PLAN.json").read_text()), contract)
    binding = shared.endpoint(
        adapter, training_plan_sha256=public.PLAN_SHA, rows_sha256=public.ROWS_SHA
    )
    binding.update(
        teacher_contract_sha256=public.CONTRACT_SHA, teaching_policy=contract["teaching_policy"]
    )
    plan.update(
        schema="textcraft-train-rollout-readiness-v1",
        profile="original",
        split="train",
        prepared=str(args.prepared.resolve()),
        tasks_sha256=manifest["tasks_sha256"],
        manifest_sha256=c.inputs.sha(args.prepared / "MANIFEST.json"),
        jobs=jobs(tasks),
        planned_episodes=32,
        planned_per_policy=32,
        planned_per_condition=32,
        parent_tasks=8,
        seeds=list(SEEDS),
        max_native_calls=3072,
        max_agent_depth={"flat": 0},
        conditions=["train_public056"],
        adapter=binding,
        fixed_adapter=str(adapter),
        adapters="Fixed public056 checkpoint23, enabled on every flat call; all weights frozen",
        training_plan_sha256=public.PLAN_SHA,
        teacher_contract_sha256=public.CONTRACT_SHA,
        initial_token_audit={
            "prompt_tokens": manifest["initial_prompt_tokens"],
            "max_prompt_plus_cap": manifest["max_initial_prompt_plus_cap"],
        },
        task_order="repeat-major: all8taskids before next sample; four fixed sample seeds",
        prompt_difference="None: original public flat prompt and native action interface",
        readiness_question="Within-task native terminal-success diversity and usable execution "
        "on exact SFT TRAIN tasks, not a training update or held-out score",
        reporting="Report all8groups including all-success/all-zero/mixed, observed versus "
        "missing and protocol/action errors separately; "
        "zero native reward is not transport failure",
        caveat=manifest["caveat"],
    )
    for module in (public, shared):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = c.inputs.sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = c.inputs.sha(Path(__file__))
    path = args.output / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable TRAIN readiness PLAN changed")
    else:
        c.save(path, plan)
    return plan, tasks, binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan, tasks, binding = prepare(args)
    c.run(args, prepared_run=(plan, tasks), adapter=binding)
