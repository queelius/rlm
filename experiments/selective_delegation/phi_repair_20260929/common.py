"""Small private binding of frozen Phi teaching, optimizer, and native interfaces."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
E = HERE.parent
R = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
ROOT = R / "textcraft-phi-repair-20260929-001"
PREPARED = ROOT / "stable_visible"
TRAINING = ROOT / "train-stable_visible-seed2026092208"
KNOWN = R / "textcraft-phi-known-sft-20260928-001"
SOURCE = R / "textcraft-teaching-order-20260928-001/stable_visible"
PYTHON = Path("/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip") / (
    "gpu/training/.venv/bin/python"
)
SEED = 2026092208


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.loads(json.dumps(value))


def save(path, value):
    """Only new artifacts, or exact idempotent reuse; never replace differing bytes."""
    path = Path(path)
    if path.exists():
        require(read(path) == canonical(value), "immutable artifact differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


@lru_cache(maxsize=1)
def phi():
    sys.path.insert(0, str(E / "phi_transfer_20260928"))
    evaluate = importlib.import_module("evaluate")
    train = importlib.import_module("train")
    require(Path(evaluate.__file__).parent == E / "phi_transfer_20260928", "wrong Phi module")
    require(Path(train.__file__).parent == E / "phi_transfer_20260928", "wrong Phi trainer")
    acquire, qualify = evaluate.acquire, evaluate.qualify
    require(
        sha(acquire.OUTPUT / "QUALIFICATION.json") == train.QUALIFICATION_SHA,
        "original Phi qualification changed",
    )
    require(sha(train.recipe.__file__) == train.RECIPE_SHA, "qualified training recipe changed")
    for path, digest in read(acquire.OUTPUT / "QUALIFICATION.json")["source_sha256"].items():
        require(sha(path) == digest, "qualified source changed: " + path)
    return evaluate, train, acquire, qualify


def match_targets(rows, known, examples, known_examples):
    require(len(rows) == len(known) == len(examples) == len(known_examples), "row count differs")
    for row, base, tokens, baseline in zip(rows, known, examples, known_examples, strict=True):
        require(
            all(row[k] == base[k] for k in ("task_id", "target"))
            and row.get("source_action_index", row["step"]) == base["step"],
            "target row or original presentation order changed",
        )
        require(tokens["target_ids"] == baseline["target_ids"], "native target IDs differ")
    return sum(len(row["target_ids"]) for row in examples)


def runtime_args(world, output):
    require(world in (42, 50), "fixed worlds42/50 only")
    return argparse.Namespace(
        teacher="stable_visible",
        assistance="raw",
        world=world,
        output=Path(output).resolve(),
        hours=0.5,
        prepare_only=False,
        profile="original",
    )


def prepare_contract(teacher, output):
    require(
        teacher == "stable_visible" and output.resolve() == TRAINING,
        "fixed repair teacher/output only",
    )
    _, train, acquire, qualify = phi()
    manifest = read(PREPARED / "MANIFEST.json")
    dose = read(PREPARED / "TOKEN-AUDIT.json")
    require(sha(PREPARED / "rows.jsonl") == manifest["rows_sha256"], "repair rows changed")
    require(sha(PREPARED / "tasks.jsonl") == manifest["tasks_sha256"], "repair tasks changed")
    require(
        sha(acquire.MODEL / "local-research-manifest.json") == manifest["model_manifest_sha256"],
        "qualified model manifest changed",
    )
    contract = dict(
        schema="phi-stable-repair-training-contract-20260929-v1",
        teacher=teacher,
        prepared=str(PREPARED),
        rows_sha256=manifest["rows_sha256"],
        prepared_manifest_sha256=sha(PREPARED / "MANIFEST.json"),
        token_audit_sha256=sha(PREPARED / "TOKEN-AUDIT.json"),
        model=str(acquire.MODEL),
        model_revision=acquire.REVISION,
        qualification_sha256=train.QUALIFICATION_SHA,
        seed=SEED,
        rows=366,
        tasks=32,
        updates=23,
        endpoint="checkpoint-0023",
        supervised_tokens=dose["supervised_tokens"],
        prompt_tokens=dose["prompt_tokens"],
        target_suffix=qualify.STOP_IDS,
        lora=dict(rank=8, alpha=16, dropout=0, target_modules=qualify.TARGET_MODULES),
        dtype=dict(base="bfloat16", lora="float32"),
        exact_known_targets_and_minibatch_denominators=True,
        legacy_plan_interpretation="Native Phi assistant turn-end AND EOS are target labels; "
        "no prompt tokens supervised. History/input dose changes, not a visibility-only contrast.",
        source_sha256={
            str(p): sha(p)
            for p in (
                HERE / "common.py",
                HERE / "experiment.py",
                Path(train.__file__),
                Path(qualify.__file__),
                Path(train.recipe.__file__),
            )
        },
    )
    save(output / "PHI-CONTRACT.json", contract)
    return PREPARED, contract


def train_run(prepare_only=False):
    _, train, _, _ = phi()
    args = argparse.Namespace(
        teacher="stable_visible", output=TRAINING, hours=0.5, prepare_only=prepare_only, resume=True
    )
    try:
        with patch.object(train, "prepare", prepare_contract):
            train.run(args)
    except Exception as exc:
        train.recipe.ACTIVE_FAILURE = f"{type(exc).__name__}: {exc}"
        raise


def stamp(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def cached_endpoint(root=ROOT):
    audit = read(Path(root) / "ENDPOINT-AUDIT.json")
    require(audit["passed"] and audit["teacher"] == "stable_visible", "wrong endpoint audit")
    for path, digest in audit["small_files"].items():
        require(sha(path) == digest, "verified endpoint receipt changed: " + path)
    adapter = Path(audit["binding"]["path"]) / "adapter_model.safetensors"
    require(stamp(adapter) == audit["adapter_stat"], "verified adapter changed")
    return audit["binding"]


def readout_output(world):
    return ROOT / f"eval-stable_visible-cp23-p00-w{world}-raw"


def build_readout(world, actual=False):
    evaluate, _, _, _ = phi()
    binding = cached_endpoint() if actual else None
    args = runtime_args(world, readout_output(world))
    plan, tasks, native_world, tokenizer, collector, auditor = evaluate.build(
        args, require_endpoint=False
    )
    plan.update(
        schema="phi-stable-repair-fixed-panel-20260929-v1",
        adapter=binding,
        endpoint_pending=not actual,
        training_plan_sha256=sha(TRAINING / "PLAN.json"),
        training_contract_sha256=sha(TRAINING / "PHI-CONTRACT.json"),
        fixed_checkpoint=str(TRAINING / "checkpoint-0023"),
        comparison="Stable-visible versus existing corrected-known and discovery Phi cp23, "
        "same panel00 eight roots/one rollout seed/world. Known targets/minibatches matched; "
        "conditioning histories differ. This is repair-method transfer, not binder assistance.",
    )
    plan["source_sha256"].update(
        {str(p): sha(p) for p in (HERE / "common.py", HERE / "experiment.py")}
    )
    return canonical(plan), tasks, native_world, tokenizer, collector, auditor


def run_readout(world):
    plan, tasks, native_world, _, collector, _ = build_readout(world, actual=True)
    output = readout_output(world)
    pending = read(output / "PENDING-PLAN.json")
    stripped = {**plan, "adapter": None, "endpoint_pending": True}
    require(stripped == pending, "actual actor readout differs from fixed pending template")
    save(output / "PLAN.json", plan)
    collector.run(
        runtime_args(world, output),
        prepared_run=(plan, tasks),
        adapter=plan["adapter"],
        world=native_world,
    )
