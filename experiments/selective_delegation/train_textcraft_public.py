"""Qualified public-discovery inputs around the unchanged source048 SFT recipe."""

import argparse
import json
from pathlib import Path

import train_textcraft_sft as recipe

MANIFEST_SHA = "c69ef258a07f4c4f9b45b5bc044880f1cc9aa884e510fe590f3ddf77f19441da"
SOURCE_SHA = "80bf7cc468d029cbab97307146ae67c5c5c220bbfe6e725eb0566a35fbfafb59"
AUDIT_SHA = "885be2299ad24139709c6b7505ea2d460db81cb0de1ee4534704d60c490272c7"
RECIPE_SHA = {
    "train_textcraft_sft.py": "274daede3a32aab96f3ad7914ae22b00cb9c4301efb39f457eeb8c786e6ec439",
    "train_planner.py": "9ce66e6f2bc8774bf5f4a5ae0e4934fbad5fd04da8886cb799231b8582f562f8",
    "probe.py": "8a73f3fab4d9c03fec8b51cc2a014b6f691915db4c972e1b1bef5a2acdae9302",
}
sha = recipe.probe.campaign.sha


def read(path):
    return json.loads(path.read_text())


def validate_qualification(manifest, audit):
    if (
        manifest["schema"] != "textcraft-public-discovery-prototype-v1"
        or manifest["selected_task_count"] != 32
        or manifest["eligible_task_count"] != 32
        or manifest["rows"] != 366
        or len(manifest["tasks"]) != 32
        or any(
            not t["eligible"] or t["native_score"] != 1 or not t["finish_included"]
            for t in manifest["tasks"]
        )
        or audit["native_successful_tasks"] != 32
        or audit["root_first_tasks"] != 32
        or audit["independently_reconstructed_prompt_and_mask_rows"] != 366
        or audit["past_query_provenance_checked"] is not True
        or audit["model_calls"] != 0
        or audit["manifest_sha256"] != MANIFEST_SHA
    ):
        raise ValueError("complete fixed32 public-only teacher qualification required")


def validate_prepared(prepared):
    prepared = prepared.resolve()
    manifest_path = prepared / "MANIFEST.json"
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("fixed055 public-discovery manifest differs")
    manifest = read(manifest_path)
    audit_path = prepared / "PUBLIC-REPLAY-AUDIT.json"
    if sha(audit_path) != AUDIT_SHA:
        raise ValueError("frozen public replay audit differs")
    audit = read(audit_path)
    validate_qualification(manifest, audit)
    for name, key in (("rows.jsonl", "rows_sha256"), ("tasks.jsonl", "tasks_sha256")):
        if sha(prepared / name) != manifest[key]:
            raise ValueError("frozen public-discovery input differs")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    if (
        len(tasks) != 32
        or {t["id"] for t in tasks} != set(manifest["task_ids"])
        or any(not t["id"].startswith("textcraft_synth.train.") for t in tasks)
    ):
        raise ValueError("exact frozen TRAIN task identities required")
    source_path = prepared.parent / "source-055-textcraft-public-discovery/SOURCE.json"
    if sha(source_path) != SOURCE_SHA:
        raise ValueError("source055 immutable teacher identity differs")
    for path, digest in read(source_path)["files"].items():
        if sha(Path(path)) != digest:
            raise ValueError("source055 teacher/import content changed")
    for name, path in (
        ("train_textcraft_sft.py", Path(recipe.__file__)),
        ("train_planner.py", Path(recipe.target_loss.__code__.co_filename)),
        ("probe.py", Path(recipe.probe.__file__)),
    ):
        if sha(path) != RECIPE_SHA[name]:
            raise ValueError("source048 training recipe must remain byte-identical")
    return {
        "schema": "textcraft-public-discovery-training-contract-v1",
        "teaching_policy": "public_observation_prerequisite_discovery",
        "prepared": str(prepared),
        "manifest_sha256": MANIFEST_SHA,
        "rows_sha256": manifest["rows_sha256"],
        "tasks_sha256": manifest["tasks_sha256"],
        "public_replay_audit_sha256": AUDIT_SHA,
        "teacher_source_receipt_sha256": SOURCE_SHA,
        "wrapper_sha256": sha(Path(__file__)),
        "source048_recipe_sha256": RECIPE_SHA,
        "tasks": 32,
        "rows": 366,
        "planned_updates": 23,
        "fixed_checkpoint": 23,
        "prompt_tokens": manifest["prompt_tokens"],
        "supervised_tokens": manifest["supervised_tokens_including_eos"],
        "comparison048": {
            "prompt_tokens": 414754,
            "supervised_tokens": 8821,
            "rows": 366,
            "updates": 23,
        },
        "dose_caveat": "Equal tasks/rows/updates, not identical histories, token dose or FLOPs",
        "selection": "No changed tasks, dropped traces, output-dependent choice "
        "or gold future actions",
        "acceptance": "Preparation is not GPU acceptance; main owns launch decision",
    }


def run(args):
    contract = validate_prepared(args.prepared)
    path = args.output.resolve() / "TEACHER-CONTRACT.json"
    if path.exists():
        if read(path) != contract:
            raise ValueError("immutable teacher contract differs")
    else:
        recipe.probe.runtime.save(path, contract)
    recipe.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    try:
        run(parser.parse_args())
    except Exception as exc:
        recipe.ACTIVE_FAILURE = f"{type(exc).__name__}: {exc}"
        raise
