"""Matched source048 one-epoch SFT around a visibility-repaired frozen teacher."""

import argparse
import json
import sys
from pathlib import Path

import prepare

SOURCE = prepare.ROOT / "source-048-textcraft-action-sft"
RECIPE_SHA = "274daede3a32aab96f3ad7914ae22b00cb9c4301efb39f457eeb8c786e6ec439"
SEEDS = (2026092208, 2026092291)


def run(args: argparse.Namespace) -> None:
    directory = args.root / args.mode
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((args.root / "SUMMARY.json").read_text())[args.mode]
    prepare.require(manifest["rows_sha256"] == summary["rows_sha256"], "changed mode manifest")
    prepare.require(
        prepare.sha(directory / "rows.jsonl") == manifest["rows_sha256"], "rows changed"
    )
    prepare.require(prepare.sha(directory / "tasks.jsonl") == prepare.TASKS_SHA, "tasks changed")
    prepare.require(
        manifest["all_query_names_public"]
        and manifest["original_craft_order_preserved"]
        and manifest["training_target_order_preserved"]
        and manifest["same_action_and_label_multiset_tasks"] == 32,
        "teacher control qualification changed",
    )
    prepare.require(prepare.sha(SOURCE / "train_textcraft_sft.py") == RECIPE_SHA, "recipe changed")
    sys.path.insert(0, str(SOURCE))
    import train_textcraft_sft as recipe

    output = args.root / f"train-{args.mode}-seed{args.seed}"
    contract = {
        "schema": "textcraft-visible-order-training-v1",
        "mode": args.mode,
        "seed": args.seed,
        "prepared": str(directory),
        "manifest_sha256": prepare.sha(manifest_path),
        "rows_sha256": manifest["rows_sha256"],
        "source048_sha256": RECIPE_SHA,
        "wrapper_sha256": prepare.sha(Path(__file__)),
        "fixed_endpoint": "checkpoint-0023",
        "pairing": "Same base/LoRA init seed, tasks, action/label multisets, target row order, "
        "training shuffle and target-token denominators; only conditioning histories differ.",
        "caveat": "Offline oracle schedule with visible queried names; prompt-token dose differs.",
    }
    path = output / "ORDER-CONTROL-CONTRACT.json"
    if path.exists():
        prepare.require(json.loads(path.read_text()) == contract, "training contract differs")
    else:
        prepare.save(path, contract)
    original_seed = recipe.SEED
    recipe.SEED = args.seed
    try:
        recipe.run(
            argparse.Namespace(
                prepared=directory,
                output=output,
                epochs=1,
                learning_rate=1e-4,
                hours=0.5,
                prepare_only=args.prepare_only,
                resume=args.resume,
            )
        )
    except Exception as exc:
        recipe.ACTIVE_FAILURE = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        recipe.SEED = original_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=prepare.OUTPUT)
    parser.add_argument("--mode", choices=prepare.MODES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
