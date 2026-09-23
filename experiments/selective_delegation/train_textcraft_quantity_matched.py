"""Train the corrected-original TextCraft control with the unchanged source048 recipe."""

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
RECIPE_SOURCE = ROOT / "source-048-textcraft-action-sft"
PREPARED = ROOT / "textcraft-quantity-matched-inputs-004"
MANIFEST_SHA = "25ed7f91209365533d1d93222d76ea55b0b777c84461d1a7e3ddbf3563d89c48"
ROWS_SHA = "24ea72cb1242f2e0d819d8fb115737864de03fb750e064f145f9ec48a245e6d6"
SEEDS = (2026092208, 2026092291)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_contract() -> dict:
    if sha(PREPARED / "MANIFEST.json") != MANIFEST_SHA or sha(PREPARED / "rows.jsonl") != ROWS_SHA:
        raise ValueError("corrected replay inputs differ from their immutable contract")
    manifest = json.loads((PREPARED / "MANIFEST.json").read_text())
    required = {
        "schema": "textcraft-quantity-matched-privileged-replay-v1",
        "tasks": 32,
        "rows": 366,
        "eligible_task_count": 32,
        "native_successful_tasks": 32,
        "action_target_differences": 1,
        "prompt_differences": 29,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("incomplete corrected whole-trajectory replay")
    return manifest


@contextmanager
def seed_override(recipe, seed: int):
    if seed not in SEEDS or recipe.SEED != 2026092208:
        raise ValueError("only the two fixed original/public teacher seeds are permitted")
    original = recipe.SEED
    recipe.SEED = seed
    try:
        yield
    finally:
        recipe.SEED = original


def run(args: argparse.Namespace) -> None:
    prepared_contract()
    sys.path.insert(0, str(RECIPE_SOURCE))
    import train_textcraft_sft as recipe

    output = args.output.resolve()
    receipt = {
        "schema": "textcraft-quantity-matched-control-v1",
        "prepared": str(PREPARED),
        "prepared_manifest_sha256": MANIFEST_SHA,
        "rows_sha256": ROWS_SHA,
        "seed": args.seed,
        "source048_recipe_sha256": sha(RECIPE_SOURCE / "train_textcraft_sft.py"),
        "wrapper_sha256": sha(Path(__file__).resolve()),
        "fixed_endpoint": "checkpoint-0023; no validation selection",
        "caveat": "All 32 native trajectories were regenerated after the quantity correction; "
        "29 history prompts change, so this controls its mismatch but not package order.",
    }
    path = output / "QUANTITY-CONTROL-CONTRACT.json"
    output.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError("immutable quantity-control contract differs")
    if not path.exists():
        path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    with seed_override(recipe, args.seed):
        recipe.run(
            argparse.Namespace(
                prepared=PREPARED,
                output=output,
                epochs=1,
                learning_rate=1e-4,
                hours=args.hours,
                prepare_only=args.prepare_only,
                resume=args.resume,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
