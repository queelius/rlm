"""One matched command-target epoch through the untouched original33-step optimizer."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import alf_rep as r
import alf_rep_data as data

if r.sha(r.TRAIN_SOURCE) != r.TRAIN_SOURCE_SHA:
    raise ValueError("accepted original optimizer changed")
recipe = r.module(r.TRAIN_SOURCE, "alf_rep_private_sft")


def validate_rows(rows, manifest):
    if (
        len(rows) != 524
        or manifest.get("examples") != 524
        or (manifest.get("successful_games_only") is not True)
    ):
        raise ValueError("same524 successful-game TRAIN examples required")
    expected = data.project_rows()
    for row, source in zip(rows, expected, strict=True):
        if any(row[key] != source[key] for key in ("id", "prompt", "target_json")):
            raise ValueError("same ordered examples and exact native-command target required")
        public = json.loads(row["prompt"].split("\n", 1)[1])["public_context"]
        r.parse_command(
            row["target_json"], "flat", [a["command"] for a in public["admissible_commands"]]
        )
    return 524


def run(args):
    if args.prepared.resolve() != data.COMMAND_DATA or args.output.resolve() != r.COMMAND_TRAIN:
        raise ValueError("fixed command data/output required")
    if not args.prepare_only and list(args.output.glob("OWNER-*.json")):
        raise ValueError("existing command training attempt; no implicit retry")
    manifest = r.read(data.COMMAND_DATA / "MANIFEST.json")
    old = r.read(r.INDEX_CHECKPOINT.parent / "PLAN.json")
    for name in ("torch", "transformers", "peft"):
        if importlib.metadata.version(name) != old["environment"][name]:
            raise ValueError("matched training software version changed: " + name)
    if manifest["original_examples_sha256"] != r.ORIGINAL_DATA_SHA:
        raise ValueError("unmatched demonstration provenance")
    contract = dict(
        schema="alfworld-command-target-sft-contract-20260928-v1",
        representation="command",
        prepared=str(data.COMMAND_DATA),
        examples_sha256=manifest["examples_sha256"],
        original_examples_sha256=r.ORIGINAL_DATA_SHA,
        original_index_training_plan_sha256=r.sha(r.INDEX_CHECKPOINT.parent / "PLAN.json"),
        fixed_checkpoint=33,
        supervised_tokens=manifest["supervised_tokens"],
        original_supervised_tokens=manifest["original_supervised_tokens"],
        modification="Private strict row validator only; original tokenize/loss/optimizer/order/"
        "initialization/checkpoint recipe unchanged. "
        "Scientific training PLAN uses command prepared data.",
        source_sha256=r.source_pins(),
        caveat="Same steps/rows/native actions, not equal tokens or per-example gradient weights.",
    )
    r.persist(args.output / "REPRESENTATION-CONTRACT.json", contract)
    recipe.validate_rows = validate_rows
    recipe.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=data.COMMAND_DATA)
    parser.add_argument("--output", type=Path, default=r.COMMAND_TRAIN)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    # The frozen trainer requires resume=True to admit its unchanged prewritten PLAN.
    # Existing GPU owners are rejected above, so this is not automatic rollout/training reuse.
    args.epochs, args.learning_rate, args.hours, args.resume = 1, 1e-4, 1 / 3, True
    try:
        run(args)
    except Exception as exc:
        recipe.ACTIVE_FAILURE = f"{type(exc).__name__}: {exc}"
        raise
