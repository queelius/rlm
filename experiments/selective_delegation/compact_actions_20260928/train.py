"""Matched-epoch compact SFT around the unchanged accepted optimizer/checkpoint recipe."""

import argparse
from pathlib import Path

import compact_bridge as compact
import prepare

recipe = prepare.accepted.recipe


def validate_prepared(prepared: Path) -> dict:
    manifest = prepare.accepted.read(prepared / "MANIFEST.json")
    if (
        manifest["schema"] != "textcraft-compact-public-discovery-projection-20260928-v1"
        or not manifest["ready"]
        or manifest["rows"] != 366
        or manifest["eligible_task_count"] != 32
        or not manifest["same_source_row_order"]
    ):
        raise ValueError("complete32/366 compact native qualification required")
    for filename, key in (
        ("rows.jsonl", "rows_sha256"),
        ("tasks.jsonl", "tasks_sha256"),
        ("DESIGN.json", "design_sha256"),
    ):
        if prepare.sha(prepared / filename) != manifest[key]:
            raise ValueError("compact prepared input changed")
    for source, digest in manifest["source_sha256"].items():
        if prepare.sha(Path(source)) != digest:
            raise ValueError("compact preparation/interface source changed")
    if prepare.sha(Path(recipe.__file__)) != prepare.accepted.RECIPE_SHA["train_textcraft_sft.py"]:
        raise ValueError("accepted SFT optimizer recipe changed")
    return dict(
        schema="textcraft-compact-action-sft-contract-20260928-v1",
        interface="compact",
        prepared=str(prepared.resolve()),
        prepared_manifest_sha256=prepare.sha(prepared / "MANIFEST.json"),
        rows_sha256=manifest["rows_sha256"],
        tasks_sha256=manifest["tasks_sha256"],
        fixed_checkpoint=23,
        tasks=32,
        rows=366,
        seed=recipe.SEED,
        prompt_tokens=manifest["prompt_tokens"],
        supervised_tokens=manifest["supervised_tokens_including_eos"],
        comparison_full_format=manifest["original"],
        dose_caveat=manifest["dose_caveat"],
        modification="Only in-process strict_action validator changed to compact.parse_action; "
        "same immutable run/tokenization/loss/optimizer/checkpoint code and same base "
        "initialization.",
        source_sha256={
            str(path.resolve()): prepare.sha(path)
            for path in (
                Path(__file__),
                Path(compact.__file__),
                Path(prepare.__file__),
                Path(recipe.__file__),
                Path(recipe.target_loss.__code__.co_filename),
                Path(recipe.probe.__file__),
            )
        },
    )


def run(args):
    contract = validate_prepared(args.prepared)
    path = args.output / "COMPACT-CONTRACT.json"
    if path.exists():
        if prepare.accepted.read(path) != contract:
            raise ValueError("immutable compact training contract changed")
    else:
        prepare.save(path, contract)
    # Local process binding, not an edit to the accepted source or active owner.
    recipe.strict_action = compact.parse_action
    recipe.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=prepare.OUTPUT)
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
