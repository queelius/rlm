"""Compare corrected-original and existing public-teacher endpoints on matched fresh32 slots."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
SOURCE = ROOT / "source-textcraft-teacher-seed2291-001"
SEEDS = (2026092208, 2026092291)
PUBLIC = {
    2026092208: ROOT / "textcraft-fresh-public-001",
    2026092291: ROOT / "textcraft-fresh-seed2291-public-001",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corrected(seed: int) -> Path:
    return ROOT / f"textcraft-fresh-quantity-matched-seed{seed}-001"


def strip_conditions(jobs: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "condition"} for row in jobs]


def public_minus_corrected(profiles, jobs: list[dict], public: dict, corrected: dict) -> dict:
    """`profiles.compare(left, right)` reports right minus left."""
    return profiles.compare(jobs, corrected, public)


def run(args: argparse.Namespace) -> None:
    report = args.report.resolve()
    if report.exists() or report.with_suffix(".md").exists():
        raise FileExistsError("immutable analysis exists")
    sys.path.insert(0, str(SOURCE))
    import analyze_textcraft_profiles as profiles

    left, right = PUBLIC[args.seed], corrected(args.seed)
    plans = [profiles.audit.read(path / "PLAN.json") for path in (left, right)]
    keys = (
        "tasks_sha256", "manifest_sha256", "world_sha256", "sampling", "seeds", "seed_rule",
        "model_manifest_sha256", "max_global_calls", "max_global_output_tokens", "max_new_tokens",
        "input_plus_output_limit", "budget_seconds", "truncation",
    )
    if any(plans[0][key] != plans[1][key] for key in keys):
        raise ValueError("public/corrected native readout contracts differ")
    if strip_conditions(plans[0]["jobs"]) != strip_conditions(plans[1]["jobs"]):
        raise ValueError("public/corrected slots differ")
    rows = [
        {(row["task_id"], row["repeat"]): row for row in (
            profiles.audit.read(path) for path in (output / "episodes").glob("*.json")
        )}
        for output in (left, right)
    ]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(plans[0]["model"], local_files_only=True)
    result = {
        "schema": "textcraft-quantity-matched-public-comparison-v1",
        "training_seed": args.seed,
        "public_minus_corrected_original": public_minus_corrected(
            profiles, plans[0]["jobs"], rows[0], rows[1]
        ),
        "arms": [
            profiles.audit.analyze(path, tokenizer, expected_task_count=16)
            for path in (left, right)
        ],
        "method": {
            "parents": 16,
            "repeats": 2,
            "planned_per_arm": 32,
            "bootstrap_draws": 20000,
            "bootstrap_seed": 2026092206,
            "unit": "root task with both sampling seeds; not independent worlds",
        },
        "provenance": {
            "public_plan_sha256": sha(left / "PLAN.json"),
            "corrected_plan_sha256": sha(right / "PLAN.json"),
            "analyzer_sha256": sha(Path(__file__).resolve()),
        },
        "caveat": "Corrects one quantity mismatch through complete native trajectory replay, "
        "not package history/order/token dose. Existing fresh32 is exposed exploratory readout.",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    report.with_suffix(".md").write_text(
        "# Quantity-matched original teacher control\n\n"
        + result["caveat"]
        + "\n\n```json\n"
        + json.dumps(result["public_minus_corrected_original"], indent=2)
        + "\n```\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--report", type=Path, required=True)
    run(parser.parse_args())
