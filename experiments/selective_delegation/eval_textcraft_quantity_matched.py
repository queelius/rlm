"""Read fixed checkpoint-23 corrected-original controls on the existing fresh32 panel."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
SOURCE = ROOT / "source-textcraft-teacher-seed2291-001"
ROWS_SHA = "24ea72cb1242f2e0d819d8fb115737864de03fb750e064f145f9ec48a245e6d6"
SEEDS = (2026092208, 2026092291)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths(seed: int) -> tuple[Path, Path]:
    root = ROOT / f"textcraft-quantity-matched-seed{seed}-001"
    return root, ROOT / f"textcraft-fresh-quantity-matched-seed{seed}-001"


def run(args: argparse.Namespace) -> None:
    train, output = paths(args.seed)
    sys.path.insert(0, str(SOURCE))
    import eval_textcraft_endpoint_fresh as fixed
    import eval_textcraft_trained as shared

    plan_path = train / "PLAN.json"
    if not plan_path.exists() or not (train / "checkpoint-0023" / "COMMIT.json").exists():
        raise ValueError("only the complete fixed checkpoint-23 endpoint may be read")
    training_plan = json.loads(plan_path.read_text())
    if training_plan.get("planned_updates") != 23 or training_plan.get("seed") != args.seed:
        raise ValueError("wrong corrected-original training endpoint")
    binding = shared.endpoint(
        train / "checkpoint-0023",
        training_plan_sha256=sha(plan_path),
        rows_sha256=ROWS_SHA,
    )
    template, tasks = fixed.template_inputs()
    plan = fixed.bound_plan(template, f"quantity_matched_seed{args.seed}", binding, train)
    plan.update(
        schema="textcraft-quantity-matched-fixed-fresh32-v1",
        training_seed=args.seed,
        endpoint_selection="Fixed23 after one complete corrected replay epoch; no selection",
        caveat="Existing fresh32 panel is exposed exploratory readout; controls quantity mismatch "
        "but not trajectory history/order or token dose.",
    )
    # Bind aliases actually used to construct the adapter identity and run the collector,
    # rather than relying on same-basename matching during later analysis.
    for module in (
        fixed,
        fixed.c,
        fixed.c.bridge,
        fixed.c.inputs,
        fixed.c.probe,
        fixed.c.probe.runtime,
        fixed.c.probe.campaign,
        shared,
    ):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    stored = output / "PLAN.json"
    if stored.exists() and json.loads(stored.read_text()) != plan:
        raise ValueError("immutable corrected-original readout plan differs")
    if not stored.exists():
        fixed.c.save(stored, plan)
    fixed.c.run(
        argparse.Namespace(output=output, prepare_only=args.prepare_only),
        prepared_run=(plan, tasks),
        adapter=binding,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
