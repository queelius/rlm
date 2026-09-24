"""World47 paired readout: public teacher versus completed quantity-corrected original."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
SOURCE = ROOT / "source-textcraft-reserve-readout-001"
WORLD, TRAINING_SEED = 47, 2026092208
ARMS = ("quantity_corrected_original", "public")
CORRECTED = ROOT / "textcraft-quantity-matched-seed2026092208-001"
ROWS_SHA = "24ea72cb1242f2e0d819d8fb115737864de03fb750e064f145f9ec48a245e6d6"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(arm: str) -> Path:
    return ROOT / f"textcraft-world47-quantity-corrected-seed2208-{arm}-003"


def strip_conditions(jobs: list[dict]) -> list[dict]:
    return [{key: value for key, value in job.items() if key != "condition"} for job in jobs]


def corrected_binding(original):
    plan = CORRECTED / "PLAN.json"
    if not (CORRECTED / "checkpoint-0023" / "COMMIT.json").exists():
        raise ValueError("completed fixed corrected checkpoint23 required")
    return original.shared.endpoint(
        CORRECTED / "checkpoint-0023",
        training_plan_sha256=sha(plan),
        rows_sha256=ROWS_SHA,
    )


def build(arm: str):
    if arm not in ARMS:
        raise ValueError("paired arms only")
    sys.path.insert(0, str(SOURCE))
    import textcraft_multiworld as multi

    public_plan, tasks, public_binding = multi.build(WORLD, "original", "public")
    binding = public_binding if arm == "public" else corrected_binding(multi.original)
    plan = copy.deepcopy(public_plan)
    condition = f"world47_quantity_corrected_seed2208_{arm}"
    plan.update(
        schema="textcraft-world47-quantity-corrected-paired-v1",
        teacher=arm,
        training_seed=TRAINING_SEED,
        adapter=binding,
        fixed_adapter=binding["path"],
        training_plan_sha256=binding["training_plan_sha256"],
        conditions=[condition],
        budget_seconds=1800,
        caveat="World47 changed-recipe paired control: public teacher versus original teacher "
        "trained on whole-trajectory quantity-corrected inputs. Eight exposed roots/two "
        "correlated sampling seeds; not fresh roots or independent worlds.",
    )
    plan["jobs"] = [dict(job, condition=condition) for job in plan["jobs"]]
    for module in (multi, multi.c, multi.c.bridge, multi.c.inputs, multi.c.probe, multi.panel):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    return multi, plan, tasks, binding


def analyze(report: Path) -> None:
    if report.exists() or report.with_suffix(".md").exists():
        raise FileExistsError("immutable world47 quantity analysis exists")
    sys.path.insert(0, str(SOURCE))
    import analyze_textcraft_profiles as profiles

    plans, rows, arms = [], [], []
    multi, _, _, _ = build("quantity_corrected_original")
    world = None
    for arm in ARMS:
        _, expected, _, _ = build(arm)
        actual = json.loads((output(arm) / "PLAN.json").read_text())
        if actual != expected:
            raise ValueError("saved PLAN differs from prepared quantity-world47 contract")
        plans.append(actual)
        world = multi.checked_world(actual)
    if strip_conditions(plans[0]["jobs"]) != strip_conditions(plans[1]["jobs"]):
        raise ValueError("paired world47 slots differ")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(plans[0]["model"], local_files_only=True)
    for arm in ARMS:
        directory = output(arm)
        summary = profiles.audit.analyze(directory, tokenizer, world=world)
        summary.pop("paired", None)
        summary.pop("depth_strata", None)
        arms.append(summary)
        rows.append(
            {
                (row["task_id"], row["repeat"]): row
                for row in (
                    json.loads(path.read_text())
                    for path in (directory / "episodes").glob("*.json")
                )
            }
        )
    result = {
        "schema": "textcraft-world47-quantity-corrected-paired-native-v1",
        "world_seed": WORLD,
        "training_seed": TRAINING_SEED,
        "arms": arms,
        "public_minus_quantity_corrected_original": profiles.compare(
            plans[0]["jobs"], rows[0], rows[1]
        ),
        "method": {"parents": 8, "repeats": 2, "planned_per_arm": 16},
        "source_sha256": sha(Path(__file__).resolve()),
        "caveat": plans[0]["caveat"],
    }
    report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    report.with_suffix(".md").write_text(
        "# World47 quantity-corrected paired control\n\n"
        + result["caveat"]
        + "\n\n```json\n"
        + json.dumps(result["public_minus_quantity_corrected_original"], indent=2)
        + "\n```\n"
    )


def run(args):
    multi, plan, tasks, binding = build(args.arm)
    path = output(args.arm) / "PLAN.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("immutable quantity-world47 PLAN differs")
    if not path.exists():
        multi.c.save(path, plan)
    multi.c.run(
        argparse.Namespace(output=output(args.arm), prepare_only=args.prepare_only),
        prepared_run=(plan, tasks),
        adapter=binding,
        world=None if args.prepare_only else multi.checked_world(plan),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report:
        analyze(args.report.resolve())
    elif args.arm:
        run(args)
    else:
        parser.error("--arm or --report required")
