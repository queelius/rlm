"""Second fixed training-seed teacher-package replication around unchanged048 recipe."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import analyze_textcraft_endpoint_fresh as comparison
import eval_textcraft_endpoint_fresh as fixed
import eval_textcraft_trained as shared
import train_textcraft_public as public

recipe, c = public.recipe, fixed.c
SEED = 2026092291
TEACHERS = ("privileged", "public")
PLANS = c.ROOT / "TEXTCRAFT-SECOND-SEED-TRAIN-PLANS-001.json"
TRAIN = {t: c.ROOT / f"textcraft-teacher-seed2291-{t}-001" for t in TEACHERS}
EVAL = {t: c.ROOT / f"textcraft-fresh-seed2291-{t}-001" for t in TEACHERS}
PREPARED = {
    "privileged": c.ROOT / "textcraft-train-inputs-001",
    "public": c.ROOT / "textcraft-public-discovery-prototype-001",
}
MANIFESTS = {
    "privileged": "670c808d563ff9fd46aa511089acfb475ce377cea803b7ffcc5d285f364d0a98",
    "public": public.MANIFEST_SHA,
}
ROWS = {
    "privileged": "caa78390f9d4ac28e600674b26e56375203b72d8cdad1c3f9471da3fb25776a9",
    "public": "dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a",
}


@contextmanager
def seed_override():
    if recipe.SEED != 2026092208:
        raise ValueError("unexpected source048 seed configuration")
    previous = recipe.SEED
    recipe.SEED = SEED  # Explicit process-local configuration; recipe bytes remain unchanged.
    try:
        yield
    finally:
        recipe.SEED = previous


def teacher_inputs(teacher):
    prepared = PREPARED[teacher]
    if c.inputs.sha(Path(recipe.__file__)) != public.RECIPE_SHA["train_textcraft_sft.py"]:
        raise ValueError("source048 recipe must remain unchanged")
    if (
        c.inputs.sha(prepared / "MANIFEST.json") != MANIFESTS[teacher]
        or c.inputs.sha(prepared / "rows.jsonl") != ROWS[teacher]
    ):
        raise ValueError("original whole366-row teacher input changed")
    qualification = public.validate_prepared(prepared) if teacher == "public" else None
    return prepared, dict(
        teacher=teacher,
        manifest_sha256=MANIFESTS[teacher],
        rows_sha256=ROWS[teacher],
        public_qualification=qualification,
    )


def train(args):
    prepared, identity = teacher_inputs(args.teacher)
    output = TRAIN[args.teacher]
    receipt = dict(
        schema="textcraft-explicit-second-seed-v1",
        seed=SEED,
        original_seed=2026092208,
        identity=identity,
        unchanged_recipe_sha256=public.RECIPE_SHA,
        wrapper_sha256=c.inputs.sha(Path(__file__)),
        rows=366,
        tasks=32,
        updates=23,
        caveat="Teacher package/history replication, not isolated query-order causality; "
        "known train.1029 quantity difference retained. Equal rows/updates not tokens/FLOPs.",
    )
    path = output / "SEED-OVERRIDE.json"
    if path.exists():
        if public.read(path) != receipt:
            raise ValueError("immutable seed/input override changed")
    else:
        c.save(path, receipt)
    train_args = argparse.Namespace(
        prepared=prepared,
        output=output,
        epochs=1,
        learning_rate=1e-4,
        hours=0.5,
        prepare_only=args.prepare_only,
        resume=args.resume,
    )
    with seed_override():
        (public.run if args.teacher == "public" else recipe.run)(train_args)


def endpoint(teacher):
    _, identity = teacher_inputs(teacher)
    pins = public.read(PLANS)
    path = TRAIN[teacher] / "PLAN.json"
    override = TRAIN[teacher] / "SEED-OVERRIDE.json"
    if (
        c.inputs.sha(path) != pins[teacher]["plan_sha256"]
        or c.inputs.sha(override) != pins[teacher]["seed_override_sha256"]
    ):
        raise ValueError("prospectively prepared training plan/seed differs")
    plan = public.read(path)
    if (
        plan["seed"] != SEED
        or plan["learning_rate"] != 1e-4
        or plan["planned_updates"] != 23
        or plan["source_sha256"] != public.RECIPE_SHA["train_textcraft_sft.py"]
    ):
        raise ValueError("not second-seed unchanged recipe/dose")
    return shared.endpoint(
        TRAIN[teacher] / "checkpoint-0023",
        training_plan_sha256=pins[teacher]["plan_sha256"],
        rows_sha256=identity["rows_sha256"],
    )


def readout(args):
    template, tasks = fixed.template_inputs()
    binding = endpoint(args.teacher)
    plan = fixed.bound_plan(
        template, "teacher_seed2291_" + args.teacher, binding, TRAIN[args.teacher]
    )
    plan.update(
        schema="textcraft-second-training-seed-fixed-fresh16-v1",
        teacher=args.teacher,
        training_seed=SEED,
        endpoint_selection="Fixed23 after one complete epoch; no selection",
        caveat="Second shared training seed, same exposed fresh16 panel; not independent world. "
        "Teacher package/history and known train.1029 quantity confound retained.",
    )
    for module in (public, recipe, shared):
        plan["source_sha256"][str(Path(module.__file__).resolve())] = c.inputs.sha(
            Path(module.__file__)
        )
    plan["source_sha256"][str(Path(__file__).resolve())] = c.inputs.sha(Path(__file__))
    plan["training_plan_pins_sha256"] = c.inputs.sha(PLANS)
    output = EVAL[args.teacher]
    path = output / "PLAN.json"
    if path.exists():
        if public.read(path) != plan:
            raise ValueError("immutable second-seed readout changed")
    else:
        c.save(path, plan)
    c.run(
        argparse.Namespace(output=output, prepare_only=args.prepare_only),
        prepared_run=(plan, tasks),
        adapter=binding,
    )


def analyze(report):
    from transformers import AutoTokenizer

    if report.exists() or report.with_suffix(".md").exists():
        raise FileExistsError("immutable report already exists")
    plans = [public.read(EVAL[t] / "PLAN.json") for t in TEACHERS]
    for teacher, plan in zip(TEACHERS, plans, strict=True):
        if plan["adapter"] != endpoint(teacher):
            raise ValueError("teacher endpoint changed")
    comparison.match_slots(plans)
    tokenizer = AutoTokenizer.from_pretrained(
        plans[0]["model"], local_files_only=True, trust_remote_code=False
    )
    audit, profiles = comparison.profiles.audit, comparison.profiles
    arms = [audit.analyze(EVAL[t], tokenizer, expected_task_count=16) for t in TEACHERS]
    rows = [
        {
            (r["task_id"], r["repeat"]): r
            for r in (public.read(p) for p in (EVAL[t] / "episodes").glob("*.json"))
        }
        for t in TEACHERS
    ]
    effect = profiles.compare(plans[0]["jobs"], rows[0], rows[1])
    result = dict(
        schema="textcraft-second-training-seed-comparison-v1",
        training_seed=SEED,
        arms=arms,
        public_minus_privileged=effect,
        parents=16,
        repeats=2,
        planned_per_arm=32,
        bootstrap_draws=20000,
        bootstrap_seed=2026092206,
        caveat=plans[0]["caveat"],
        native_source_sha256=c.inputs.sha(Path(__file__)),
        training_plan_pins_sha256=c.inputs.sha(PLANS),
    )
    c.save(report, result)
    with report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Second training-seed teacher replication\n\n"
            + result["caveat"]
            + "\n\n```json\n"
            + json.dumps(effect, indent=2)
            + "\n```\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "readout", "analyze"), required=True)
    parser.add_argument("--teacher", choices=TEACHERS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "analyze":
        if args.report is None:
            parser.error("--report required")
        analyze(args.report)
    elif args.teacher is None:
        parser.error("--teacher required")
    elif args.mode == "train":
        try:
            train(args)
        except Exception as exc:
            recipe.ACTIVE_FAILURE = f"{type(exc).__name__}: {exc}"
            raise
    else:
        readout(args)
