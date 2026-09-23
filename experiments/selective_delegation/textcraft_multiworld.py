"""Frozen worlds44–46, two teacher packages and two fixed training seeds."""

import argparse
import copy
import json
from pathlib import Path

import eval_textcraft_world43 as original
import prepare_textcraft_multiworld as panel
import textcraft_teacher_seed as seed

c = seed.c
TEMPLATE = c.ROOT / "textcraft-world43-privileged-001/PLAN.json"
TEMPLATE_SHA = "f2dbd40dc6ecc588e406d88e85b12d8aeef5d1d7c48aa82198c0003d94fdf00a"
PREPARED = c.ROOT / "textcraft-multiworld-inputs-001"
MANIFEST_SHA = "03082cf977c95d482b385943297dbe7128de24f19f49453d1f61e3297a45bd13"
TRAINING_SEEDS = {"original": 2026092208, "2291": 2026092291}


def output(world_seed, training_seed, teacher):
    return c.ROOT / f"textcraft-world{world_seed}-seed{TRAINING_SEEDS[training_seed]}-{teacher}-001"


def normalize(jobs):
    return [{k: v for k, v in j.items() if k != "condition"} for j in jobs]


def binding(training_seed, teacher):
    if training_seed == "2291":
        return seed.endpoint(teacher)
    if training_seed != "original" or teacher not in seed.TEACHERS:
        raise ValueError("unqualified teacher/training seed")
    directory = c.ROOT / (
        "textcraft-action-sft-001"
        if teacher == "privileged"
        else "textcraft-public-discovery-sft-001"
    )
    if teacher == "privileged":
        return original.shared.endpoint(directory / "checkpoint-0023")
    public = original.public
    if c.inputs.sha(directory / "TEACHER-CONTRACT.json") != public.CONTRACT_SHA:
        raise ValueError("fixed public teacher contract changed")
    public.validate_teacher(
        json.loads((directory / "PLAN.json").read_text()),
        json.loads((directory / "TEACHER-CONTRACT.json").read_text()),
    )
    return original.shared.endpoint(
        directory / "checkpoint-0023",
        training_plan_sha256=public.PLAN_SHA,
        rows_sha256=public.ROWS_SHA,
    )


def build(world_seed, training_seed, teacher):
    if world_seed not in panel.WORLD_SEEDS:
        raise ValueError("only frozen worlds44/45/46")
    if (
        c.inputs.sha(TEMPLATE) != TEMPLATE_SHA
        or c.inputs.sha(PREPARED / "MANIFEST.json") != MANIFEST_SHA
    ):
        raise ValueError("frozen template/panel changed")
    master = json.loads((PREPARED / "MANIFEST.json").read_text())
    selected = next(w for w in master["worlds"] if w["world_seed"] == world_seed)
    prepared = PREPARED / f"world{world_seed}"
    manifest = json.loads((prepared / "MANIFEST.json").read_text())
    if (
        manifest != selected
        or not manifest["ready"]
        or c.inputs.sha(prepared / "tasks.jsonl") != manifest["tasks_sha256"]
    ):
        raise ValueError("native-qualified world inventory changed")
    bound = binding(training_seed, teacher)
    plan = copy.deepcopy(json.loads(TEMPLATE.read_text()))
    for module in (c, c.bridge, c.inputs):
        expected = [
            digest
            for path, digest in plan["source_sha256"].items()
            if Path(path).name == Path(module.__file__).name
        ]
        if expected != [c.inputs.sha(Path(module.__file__))]:
            raise ValueError("native runtime must remain byte-identical to source062")
    tasks = list(map(json.loads, (prepared / "tasks.jsonl").read_text().splitlines()))
    reference = list(
        map(json.loads, (c.ROOT / "textcraft-inputs-001/tasks.jsonl").read_text().splitlines())
    )
    if [(t["id"], t["goal"], t["misc"]["target_items"]) for t in tasks] != [
        (t["id"], t["goal"], t["misc"]["target_items"]) for t in reference
    ]:
        raise ValueError("all original goals/quantities required")
    condition = f"world{world_seed}_seed{training_seed}_{teacher}"
    plan.update(
        schema="textcraft-multiworld-fixed-teacher-v1",
        teacher=teacher,
        training_seed=TRAINING_SEEDS[training_seed],
        training_seed_label=training_seed,
        adapter=bound,
        fixed_adapter=bound["path"],
        training_plan_sha256=bound["training_plan_sha256"],
        prepared=str(prepared),
        tasks_sha256=manifest["tasks_sha256"],
        manifest_sha256=c.inputs.sha(prepared / "MANIFEST.json"),
        panel_manifest_sha256=MANIFEST_SHA,
        world_seed=world_seed,
        world_sha256=manifest["world_sha256"],
        conditions=[condition],
        template_sha256=TEMPLATE_SHA,
        initial_token_audit={
            "prompt_tokens": [
                a["qualification"]["initial_prompt_tokens"] for a in manifest["audits"]
            ]
        },
        prompt_difference="Exact original flat interface; frozen recipe assignment/inventory "
        "changes across worlds, fixed teacher checkpoint changes across paired arms.",
        caveat="Eight exposed roots/two correlated evaluation seeds; worlds share names and "
        "generator. Teacher-package comparison, not query-order causality; train.1029 quantity "
        "confound retained. Missing/unknown outcomes are not failures.",
    )
    plan["jobs"] = [dict(j, condition=condition) for j in plan["jobs"]]
    for module in (
        c,
        c.bridge,
        c.inputs,
        c.probe,
        seed,
        original,
        original.shared,
        original.public,
        panel,
        panel.prior,
        panel.prior.inventory,
    ):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = c.inputs.sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = c.inputs.sha(Path(__file__))
    return plan, tasks, bound


def prepare(world_seed, training_seed, teacher):
    result = build(world_seed, training_seed, teacher)
    path = output(world_seed, training_seed, teacher) / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != result[0]:
            raise ValueError("immutable prepared PLAN changed")
    else:
        c.save(path, result[0])
    return result


def checked_world(plan):
    native = panel.world(plan["world_seed"])
    if panel.prior.inventory.digest(panel.prior.inventory.snapshot(native)) != plan["world_sha256"]:
        raise ValueError("native world differs from frozen panel")
    return native


def analyze(world_seed, training_seed, report):
    import analyze_textcraft_profiles as profiles
    from transformers import AutoTokenizer

    if report.exists() or report.with_suffix(".md").exists():
        raise FileExistsError(report)
    plans = []
    for teacher in seed.TEACHERS:
        expected, _, _ = build(world_seed, training_seed, teacher)
        actual = json.loads((output(world_seed, training_seed, teacher) / "PLAN.json").read_text())
        if actual != expected:
            raise ValueError("runtime PLAN does not match fixed prepared contract")
        plans.append(actual)
    if normalize(plans[0]["jobs"]) != normalize(plans[1]["jobs"]):
        raise ValueError("paired native slots differ")
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    world, arms, rows = checked_world(plans[0]), [], []
    for teacher in seed.TEACHERS:
        directory = output(world_seed, training_seed, teacher)
        arm = profiles.audit.analyze(directory, tokenizer, world=world)
        for key in ("paired", "depth_strata"):
            arm.pop(key, None)
        arms.append(arm)
        rows.append(
            {
                (r["task_id"], r["repeat"]): r
                for r in (
                    json.loads(p.read_text()) for p in (directory / "episodes").glob("*.json")
                )
            }
        )
    result = dict(
        schema="textcraft-multiworld-paired-native-v1",
        world_seed=world_seed,
        training_seed=TRAINING_SEEDS[training_seed],
        arms=arms,
        public_minus_privileged=profiles.compare(plans[0]["jobs"], *rows),
        parent_count=8,
        repeats=2,
        planned_per_arm=16,
        caveat=plans[0]["caveat"],
        source_sha256=c.inputs.sha(Path(__file__)),
    )
    c.save(report, result)
    with report.with_suffix(".md").open("x") as stream:
        stream.write("# Multiworld fixed teacher comparison\n\n" + result["caveat"] + "\n\n")
        stream.write(
            json.dumps(
                {k: v for k, v in result["public_minus_privileged"].items() if k != "rows"},
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, choices=panel.WORLD_SEEDS, required=True)
    parser.add_argument("--training-seed", choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--teacher", choices=seed.TEACHERS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report:
        analyze(args.world, args.training_seed, args.report)
    else:
        if args.teacher is None:
            parser.error("--teacher required for readout/preparation")
        plan, tasks, adapter = prepare(args.world, args.training_seed, args.teacher)
        c.run(
            argparse.Namespace(
                output=output(args.world, args.training_seed, args.teacher),
                prepare_only=args.prepare_only,
            ),
            prepared_run=(plan, tasks),
            adapter=adapter,
            world=None if args.prepare_only else checked_world(plan),
        )
