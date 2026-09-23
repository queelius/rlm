"""Fixed second-training-seed adapters on the unchanged sixteen world43 slots."""

import argparse
import copy
import json
from pathlib import Path

import analyze_textcraft_profiles as profiles
import prepare_textcraft_world43 as panel
import textcraft_teacher_seed as seed

c = seed.c
TEMPLATE = c.ROOT / "textcraft-world43-privileged-001/PLAN.json"
TEMPLATE_SHA = "f2dbd40dc6ecc588e406d88e85b12d8aeef5d1d7c48aa82198c0003d94fdf00a"
OUTPUTS = {t: c.ROOT / f"textcraft-world43-seed2291-{t}-001" for t in seed.TEACHERS}


def prepare(teacher):
    if c.inputs.sha(TEMPLATE) != TEMPLATE_SHA:
        raise ValueError("fixed completed world43 template changed")
    template = json.loads(TEMPLATE.read_text())
    for module in (c, c.bridge, c.inputs):
        old = [
            v
            for k, v in template["source_sha256"].items()
            if Path(k).name == Path(module.__file__).name
        ]
        if old != [c.inputs.sha(Path(module.__file__))]:
            raise ValueError("native world43 runtime changed")
    prepared = Path(template["prepared"])
    if c.inputs.sha(prepared / "tasks.jsonl") != template["tasks_sha256"]:
        raise ValueError("fixed world43 task bytes changed")
    binding = seed.endpoint(teacher)
    plan = copy.deepcopy(template)
    condition = "world43_seed2291_" + teacher
    plan.update(
        schema="textcraft-world43-second-seed-v1",
        teacher=teacher,
        training_seed=seed.SEED,
        adapter=binding,
        fixed_adapter=binding["path"],
        training_plan_sha256=binding["training_plan_sha256"],
        conditions=[condition],
        template_sha256=TEMPLATE_SHA,
        caveat="Same eight exposed roots/two correlated evaluation seeds in world43; "
        "second training seed, not independent worlds or query-order causality. "
        "Known train.1029 teacher quantity difference retained.",
    )
    plan["jobs"] = [dict(j, condition=condition) for j in template["jobs"]]
    for module in (seed, panel):
        plan["source_sha256"][str(Path(module.__file__).resolve())] = c.inputs.sha(
            Path(module.__file__)
        )
    plan["source_sha256"][str(Path(__file__).resolve())] = c.inputs.sha(Path(__file__))
    path = OUTPUTS[teacher] / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable prepared plan changed")
    else:
        c.save(path, plan)
    return plan, list(map(json.loads, (prepared / "tasks.jsonl").read_text().splitlines())), binding


def analyze(report):
    from transformers import AutoTokenizer

    if report.exists() or report.with_suffix(".md").exists():
        raise FileExistsError(report)
    prepared = [prepare(t) for t in seed.TEACHERS]
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    world = panel.worlds()[1]
    arms, rows = [], []
    for teacher in seed.TEACHERS:
        output = OUTPUTS[teacher]
        arm = profiles.audit.analyze(output, tokenizer, world=world)
        for key in ("paired", "depth_strata"):
            arm.pop(key, None)
        arms.append(arm)
        rows.append(
            {
                (r["task_id"], r["repeat"]): r
                for r in (json.loads(p.read_text()) for p in (output / "episodes").glob("*.json"))
            }
        )
    result = dict(
        schema="textcraft-world43-second-seed-analysis-v1",
        arms=arms,
        public_minus_privileged=profiles.compare(prepared[0][0]["jobs"], *rows),
        parents=8,
        repeats=2,
        planned_per_arm=16,
        caveat=prepared[0][0]["caveat"],
        source_sha256=c.inputs.sha(Path(__file__)),
    )
    c.save(report, result)
    with report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Second-seed changed-world teacher comparison\n\n" + result["caveat"] + "\n\n"
        )
        stream.write(
            json.dumps(
                {k: v for k, v in result["public_minus_privileged"].items() if k != "rows"},
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=seed.TEACHERS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report:
        analyze(args.report)
    else:
        plan, tasks, binding = prepare(args.teacher)
        c.run(
            argparse.Namespace(output=OUTPUTS[args.teacher], prepare_only=args.prepare_only),
            prepared_run=(plan, tasks),
            adapter=binding,
            world=None if args.prepare_only else panel.worlds()[1],
        )
