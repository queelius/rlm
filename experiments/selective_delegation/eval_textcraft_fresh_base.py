"""Proposed no-adapter baseline on the already frozen fresh16 TextCraft panel."""

import argparse
import json
from pathlib import Path

import eval_textcraft_fresh as fresh

c = fresh.c
TEMPLATE = c.ROOT / "textcraft-fresh-privileged-001/PLAN.json"
TEMPLATE_SHA = "92c39e0297c641451376070ef05c5dd531b768f227a48593dfc3d6bcf048f343"
TEACHER_REPORT = c.ROOT / "analysis-textcraft-fresh-001.json"
TEACHER_REPORT_SHA = "574b545bfb6bd7b00c3e83758d7bc4007464afca86ca9a958d4679f96995e480"
COLLECTION_CONTRACT = dict(
    base_collection_seconds=7200,
    teacher_collection_seconds=3600,
    per_episode_reasoning_budget_changed=False,
    teacher_report=str(TEACHER_REPORT),
    teacher_report_sha256=TEACHER_REPORT_SHA,
    declaration="Prospective before any base outcomes: collection wall cap only. Both teacher "
    "arms completed all32 observed slots within their one-hour collection cap. All per-episode "
    "calls/tokens/context/sampling/seed limits unchanged. Unknown base slots stay unknown; "
    "no complete-panel effect/CI unless all paired outcomes observed.",
)


def prepare(args):
    if args.hours not in (1, 2) or c.inputs.sha(TEMPLATE) != TEMPLATE_SHA:
        raise ValueError("declared one/two-hour collection cap and frozen fresh PLAN required")
    plan = json.loads(TEMPLATE.read_text())
    prepared = Path(plan["prepared"])
    if (
        c.inputs.sha(prepared / "MANIFEST.json") != fresh.MANIFEST_SHA
        or c.inputs.sha(prepared / "tasks.jsonl") != fresh.TASKS_SHA
        or plan["world_sha256"] != fresh.WORLD_SHA
    ):
        raise ValueError("frozen fresh panel/world changed")
    tasks = list(map(json.loads, (prepared / "tasks.jsonl").read_text().splitlines()))
    if len(tasks) != 16 or len(plan["jobs"]) != 32:
        raise ValueError("all16 tasks/two seeds required")
    for key in ("adapter", "fixed_adapter", "training_plan_sha256", "teacher"):
        plan.pop(key, None)
    plan.update(
        schema="textcraft-fresh16-base-readout-v1",
        adapters=None,
        conditions=["fresh_base"],
        jobs=[dict(j, condition="fresh_base") for j in plan["jobs"]],
        baseline_template=str(TEMPLATE),
        baseline_template_sha256=TEMPLATE_SHA,
        prompt_difference="None: fixed fresh original flat interface; no PEFT adapter loaded.",
        caveat="Same frozen16roots/two correlated seeds in world42; shared prerequisites. "
        "Collection cap may leave unknown slots; no outcome-based replacement or imputation.",
    )
    if args.hours == 2:
        if c.inputs.sha(TEACHER_REPORT) != TEACHER_REPORT_SHA:
            raise ValueError("completed teacher native report changed")
        completed = json.loads(TEACHER_REPORT.read_text())
        groups = [next(iter(arm["groups"].values())) for arm in completed["arms"]]
        if len(groups) != 2 or any(g["planned"] != 32 or g["observed"] != 32 for g in groups):
            raise ValueError("both teacher arms must have all32 native-observed outcomes")
        plan.update(budget_seconds=7200, collection_wall_cap_contract=COLLECTION_CONTRACT)
    for module in (c, fresh):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = c.inputs.sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = c.inputs.sha(Path(__file__))
    target = args.output / "PLAN.json"
    if target.exists():
        if json.loads(target.read_text()) != plan:
            raise ValueError("immutable base fresh PLAN changed")
    else:
        c.save(target, plan)
    return plan, tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    c.run(args, prepared_run=prepare(args), adapter=None)
