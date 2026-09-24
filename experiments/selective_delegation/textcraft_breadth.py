"""Frozen new-goal raw/binder panels; run with the matching sealed import package."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import textcraft_multiworld as m

ROOT = m.c.ROOT
INPUT = ROOT / "textcraft-breadth-inputs-001"
WORLDS = (42, 50, 51, 52)
SEED = 2026092401


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def select(rows, excluded, count=8):
    used = {k for t in excluded for k in t["misc"]["target_items"]}
    ids = {t["id"] for t in excluded}
    panels = []
    for _ in range(count):
        selected = []
        for depth, n in ((2, 2), (3, 2), (4, 1), (5, 3)):
            candidates = sorted(
                [t for t in rows if t["id"] not in ids and t["misc"]["max_depth"] == depth],
                key=lambda t: hashlib.sha256(f"{SEED}:{t['id']}".encode()).hexdigest(),
            )
            taken = 0
            for task in candidates:
                roots = set(task["misc"]["target_items"])
                if roots & used:
                    continue
                selected.append(task)
                used.update(roots)
                ids.add(task["id"])
                taken += 1
                if taken == n:
                    break
            if taken != n:
                raise ValueError("insufficient fixed stratum; no replacement")
        panels.append(selected)
    return panels


def freeze():
    from transformers import AutoTokenizer

    if INPUT.exists():
        raise FileExistsError(INPUT)
    official = m.panel.prior.inputs.TASKS
    train = official.with_name("textcraft_synth_train.jsonl")
    paths = sorted(ROOT.glob("textcraft*/tasks.jsonl"))
    # PLAN inventory includes task IDs whose prepared directory is nested.
    plans = sorted(ROOT.glob("textcraft*/PLAN.json"))
    rows = read(official)
    by_id = {t["id"]: t for t in rows}
    excluded = read(train) + [t for p in paths for t in read(p)]
    for p in plans:
        for job in json.loads(p.read_text()).get("jobs", []):
            if job.get("task_id") in by_id:
                excluded.append(by_id[job["task_id"]])
    panels = select(rows, excluded)
    m.c.save(
        INPUT / "SELECTION.json",
        dict(
            seed=SEED,
            selected_before_native_replay=True,
            model_outcomes_used=False,
            panels=[[t["id"] for t in tasks] for tasks in panels],
            worlds=WORLDS,
            strata={2: 2, 3: 2, 4: 1, 5: 3},
            scope="All official TRAIN roots, current top-level task inventories and PLAN VAL IDs; "
            "not an unbounded guarantee against all historical exposure.",
            input_sha256={str(p): m.c.inputs.sha(p) for p in [official, train, *paths, *plans]},
            no_replacement=True,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(m.c.BASE, local_files_only=True)
    manifests = []
    for world_seed in WORLDS:
        world = m.panel.world(world_seed)
        digest = m.panel.prior.inventory.digest(m.panel.prior.inventory.snapshot(world))
        for index, tasks in enumerate(panels):
            dest = INPUT / f"p{index:02d}-w{world_seed}"
            dest.mkdir()
            changed, audits = [], []
            for task in tasks:
                # Selection already frozen. Construction failures are fatal, never replaced.
                item = dict(m.panel.prior.construct(task, world), derived_world_seed=world_seed)
                audit = dict(task_id=task["id"], statistics=m.panel.prior.statistics(item, world))
                try:
                    audit["qualification"] = m.panel.prior.qualify(item, world, tokenizer)
                except ValueError as exc:
                    audit["qualification_failure"] = str(exc)
                changed.append(item)
                audits.append(audit)
            with (dest / "tasks.jsonl").open("x") as stream:
                for item in changed:
                    stream.write(json.dumps(item) + "\n")
            manifest = dict(
                panel=index,
                world_seed=world_seed,
                world_sha256=digest,
                task_count=8,
                tasks_sha256=m.c.inputs.sha(dest / "tasks.jsonl"),
                audits=audits,
                no_replacement=True,
            )
            m.c.save(dest / "MANIFEST.json", manifest)
            manifests.append(dict(path=str(dest), sha256=m.c.inputs.sha(dest / "MANIFEST.json")))
    m.c.save(
        INPUT / "MANIFEST.json",
        dict(
            panels=manifests,
            selection_sha256=m.c.inputs.sha(INPUT / "SELECTION.json"),
            preparer_sha256=m.c.inputs.sha(Path(__file__)),
            caveat="New-to-inventory goals, shared generator/item grammar; derived sufficient "
            "inventory changes by world. Depth5 is an explicitly expanded difficulty stratum. "
            "Native qualification failures retained, not model outcome selection.",
        ),
    )


def output(panel, world, seed, mode):
    return ROOT / f"textcraft-breadth-p{panel:02d}-w{world}-s{seed}-{mode}-001"


def build(panel, world, seed, mode):
    if panel not in range(8) or world not in WORLDS or mode not in ("raw", "binder"):
        raise ValueError("undeclared panel/world/mode")
    collector = Path(m.c.__file__).read_text()
    if ("bind_observed_recipe_arguments" in collector) != (mode == "binder"):
        raise ValueError("wrong raw/binder native collector import")
    plan, _, adapter = m.build(47, seed, "public")
    plan = copy.deepcopy(plan)
    prepared = INPUT / f"p{panel:02d}-w{world}"
    manifest = json.loads((prepared / "MANIFEST.json").read_text())
    master = json.loads((INPUT / "MANIFEST.json").read_text())
    expected = next(x for x in master["panels"] if x["path"] == str(prepared))
    if m.c.inputs.sha(prepared / "MANIFEST.json") != expected["sha256"]:
        raise ValueError("panel manifest changed")
    if m.c.inputs.sha(prepared / "tasks.jsonl") != manifest["tasks_sha256"]:
        raise ValueError("task bytes changed")
    tasks = read(prepared / "tasks.jsonl")
    condition = f"breadth_p{panel}_w{world}_{seed}_{mode}"
    plan.update(
        schema="textcraft-paired-breadth-v1",
        prepared=str(prepared),
        tasks_sha256=manifest["tasks_sha256"],
        manifest_sha256=expected["sha256"],
        world_seed=world,
        world_sha256=manifest["world_sha256"],
        budget_seconds=2700,
        conditions=[condition],
        breadth_panel=panel,
        execution_mode=mode,
        caveat=master["caveat"],
        initial_token_audit=manifest["audits"],
    )
    plan["jobs"] = [
        dict(j, task_id=tasks[int(j["episode_id"][1:3])]["id"], condition=condition)
        for j in plan["jobs"]
    ]
    for path in (Path(__file__).resolve(), INPUT / "MANIFEST.json", INPUT / "SELECTION.json"):
        plan["source_sha256"][str(path)] = m.c.inputs.sha(path)
    return plan, tasks, adapter


def run(args):
    plan, tasks, adapter = build(args.panel, args.world, args.seed, args.mode)
    directory = output(args.panel, args.world, args.seed, args.mode)
    if (directory / "PLAN.json").exists():
        if json.loads((directory / "PLAN.json").read_text()) != plan:
            raise ValueError("immutable PLAN differs")
    else:
        m.c.save(directory / "PLAN.json", plan)
    if args.report:
        import analyze_textcraft_profiles as profiles
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(m.c.BASE, local_files_only=True)
        result = profiles.audit.analyze(directory, tokenizer, world=m.checked_world(plan))
        result.pop("paired", None)
        result.pop("depth_strata", None)
        m.c.save(args.report, result)
        return
    m.c.run(
        argparse.Namespace(output=directory, prepare_only=args.prepare_only),
        prepared_run=(plan, tasks),
        adapter=adapter,
        world=None if args.prepare_only else m.checked_world(plan),
    )


def compare(panel, world, seed, report):
    import analyze_textcraft_profiles as profiles

    arms, rows = [], []
    for mode in ("raw", "binder"):
        directory = output(panel, world, seed, mode)
        audit_path = directory / "NATIVE-AUDIT.json"
        arm = json.loads(audit_path.read_text())
        for path in [directory / "PLAN.json", *(directory / "episodes").glob("*.json")]:
            if arm["sha256"].get(str(path)) != m.c.inputs.sha(path):
                raise ValueError("native-audited receipt changed")
        arms.append(
            dict(
                native_audit=str(audit_path),
                sha256=m.c.inputs.sha(audit_path),
                groups=arm["groups"],
            )
        )
        rows.append(
            {
                (r["task_id"], r["repeat"]): r
                for r in (
                    json.loads(p.read_text()) for p in (directory / "episodes").glob("*.json")
                )
            }
        )
    plans = [
        json.loads((output(panel, world, seed, mode) / "PLAN.json").read_text())
        for mode in ("raw", "binder")
    ]
    if m.normalize(plans[0]["jobs"]) != m.normalize(plans[1]["jobs"]):
        raise ValueError("paired slots changed")
    for key in ("budget_seconds", "tasks_sha256", "world_sha256", "fixed_adapter"):
        if plans[0][key] != plans[1][key]:
            raise ValueError("paired scientific contract changed")
    m.c.save(
        report,
        dict(
            arms=arms,
            binder_minus_raw=profiles.compare(plans[0]["jobs"], *rows),
            caveat=plans[0]["caveat"],
            planned_per_arm=16,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--panel", type=int, default=0)
    parser.add_argument("--world", type=int, default=42)
    parser.add_argument("--seed", choices=("original", "2291"), default="original")
    parser.add_argument("--mode", choices=("raw", "binder"), default="raw")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.compare:
        compare(args.panel, args.world, args.seed, args.compare)
    else:
        run(args)
