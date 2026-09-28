"""Fixed panel00 Phi teacher/assistance readout; one paired seed, unchanged native worlds."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import acquire  # noqa: E402
import eval_textcraft_trained as shared  # noqa: E402
import inspect_textcraft_worlds as worlds  # noqa: E402
import native_adapter  # noqa: E402
import qualify  # noqa: E402
import train  # noqa: E402

PLAN_SHAS = {
    "discovery": "c0294328161bbd0cd603140fc950c821c6581f617336e5941f9d80d06bd20321",
    "known": "4eda31f6857a6a823d3d2bac61c4a19b43da3cd9dc5e494fe129f879b8b5ea71",
}


def world_for(seed):
    world = qualify.bridge.load_world()
    if seed == 50:
        module = sys.modules["pinned_textcraft_synth_generator_d9c5857d"]
        module.set_naming_mode(semantic=False)
        world = module.SynthRecipeDatabase()
        world.generate_all_recipes(seed=seed, items_per_domain_tier=25)
    elif seed != 42:
        raise ValueError("only pinned worlds42/50")
    return world


def endpoint(teacher):
    if teacher == "base":
        return None
    output = acquire.ROOT / f"textcraft-phi-{teacher}-sft-20260928-001"
    _, contract = train.prepare(teacher, output)
    shared.collector.BASE = acquire.MODEL
    return shared.endpoint(
        output / "checkpoint-0023",
        training_plan_sha256=PLAN_SHAS[teacher],
        rows_sha256=contract["rows_sha256"],
    )


def build(args, require_endpoint=True):
    from transformers import AutoTokenizer

    if not 0 < args.hours <= 0.5 or args.teacher == "base" and args.hours > 0.25:
        raise ValueError("cap30minutes per trained8-episode arm,15minutes per base2-episode arm")
    if args.teacher == "base" and args.assistance != "raw":
        raise ValueError("tiny base control is raw-only; no additional grid")
    if acquire.sha(acquire.OUTPUT / "QUALIFICATION.json") != train.QUALIFICATION_SHA:
        raise ValueError("frozen Phi qualification changed")
    qualification = acquire.read(acquire.OUTPUT / "QUALIFICATION.json")
    panel = qualification["fixed_panels"][str(args.world)]
    prepared = Path(panel["prepared"])
    if acquire.sha(prepared / "tasks.jsonl") != panel["tasks_sha256"] or (
        acquire.sha(prepared / "MANIFEST.json") != panel["manifest_sha256"]
    ):
        raise ValueError("frozen panel00 input changed")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    if [task["id"] for task in tasks] != panel["task_ids"]:
        raise ValueError("exact original panel task order required")
    baseline = acquire.ROOT / f"textcraft-breadth-p00-w{args.world}-soriginal-binder-001"
    if acquire.sha(baseline / "PLAN.json") != panel["baseline_plan_sha256"]:
        raise ValueError("pinned original panel schedule changed")
    original = acquire.read(baseline / "PLAN.json")
    allowed = (
        {tasks[0]["id"], tasks[-1]["id"]} if args.teacher == "base" else set(panel["task_ids"])
    )
    jobs = []
    for job in original["jobs"]:
        if job["seed"] == 2026092204 and job["task_id"] in allowed:
            jobs.append(
                dict(
                    episode_id=f"phi-{args.teacher}-{args.assistance}-w{args.world}-"
                    + job["episode_id"],
                    task_id=job["task_id"],
                    policy="flat",
                    repeat=0,
                    seed=2026092204,
                )
            )
    if len(jobs) != len(allowed):
        raise ValueError("exact one seed per fixed goal required")
    collector, auditor, binding = native_adapter.implementation(args.assistance)
    fixture = acquire.read(acquire.OUTPUT / "native-fixture-001/VERIFICATION.json")
    bound = next(
        case["binding"] for case in fixture["cases"] if case["assistance"] == args.assistance
    )
    if json.loads(json.dumps(binding)) != bound or not fixture["passed"]:
        raise ValueError("native Phi client differs from actual saved-request CPU fixture")
    world = world_for(args.world)
    if worlds.digest(worlds.snapshot(world)) != panel["world_sha256"]:
        raise ValueError("native world identity differs")
    tokenizer = AutoTokenizer.from_pretrained(
        acquire.MODEL, local_files_only=True, trust_remote_code=False
    )
    paths = {
        Path(__file__),
        Path(native_adapter.__file__),
        Path(qualify.__file__),
        Path(acquire.__file__),
        Path(train.__file__),
        Path(collector.__file__),
        Path(auditor.__file__),
        Path(collector.bridge.__file__),
        Path(collector.inputs.__file__),
        Path(collector.probe.__file__),
        Path(collector.probe.runtime.__file__),
        Path(collector.probe.campaign.__file__),
        Path(shared.__file__),
        Path(worlds.__file__),
    }
    if args.assistance == "binder":
        paths.add(Path(collector.recipe_binder.__file__))
    plan = dict(
        schema="phi-teaching-assistance-fixed-panel-20260928-v1",
        teacher=args.teacher,
        assistance=args.assistance,
        prepared=str(prepared),
        tasks_sha256=panel["tasks_sha256"],
        manifest_sha256=panel["manifest_sha256"],
        jobs=jobs,
        planned_episodes=len(jobs),
        parent_tasks=len(allowed),
        seeds=[2026092204],
        profile="original",
        model=str(acquire.MODEL),
        model_revision=acquire.REVISION,
        model_manifest_sha256=acquire.sha(acquire.MODEL / "local-research-manifest.json"),
        adapter=endpoint(args.teacher) if require_endpoint else None,
        endpoint_pending=not require_endpoint and args.teacher != "base",
        qualification_sha256=train.QUALIFICATION_SHA,
        native_stop_binding=binding,
        native_fixture_sha256=acquire.sha(acquire.OUTPUT / "native-fixture-001/VERIFICATION.json"),
        source_sha256={str(p.resolve()): acquire.sha(p) for p in sorted(paths)},
        trusted_source=collector.bridge.trusted_provenance(),
        world_seed=args.world,
        world_sha256=panel["world_sha256"],
        max_global_calls=96,
        max_global_output_tokens=8192,
        max_new_tokens=256,
        input_plus_output_limit=8192,
        truncation=False,
        max_agent_depth=0,
        max_native_calls=96 * len(jobs),
        budget_seconds=args.hours * 3600,
        optimizer=None,
        initial_token_audit=panel["initial_prompt_tokens"],
        sampling=dict(temperature=0.5, top_p=1.0, top_k=0, max_new_tokens=256, do_sample=True),
        comparison="Within Phi: discovery versus corrected-known at23 updates, each raw versus "
        "observed-history binder; same8 fixed roots, one paired seed, worlds42/50. Base control "
        "uses panel positions0/7 only. Eight distinct roots, not64 independent examples.",
        caveat="Across families tokenizer, stop IDs, parameterization, LoRA target count and "
        "compute differ. This tests package/model dependence, not an isolated causal mechanism.",
    )
    return plan, tasks, world, tokenizer, collector, auditor


def audit(output):
    import psutil
    from transformers import AutoTokenizer

    plan = acquire.read(output / "PLAN.json")
    collector, auditor, binding = native_adapter.implementation(plan["assistance"])
    if json.loads(json.dumps(binding)) != plan["native_stop_binding"]:
        raise ValueError("recorded Phi stop binding changed")
    owners = list(output.glob("OWNER-*.json"))
    if (
        len(owners) != 1
        or not owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-")).exists()
    ):
        raise ValueError("single terminal native owner required")
    owner = acquire.read(owners[0])
    try:
        process = psutil.Process(owner["pid"])
        if (
            abs(process.create_time() - owner["create_time"]) < 0.01
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            raise ValueError("scientific owner still live")
    except psutil.NoSuchProcess:
        pass
    if str(Path(owner["source"]).resolve()) not in plan["source_sha256"]:
        raise ValueError("owner source unbound")
    for path, digest in plan["source_sha256"].items():
        if acquire.sha(path) != digest:
            raise ValueError("sealed Phi source changed: " + path)
    prepared = Path(plan["prepared"])
    if acquire.sha(prepared / "tasks.jsonl") != plan["tasks_sha256"] or (
        acquire.sha(prepared / "MANIFEST.json") != plan["manifest_sha256"]
    ):
        raise ValueError("sealed Phi panel changed")
    tasks = {
        t["id"]: t
        for t in (json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines())
    }
    calls = {p.stem: acquire.read(p) for p in (output / "calls").glob("*.json")}
    starts = {p.stem: acquire.read(p) for p in (output / "starts").glob("*.json")}
    rows = {p.stem: acquire.read(p) for p in (output / "episodes").glob("*.json")}
    if set(calls) != set(starts) or not set(rows) <= {job["episode_id"] for job in plan["jobs"]}:
        raise ValueError("unresolved request or unplanned episode")
    for cid in calls:
        if starts[cid]["request"] != calls[cid]["request"]:
            raise ValueError("native request changed after start")
    if set(calls) != {cid for row in rows.values() for cid in row["call_ids"]}:
        raise ValueError("extra/unbound native call")
    tokenizer = AutoTokenizer.from_pretrained(
        acquire.MODEL, local_files_only=True, trust_remote_code=False
    )
    world = world_for(plan["world_seed"])
    if worlds.digest(worlds.snapshot(world)) != plan["world_sha256"]:
        raise ValueError("audit world changed")
    episodes = []
    for job in plan["jobs"]:
        row = rows.get(job["episode_id"])
        if row is None:
            episodes.append(
                dict(task_id=job["task_id"], observed=False, native_score=None, missing=True)
            )
            continue
        nodes = {
            nid: acquire.read(output / "nodes" / f"{job['episode_id']}-{nid}.json")
            for nid in row["node_ids"]
        }
        checked = auditor.audit_episode(
            tasks[job["task_id"]],
            job,
            row,
            calls,
            nodes,
            plan,
            acquire.sha(output / "PLAN.json"),
            tokenizer,
            world,
        )
        episodes.append(dict(task_id=job["task_id"], observed=row["observed"], **checked))
    return dict(
        schema="phi-textcraft-native-audit-20260928-v1",
        passed=True,
        teacher=plan["teacher"],
        assistance=plan["assistance"],
        world_seed=plan["world_seed"],
        episodes=episodes,
        observed=sum(e["observed"] for e in episodes),
        successes=sum(e["native_score"] == 1 for e in episodes if e["observed"]),
        unknown=sum(not e["observed"] for e in episodes),
        physical_cost=collector.cost(list(calls.values())),
        plan_sha256=acquire.sha(output / "PLAN.json"),
        caveat="Unknown/unfinished roots are not imputed. Complete episodes replay native "
        "transitions; external failures receive the narrower request/response binding audit.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("discovery", "known", "base"), required=True)
    parser.add_argument("--assistance", choices=("raw", "binder"), default="raw")
    parser.add_argument("--world", type=int, choices=(42, 50), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit(args.output.resolve())
        acquire.save(args.output / "PHI-AUDIT.json", result)
        print(json.dumps(result, indent=2))
    else:
        plan, tasks, world, _, collector, _ = build(args, not args.validate_inputs_only)
        if args.validate_inputs_only:
            print(
                json.dumps(
                    dict(
                        episodes=len(plan["jobs"]),
                        teacher=args.teacher,
                        assistance=args.assistance,
                        world=args.world,
                        endpoint_pending=plan["endpoint_pending"],
                        GPU_loaded=False,
                    )
                )
            )
        else:
            path = args.output / "PLAN.json"
            if path.exists():
                if acquire.read(path) != plan:
                    raise ValueError("immutable Phi readout PLAN changed")
            else:
                acquire.save(path, plan)
            collector.run(args, prepared_run=(plan, tasks), adapter=plan["adapter"], world=world)
