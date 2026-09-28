"""One-boundary, public-information admission probe; parent alone launches GPUs."""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_textcraft_public as public  # noqa: E402
import eval_textcraft_trained as shared  # noqa: E402
import prepare_data as data  # noqa: E402
import routing  # noqa: E402

BINDER = data.ROOT / "source-textcraft-recipe-binder-003"
BINDER_PINS = {
    "eval_textcraft.py": "1e8156909b2c17cbed20d3add5e429daece4d602979e6e44bde628ad7296c92b",
    "analyze_textcraft.py": "f1e095401698432fcf630b62baba88df9852e3b0699f711553ba583a382344d1",
}
RESPONSE_CAP = 32
SEED = 2026092806


class AdmissionStop(RuntimeError):
    """Planned screening cutoff, NOT a failed native root reward."""


@lru_cache(maxsize=3)
def implementation(mode):
    for filename, digest in BINDER_PINS.items():
        if data.sha(BINDER / filename) != digest:
            raise ValueError("sealed binder implementation changed")
    collector = data.load(BINDER / "eval_textcraft.py", "decomposition_collector_" + mode)
    auditor = data.load(BINDER / "analyze_textcraft.py", "decomposition_auditor_" + mode)
    collector.bridge = routing.make_bridge(mode)
    auditor.collector, auditor.bridge = collector, collector.bridge

    class AdmissionClient(collector.NativeClient):
        response_cap = RESPONSE_CAP

        def ids(self, prompt):
            control = json.loads(prompt.rsplit(routing.MARKER, 1)[1])
            if self.returned >= self.response_cap:
                raise AdmissionStop(f"response_limit={self.response_cap}; root outcome unknown")
            if control["abort_requested"]:
                raise AdmissionStop("two instruction rejections; root outcome unknown")
            return super().ids(prompt)

    collector.NativeClient = AdmissionClient
    return collector, auditor


def build(args, fixture=False):
    if not 0 < args.hours <= 0.25:
        raise ValueError("admission arm cap must be at most15minutes")
    prepared = data.OUTPUT
    manifest = data.read(prepared / "MANIFEST.json")
    if (
        data.sha(prepared / "tasks.jsonl") != manifest["tasks_sha256"]
        or data.sha(prepared / "SELECTION.json") != manifest["selection_sha256"]
        or not manifest["all_native_feasible"]
    ):
        raise ValueError("frozen fresh admission task changed/unqualified")
    for path, digest in manifest["source_sha256"].items():
        if data.sha(path) != digest:
            raise ValueError("data preparer changed after selection")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    if len(tasks) != 1 or tasks[0]["id"] != "textcraft_synth.val.494":
        raise ValueError("exact prospective fresh task required")
    collector, auditor = implementation(args.mode)
    world = routing.native.load_world()
    if data.worlds.digest(data.worlds.snapshot(world)) != manifest["world_sha256"]:
        raise ValueError("native world42 changed")
    adapter = (
        None
        if fixture
        else shared.endpoint(
            data.ROOT / "textcraft-public-discovery-sft-001/checkpoint-0023",
            training_plan_sha256=public.PLAN_SHA,
            rows_sha256=public.ROWS_SHA,
        )
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        collector.BASE, local_files_only=True, trust_remote_code=False
    )
    task = tasks[0]
    frame = collector.bridge.Frame(
        world,
        dict(task["misc"]["initial_inventory"]),
        task["misc"]["target_items"],
        routing.native.Budget(),
        0 if args.mode == "flat" else 1,
    )
    initial = collector.bridge.public_prompt(frame, [], goal=task["goal"])
    count = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": initial}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if count + 256 > 8192:
        raise ValueError("initial context exceeds native cap")
    paths = {
        Path(__file__).resolve(),
        Path(routing.__file__),
        Path(routing.native.__file__),
        Path(data.__file__),
        Path(collector.__file__),
        Path(auditor.__file__),
        Path(collector.recipe_binder.__file__),
        Path(collector.inputs.__file__),
        Path(collector.probe.__file__),
        Path(collector.probe.runtime.__file__),
        Path(collector.probe.campaign.__file__),
        Path(shared.__file__),
        Path(public.__file__),
    }
    job = dict(
        episode_id=f"decomp-val494-{args.mode}-r0",
        task_id=task["id"],
        policy="flat" if args.mode == "flat" else "recursive",
        repeat=0,
        seed=SEED,
    )
    plan = dict(
        schema="textcraft-decomposition-admission-20260928-v1",
        routing=args.mode,
        prepared=str(prepared),
        manifest_sha256=data.sha(prepared / "MANIFEST.json"),
        tasks_sha256=manifest["tasks_sha256"],
        jobs=[job],
        planned_episodes=1,
        parent_tasks=1,
        seeds=[SEED],
        profile="original",
        model=str(collector.BASE),
        model_manifest_sha256=data.sha(collector.BASE / "local-research-manifest.json"),
        adapter=adapter,
        optimizer=None,
        world_seed=42,
        world_sha256=manifest["world_sha256"],
        max_global_calls=96,
        max_global_output_tokens=8192,
        max_new_tokens=256,
        input_plus_output_limit=8192,
        truncation=False,
        root_depth=0,
        max_agent_depth=0 if args.mode == "flat" else 1,
        max_helper_boundaries=1,
        admission_response_cap=RESPONSE_CAP,
        instruction_rejection_cap=2,
        budget_seconds=args.hours * 3600,
        max_native_calls=RESPONSE_CAP,
        initial_token_audit=dict(prompt_tokens=count, prompt_plus_cap=count + 256),
        source_sha256={str(p.resolve()): data.sha(p) for p in sorted(paths)},
        trusted_source=routing.native.trusted_provenance(),
        fixture=fixture,
        model_generation=not fixture,
        comparison="Same publiccp23 root/helper, original task bytes, stock, seed, sampling, "
        "common mandatory public discovery and candidate facts. Flat versus one fixed "
        "lexical helper boundary versus public stock/budget/observed-depth branching gate.",
        stock_contract="UNCHANGED native shared inventory, child snapshots at entry, no stock "
        "reservation. Subgoal quantity is new shortage, root scored against original inventory.",
        stop_contract="Before a next request, after32 returned responses or two instruction "
        "rejections. No phantom calls/repairs. Planned stops are UNKNOWN root outcomes, not0.",
        scientific_scope="Admission screen, not recursion efficacy or trained recursion. "
        "Oracle-free deterministic harness routing; observed depth is partial knowledge. "
        "One task/seed; no confidence interval or promotion to robust claim.",
    )
    return plan, tasks, world, tokenizer, collector, auditor


def audit(output, require_terminal=True):
    import psutil
    from transformers import AutoTokenizer

    plan = data.read(output / "PLAN.json")
    collector, auditor = implementation(plan["routing"])
    if require_terminal:
        owners = list(output.glob("OWNER-*.json"))
        if len(owners) != 1:
            raise ValueError("single scientific owner required")
        owner = data.read(owners[0])
        if not owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-")).exists():
            raise ValueError("terminal owner required")
        try:
            process = psutil.Process(owner["pid"])
            if abs(process.create_time() - owner["create_time"]) < 0.01 and (
                process.status() != psutil.STATUS_ZOMBIE
            ):
                raise ValueError("scientific owner still live")
        except psutil.NoSuchProcess:
            pass
        if str(Path(owner["source"]).resolve()) not in plan["source_sha256"]:
            raise ValueError("owner source unbound")
    for path, digest in plan["source_sha256"].items():
        if data.sha(path) != digest:
            raise ValueError("sealed admission source changed: " + path)
    prepared = Path(plan["prepared"])
    if data.sha(prepared / "MANIFEST.json") != plan["manifest_sha256"] or (
        data.sha(prepared / "tasks.jsonl") != plan["tasks_sha256"]
    ):
        raise ValueError("sealed admission data changed")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    calls = {p.stem: data.read(p) for p in (output / "calls").glob("*.json")}
    starts = {p.stem: data.read(p) for p in (output / "starts").glob("*.json")}
    job = plan["jobs"][0]
    row = data.read(output / "episodes" / (job["episode_id"] + ".json"))
    if set(calls) != set(starts) or set(calls) != set(row["call_ids"]):
        raise ValueError("extra/unresolved calls or starts")
    for cid, call in calls.items():
        if starts[cid]["request"] != call["request"] or (
            starts[cid]["request_digest"] != call["request_digest"]
        ):
            raise ValueError("saved request changed between start and response")
    nodes = {
        nid: data.read(output / "nodes" / f"{job['episode_id']}-{nid}.json")
        for nid in row["node_ids"]
    }
    if len(calls) > plan["admission_response_cap"]:
        raise ValueError("admission scientific response cap exceeded")
    tokenizer = AutoTokenizer.from_pretrained(
        collector.BASE, local_files_only=True, trust_remote_code=False
    )
    world = routing.native.load_world()
    if data.worlds.digest(data.worlds.snapshot(world)) != plan["world_sha256"]:
        raise ValueError("audit world changed")
    checked = auditor.audit_episode(
        tasks[0], job, row, calls, nodes, plan, data.sha(output / "PLAN.json"), tokenizer, world
    )
    children = [node for node in nodes.values() if node["depth"] > 0]
    deltas = []
    for child in children:
        before, after = child["initial_inventory"], child["final_inventory"]
        deltas.append(
            dict(
                node_id=child["node_id"],
                targets=child["targets"],
                score=child["native_score"],
                stock_delta={
                    k: after.get(k, 0) - before.get(k, 0)
                    for k in sorted(set(before) | set(after))
                    if after.get(k, 0) != before.get(k, 0)
                },
            )
        )
    returned = [call for call in calls.values() if call["available"]]
    output_tokens = sum(len(call["output_token_ids"]) for call in returned)
    if output_tokens != row["global_output_tokens"]:
        raise ValueError("all returned root/child/error tokens must be charged")
    return dict(
        schema="textcraft-decomposition-admission-audit-20260928-v1",
        passed=True,
        routing=plan["routing"],
        task_id=tasks[0]["id"],
        native_audit=checked,
        root_observed=row["observed"],
        root_score=row["native_score"],
        failure=row["failure"],
        calls=len(calls),
        returned=len(returned),
        failed=len(calls) - len(returned),
        output_tokens=output_tokens,
        prompt_tokens=sum(call["usage"]["prompt_tokens"] for call in calls.values()),
        native_service_seconds=sum(call["ended"] - call["started"] for call in calls.values()),
        child_nodes=len(children),
        child_calls=sum(len(c["call_ids"]) for c in children),
        errors=row["errors"],
        child_stock_deltas=deltas,
        criterion="Admission established only when a real delegate creates a child with real "
        "responses. Full native state replay only for completed/observed episodes; planned "
        "screening stops retain unknown root outcome and receive receipt binding audit only.",
        artifacts_sha256={str(p): data.sha(p) for p in output.rglob("*.json")},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("flat", "fixed", "adaptive"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.25)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit(args.output.resolve())
        data.save(args.output / "ADMISSION-AUDIT.json", result)
        print(json.dumps(result, indent=2))
    else:
        plan, tasks, world, _, collector, _ = build(args)
        path = args.output / "PLAN.json"
        if path.exists():
            if data.read(path) != plan:
                raise ValueError("immutable admission PLAN changed")
        else:
            data.save(path, plan)
        try:
            collector.run(args, prepared_run=(plan, tasks), adapter=plan["adapter"], world=world)
        except RuntimeError as exc:
            if "AdmissionStop:" not in str(exc):
                raise
            data.save(args.output / "ADMISSION-STOP.json", dict(reason=str(exc), root_score=None))
            print(json.dumps(dict(planned_admission_stop=str(exc), root_outcome="unknown")))
