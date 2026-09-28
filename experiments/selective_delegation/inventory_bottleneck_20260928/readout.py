"""Fixed 8+8 public-demand versus masked-table pilot; parent owns GPU launch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import demand_table as table

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
OUTPUT = ROOT / "textcraft-public-demand-20260928-001"
BASELINE = ROOT / "textcraft-breadth-p00-w42-soriginal-binder-001"
RUNTIME = ROOT / "source-textcraft-recipe-binder-003"
SEEDS = (2026092804, 2026092805)
DEEP_IDS = (
    "textcraft_synth.val.628",
    "textcraft_synth.val.599",
    "textcraft_synth.val.401",
    "textcraft_synth.val.294",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


@cache
def runtime():
    sys.path.insert(0, str(RUNTIME))
    import analyze_textcraft as audit
    import eval_textcraft as collector
    import eval_textcraft_trained as trained
    import inspect_textcraft_worlds as worlds

    for module in (collector, audit, collector.bridge, trained, worlds):
        require(Path(module.__file__).parent == RUNTIME, "mixed native runtime imports")
    return collector, audit, trained, worlds


@cache
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        runtime()[0].BASE, local_files_only=True, trust_remote_code=False
    )


def build(mode, root=OUTPUT, persist=True):
    require(mode in table.MODES, "undeclared table arm")
    collector, auditor, trained, worlds = runtime()
    baseline = read(BASELINE / "PLAN.json")
    baseline_audit = read(BASELINE / "NATIVE-AUDIT.json")
    require(
        sha(BASELINE / "PLAN.json") == baseline_audit["sha256"][str(BASELINE / "PLAN.json")],
        "audited baseline changed",
    )
    prepared = Path(baseline["prepared"])
    for name, key in (("tasks.jsonl", "tasks_sha256"), ("MANIFEST.json", "manifest_sha256")):
        require(sha(prepared / name) == baseline[key], "paired panel bytes changed")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    deep = [t for t in tasks if t["misc"]["max_depth"] in (4, 5)]
    require(tuple(t["id"] for t in deep) == DEEP_IDS, "fixed deep selection differs")
    binding = trained.endpoint(
        Path(baseline["adapter"]["path"]),
        training_plan_sha256=baseline["adapter"]["training_plan_sha256"],
        rows_sha256=baseline["adapter"]["training_rows_sha256"],
    )
    require(binding == baseline["adapter"], "actor differs from audited discovery checkpoint23")
    world = collector.bridge.load_world()
    require(worlds.digest(worlds.snapshot(world)) == baseline["world_sha256"], "world42 changed")
    lengths = []
    with table.installed(collector.bridge, mode, tokenizer()):
        for task in deep:
            lengths.append(
                len(table.input_ids(tokenizer(), collector.bridge.initial_prompt(task, "flat")))
            )
    require(max(lengths) + 256 <= 8192, "table initial prompt exceeds unchanged context cap")
    condition = f"public_demand_{mode}_p00_w42_original"
    plan = copy.deepcopy(baseline)
    plan.update(
        schema="textcraft-public-demand-pilot-20260928-v1",
        table_mode=mode,
        conditions=[condition],
        jobs=[
            dict(job, condition=condition, seed=SEEDS[job["repeat"]])
            for job in baseline["jobs"]
            if job["task_id"] in DEEP_IDS
        ],
        seeds=list(SEEDS),
        parent_tasks=4,
        prepared_task_file_contains=8,
        planned_episodes=8,
        planned_per_policy=8,
        planned_per_condition=8,
        max_native_calls=768,
        budget_seconds=2700,
        initial_token_audit=dict(prompt_tokens=lengths, max_prompt_plus_cap=max(lengths) + 256),
        baseline_plan=str(BASELINE / "PLAN.json"),
        baseline_plan_sha256=sha(BASELINE / "PLAN.json"),
        table_contract=dict(
            description=table.DESCRIPTION,
            computed_columns=list(table.DERIVED),
            fact_source="Only original serialized public prompt; no native world or hidden graph",
            history="Unmodified full history in both arms",
            ordering="Alphabetical item order, never a selected next action",
            token_matching="Equal complete encoded prompt lengths at the same public state; "
            "shorter rendering padded with semantically empty x tokens. No free prompt tokens.",
            semantics="Existing observed-recipe binder unchanged; no new execution assistance",
        ),
        prompt_difference="Public-fact table plus derived quantities versus masked computed "
        "columns; same instructions, facts, full history and state-conditional token length",
        caveat="Adaptive exploratory reuse of four already-exposed deep VAL task identities in "
        "world42, not held-out confirmation. Fresh fixed paired seeds. Host arithmetic assistance, "
        "not pure memory. Padding/presentation effects and differing trajectories remain possible. "
        "All planned slots accounted for; incomplete outcomes remain unknown, never failures.",
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            packages={
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "tokenizers")
            },
        ),
    )
    for module in (collector, collector.bridge, collector.recipe_binder, auditor, trained, worlds):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = sha(path)
    for path in (
        Path(__file__).resolve(),
        Path(table.__file__).resolve(),
        Path(table.math.__file__),
    ):
        plan["source_sha256"][str(path)] = sha(path)
    for path, digest in plan["source_sha256"].items():
        require(sha(path) == digest, "sealed source changed: " + path)
    directory = root / mode
    if persist:
        path = directory / "PLAN.json"
        if path.exists():
            require(read(path) == plan, "immutable table PLAN differs")
        else:
            save(path, plan)
    return plan, deep, binding, world, directory


@contextmanager
def correct_summary_counts(collector):
    """The historical collector's 16-slot display must not mislabel this 8-slot slice."""
    original = collector.summarize

    def summarize(output, plan):
        result = original(output, plan)
        for condition, group in result["groups"].items():
            planned = sum(j["condition"] == condition for j in plan["jobs"])
            group["planned"] = planned
            group["missing_or_unknown"] = planned - group["observed"]
        return result

    collector.summarize = summarize
    try:
        yield
    finally:
        collector.summarize = original


def audit_arm(mode, root=OUTPUT):
    import psutil

    plan, tasks, _, world, directory = build(mode, root)
    collector, auditor, _, _ = runtime()
    owners = list(directory.glob("OWNER-*.json"))
    require(len(owners) == 1, "single scientific owner required")
    owner = read(owners[0])
    terminal_path = owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-"))
    terminal = read(terminal_path)
    try:
        proc = psutil.Process(owner["pid"])
        require(
            abs(proc.create_time() - owner["create_time"]) >= 0.01
            or proc.status() == psutil.STATUS_ZOMBIE,
            "scientific owner still live",
        )
    except psutil.NoSuchProcess:
        pass
    require(owner["source"] in plan["source_sha256"], "unbound collector owner")
    episodes = {p.stem: read(p) for p in (directory / "episodes").glob("*.json")}
    calls = {p.stem: read(p) for p in (directory / "calls").glob("*.json")}
    starts = {p.stem: read(p) for p in (directory / "starts").glob("*.json")}
    require(set(episodes) <= {j["episode_id"] for j in plan["jobs"]}, "unplanned episode")
    require(set(calls) <= set(starts), "call has no start receipt")
    for cid, call in calls.items():
        require(
            starts[cid]["request"] == call["request"]
            and starts[cid]["request_digest"] == call["request_digest"],
            "saved native start/call mismatch",
        )
    lookup, audits, used, diagnoses = {t["id"]: t for t in tasks}, {}, [], []
    with table.installed(collector.bridge, mode, tokenizer()):
        for job in plan["jobs"]:
            eid = job["episode_id"]
            if eid not in episodes:
                continue
            row = episodes[eid]
            nodes = {n: read(directory / "nodes" / f"{eid}-{n}.json") for n in row["node_ids"]}
            audits[eid] = auditor.audit_episode(
                lookup[job["task_id"]],
                job,
                row,
                calls,
                nodes,
                plan,
                sha(directory / "PLAN.json"),
                tokenizer(),
                world,
            )
            used.extend(row["call_ids"])
            if row["observed"]:
                diagnostic = table.math.analyze_node(nodes["n0"])
                diagnoses.append({k: diagnostic[k] for k in ("counts", "flags", "native_score")})
    require(len(used) == len(set(used)), "sampled calls reused across episodes")
    require(set(calls) <= set(used), "orphan completed native calls")
    known = [r for r in episodes.values() if r["observed"]]
    successes = sum(r["native_score"] for r in known)
    report = dict(
        schema="textcraft-public-demand-native-audit-20260928-v1",
        table_mode=mode,
        planned=8,
        recorded=len(episodes),
        observed=len(known),
        missing=8 - len(episodes),
        unavailable=len(episodes) - len(known),
        successes=successes,
        success_rate_bounds=[successes / 8, (successes + 8 - len(known)) / 8],
        audits=audits,
        terminal=terminal,
        physical_cost=collector.cost(list(calls.values())),
        unresolved_starts=sorted(set(starts) - set(calls)),
        diagnostic_counts=dict(sum((Counter(d["counts"]) for d in diagnoses), Counter())),
        diagnostic_episode_flags=dict(
            sum((Counter({k: int(v) for k, v in d["flags"].items()}) for d in diagnoses), Counter())
        ),
        sha256={str(p): sha(p) for p in directory.rglob("*.json") if p.name != "NATIVE-AUDIT.json"},
        scope="Saved requests/tokens/decodes and binder native replay; no statistical "
        "independence attributed to repeated attempts. Unknown outcomes remain unknown.",
    )
    save(directory / "NATIVE-AUDIT.json", report)
    return {k: report[k] for k in ("planned", "observed", "successes", "success_rate_bounds")}


def main(args):
    if args.audit:
        print(json.dumps(audit_arm(args.mode, args.root), indent=2))
        return
    plan, tasks, binding, world, directory = build(args.mode, args.root)
    collector = runtime()[0]
    with (
        correct_summary_counts(collector),
        table.installed(collector.bridge, args.mode, tokenizer()),
    ):
        collector.run(
            argparse.Namespace(output=directory, prepare_only=args.prepare_only),
            prepared_run=(plan, tasks),
            adapter=binding,
            world=world,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--mode", choices=table.MODES, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    main(parser.parse_args())
