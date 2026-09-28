"""Small TRAIN-only raw/binder probe; reuse the audited native implementation."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

LIBRARY = Path(
    "/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/"
    "experiments/selective_delegation"
)
sys.path.insert(0, str(LIBRARY))

import rl_textcraft_terminal as rl  # noqa: E402

c, read, sha = rl.c, rl.audit.read, rl.c.inputs.sha
ROOT = c.ROOT
WARM = ROOT / "textcraft-public-discovery-sft-001/checkpoint-0023"
BINDER = ROOT / "source-textcraft-recipe-binder-003"
BINDER_PINS = {
    "eval_textcraft.py": "1e8156909b2c17cbed20d3add5e429daece4d602979e6e44bde628ad7296c92b",
    "analyze_textcraft.py": "f1e095401698432fcf630b62baba88df9852e3b0699f711553ba583a382344d1",
}


@lru_cache(maxsize=2)
def implementation(mode: str):
    if mode == "raw":
        return c, rl.audit
    if mode != "binder":
        raise ValueError("only raw and binder are declared")
    modules = []
    for filename, digest in BINDER_PINS.items():
        path = BINDER / filename
        if sha(path) != digest:
            raise ValueError("sealed binder implementation changed")
        spec = importlib.util.spec_from_file_location("resume_bound_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def tasks() -> list[dict]:
    plan_path = ROOT / "textcraft-train-readiness-001/PLAN.json"
    if sha(plan_path) != rl.readiness.PLAN_SHA:
        raise ValueError("accepted TRAIN selection provenance changed")
    return rl.readiness.runtime_tasks(read(plan_path))


def jobs(rows: list[dict], mode: str, phase: str, update: int) -> list[dict]:
    if (
        len(rows) != 8
        or len({t["id"] for t in rows}) != 8
        or any(not t["id"].startswith("textcraft_synth.train.") for t in rows)
    ):
        raise ValueError("exactly eight unique official TRAIN tasks required")
    if mode not in ("raw", "binder") or phase not in ("collect", "readout"):
        raise ValueError("undeclared mode or phase")
    if update not in range(1, 5):
        raise ValueError("at most four prospectively numbered fresh batches")
    repeats = 4 if phase == "collect" else 2
    # Readout seeds are fixed across checkpoints and disjoint from every training batch.
    seed = 2026092800 + 10 * update if phase == "collect" else 2026092890
    return [
        dict(
            episode_id=f"t{i:02d}-r{repeat}-flat",
            task_id=task["id"],
            repeat=repeat,
            seed=seed + repeat,
            policy="flat",
            prompt_profile="original",
            condition=f"resume_{mode}_{phase}",
        )
        for repeat in range(repeats)
        for i, task in enumerate(rows)
    ]


def adapter_binding(checkpoint: Path) -> dict:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(ROOT):
        raise ValueError("checkpoint outside this campaign")
    commit = read(checkpoint / "COMMIT.json")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if sha(checkpoint / name) != commit["files"][name]:
            raise ValueError("adapter differs from committed bytes")
    return rl.binding(checkpoint)


def source_pins(entrypoint: Path, mode: str) -> dict[str, str]:
    collector, auditor = implementation(mode)
    modules = (
        rl,
        rl.loss_math,
        rl.checkpoints,
        c,
        c.bridge,
        c.inputs,
        c.probe,
        c.probe.runtime,
        c.probe.campaign,
        rl.audit,
        rl.reader,
        rl.readiness,
        collector,
        auditor,
    )
    paths = {Path(m.__file__).resolve() for m in modules}
    paths.update(
        (entrypoint.resolve(), Path(__file__).resolve(), LIBRARY / "textcraft_recipe_binder.py")
    )
    return {str(p): sha(p) for p in sorted(paths)}


def prepare_collection(args, entrypoint: Path) -> tuple[dict, list[dict]]:
    rows = tasks()
    if not 0 < args.hours <= 1.5 or not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("campaign output and at most 90 minutes required")
    adapter = adapter_binding(args.checkpoint)
    old = read(ROOT / "textcraft-train-readiness-001/PLAN.json")
    plan = dict(
        schema="textcraft-assisted-rl-train-collection-20260928-v1",
        split="train",
        execution_mode=args.mode,
        phase=args.phase,
        update=args.update,
        model=str(c.BASE),
        model_manifest_sha256=sha(c.BASE / "local-research-manifest.json"),
        base_dtype="float16",
        lora_dtype="float32",
        adapter=adapter,
        prepared=old["prepared"],
        tasks_sha256=old["tasks_sha256"],
        manifest_sha256=old["manifest_sha256"],
        selection_plan_sha256=rl.readiness.PLAN_SHA,
        runtime_tasks_source=str(ROOT / rl.reader.SOURCE_TASKS),
        runtime_tasks_sha256=rl.reader.SOURCE_TASKS_SHA,
        jobs=jobs(rows, args.mode, args.phase, args.update),
        planned_episodes=32 if args.phase == "collect" else 16,
        profile="original",
        sampling=dict(temperature=0.5, top_p=1.0, top_k=0),
        max_global_calls=96,
        max_global_output_tokens=8192,
        max_new_tokens=256,
        input_plus_output_limit=8192,
        truncation=False,
        budget_seconds=args.hours * 3600,
        source_sha256=source_pins(entrypoint, args.mode),
        likelihood_target="Exact original sampled output_token_ids, including EOS/errors; "
        "never retokenized executed arguments. Binder is a deterministic environment transition.",
        caveat="Exploratory exposed TRAIN panel; no held-out goal touched. Paired raw/binder "
        "seeds and call/token caps, but realized trajectories and credited-token doses can differ.",
    )
    path = args.output / "PLAN.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("immutable collection PLAN differs")
    else:
        c.save(path, plan)
    return plan, rows


def audit_collection(directory, plan, rows, tokenizer, world, mode) -> dict:
    """Use the native auditors, including requested/executed binder transitions."""
    _, auditor = implementation(mode)
    episodes = {p.stem: read(p) for p in (directory / "episodes").glob("*.json")}
    calls = {p.stem: read(p) for p in (directory / "calls").glob("*.json")}
    starts = {p.stem: read(p) for p in (directory / "starts").glob("*.json")}
    if set(episodes) != {j["episode_id"] for j in plan["jobs"]} or set(starts) != set(calls):
        raise ValueError("incomplete collection or unresolved native request")
    for cid, call in calls.items():
        if starts[cid]["request"] != call["request"] or (
            starts[cid]["request_digest"] != call["request_digest"]
        ):
            raise ValueError("native start/call identity differs")
    lookup, audits, covered = {t["id"]: t for t in rows}, {}, []
    for job in plan["jobs"]:
        row = episodes[job["episode_id"]]
        if not row["observed"]:
            raise ValueError("unobserved episodes cannot become training failures")
        nodes = {
            n: read(directory / "nodes" / f"{job['episode_id']}-{n}.json") for n in row["node_ids"]
        }
        audits[job["episode_id"]] = auditor.audit_episode(
            lookup[job["task_id"]],
            job,
            row,
            calls,
            nodes,
            plan,
            sha(directory / "PLAN.json"),
            tokenizer,
            world,
        )
        covered.extend(row["call_ids"])
    if len(covered) != len(set(covered)) or set(covered) != set(calls):
        raise ValueError("orphan or reused sampled actor calls")
    groups = {}
    for task in rows:
        group = [e for e in episodes.values() if e["task_id"] == task["id"]]
        won = sum(e["native_score"] for e in group)
        groups[task["id"]] = dict(observed=len(group), successes=won, mixed=0 < won < len(group))
    diversity = {}
    for task in rows:
        first = [
            calls[e["call_ids"][0]]["text"] for e in episodes.values() if e["task_id"] == task["id"]
        ]
        diversity[task["id"]] = dict(samples=len(first), unique_first_responses=len(set(first)))
    return dict(
        planned=len(plan["jobs"]),
        observed=len(episodes),
        successes=sum(e["native_score"] for e in episodes.values()),
        groups=groups,
        mixed_groups=sum(g["mixed"] for g in groups.values()),
        audits=audits,
        physical_cost=c.cost(list(calls.values())),
        first_response_diversity=diversity,
        errors=dict(sum((Counter(e["errors"]) for e in episodes.values()), Counter())),
        receipt_sha256={
            str(p): sha(p)
            for folder in ("calls", "starts", "nodes", "episodes")
            for p in sorted((directory / folder).glob("*.json"))
        },
    )
