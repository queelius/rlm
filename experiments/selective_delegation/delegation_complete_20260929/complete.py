"""Two complete-goal readouts per frozen oracle-free one-boundary routing policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLEX = HERE.parent / "decomposition_flexible_20260928"
sys.path.insert(0, str(FLEX))
_spec = importlib.util.spec_from_file_location(
    "complete_private_flexible", FLEX / "run_flexible.py"
)
flex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flex)
data = flex.data
ROOT = data.ROOT
STUDY = ROOT / "textcraft-delegation-complete-20260929-001"
SEEDS = (2026092901, 2026092902)
MODES = ("flat", "fixed", "adaptive")
SCHEMA = "textcraft-complete-goal-delegation-20260929-v1"
PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)
read, sha = data.read, data.sha


def persist(path: Path, value: dict) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError("immutable complete-goal artifact differs: " + str(path))
    else:
        data.save(path, value)


def jobs(mode: str) -> list[dict]:
    if mode not in MODES:
        raise ValueError("undeclared routing policy")
    return [
        dict(
            episode_id=f"decomp-val494-{mode}-r{repeat}",
            task_id="textcraft_synth.val.494",
            policy="flat" if mode == "flat" else "recursive",
            repeat=repeat,
            seed=seed,
        )
        for repeat, seed in enumerate(SEEDS)
    ]


def implementation(mode: str):
    collector, auditor = flex.base.implementation(mode)
    if not getattr(collector, "complete_client_installed", False):
        native = collector.NativeClient.__bases__[0]
        if native.__name__ != "NativeClient":
            raise ValueError("unexpected sealed admission client inheritance")

        class CompleteClient(native):
            def ids(self, prompt):
                control = json.loads(prompt.rsplit(flex.routing.MARKER, 1)[1])
                if control["abort_requested"]:
                    raise flex.base.AdmissionStop(
                        "two instruction rejections; root outcome unknown"
                    )
                return super().ids(prompt)

        collector.NativeClient = CompleteClient
        collector.complete_client_installed = True
    return collector, auditor


def build(mode: str, output: Path, *, fixture: bool = False):
    args = argparse.Namespace(mode=mode, hours=0.25, output=output)
    plan, tasks, world, tokenizer, _, _ = flex.build(args, fixture=fixture)
    collector, auditor = implementation(mode)
    plan.update(
        schema=SCHEMA,
        jobs=jobs(mode),
        planned_episodes=2,
        seeds=list(SEEDS),
        budget_seconds=1200,
        admission_response_cap=None,
        max_native_calls=192,
        comparison="Three fixed routing arms on exposed val494/world42, two new paired seeds. "
        "Same publiccp23/binder and native global96-call/8192-token budgets; one child at most. "
        "Remove only32-response screening ceiling, not the two-delegate-refusal guard.",
        stop_contract="Native finish or original global-call/output/context termination. "
        "Two delegate refusals, transport or owner interruption remain UNKNOWN; later slots "
        "may be unattempted. All returned/error responses are charged. No retry or repair.",
        scientific_scope="One exposed root, two correlated seeds, one fit checkpoint. "
        "Deterministic public-information routing, not learned recursion or robust transfer.",
    )
    for path in HERE.glob("*.py"):
        plan["source_sha256"][str(path)] = sha(path)
    return plan, tasks, world, tokenizer, collector, auditor


def check_plan(plan: dict) -> None:
    if (
        plan.get("schema") != SCHEMA
        or plan["jobs"] != jobs(plan["routing"])
        or plan["max_global_calls"] != 96
        or plan["max_global_output_tokens"] != 8192
        or plan["input_plus_output_limit"] != 8192
        or plan["world_seed"] != 42
        or plan["planned_episodes"] != 2
        or plan["admission_response_cap"] is not None
        or plan["instruction_rejection_cap"] != 2
        or plan["budget_seconds"] != 1200
    ):
        raise ValueError("complete-goal scientific contract changed")
    for name, digest in plan["source_sha256"].items():
        if sha(name) != digest:
            raise ValueError("sealed dependency changed: " + name)
    prepared = Path(plan["prepared"])
    if sha(prepared / "tasks.jsonl") != plan["tasks_sha256"] or (
        sha(prepared / "MANIFEST.json") != plan["manifest_sha256"]
    ):
        raise ValueError("original qualified val494 data changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.study / args.mode
    plan, tasks, world, _, collector, _ = build(args.mode, output)
    check_plan(plan)
    persist(output / "PLAN.json", plan)
    try:
        collector.run(
            argparse.Namespace(output=output, prepare_only=args.prepare_only),
            prepared_run=(plan, tasks),
            adapter=plan["adapter"],
            world=world,
        )
    except RuntimeError as exc:
        if "AdmissionStop:" not in str(exc):
            raise
        persist(output / "ADMISSION-STOP.json", dict(reason=str(exc), root_score=None))
        print(json.dumps(dict(admission_failure=str(exc), root_outcome="unknown")))
