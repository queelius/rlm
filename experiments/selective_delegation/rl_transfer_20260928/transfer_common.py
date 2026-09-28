"""Frozen fresh-B readouts of the two existing familiar-goal RL checkpoints."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl_fresh_20260928"))
import fresh_common as f  # noqa: E402

STUDY = f.ROOT / "textcraft-rl-transfer-20260928-001"
FAMILIAR = f.ROOT / "textcraft-rl-assist-20260928-001"
WARM_STUDY = f.ROOT / "textcraft-fresh-rl-20260928-002"
WARM_RECEIPT = WARM_STUDY / "PREPARED-CAMPAIGN-0004.json"
WARM_RECEIPT_SHA = "4e10be9cd54480c7deee72118b31f5d49ee56042a9889213dc08fd4a6de7f3d6"
SCHEMA = "textcraft-familiar-rl-fresh-diagnostic-transfer-20260928-v1"
PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=4)
def admit_endpoint(mode: str, directory: Path | None = None) -> dict:
    """Check only actual readout dependencies, not unused optimizer tensors or base ancestry."""
    if mode not in ("raw", "binder"):
        raise ValueError("only the two fixed familiar-goal RL actors are declared")
    directory = directory or FAMILIAR / mode / "train-0001"
    summary = f.read(directory / "SUMMARY.json")
    plan = f.read(directory / "PLAN.json")
    checkpoint = Path(summary.get("endpoint") or "/absent").resolve()
    if (
        summary.get("complete") is not True
        or summary.get("failure") is not None
        or summary.get("endpoint_usable") is not True
        or summary.get("actual_new_optimizer_steps") != 1
        or summary.get("committed_optimizer_steps") != 1
        or checkpoint != (directory / "boundaries/sample-0001/checkpoint-0001").resolve()
        or plan.get("schema") != "textcraft-assisted-terminal-rloo-20260928-v1"
        or plan.get("execution_mode") != mode
        or plan.get("update") != 1
        or plan.get("adapter") != f.adapter_binding(f.WARM)
    ):
        raise ValueError("actual usable familiar-goal one-update endpoint required")
    state, commit = (f.read(checkpoint / name) for name in ("STATE.json", "COMMIT.json"))
    if (
        state.get("step") != 1
        or commit.get("step") != 1
        or state.get("plan_sha256") != f.sha(directory / "PLAN.json")
        or commit.get("files", {}).get("STATE.json") != f.sha(checkpoint / "STATE.json")
    ):
        raise ValueError("committed STATE or training plan identity changed")
    adapter = f.adapter_binding(checkpoint)
    paths = [directory / name for name in ("PLAN.json", "SUMMARY.json")]
    paths += [
        checkpoint / name
        for name in (
            "STATE.json",
            "COMMIT.json",
            "adapter_config.json",
            "adapter_model.safetensors",
        )
    ]
    return dict(
        adapter=adapter,
        training_execution_mode=mode,
        training_directory=str(directory),
        optimizer_steps=1,
        artifact_sha256={str(path): f.sha(path) for path in paths},
        validation="Actual usable SUMMARY; one-update STATE/COMMIT and plan linkage; "
        "adapter/config committed bytes. Optimizer/RNG are not loaded by readout.",
    )


def arguments(actor: str, mode: str, study: Path, *, prepare_only: bool = False):
    endpoint = admit_endpoint(actor)
    if mode not in ("raw", "binder") or not study.resolve().is_relative_to(f.ROOT):
        raise ValueError("declared execution interface and campaign output required")
    return SimpleNamespace(
        actor=actor,
        mode=mode,
        dataset=f.DATA / "diagnostic",
        phase="readout",
        update=1,
        checkpoint=Path(endpoint["adapter"]["path"]),
        hours=1.5,
        output=study / f"actor-{actor}" / f"readout-{mode}",
        prepare_only=prepare_only,
    )


def build_plan(args) -> tuple[dict, list[dict]]:
    # Private planner: suppress its early persistence, then add true actor lineage before saving.
    planner = load_file("transfer_private_planner", f.HERE / "fresh_common.py")
    planner.persist = lambda _path, _plan: None
    plan, rows = planner.prepare_collection(args, f.LEGACY / "collect.py")
    endpoint = admit_endpoint(args.actor)
    if args.phase != "readout" or plan["adapter"] != endpoint["adapter"]:
        raise ValueError("transfer is readout-only at the admitted fixed endpoint")
    plan.update(
        schema=SCHEMA,
        actor_training_execution_mode=args.actor,
        actor_lineage=endpoint,
        original_collection_schema="textcraft-fresh-train-collection-20260928-v1",
        diagnostic_role="Existing frozen fresh-B official TRAIN transfer diagnostic; "
        "no optimization, reselection, or endpoint selection from its outcomes.",
        matched_warm_readout=str(WARM_STUDY / args.mode / "readout-warm"),
    )
    for name in ("transfer_common.py", "collect.py", "prepare.py", "compare.py"):
        path = HERE / name
        plan["source_sha256"][str(path)] = f.sha(path)
    return plan, rows


def prepare_collection(args, _entrypoint: Path):
    plan, rows = build_plan(args)
    f.persist(args.output / "PLAN.json", plan)
    return plan, rows


def load_collector():
    native = f.load_legacy("collect.py")
    dependencies = dict(vars(f))
    dependencies["prepare_collection"] = prepare_collection
    native.p = SimpleNamespace(**dependencies)
    return native


def warm_evidence() -> dict:
    if f.sha(WARM_RECEIPT) != WARM_RECEIPT_SHA:
        raise ValueError("accepted fresh campaign receipt changed")
    names = {f"fresh-{mode}-warm-readout-0001" for mode in ("raw", "binder")}
    jobs = [job for job in f.read(WARM_RECEIPT)["jobs"] if job["name"] in names]
    if {job["name"] for job in jobs} != names or len(jobs) != 2:
        raise ValueError("expected exactly two original warm descriptors")
    for job in jobs:
        for path, digest in job["pins"].items():
            if f.sha(Path(path)) != digest:
                raise ValueError("accepted warm dependency changed: " + path)
    paths = [f.HERE / name for name in ("stage.py", "collect.py", "fresh_common.py")]
    paths += [f.LEGACY / "collect.py", f.LEGACY / "probe_common.py"]
    return dict(
        independent=True,
        accepted_receipt=str(WARM_RECEIPT),
        accepted_receipt_sha256=WARM_RECEIPT_SHA,
        original_descriptors=jobs,
        source_sha256={str(path): f.sha(path) for path in paths},
        reasoning="stage.resolve(kind=readout, actor=warm) directly sets checkpoint=f.WARM "
        "and output=study/mode/readout-warm; it never calls endpoint() or reads any A result. "
        "The native collector consumes that checkpoint, diagnostic tasks, world42 and fixed "
        "readout jobs. f.jobs uses seeds202609280900/901 independently of update. "
        "Original descriptor pins mention training SOURCE/data, not future training OUTPUTS. "
        "Moving these unchanged jobs changes schedule, not their scientific dependencies.",
        reuse="Parent may move each exact descriptor once; none is in this package's jobs list.",
    )
