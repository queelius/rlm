"""Truthful compact-interface adapters around unchanged native collection/RLOO loops."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl_fresh_20260928"))
import fresh_common as f  # noqa: E402

COMPACT = HERE.parent / "compact_actions_20260928"
sys.path.insert(0, str(COMPACT))


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


interface = module(COMPACT / "evaluate.py", "compact_rl_native_interface")
if Path(interface.data.__file__).resolve() != COMPACT / "prepare.py":
    raise ValueError("ambiguous compact preparation module import")
MODE = "compact_observed"
ROOT, rl, read, sha = f.ROOT, f.rl, f.read, f.sha
STUDY = ROOT / "textcraft-compact-rl-20260928-001"
BASELINE = ROOT / "textcraft-fresh-rl-20260928-002"
WARM = interface.TRAIN_OUTPUT / "checkpoint-0023"
c, auditor = interface.implementation("compact")


def implementation(mode: str):
    if mode != MODE:
        raise ValueError("compact observed-only interface must be explicitly declared")
    return c, auditor


# Private namespace reuse keeps the frozen auditor's logic and changes its interface dependency.
_audit = module(f.LEGACY / "probe_common.py", "compact_rl_native_audit_common")
_audit.implementation = implementation
_audit.c = c
audit_collection = _audit.audit_collection


class PendingCheckpoint(ValueError):
    """A future checkpoint path is not a scientific adapter binding."""


@lru_cache(maxsize=2)
def _warm_binding(path: Path) -> dict:
    if not (path / "COMMIT.json").exists():
        raise PendingCheckpoint("compact SFTcp23 has not committed; weights remain unknown")
    if path != interface.TRAIN_OUTPUT / "checkpoint-0023":
        raise ValueError("only the accepted compact SFT checkpoint23 may initialize this cycle")
    verified = interface.binding("compact")
    # Existing compact endpoint validation checks completed owner, SFT dose and committed bytes.
    f.persist(STUDY / "WARM-ENDPOINT.json", verified)
    return {key: verified[key] for key in ("path", "sha256", "commit_sha256")}


def warm_binding() -> dict:
    return _warm_binding(WARM.resolve())


def endpoint() -> str | None:
    path = STUDY / "train-0001/SUMMARY.json"
    result = read(path) if path.exists() else {}
    return result.get("endpoint") if result.get("endpoint_usable") else None


def adapter_binding(checkpoint: Path) -> dict:
    checkpoint = checkpoint.resolve()
    if checkpoint == WARM:
        return warm_binding()
    expected = endpoint()
    if not expected or checkpoint != Path(expected):
        raise ValueError("only the prespecified usable compact first-step endpoint is allowed")
    plan = read(STUDY / "train-0001/PLAN.json")
    state = read(checkpoint / "STATE.json")
    if (
        plan["schema"] != "textcraft-compact-terminal-rloo-20260928-v1"
        or plan["execution_mode"] != MODE
        or plan["adapter"] != warm_binding()
        or state["step"] != 1
        or state["plan_sha256"] != sha(STUDY / "train-0001/PLAN.json")
        or not checkpoint.is_relative_to(STUDY / "train-0001")
    ):
        raise ValueError("compact optimizer lineage changed")
    return f.adapter_binding(checkpoint)


def jobs(rows: list[dict], phase: str) -> list[dict]:
    # Reuse the fixed fresh seed/task schedule only; scientific interface identity is new.
    return [dict(j, condition=f"{MODE}_{phase}") for j in f.jobs(rows, "binder", phase, 1)]


def source_pins() -> dict:
    pins = f.prior.source_pins(f.LEGACY / "collect.py", "raw")
    pins.update(f.prior.source_pins(f.LEGACY / "train.py", "raw"))
    paths = [
        f.HERE / "fresh_common.py",
        f.LEGACY / "compare.py",
        *[
            HERE / name
            for name in (
                "compact_common.py",
                "compact_stage.py",
                "compact_compare.py",
                "prepare_compact_rl.py",
                "compact_rl_fixture.py",
            )
        ],
        *[
            COMPACT / name
            for name in ("compact_bridge.py", "evaluate.py", "prepare.py", "train.py")
        ],
        Path(interface.shared.__file__),
        Path(interface.public.__file__),
        Path(interface.data.accepted.__file__),
    ]
    pins.update({str(path.resolve()): sha(path) for path in paths})
    return pins


def prepare_collection(args, entrypoint: Path):
    if args.mode != MODE or args.update != 1 or args.phase not in ("collect", "readout"):
        raise ValueError("exactly one declared compact RL cycle")
    maximum = 2.5 if args.phase == "collect" else 1.5
    if not 0 < args.hours <= maximum:
        raise ValueError("declared compact collection/readout cap exceeded")
    group = "train" if args.phase == "collect" else "diagnostic"
    dataset = f.DATA / group
    manifest, rows = f.load_dataset(dataset, args.phase)
    suffix = (
        "collect-0001"
        if args.phase == "collect"
        else ("readout-warm" if args.checkpoint.resolve() == WARM else "readout-0001")
    )
    if args.output.resolve() != STUDY / suffix:
        raise ValueError("prospectively fixed compact output path required")
    binding = adapter_binding(args.checkpoint)
    if args.phase == "collect" and binding != warm_binding():
        raise ValueError("first compact training batch must use its actual compact SFT warm actor")
    plan = dict(
        schema="textcraft-compact-rl-collection-20260928-v1",
        split="train",
        execution_mode=MODE,
        phase=args.phase,
        update=1,
        dataset_group=group,
        prepared=str(dataset),
        tasks_sha256=manifest["tasks_sha256"],
        manifest_sha256=sha(dataset / "MANIFEST.json"),
        dataset_provenance=manifest,
        **f.dataset_binding(dataset),
        world_seed=42,
        world_sha256=manifest["world_sha256"],
        model=str(c.BASE),
        model_manifest_sha256=sha(c.BASE / "local-research-manifest.json"),
        adapter=binding,
        base_dtype="float16",
        lora_dtype="float32",
        jobs=jobs(rows, args.phase),
        planned_episodes=32 if group == "train" else 16,
        profile="original",
        prompt_template="compact_actions_20260928.compact_bridge.public_prompt",
        sampling=dict(temperature=0.5, top_p=1.0, top_k=0),
        max_global_calls=96,
        max_global_output_tokens=8192,
        max_new_tokens=256,
        input_plus_output_limit=8192,
        truncation=False,
        budget_seconds=args.hours * 3600,
        source_sha256=source_pins(),
        compact_sft_plan_sha256=interface.TRAIN_PLAN_SHA,
        compact_sft_contract_sha256=interface.TRAIN_CONTRACT_SHA,
        interface_contract="Strict compact craft(target_item,output_count); exact observed-only "
        "recipe expansion; unobserved/ambiguous/nondivisible crafts rejected. Public history "
        "retains compact requests, native stock/score unchanged. Flat-only.",
        likelihood_target="Every original sampled compact token including EOS and invalid outputs; "
        "executed ingredients are environment transitions, never substituted actor targets.",
        caveat="Official TRAIN groupB is diagnostic-only. Same SFT updates do not match fullformat "
        "token exposure; strict unknown-recipe rejection differs from full binder. "
        "No inference from raw cross-interface score differences to a learning gain.",
    )
    f.persist(args.output / "PLAN.json", plan)
    return plan, rows


def prepare_training(args) -> dict:
    batch = read(args.collection / "PLAN.json")
    if (
        batch.get("schema") != "textcraft-compact-rl-collection-20260928-v1"
        or batch.get("dataset_group") != "train"
        or batch.get("execution_mode") != MODE
        or batch.get("phase") != "collect"
        or batch.get("update") != 1
        or batch.get("planned_episodes") != 32
    ):
        raise ValueError("complete native compact groupA32 collection required")
    manifest, rows = f.load_dataset(f.DATA / "train", "collect")
    if (
        args.collection.resolve() != STUDY / "collect-0001"
        or args.output.resolve() != STUDY / "train-0001"
        or not 0 < args.hours <= 2
        or batch["base_dtype"] != "float16"
        or batch["adapter"] != warm_binding()
        or batch["manifest_sha256"] != f.GROUP_SHA["train"]
        or batch["tasks_sha256"] != manifest["tasks_sha256"]
        or batch["jobs"] != jobs(rows, "collect")
    ):
        raise ValueError("one exact-behavior compact update under the real warm actor required")
    plan = dict(
        schema="textcraft-compact-terminal-rloo-20260928-v1",
        split="train",
        execution_mode=MODE,
        update=1,
        collection=str(args.collection.resolve()),
        collection_plan_sha256=sha(args.collection / "PLAN.json"),
        dataset=str(f.DATA / "train"),
        dataset_manifest_sha256=f.GROUP_SHA["train"],
        dataset_tasks_sha256=batch["tasks_sha256"],
        **f.dataset_binding(f.DATA / "train"),
        adapter=batch["adapter"],
        compact_sft_plan_sha256=interface.TRAIN_PLAN_SHA,
        compact_sft_contract_sha256=interface.TRAIN_CONTRACT_SHA,
        base_dtype="float16",
        lora_dtype="float32",
        learning_rate=2e-5,
        weight_decay=0,
        clip_grad_norm=1.0,
        optimizer="fresh_AdamW",
        budget_seconds=args.hours * 3600,
        maximum_new_optimizer_updates=1,
        max_replay_gap=0.25,
        mean_replay_gap=0.025,
        objective="-sum_i[(native_success_i - mean(other three same-task native rewards)) "
        "* sum_original_emitted_tokens log pi_T0.5]/32",
        reward="Unchanged native terminal success; all emitted errors retain trajectory reward",
        target="Original compact sampled token IDs including EOS/invalid outputs, "
        "not executed args",
        on_policy="One step from exact collection weights. "
        "No rollout reuse or optimizer continuation.",
        source_sha256=source_pins(),
        controls="Own compact warm/post groupB gain, compared with full+binder first-fresh gain. "
        "SFT token dose and strict recipe semantics differ; not an isolated schema-only effect.",
    )
    f.persist(args.output / "PLAN.json", plan)
    return plan


def load_legacy(filename: str):
    result = f.load_legacy(filename)
    result.p = sys.modules[__name__]
    if filename == "train.py":
        result.prepare = prepare_training
    return result


def skip(output: Path, reason: str):
    f.persist(
        output / "CONDITIONAL-SKIP.json",
        dict(
            reason=reason,
            execution_mode=MODE,
            GPU_loaded=False,
            scientific_calls=0,
            actual_optimizer_steps=0,
            endpoint=None,
            endpoint_usable=False,
        ),
    )
