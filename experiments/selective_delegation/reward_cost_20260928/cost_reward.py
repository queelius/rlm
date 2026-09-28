"""Independent composite rewards; native scores and original actor tokens never change."""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl_fresh_20260928"))
import fresh_common as f  # noqa: E402

FRESH_STUDY = f.ROOT / "textcraft-fresh-rl-20260928-002"
STUDY = f.ROOT / "textcraft-error-cost-20260928-001"
CHARGED_ERRORS = ("invalid_schema", "native_action_error")


def reward_row(episode: dict) -> dict:
    """Use integer milli-rewards so bounds and zero composite credit are exact."""
    calls, errors = episode["global_calls"], episode["errors"]
    if type(calls) is not int or not 0 <= calls <= 96 or calls != len(episode["call_ids"]):
        raise ValueError("physical calls must equal saved calls and lie in [0,96]")
    if any(type(n) is not int or n < 0 for n in errors.values()):
        raise ValueError("nonnegative integer error counts required")
    if sum(errors.values()) > calls:
        raise ValueError("mutually exclusive error counts exceed physical calls")
    score = episode["native_score"]
    if type(score) not in (int, float) or score not in (0, 1):
        raise ValueError("unaltered binary native success label required")
    charged = sum(errors.get(key, 0) for key in CHARGED_ERRORS)
    milli = 1000 * int(score) - charged
    return dict(
        native_score=score,
        physical_calls=calls,
        error_counts=dict(errors),
        charged_errors=charged,
        composite_reward_milli=milli,
        composite_reward=milli / 1000,
    )


def derive_credits(episodes: list[dict], calls: dict) -> tuple[list, dict]:
    """Retain the validated terminal trainer's call coverage; replace advantages only."""
    native_credits = f.rl.loss_math.batch_credits(episodes, calls)
    rows, groups = {}, defaultdict(list)
    native_advantages = {c.episode_id: c.advantage for c in native_credits}
    for episode in episodes:
        eid = episode["episode_id"]
        rows[eid] = dict(
            reward_row(episode),
            task_id=episode["task_id"],
            repeat=episode["repeat"],
            native_advantage=native_advantages[eid],
        )
        groups[episode["task_id"]].append(eid)
    group_stats = {}
    for task_id, ids in groups.items():
        total = sum(rows[eid]["composite_reward_milli"] for eid in ids)
        for eid in ids:
            # r_i - mean(other three rewards) = (4*r_i - sum_j r_j)/3.
            rows[eid]["composite_advantage"] = (
                4 * rows[eid]["composite_reward_milli"] - total
            ) / 3000
        successes = sum(rows[eid]["native_score"] for eid in ids)
        varied = len({rows[eid]["composite_reward_milli"] for eid in ids}) > 1
        group_stats[task_id] = dict(
            native_successes=successes,
            mixed_native=0 < successes < 4,
            mixed_composite=varied,
            all_failure_cost_varying=successes == 0 and varied,
            all_success_cost_varying=successes == 4 and varied,
            episode_ids=ids,
        )
    credits = [
        replace(credit, advantage=rows[credit.episode_id]["composite_advantage"])
        for credit in native_credits
    ]
    return credits, dict(
        schema="textcraft-native-success-minus-error-cost-rewards-20260928-v1",
        reward="native_success - 0.001*(invalid_schema + native_action_error)",
        charged_error_categories=list(CHARGED_ERRORS),
        explicitly_uncharged="rejected_action (bridge-level validation); reported separately",
        native_labels_modified=False,
        episodes=rows,
        groups=group_stats,
        native_successes=sum(row["native_score"] for row in rows.values()),
        mixed_native_groups=sum(g["mixed_native"] for g in group_stats.values()),
        mixed_composite_groups=sum(g["mixed_composite"] for g in group_stats.values()),
        all_failure_cost_varying_groups=sum(
            g["all_failure_cost_varying"] for g in group_stats.values()
        ),
        all_success_cost_varying_groups=sum(
            g["all_success_cost_varying"] for g in group_stats.values()
        ),
        successful_reward_lower_bound=0.904,
        failed_reward_upper_bound=0.0,
        warning="Changed multiobjective reward, not potential shaping or process credit. "
        "Full trajectory-token likelihood is credited, including uncharged actions. "
        "A lower-cost all-failure trajectory can receive positive advantage.",
    )


def collection_ready(directory: Path) -> bool:
    path = directory / "SUMMARY.json"
    if not path.exists():
        return False
    summary = f.read(path)
    return bool(
        summary.get("complete")
        and not summary.get("failure")
        and (directory / "NATIVE-AUDIT.json").exists()
    )


def source_pins(mode: str) -> dict:
    pins = f.source_pins(f.LEGACY / "train.py", mode)
    pins.update(f.source_pins(f.LEGACY / "collect.py", mode))
    for filename in (
        "cost_reward.py",
        "train_cost.py",
        "stage_cost.py",
        "analyze_cost.py",
        "prepare_jobs.py",
    ):
        path = HERE / filename
        pins[str(path)] = f.sha(path)
    for path in (f.HERE / "compare.py", f.LEGACY / "compare.py"):
        pins[str(path)] = f.sha(path)
    return pins


def skip(output: Path, reason: str, *, admission: dict | None = None) -> None:
    """Explicit unavailable branch; never replace it with fabricated zero rewards."""
    record = dict(
        reason=reason,
        scientific_calls=0,
        GPU_loaded=False,
        actual_new_optimizer_steps=0,
        endpoint=None,
        endpoint_usable=False,
        native_success=None,
    )
    f.persist(output / "CONDITIONAL-SKIP.json", record)
    if admission is not None:
        f.persist(output / "ADMISSION.json", admission)
        f.persist(
            output / "SUMMARY.json",
            dict(record, complete=True, failure=None, admission=admission),
        )


def prepare_training(args) -> dict:
    collected = f.read(args.collection / "PLAN.json")
    mode = collected["execution_mode"]
    if (
        mode not in ("raw", "binder")
        or args.collection.resolve() != FRESH_STUDY / mode / "collect-0001"
        or collected.get("schema") != "textcraft-fresh-train-collection-20260928-v1"
        or collected.get("split") != "train"
        or collected.get("phase") != "collect"
        or collected.get("dataset_group") != "train"
        or collected.get("update") != 1
        or collected.get("planned_episodes") != 32
        or collected.get("base_dtype") != "float16"
    ):
        raise ValueError("use only v002 first fresh groupA raw/binder complete32 collection")
    if args.output.resolve() != STUDY / mode / "train-0001" or not 0 < args.hours <= 2:
        raise ValueError("one prospectively named cost branch per interface; cap120minutes")
    dataset = Path(collected["prepared"])
    manifest, _ = f.load_dataset(dataset, "collect")
    curriculum = f.dataset_binding(dataset)
    if (
        collected["manifest_sha256"] != f.GROUP_SHA["train"]
        or collected["tasks_sha256"] != manifest["tasks_sha256"]
        or collected["adapter"] != f.adapter_binding(f.WARM)
    ):
        raise ValueError("actual groupA and identical public warmcp23 behavior required")
    terminal_path = FRESH_STUDY / mode / "train-0001/PLAN.json"
    terminal = f.read(terminal_path)
    if (
        terminal["collection_plan_sha256"] != f.sha(args.collection / "PLAN.json")
        or terminal["adapter"] != collected["adapter"]
        or terminal["update"] != 1
        or terminal["schema"] != "textcraft-fresh-train-rloo-20260928-v1"
    ):
        raise ValueError("terminal first-step control must use the exact same batch and actor")
    plan = dict(
        schema="textcraft-error-cost-first-step-rloo-20260928-v1",
        split="train",
        execution_mode=mode,
        update=1,
        collection=str(args.collection.resolve()),
        collection_plan_sha256=f.sha(args.collection / "PLAN.json"),
        terminal_control_plan=str(terminal_path),
        terminal_control_plan_sha256=f.sha(terminal_path),
        dataset=str(dataset),
        dataset_manifest_sha256=collected["manifest_sha256"],
        dataset_tasks_sha256=collected["tasks_sha256"],
        **curriculum,
        adapter=collected["adapter"],
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
        objective="-sum_i[(composite_reward_i - mean(other three composite rewards)) "
        "* sum_original_emitted_tokens log pi_T0.5]/32",
        reward="native_success - 0.001*(invalid_schema + native_action_error)",
        actual_native_labels="Unchanged binary native_score; separate derived REWARD-TABLE.json",
        action_target="All original saved emitted token IDs including EOS/errors; never "
        "executed binder arguments. Trajectory-token SUM, not per-call or per-token mean.",
        matched_control="Exact same per-interface batch, warmcp23 weights, fresh AdamW, LR, "
        "clip, sampling temperature and one step as terminal-only update1. Nonzero credit "
        "coverage/token dose can differ because composite rewards unflatten native-flat groups.",
        on_policy="Same warm behavior for both objectives. This new objective receives one "
        "step only, never the terminal-updated actor or its optimizer. No four-cycle branch.",
        original_estimator=str(f.LEGACY / "train.py"),
        source_sha256=source_pins(mode),
        diagnostic_dataset=str(f.DATA / "diagnostic"),
        diagnostic_manifest_sha256=f.GROUP_SHA["diagnostic"],
        primary_metric="Paired native success on fixed groupB16 versus terminal-only step1 "
        "and warm. Report calls/errors/failed finishes and get_info exploration separately.",
        retirement="Do not promote lower errors/calls alone. Retire this cost weight/objective "
        "if it lowers cost without success improvement and increases unsuccessful early finish "
        "or reduces information exploration; report inconclusive small-N outcomes honestly.",
        caveat="Changed multiobjective reward, not reward-equivalent potential shaping or "
        "process credit. Strict per-trajectory success ordering does not guarantee policy-level "
        "success preservation. Adam/gradient normalization means a tiny reward coefficient "
        "need not cause a tiny parameter update. No official VAL/HOLDOUT efficacy claim.",
    )
    f.persist(args.output / "PLAN.json", plan)
    return plan
