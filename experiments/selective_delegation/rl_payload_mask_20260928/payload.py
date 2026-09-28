"""Additive public-recipe payload masks and a privately bound legacy trainer."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "rl_resume_20260928"
sys.path.insert(0, str(LEGACY))

import probe_common as p  # noqa: E402
import textcraft_recipe_binder as binder  # noqa: E402

ROOT = p.ROOT
STUDY = ROOT / "textcraft-rl-payload-mask-20260928-001"
ORIGINAL = ROOT / "textcraft-rl-assist-20260928-001/binder"
COLLECTION = ORIGINAL / "collect-0001"
TRAIN = STUDY / "train-0001"
READOUT = STUDY / "readout-0001"
PINS = {
    LEGACY / "train.py": "171dacfa3f1063ac696bd4dca1291c8387ed45573fa8e1d2441391ab902226ff",
    LEGACY / "collect.py": "f1457237edcbf7fa907c56a48069841b744a071e901f11998a86c1e031b69d3b",
    LEGACY / "compare.py": "877162ed42c1a930811662270ff6aef7099ae50d48f27ffff05b30e0d7fed29a",
    Path(p.rl.__file__): "591e06b194a0b1818f897223244d53a3430f18453785d8870662d0a8714db345",
}


def private(path: Path):
    if path in PINS and p.sha(path) != PINS[path]:
        raise ValueError("accepted source changed: " + str(path))
    spec = importlib.util.spec_from_file_location("payload_private_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def persist(path: Path, value: dict):
    if path.exists():
        if p.read(path) != value:
            raise ValueError("immutable prepared record differs: " + str(path))
    else:
        p.c.save(path, value)


def source_pins() -> dict:
    pins = p.source_pins(LEGACY / "train.py", "binder")
    pins.update(p.source_pins(LEGACY / "collect.py", "binder"))
    pins.update({str(path.resolve()): digest for path, digest in PINS.items()})
    pins.update({str(path.resolve()): p.sha(path) for path in HERE.glob("*.py")})
    return pins


def payload_offsets_mask(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    decoder, pos = json.JSONDecoder(), len(text) - len(text.lstrip()) + 1
    span = None
    while True:
        while text[pos].isspace():
            pos += 1
        if text[pos] == "}":
            break
        key, pos = decoder.raw_decode(text, pos)
        while text[pos].isspace():
            pos += 1
        if text[pos] != ":":
            raise ValueError("invalid JSON separator")
        pos += 1
        while text[pos].isspace():
            pos += 1
        start = pos
        _, pos = decoder.raw_decode(text, pos)
        if key == "ingredients":
            if span is not None:
                raise ValueError("duplicate ingredients member")
            span = (start, pos)
        while text[pos].isspace():
            pos += 1
        if text[pos] == "}":
            break
        if text[pos] != ",":
            raise ValueError("invalid JSON separator")
        pos += 1
    if span is None:
        raise ValueError("ingredient payload absent")
    # Strict interior retains braces and every token touching/crossing a value boundary.
    return [int(not (span[0] < a and b < span[1])) for a, b in offsets]


def eligibility(action: dict | None, history: list[dict], assist: dict | None):
    if action is None:
        if assist is not None:
            raise ValueError("invalid schema cannot have a saved binding")
        return False, "invalid_schema"
    executed, reason = binder.bind_observed_recipe_arguments(action, history)
    if assist is not None and (
        assist["requested_action"] != action
        or assist["executed_action"] != executed
        or assist["binding_reason"] != reason
    ):
        raise ValueError("saved binding differs from prior public observations")
    return action["action"] == "craft" and reason == "bound_observed_single_recipe", reason


def call_mask(call: dict, history: list[dict], assist: dict | None, tokenizer) -> dict:
    try:
        action = p.c.bridge.parse_action(call["text"])
    except ValueError:
        action = None
    eligible, reason = eligibility(action, history, assist)
    emitted = call["output_token_ids"]
    mask = [1] * len(emitted)
    if eligible:
        encoded = tokenizer(call["text"], add_special_tokens=False, return_offsets_mapping=True)
        specials = set(tokenizer.all_special_ids)
        if encoded["input_ids"] != [t for t in emitted if t not in specials]:
            raise ValueError("eligible text does not align with exact emitted IDs")
        plain = iter(payload_offsets_mask(call["text"], encoded["offset_mapping"]))
        mask = [1 if token in specials else next(plain) for token in emitted]
    return dict(
        mask=mask,
        eligible=eligible,
        reason=reason,
        original_tokens=len(emitted),
        retained_tokens=sum(mask),
        masked_tokens=mask.count(0),
        original_ids_sha256=hashlib.sha256(json.dumps(emitted).encode()).hexdigest(),
    )


def masked_objective(logps, advantage: float, mask: list[int]):
    import torch

    if logps.numel() != len(mask) or not set(mask) <= {0, 1}:
        raise ValueError("one binary mask bit per original output token required")
    tensor = torch.tensor(mask, dtype=logps.dtype, device=logps.device)
    return -advantage * (logps.flatten() * tensor).sum() / 32


def masked_batch_objective(credits, logps: dict, masks: dict) -> float:
    return (
        -sum(
            c.advantage
            * sum(v * bit for v, bit in zip(logps[c.call_id], masks[c.call_id], strict=True))
            for c in credits
        )
        / 32
    )


def loss_counts(credits, masks: dict) -> dict:
    retained = sum(sum(masks[c.call_id]) for c in credits)
    original = sum(len(masks[c.call_id]) for c in credits)
    return dict(retained_loss_tokens=retained, masked_loss_tokens=original - retained)


SUBSTITUTIONS = {
    "objective = signed_objective(logps, credit.advantage)": (
        "objective = masked_objective(logps, credit.advantage, MASKS[credit.call_id])"
    ),
    "objective_before=value,": "objective_before=value,\n            objective_before_eval="
    "masked_batch_objective(nonzero, before, MASKS),",
    "credited_tokens=count,": (
        "original_replay_tokens=count,\n            **loss_counts(nonzero, MASKS),"
    ),
    "objective_after=-sum(v.advantage * sum(after[v.call_id]) for v in nonzero) / 32,": (
        "objective_after=masked_batch_objective(nonzero, after, MASKS),"
    ),
}


def rewrite_run(source: str) -> str:
    for old, new in SUBSTITUTIONS.items():
        if source.count(old) != 1:
            raise ValueError("masked trainer site must occur exactly once: " + old)
        source = source.replace(old, new)
    return source


def binding_receipt() -> dict:
    original = inspect.getsource(private(LEGACY / "train.py").run)
    changed = rewrite_run(original)
    return dict(
        source=str(LEGACY / "train.py"),
        source_sha256=PINS[LEGACY / "train.py"],
        original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
        transformed_function_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        exact_substitutions=SUBSTITUTIONS,
        unchanged="Full original-token before-generation and train/eval replay, guard count, "
        "optimizer, clipping, adapter/RNG commit, all-token probability diagnostics",
        early_replay="Private action_logps wrapper checks first full saved call and writes "
        "FIRST-REPLAY.json; full original numerical checks remain in the trainer.",
    )


def prepare_masks() -> dict:
    from transformers import AutoTokenizer

    trainer = private(LEGACY / "train.py")
    plan = p.read(COLLECTION / "PLAN.json")
    calls, credits, _, admission = trainer.load_batch(
        COLLECTION, {"collection_plan_sha256": p.sha(COLLECTION / "PLAN.json")}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        p.c.BASE, local_files_only=True, trust_remote_code=False
    )
    records = {}
    for job in plan["jobs"]:
        node = p.read(COLLECTION / "nodes" / (job["episode_id"] + "-n0.json"))
        assists = {a["call_id"]: a for a in node["execution_assists"]}
        history = []
        for cid, row in zip(node["call_ids"], node["public_history"], strict=True):
            action = row.get("action")
            if action is not None and cid not in assists:
                raise ValueError("parsed binder call has no assist receipt")
            records[cid] = call_mask(calls[cid], history, assists.get(cid), tokenizer)
            history.append(row)
    if set(records) != set(calls):
        raise ValueError("mask inventory differs from native collection")
    nonzero = [c for c in credits if c.advantage]
    masked = sum(records[c.call_id]["masked_tokens"] for c in nonzero)
    mass = sum(abs(c.advantage) * records[c.call_id]["masked_tokens"] for c in nonzero)
    total_mass = sum(abs(c.advantage) * records[c.call_id]["original_tokens"] for c in nonzero)
    fraction = mass / total_mass if total_mass else 0
    result = dict(
        schema="observed-binder-payload-mask-20260928-v1",
        collection=str(COLLECTION),
        collection_plan_sha256=p.sha(COLLECTION / "PLAN.json"),
        collection_audit_sha256=p.sha(COLLECTION / "NATIVE-AUDIT.json"),
        eligibility="Only bound_observed_single_recipe, recomputed from prior public history",
        mask="Zero only tokens strictly interior to ingredients JSON value; keep all other tokens",
        mixed_groups=admission["mixed_groups"],
        calls=len(calls),
        eligible_calls=sum(r["eligible"] for r in records.values()),
        nonzero_credit_calls=len(nonzero),
        nonzero_credit_trajectories=len({c.episode_id for c in nonzero}),
        original_nonzero_credit_tokens=sum(records[c.call_id]["original_tokens"] for c in nonzero),
        retained_nonzero_credit_tokens=sum(records[c.call_id]["retained_tokens"] for c in nonzero),
        masked_nonzero_credit_tokens=masked,
        absolute_advantage_masked_tokens=mass,
        absolute_advantage_original_tokens=total_mass,
        absolute_masked_mass_fraction=fraction,
        gate=dict(
            minimum_mixed_groups=1,
            minimum_credited_masked_tokens=100,
            minimum_absolute_mass_fraction=0.01,
        ),
        qualified=admission["mixed_groups"] >= 1 and masked >= 100 and fraction >= 0.01,
        source_sha256=source_pins(),
        records=records,
    )
    persist(STUDY / "MASKS.json", result)
    persist(STUDY / "PRIVATE-ADAPTER.json", binding_receipt())
    return result


def masks_record() -> dict:
    record = p.read(STUDY / "MASKS.json")
    if not record["qualified"] or record["collection_audit_sha256"] != p.sha(
        COLLECTION / "NATIVE-AUDIT.json"
    ):
        raise ValueError("qualified immutable binder batch required")
    for name, digest in record["source_sha256"].items():
        if p.sha(Path(name)) != digest:
            raise ValueError("prepared mask source changed: " + name)
    return record


def prepare_training(args) -> dict:
    if args.output.resolve() != TRAIN or args.collection.resolve() != COLLECTION or args.hours != 1:
        raise ValueError("fixed first binder batch, new training root, and 60-minute cap required")
    masks = masks_record()
    baseline = p.read(ORIGINAL / "train-0001/PLAN.json")
    collection = p.read(COLLECTION / "PLAN.json")
    if baseline["update"] != 1 or baseline["adapter"] != collection["adapter"]:
        raise ValueError("same warm first-update baseline required")
    if p.adapter_binding(p.WARM) != baseline["adapter"]:
        raise ValueError("actual warm adapter changed")
    plan = dict(baseline)
    plan.update(
        schema="textcraft-same-binder-payload-masked-update-20260928-v1",
        budget_seconds=3600,
        masks=str(STUDY / "MASKS.json"),
        masks_sha256=p.sha(STUDY / "MASKS.json"),
        private_adapter_sha256=p.sha(STUDY / "PRIVATE-ADAPTER.json"),
        objective="-sum_i A_i sum_t mask_it log pi_T0.5(original token | original prefix)/32",
        on_policy="Batch sampled from exact warm behavior; masked loss is a biased surrogate, "
        "not unbiased executed-action policy gradient or a correction to full-token REINFORCE",
        token_weighting="No token normalization; exact original advantages and denominator32",
        continuation_gate="Exactly one independently warm-started update; no continuation",
        baseline_full_training=str(ORIGINAL / "train-0001"),
        baseline_full_plan_sha256=p.sha(ORIGINAL / "train-0001/PLAN.json"),
        baseline_endpoint_status="Must authenticate real usable full baseline before GPU dispatch",
        source_sha256=source_pins(),
        masked_loss_counts={k: v for k, v in masks.items() if k.endswith("tokens")},
    )
    persist(TRAIN / "PLAN.json", plan)
    return plan


def endpoint(directory: Path) -> dict | None:
    path = directory / "SUMMARY.json"
    if not path.exists() or not p.read(path).get("endpoint_usable"):
        return None
    summary, plan = p.read(path), p.read(directory / "PLAN.json")
    checkpoint = Path(summary["endpoint"])
    if not checkpoint.is_relative_to(directory) or summary["actual_new_optimizer_steps"] != 1:
        raise ValueError("unexpected endpoint or update count")
    state, commit = p.read(checkpoint / "STATE.json"), p.read(checkpoint / "COMMIT.json")
    if state["step"] != 1 or state["plan_sha256"] != p.sha(directory / "PLAN.json"):
        raise ValueError("endpoint state/PLAN differs")
    if p.sha(checkpoint / "STATE.json") != commit["files"]["STATE.json"]:
        raise ValueError("committed state changed")
    if plan["adapter"] != p.read(COLLECTION / "PLAN.json")["adapter"]:
        raise ValueError("endpoint did not start from same actual warm actor")
    terminals = list(directory.glob("TERMINAL-*.json"))
    if len(terminals) != 1 or not p.read(terminals[0])["endpoint_usable"]:
        raise ValueError("completed usable owner boundary required")
    return dict(
        **p.adapter_binding(checkpoint),
        training_plan_sha256=p.sha(directory / "PLAN.json"),
        summary_sha256=p.sha(path),
        state_sha256=p.sha(checkpoint / "STATE.json"),
    )


def first_replay(output: Path, call: dict, logps, captured: list[float]) -> dict:
    values = logps.detach().flatten().cpu().tolist()
    gaps = [abs(a - b) for a, b in zip(values, captured, strict=True)]
    finite = all(math.isfinite(v) for v in values + gaps)
    maximum, average = max(gaps), sum(gaps) / len(gaps)
    passed = finite and maximum <= 0.25 and average <= 0.025
    result = dict(
        call_id=call["call_id"],
        original_tokens=len(values),
        finite=finite,
        maximum_generation_replay_gap=maximum,
        mean_generation_replay_gap=average,
        passed=passed,
        completed=time.time(),
        GPU_replay=logps.is_cuda,
        scope="First real complete saved-call replay only; full batch check still required",
    )
    p.c.save(output / "FIRST-REPLAY.json", result)
    print(json.dumps(dict(first_saved_call_replay=passed, tokens=len(values))), flush=True)
    if not passed:
        raise ValueError("first saved-call numerical replay failed")
    return result


def bound_trainer(output: Path, masks: dict):
    module = private(LEGACY / "train.py")
    source = rewrite_run(inspect.getsource(module.run))
    module.__dict__.update(
        MASKS=masks,
        masked_objective=masked_objective,
        masked_batch_objective=masked_batch_objective,
        loss_counts=loss_counts,
    )
    exec(compile(source, str(HERE / "train.py"), "exec"), module.__dict__)
    module.__file__, module.prepare = str(HERE / "train.py"), prepare_training
    # A private rl module preserves its own STOP/guard globals; no shared function is edited.
    module.p = SimpleNamespace(**vars(p))
    module.p.rl = private(Path(p.rl.__file__))
    original_replay, original_load = module.p.rl.action_logps, module.load_batch
    captured, seen = {}, False

    def load_batch(directory, plan):
        calls, credits, old, admission = original_load(directory, plan)
        if set(calls) != set(masks):
            raise ValueError("mask/call inventory mismatch")
        record = masks_record()
        for cid, call in calls.items():
            saved = record["records"][cid]
            digest = hashlib.sha256(json.dumps(call["output_token_ids"]).encode()).hexdigest()
            if saved["original_ids_sha256"] != digest or len(masks[cid]) != len(
                call["output_token_ids"]
            ):
                raise ValueError("mask no longer belongs to original emitted IDs")
        captured.update(old)
        admission["mask_sha256"] = p.sha(STUDY / "MASKS.json")
        admission["loss_token_counts"] = loss_counts([c for c in credits if c.advantage], masks)
        return calls, credits, old, admission

    def early_replay(model, call):
        nonlocal seen
        logps = original_replay(model, call)
        if not seen:
            first_replay(output, call, logps, captured[call["call_id"]])
            seen = True
        return logps

    module.load_batch, module.p.rl.action_logps = load_batch, early_replay
    return module
