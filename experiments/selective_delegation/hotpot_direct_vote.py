"""Frozen-extra-sample collector and official HotpotQA vote analyzer.

This intentionally reuses saved direct calls as voter zero.  It only generates
voters one and two; neither generation nor vote construction reads host gold.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import signal
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from random import Random

import eval_planner
import prepare_plan_training as planner
import probe
import score_hotpot

SEED_OFFSETS = (10_000, 20_000)
MAX_NEW_CALLS = 512
MAX_HOURS = 1 / 3
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026092179
STOP = False


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable_in_preflight_environment"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _original_call(source: Path, case_id: str, repeat: int) -> dict:
    episode_id = f"{case_id}-r{repeat}-base-direct"
    episode = _read(source / "episodes" / f"{episode_id}.json")
    if episode.get("episode_id") != episode_id or episode.get("case_id") != case_id:
        raise ValueError("original direct episode identity differs")
    call_id = episode_id + "-final"
    if episode.get("call_ids") != [call_id]:
        raise ValueError("original direct slot has unexpected call sequence")
    call = _read(source / "calls" / f"{call_id}.json")
    if call.get("call_id") != call_id:
        raise ValueError("original direct call identity differs")
    return call


def _validate_original_request(call: dict) -> int:
    request = call.get("request", {})
    sampling = request.get("sampling", {})
    if (
        request.get("condition") != "base"
        or request.get("role") != "final"
        or request.get("adapter_enabled") is not False
    ):
        raise ValueError("original receipt is not a base direct final")
    if (
        sampling.get("temperature") != 0.5
        or sampling.get("top_p") != 1.0
        or sampling.get("top_k") != 0
        or sampling.get("max_new_tokens") != 128
    ):
        raise ValueError("original direct receipt sampling differs")
    seed = request.get("seed")
    if type(seed) is not int:
        raise ValueError("original direct receipt has no actual numeric seed")
    return seed


def build_jobs(
    original: Path, case_ids: list[str], *, cases_by_id: dict[str, dict] | None = None
) -> list[dict]:
    """Bind every extra call to its actual saved direct seed, never inferred seed."""
    original = Path(original)
    plan = _read(original / "PLAN.json")
    if (
        plan.get("mode") != "direct"
        or plan.get("execution") != "direct"
        or plan.get("conditions") != ["base"]
        or plan.get("repeats") != 2
        or plan.get("case_ids") != case_ids
    ):
        raise ValueError("original evaluation is not the exact 128x2 base-direct panel")
    jobs = []
    for case_id in case_ids:
        for repeat in range(2):
            original_call = _original_call(original, case_id, repeat)
            original_seed = _validate_original_request(original_call)
            if cases_by_id is not None:
                request = original_call["request"]
                if request.get("prompt") != eval_planner.direct_prompt(cases_by_id[case_id]):
                    raise ValueError(
                        "original direct receipt prompt differs from frozen public case"
                    )
                if request.get("model") != str(planner.BASE):
                    raise ValueError("original direct receipt base model differs")
            for voter_index, offset in enumerate(SEED_OFFSETS, start=1):
                jobs.append(
                    {
                        "case_id": case_id,
                        "repeat": repeat,
                        "voter_index": voter_index,
                        "original_call_id": original_call["call_id"],
                        "original_seed": original_seed,
                        "seed": original_seed + offset,
                    }
                )
    if len(jobs) > MAX_NEW_CALLS:
        raise ValueError("new direct-vote call cap exceeded")
    return jobs


def vote(voters: list[dict]) -> dict:
    """Vote observed strict answers; malformed output is a protocol ballot, never a fallback."""
    if [row.get("index") for row in voters] != [0, 1, 2]:
        raise ValueError("vote requires original plus exactly two ordered voters")
    parsed, invalid, unavailable = [], 0, 0
    for row in voters:
        if not row.get("available", False):
            unavailable += 1
            continue
        try:
            answer = score_hotpot.parse_answer(row.get("text"))
        except (TypeError, ValueError, json.JSONDecodeError):
            invalid += 1
            parsed.append({"index": row["index"], "raw": None, "normalized": "__invalid__"})
            continue
        parsed.append(
            {
                "index": row["index"],
                "raw": answer,
                "normalized": score_hotpot.normalize_answer(answer),
            }
        )
    if unavailable:
        return {
            "status": "unavailable",
            "valid_voters": len(parsed) - invalid,
            "invalid_voters": invalid,
            "unavailable_voters": unavailable,
            "chosen_index": None,
            "chosen_raw_answer": None,
            "chosen_normalized_answer": None,
            "tied": False,
        }
    counts = Counter(row["normalized"] for row in parsed)
    highest = max(counts.values())
    leaders = {answer for answer, count in counts.items() if count == highest}
    # The saved original is index zero, so it deterministically wins any tie it joins.
    chosen = min(
        (row for row in parsed if row["normalized"] in leaders), key=lambda row: row["index"]
    )
    protocol = chosen["normalized"] == "__invalid__"
    return {
        "status": "protocol_zero" if protocol else "scored",
        "valid_voters": len(parsed) - invalid,
        "invalid_voters": invalid,
        "unavailable_voters": unavailable,
        "chosen_index": chosen["index"],
        "chosen_raw_answer": None if protocol else chosen["raw"],
        "chosen_normalized_answer": None if protocol else chosen["normalized"],
        "tied": len(leaders) > 1,
    }


def _call_path(output: Path, job: dict) -> Path:
    identity = f"{job['case_id']}-r{job['repeat']}-base-direct-vote{job['voter_index']}-final"
    return output / "calls" / f"{identity}.json"


def collect_one(client, case: dict, job: dict) -> dict:
    identity = _call_path(client.output, job).stem
    call = client.call(
        identity,
        eval_planner.direct_prompt(case),
        "base",
        "final",
        job["seed"],
        max_new_tokens=128,
    )
    episode = {
        "episode_id": identity.removesuffix("-final"),
        "case_id": job["case_id"],
        "repeat": job["repeat"],
        "voter_index": job["voter_index"],
        "original_call_id": job["original_call_id"],
        "original_seed": job["original_seed"],
        "seed": job["seed"],
        "call_ids": [identity],
        "available": call.get("available", False),
        "status": "returned" if call.get("available", False) else "unavailable_final",
        "ended": time.time(),
    }
    path = client.output / "episodes" / f"{episode['episode_id']}.json"
    if path.exists():
        old = _read(path)
        for field in ("case_id", "repeat", "voter_index", "original_call_id", "seed", "call_ids"):
            if old.get(field) != episode[field]:
                raise ValueError("resumed extra-voter episode changed")
        return old
    probe.runtime.save(path, episode)
    return episode


def _plan(cases_path: Path, original: Path, jobs: list[dict], adapter: Path) -> dict:
    original_plan = _read(original / "PLAN.json")
    binding = eval_planner.adapter_identity(adapter)
    return {
        "schema": "hotpot-direct-three-vote-collector-v1",
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": score_hotpot.sha256(cases_path),
        "original_evaluation": str(original.resolve()),
        "original_plan_sha256": score_hotpot.sha256(original / "PLAN.json"),
        "original_calls_sha256": score_hotpot.tree_sha256(original / "calls"),
        "case_ids": original_plan["case_ids"],
        "repeats": 2,
        "voters_per_slot": 3,
        "new_voter_indices": [1, 2],
        "seed_offsets": list(SEED_OFFSETS),
        "planned_original_slots": len(original_plan["case_ids"]) * 2,
        "planned_new_calls": len(jobs),
        "maximum_new_calls": MAX_NEW_CALLS,
        "model": original_plan["model"],
        "adapter": str(adapter.resolve()),
        "adapter_files_sha256": binding,
        "model_manifest_sha256": score_hotpot.sha256(planner.BASE / "local-research-manifest.json"),
        "sampling": {"temperature": 0.5, "top_p": 1.0, "top_k": 0, "max_new_tokens": 128},
        "policy": "Base direct only; adapter is loaded for native evaluator parity but disabled on "
        "every call. Original calls are reused as voter zero. Invalid output is an explicit "
        "protocol ballot; any unavailable voter makes the planned three-vote policy unobserved. "
        "Ties choose the lowest voter index. No answer fallback or gold access.",
        "source_sha256": score_hotpot.sha256(Path(__file__)),
        "dependencies": {
            str(path): score_hotpot.sha256(path)
            for path in (
                Path(eval_planner.__file__),
                Path(score_hotpot.__file__),
                Path(probe.__file__),
            )
        },
        "budget_seconds": int(MAX_HOURS * 3600),
        "environment": {
            "python": sys.version,
            **{name: _package_version(name) for name in ("torch", "transformers", "peft")},
        },
    }


def summarize(output: Path, plan: dict) -> dict:
    calls = [_read(path) for path in (output / "calls").glob("*.json")]
    episodes = [_read(path) for path in (output / "episodes").glob("*.json")]
    return {
        "planned_new_calls": plan["planned_new_calls"],
        "recorded_new_calls": len(calls),
        "missing_new_calls": plan["planned_new_calls"] - len(calls),
        "returned_new_calls": sum(row.get("available", False) for row in calls),
        "unavailable_new_calls": sum(not row.get("available", False) for row in calls),
        "recorded_episodes": len(episodes),
        "physical_cost": eval_planner.cost(calls),
        "lower_bound_note": "Missing planned calls are unavailable, not votes or zero-cost calls.",
        "updated": time.time(),
    }


def _evaluation_outcomes(source: Path, case_ids: list[str], cases: dict[str, dict]) -> dict:
    """Read one completed saved arm under its planned denominator, without score fallbacks."""
    plan = _read(source / "PLAN.json")
    if plan.get("case_ids") != case_ids or plan.get("repeats") != 2:
        raise ValueError("comparison arm is not the identical 128x2 Hotpot panel")
    outcomes = {}
    for case_id in case_ids:
        for repeat in range(2):
            suffix = "-base-direct" if plan.get("mode") == "direct" else "-sft-isolated"
            episode_id = f"{case_id}-r{repeat}{suffix}"
            path = source / "episodes" / f"{episode_id}.json"
            if not path.exists():
                outcomes[case_id, repeat] = (0.0, 0.0)
                continue
            episode = _read(path)
            call_ids = episode.get("call_ids", [])
            final_id = call_ids[-1] if call_ids and call_ids[-1].endswith("-final") else None
            call_path = source / "calls" / f"{final_id}.json" if final_id else None
            if call_path is None or not call_path.exists():
                outcomes[case_id, repeat] = (0.0, 0.0)
                continue
            call = _read(call_path)
            try:
                answer = score_hotpot.parse_answer(call["text"]) if call.get("available") else None
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                answer = None
            outcomes[case_id, repeat] = (
                score_hotpot.official_score(answer, cases[case_id]["answer"])
                if answer is not None
                else (0.0, 0.0)
            )
    return outcomes


def _paired_parent_bootstrap(left: dict, right: dict, case_ids: list[str]) -> dict:
    """Parent-clustered comparison: average its two repeats before resampling 128 parents."""
    rng = Random(BOOTSTRAP_SEED)
    result = {}
    for metric_index, metric in enumerate(("em", "f1")):
        parent_deltas = [
            sum(
                left[case_id, repeat][metric_index] - right[case_id, repeat][metric_index]
                for repeat in range(2)
            )
            / 2
            for case_id in case_ids
        ]
        draws = sorted(
            sum(parent_deltas[rng.randrange(len(parent_deltas))] for _ in parent_deltas)
            / len(parent_deltas)
            for _ in range(BOOTSTRAP_DRAWS)
        )
        result[metric] = {
            "estimate": sum(parent_deltas) / len(parent_deltas),
            "ci95": [draws[int(0.025 * BOOTSTRAP_DRAWS)], draws[int(0.975 * BOOTSTRAP_DRAWS)]],
        }
    return result


def analyze(
    cases_path: Path, original: Path, extras: Path, *, planner_output: Path | None = None
) -> dict:
    """Grade saved vote choices using official HotpotQA scoring, all 256 slots."""
    cases = score_hotpot._load_cases(cases_path)
    plan = _read(extras / "PLAN.json")
    jobs = build_jobs(original, plan["case_ids"], cases_by_id=cases)
    by_job = {(j["case_id"], j["repeat"], j["voter_index"]): j for j in jobs}
    slots, counts, em, f1, voted = [], Counter(), 0.0, 0.0, {}
    for case_id in plan["case_ids"]:
        for repeat in range(2):
            original_call = _original_call(original, case_id, repeat)
            voters = [
                {
                    "index": 0,
                    "available": original_call.get("available", False),
                    "text": original_call.get("text"),
                }
            ]
            for index in (1, 2):
                job = by_job[case_id, repeat, index]
                path = _call_path(extras, job)
                if path.exists():
                    row = _read(path)
                    voters.append(
                        {
                            "index": index,
                            "available": row.get("available", False),
                            "text": row.get("text"),
                        }
                    )
                else:
                    voters.append({"index": index, "available": False, "text": None})
            result = vote(voters)
            if result["status"] == "scored":
                slot_em, slot_f1 = score_hotpot.official_score(
                    result["chosen_raw_answer"], cases[case_id]["answer"]
                )
            else:
                slot_em = slot_f1 = 0.0
            result.update(case_id=case_id, repeat=repeat, em=slot_em, f1=slot_f1)
            slots.append(result)
            voted[case_id, repeat] = (slot_em, slot_f1)
            counts[result["status"]] += 1
            counts["invalid_voters"] += result["invalid_voters"]
            counts["unavailable_voters"] += result["unavailable_voters"]
            em += slot_em
            f1 += slot_f1
    denominator = len(plan["case_ids"]) * 2
    original_outcomes = _evaluation_outcomes(original, plan["case_ids"], cases)
    comparisons = {
        "vote_minus_original_direct": _paired_parent_bootstrap(
            voted, original_outcomes, plan["case_ids"]
        )
    }
    cost = {
        "reused_original_direct": eval_planner.cost(
            [_read(path) for path in (original / "calls").glob("*.json")]
        ),
        "new_vote_calls": eval_planner.cost(
            [_read(path) for path in (extras / "calls").glob("*.json")]
        ),
    }
    cost["hypothetical_three_vote_total"] = {
        key: cost["reused_original_direct"].get(key, 0) + cost["new_vote_calls"].get(key, 0)
        for key in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "unknown_usage_calls",
            "failed_calls",
        )
    }
    if planner_output is not None:
        planner_output = Path(planner_output)
        planner_outcomes = _evaluation_outcomes(planner_output, plan["case_ids"], cases)
        comparisons["vote_minus_planner"] = _paired_parent_bootstrap(
            voted, planner_outcomes, plan["case_ids"]
        )
        cost["planner_physical"] = eval_planner.cost(
            [_read(path) for path in (planner_output / "calls").glob("*.json")]
        )
    return {
        "schema": "hotpot-direct-three-vote-analysis-v1",
        "planned_slots": denominator,
        "slots": slots,
        "counts": dict(counts),
        "official_hotpotqa": {"em": em / denominator, "f1": f1 / denominator},
        "paired_parent_bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "clusters": len(plan["case_ids"]),
            "comparisons": comparisons,
        },
        "cost": cost,
        "denominator_note": "All 128 parents x two original repeats are retained. Any missing "
        "voter makes the three-sample policy unavailable and scores zero; malformed output is "
        "a separate protocol ballot and can win a vote, also scoring zero.",
        "sources": {
            "cases_sha256": score_hotpot.sha256(cases_path),
            "original_plan_sha256": score_hotpot.sha256(original / "PLAN.json"),
            "original_calls_sha256": score_hotpot.tree_sha256(original / "calls"),
            "extra_plan_sha256": score_hotpot.sha256(extras / "PLAN.json"),
            "extra_calls_sha256": score_hotpot.tree_sha256(extras / "calls"),
            "evaluator_sha256": score_hotpot.sha256(score_hotpot.EVALUATOR),
        },
        "code_sha256": score_hotpot.sha256(Path(__file__)),
    }


def run(args):
    global STOP
    STOP = False
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output, original, adapter = (
        args.output.resolve(),
        args.original.resolve(),
        args.adapter.resolve(),
    )
    if not 0 < args.hours <= MAX_HOURS:
        raise ValueError("direct-vote collection cap is at most 20 minutes")
    if list(output.glob("OWNER-*.json")) or any(
        (output / name).exists() and any((output / name).glob("*.json"))
        for name in ("calls", "episodes")
    ):
        raise ValueError("existing owner or receipts require explicit review; no implicit resume")
    cases = [json.loads(line) for line in args.cases.open() if line.strip()]
    by_id = {case["id"]: case for case in cases}
    jobs = build_jobs(original, [case["id"] for case in cases], cases_by_id=by_id)
    plan = _plan(args.cases, original, jobs, adapter)
    plan["budget_seconds"] = int(args.hours * 3600)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "PLAN.json"
    if plan_path.exists() and _read(plan_path) != plan:
        raise ValueError("immutable direct-vote plan differs")
    if not plan_path.exists():
        probe.runtime.save(plan_path, plan)
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + args.hours * 3600, lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient collection/allocation budget")
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started, model, failure = uuid.uuid4().hex[:12], time.time(), None, None
    try:
        probe.runtime.save(
            output / f"OWNER-{invocation}.json",
            {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "started": started,
                "deadline": deadline,
                "source": str(Path(__file__).resolve()),
            },
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("main must assign exactly one GPU")
        tokenizer = AutoTokenizer.from_pretrained(
            planner.BASE, local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            planner.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        binding = eval_planner.adapter_identity(adapter)
        model = PeftModel.from_pretrained(
            base, adapter, is_trainable=False, autocast_adapter_dtype=True
        )
        model.eval()
        model.config.use_cache = True
        client = eval_planner.HFClient(
            model, tokenizer, output, deadline, binding["adapter_model.safetensors"]
        )
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(sys.modules[__name__], "STOP", True))
        for job in jobs:
            if STOP or time.time() >= deadline - 5 or (output / "STOP").exists():
                break
            collect_one(client, by_id[job["case_id"]], job)
            probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
            if (client.returned == 0 and client.failed) or client.consecutive_failures >= 2:
                raise RuntimeError("scientific response check failed; stop faulty owner")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals():
            torch.cuda.empty_cache()
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": STOP,
                "ended": time.time(),
                "elapsed_seconds": time.time() - started,
                "deadline": deadline,
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1 / 3)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--analysis-output", type=Path)
    parser.add_argument("--planner-output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    if arguments.analyze:
        if arguments.analysis_output is None or arguments.analysis_output.exists():
            raise ValueError("new --analysis-output is required for analysis")
        report = analyze(
            arguments.cases,
            arguments.original,
            arguments.output,
            planner_output=arguments.planner_output,
        )
        arguments.analysis_output.mkdir(parents=True)
        (arguments.analysis_output / "REPORT.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    elif arguments.preflight:
        if not 0 < arguments.hours <= MAX_HOURS:
            raise ValueError("direct-vote collection cap is at most 20 minutes")
        from transformers import AutoTokenizer

        cases = [json.loads(line) for line in arguments.cases.open() if line.strip()]
        jobs = build_jobs(
            arguments.original,
            [case["id"] for case in cases],
            cases_by_id={case["id"]: case for case in cases},
        )
        plan = _plan(arguments.cases, arguments.original, jobs, arguments.adapter)
        plan["budget_seconds"] = int(arguments.hours * 3600)
        arguments.output.mkdir(parents=True, exist_ok=True)
        plan_path = arguments.output / "PLAN.json"
        if plan_path.exists() and _read(plan_path) != plan:
            raise ValueError("immutable direct-vote plan differs")
        if not plan_path.exists():
            probe.runtime.save(plan_path, plan)
        tokenizer = AutoTokenizer.from_pretrained(
            planner.BASE, local_files_only=True, trust_remote_code=False
        )
        probe.runtime.save(
            arguments.output / "PREFLIGHT.json",
            {
                "prepared": time.time(),
                "tokenizer_class": type(tokenizer).__name__,
                "gpu_loaded": False,
            },
        )
    else:
        run(arguments)
