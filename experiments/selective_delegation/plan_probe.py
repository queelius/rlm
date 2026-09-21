"""Matched fresh model-plan/reference-question helpers; annotation-privileged diagnostic."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import answer_format
import probe

campaign, runtime = probe.campaign, probe.runtime
ARMS = ("model_plan", "reference_questions")
SEED = probe.SEED + 10000
DEPENDENCY_INSTRUCTION = (
    "The questions are numbered by their order in the list. #1 means your own inferred "
    "answer to question 1, not an externally supplied answer. "
)


def read(path):
    return json.loads(Path(path).read_text())


def helper_prompt(case, state, arm):
    if arm not in ARMS:
        raise ValueError("unknown plan arm")
    helper_state = dict(state)
    if arm == "reference_questions":
        questions = [step["question"] for step in case["metadata"]["question_decomposition"]]
        if len(questions) != 2 or not all(isinstance(q, str) and q.strip() for q in questions):
            raise ValueError("reference diagnostic requires exactly two questions")
        helper_state["subquestions"] = questions
    return DEPENDENCY_INSTRUCTION + probe.helper_prompt(case, helper_state, "decompose")


def prepare_sources(source, cases_path):
    source = Path(source).resolve()
    plan = read(source / "PLAN.json")
    if campaign.sha(cases_path) != plan["cases_sha256"]:
        raise ValueError("source cases checksum changed")
    if plan["model"] != campaign.MODELS["4b"]:
        raise ValueError("diagnostic requires frozen 4B helper model")
    with Path(cases_path).open() as stream:
        cases = {c["id"]: c for c in (json.loads(line) for line in stream)}
    chosen, sources, exclusions = [], {}, Counter()
    for cid in plan["case_ids"]:
        case = cases[cid]
        if case["split"] != "train":
            raise ValueError("plan diagnostic is restricted to training parents")
        if case["metadata"]["hops"] != 2:
            exclusions["not_two_hop"] += 1
            continue
        receipt_path = source / "checkpoints" / (cid + ".json")
        if not receipt_path.exists():
            exclusions["checkpoint_missing"] += 1
            continue
        receipt = read(receipt_path)
        if receipt.get("error") or receipt.get("state") is None:
            exclusions["checkpoint_failed"] += 1
            continue
        if receipt["case_id"] != cid or receipt["call_id"] != cid + "-checkpoint":
            raise ValueError("checkpoint identity mismatch")
        call_path = source / "calls" / (receipt["call_id"] + ".json")
        record = read(call_path)
        if not record["available"]:
            raise ValueError("checkpoint availability mismatch")
        if runtime.digest(record["request"]) != record["request_digest"]:
            raise ValueError("checkpoint native request changed")
        if record["request"]["model"] != plan["model"]:
            raise ValueError("checkpoint model mismatch")
        if record["prompt"] != probe.initial_prompt(case):
            raise ValueError("checkpoint prompt differs from public source context")
        state = probe.checkpoint(record["text"])
        if state != receipt["state"]:
            raise ValueError("checkpoint state differs from source response")
        sources[cid] = {
            "state": state,
            "record": record,
            "request_digest": record["request_digest"],
            "source_hashes": {str(p): campaign.sha(p) for p in (receipt_path, call_path)},
        }
        chosen.append(case)
    return chosen, sources, dict(exclusions), plan


def identity(case, arm, repeat):
    return f"{case['id']}-r{repeat}-{arm}"


def collect(client, case, source, arm, repeat):
    key = identity(case, arm, repeat)
    path = client.output / "episodes" / (key + ".json")
    if path.exists():
        return read(path)
    seed = SEED + repeat * 100
    records = []
    result = {
        "episode_id": key,
        "case_id": case["id"],
        "split": case["split"],
        "arm": arm,
        "repeat": repeat,
        "seed": seed,
        "available": False,
        "valid": False,
        "correct": False,
        "f1": 0.0,
        "parsed": None,
        "annotation_privileged": arm == "reference_questions",
        "original_checkpoint_digest": runtime.digest(source["state"]),
        "reused_checkpoint_call_id": source["record"]["call_id"],
        "source_checkpoint_request_digest": source["request_digest"],
        "source_hashes": source["source_hashes"],
        "started": time.time(),
    }
    try:
        helper = client.call(
            key + "-helper",
            helper_prompt(case, source["state"], arm),
            seed + 1,
            384,
            temperature=0.5,
        )
        records.append(helper)
        if not helper["available"]:
            raise RuntimeError("helper_transport_failure")
        final = client.call(
            key + "-final",
            probe.final_prompt(case, source["state"], helper["text"], answer_style="span"),
            seed,
            128,
            temperature=0.5,
        )
        records.append(final)
        if not final["available"]:
            raise RuntimeError("final_transport_failure")
        result.update(probe.grade(final["text"], case), available=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result.update(
        new_call_ids=[r["call_id"] for r in records],
        new_physical_cost=answer_format.measured_cost(records),
        hypothetical_deployed_cost=answer_format.measured_cost([source["record"], *records]),
        ended=time.time(),
    )
    runtime.save(path, result)
    return result


def summarize(output, plan):
    rows = [read(p) for p in (output / "episodes").glob("*.json")]
    calls = [read(p) for p in (output / "calls").glob("*.json")]
    starts = [read(p) for p in (output / "starts").glob("*.json")]
    by_key = {(r["case_id"], r["repeat"], r["arm"]): r for r in rows}
    pairs = []
    for cid in plan["case_ids"]:
        for repeat in range(plan["repeats"]):
            model = by_key.get((cid, repeat, "model_plan"))
            reference = by_key.get((cid, repeat, "reference_questions"))
            if model and reference:
                pairs.append(
                    {
                        "case_id": cid,
                        "repeat": repeat,
                        "both_available": model["available"] and reference["available"],
                        "model_correct": model["correct"],
                        "reference_correct": reference["correct"],
                        "model_f1": model["f1"],
                        "reference_f1": reference["f1"],
                        "model_answer": model["parsed"],
                        "reference_answer": reference["parsed"],
                        "model_episode_id": model["episode_id"],
                        "reference_episode_id": reference["episode_id"],
                    }
                )
    available_pairs = [p for p in pairs if p["both_available"]]
    groups = {}
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        groups[arm] = {
            "planned": len(plan["case_ids"]) * plan["repeats"],
            "episodes": len(subset),
            "available": sum(r["available"] for r in subset),
            "valid": sum(r["valid"] for r in subset),
            "correct": sum(r["correct"] for r in subset),
            "f1_sum": sum(r["f1"] for r in subset),
        }
    return {
        "groups": groups,
        "pairs": pairs,
        "paired_available": len(available_pairs),
        "paired_wins": sum(
            not p["model_correct"] and p["reference_correct"] for p in available_pairs
        ),
        "paired_losses": sum(
            p["model_correct"] and not p["reference_correct"] for p in available_pairs
        ),
        "source_exclusions": plan["source_exclusions"],
        "new_physical_cost": answer_format.measured_cost(calls),
        "physical_start_receipts": len(starts),
        "unresolved_start_receipts": sum(
            not (output / "calls" / (s["call_id"] + ".json")).exists() for s in starts
        ),
        "extra_start_attempts": max(0, len(starts) - len({s["call_id"] for s in starts})),
        "interpretation": "Reference-question arm uses privileged annotations, not a deployable "
        "policy or learned RLM improvement. Both arms have fresh helpers/finals and the same "
        "original checkpoint in the final prompt. Repeats share that fixed checkpoint. "
        "New call receipts exclude reused checkpoint cost; "
        "hypothetical deployment includes it once. "
        "Unknown or repeated-start usage remains unmeasured, not free.",
        "updated": time.time(),
    }


def run(args):
    import psutil
    from transformers import AutoTokenizer

    source, output = args.source_output.resolve(), args.output.resolve()
    if source == output or not list(source.glob("TERMINAL-*.json")):
        raise ValueError("use a separate output after source owner termination")
    cases, sources, exclusions, source_plan = prepare_sources(source, args.cases)
    if not cases or args.repeats < 1:
        raise ValueError("no eligible parents or repeats")
    plan = {
        "source_output": str(source),
        "source_plan_sha256": campaign.sha(source / "PLAN.json"),
        "cases_sha256": campaign.sha(args.cases),
        "case_ids": [c["id"] for c in cases],
        "source_checkpoints": {
            cid: {k: v for k, v in s.items() if k in ("source_hashes", "request_digest")}
            for cid, s in sources.items()
        },
        "source_exclusions": exclusions,
        "model": source_plan["model"],
        "seed": SEED,
        "repeats": args.repeats,
        "arms": list(ARMS),
        "temperature": 0.5,
        "caps": {"helper": 384, "final": 128},
        "answer_style": "span",
        "annotation_privileged": True,
        "dependency_instruction": DEPENDENCY_INSTRUCTION,
        "dependencies": {
            str(p): campaign.sha(p)
            for p in (
                Path(__file__),
                Path(probe.__file__),
                Path(answer_format.__file__),
                probe.BREADTH / "runner.py",
                probe.BREADTH / "runner_v2.py",
                probe.BREADTH / "campaign.py",
                probe.MUSIQUE / "metrics/answer.py",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PLAN.json").exists():
        if read(output / "PLAN.json") != plan:
            raise ValueError("immutable resume plan mismatch")
    else:
        runtime.save(output / "PLAN.json", plan)
    jobs = []
    for case in cases:
        for repeat in range(args.repeats):
            order = ARMS if (int(case["id"], 16) + repeat) % 2 == 0 else ARMS[::-1]
            for arm in order:
                if not (output / "episodes" / (identity(case, arm, repeat) + ".json")).exists():
                    jobs.append((case, sources[case["id"]], arm, repeat))
    if not jobs:
        campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
        return
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + args.hours * 3600, lease - 600)
    if deadline < time.time() + 300:
        raise ValueError("insufficient allocation time")
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    lock = (campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation = uuid.uuid4().hex[:12]
    process = client = None
    failure = None
    try:
        runtime.save(
            output / f"OWNER-{invocation}.json",
            {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "deadline": deadline,
                "allocation_end": lease,
                "started": time.time(),
                "source": str(Path(__file__).resolve()),
            },
        )
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: campaign.STOP.set())
        os.environ["BREADTH_LOCAL_API_KEY"] = uuid.uuid4().hex
        campaign.snapshot(output / "STATUS.json", {"state": "starting", "updated": time.time()})
        process, log, helper = campaign.service(
            plan["model"], output / "services" / invocation, deadline
        )
        client = runtime.Client(
            tokenizer, "http://127.0.0.1:18731", plan["model"], output, deadline
        )
        case, saved, arm, repeat = jobs[0]
        first = client.call(
            identity(case, arm, repeat) + "-helper",
            helper_prompt(case, saved["state"], arm),
            SEED + repeat * 100 + 1,
            384,
            temperature=0.5,
        )
        if not first["available"]:
            raise RuntimeError("first scientific helper request failed")
        campaign.snapshot(
            output / "STATUS.json",
            {
                "state": "collecting",
                "returned": client.returned,
                "transport_errors": client.errors,
                "updated": time.time(),
                "deadline": deadline,
            },
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            for position in range(0, len(jobs), 4):
                if campaign.STOP.is_set() or client.stop.is_set() or time.time() >= deadline - 120:
                    break
                futures = [
                    pool.submit(collect, client, *job) for job in jobs[position : position + 4]
                ]
                for future in futures:
                    future.result()
                campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
                campaign.snapshot(
                    output / "STATUS.json",
                    {
                        "state": "collecting",
                        "returned": client.returned,
                        "transport_errors": client.errors,
                        "last_return": client.last_return,
                        "updated": time.time(),
                        "deadline": deadline,
                    },
                )
                if (output / "STOP").exists():
                    campaign.STOP.set()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if process is not None:
                if client is not None:
                    client.stop.set()
                helper._stop(process)
                log.close()
                runtime.save(
                    output / f"RELEASED-{invocation}.json",
                    {"pid": process.pid, "ended": time.time()},
                )
            campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
            campaign.snapshot(
                output / "STATUS.json",
                {
                    "state": "failed" if failure else "finished_or_capped",
                    "failure": failure,
                    "updated": time.time(),
                },
            )
            runtime.save(
                output / f"TERMINAL-{invocation}.json",
                {
                    "ended": time.time(),
                    "failure": failure,
                    "stop_requested": campaign.STOP.is_set(),
                    "deadline_reached": time.time() >= deadline - 120,
                },
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hours", type=float, default=2)
    run(parser.parse_args())
