"""Replay only final answers with an explicit short-answer contract; main owns GPU."""

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

import probe

campaign, runtime = probe.campaign, probe.runtime


def read(path):
    return json.loads(Path(path).read_text())


def measured_cost(records):
    return {
        "calls": len(records),
        "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in records),
        "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in records),
        "unknown_usage_calls": sum(
            not r.get("available")
            or any(
                type(r.get("usage", {}).get(k)) is not int
                for k in ("prompt_tokens", "completion_tokens")
            )
            for r in records
        ),
    }


def prepare_episode(path, case):
    path = Path(path)
    episode = read(path)
    if not episode["available"]:
        raise ValueError("source episode unavailable")
    if episode["case_id"] != case["id"] or episode["arm"] not in probe.ARMS:
        raise ValueError("source episode identity mismatch")
    expected = [case["id"] + "-checkpoint"]
    if episode["arm"] != "finish":
        expected.append(episode["episode_id"] + "-helper")
    expected.append(episode["episode_id"] + "-final")
    if episode["call_ids"] != expected:
        raise ValueError("source call roles mismatch")
    records, hashes = [], {str(path): campaign.sha(path)}
    for cid in expected:
        call_path = path.parent.parent / "calls" / (cid + ".json")
        record = read(call_path)
        if not record["available"] or record["call_id"] != cid:
            raise ValueError("source call unavailable or mismatched")
        if runtime.digest(record["request"]) != record["request_digest"]:
            raise ValueError("source request digest mismatch")
        records.append(record)
        hashes[str(call_path)] = campaign.sha(call_path)
    state = probe.checkpoint(records[0]["text"])
    if runtime.digest(state) != episode["checkpoint_digest"]:
        raise ValueError("source checkpoint digest mismatch")
    report = records[1]["text"] if episode["arm"] != "finish" else None
    original_prompt = probe.final_prompt(case, state, report, answer_style="original")
    if original_prompt != records[-1]["prompt"]:
        raise ValueError("source final prompt differs from reconstructed evidence")
    sampling = records[-1]["request"]["sampling_params"]
    if sampling["seed"] != episode["seed"] or sampling["max_tokens"] != 128:
        raise ValueError("source final sampling contract changed")
    return {
        "episode": episode,
        "records": records,
        "report": report,
        "source_hashes": hashes,
        "source_request_digests": {r["call_id"]: r["request_digest"] for r in records},
        "reused_call_ids": expected[:-1],
        "prompt": probe.final_prompt(case, state, report, answer_style="span"),
        "seed": sampling["seed"],
        "temperature": sampling["temperature"],
        "max_tokens": 128,
    }


def pair_result(job, case, new):
    episode = job["episode"]
    old_grade = probe.grade(job["records"][-1]["text"], case)
    new_grade = (
        probe.grade(new["text"], case)
        if new["available"]
        else {"valid": False, "correct": False, "f1": 0.0, "parsed": None}
    )
    return {
        "episode_id": episode["episode_id"],
        "case_id": case["id"],
        "arm": episode["arm"],
        "repeat": episode["repeat"],
        "seed": job["seed"],
        "available": new["available"],
        "old": old_grade,
        "new": new_grade,
        "old_answer": old_grade["parsed"],
        "new_answer": new_grade["parsed"],
        "reused_call_ids": job["reused_call_ids"],
        "new_call_id": new["call_id"],
        "source_hashes": job["source_hashes"],
        "source_request_digests": job["source_request_digests"],
        "new_physical_cost": measured_cost([new]),
        "hypothetical_deployed_cost": measured_cost(job["records"][:-1] + [new]),
        "error": new.get("error"),
        "ended": time.time(),
    }


def collect(client, job, case):
    identity = job["episode"]["episode_id"]
    path = client.output / "pairs" / (identity + ".json")
    if path.exists():
        return read(path)
    new = client.call(
        identity + "-short-final",
        job["prompt"],
        job["seed"],
        job["max_tokens"],
        temperature=job["temperature"],
    )
    result = pair_result(job, case, new)
    runtime.save(path, result)
    return result


def summary(output, plan):
    pairs = [read(path) for path in (output / "pairs").glob("*.json")]
    calls = [read(path) for path in (output / "calls").glob("*.json")]
    starts = list((output / "starts").glob("*.json"))
    groups = {}
    for arm in probe.ARMS:
        rows = [r for r in pairs if r["arm"] == arm]
        available = [r for r in rows if r["available"]]
        groups[arm] = {
            "pairs": len(rows),
            "available": len(available),
            "old_correct": sum(r["old"]["correct"] for r in rows),
            "new_correct": sum(r["new"]["correct"] for r in rows),
            "paired_wins": sum(not r["old"]["correct"] and r["new"]["correct"] for r in available),
            "paired_losses": sum(
                r["old"]["correct"] and not r["new"]["correct"] for r in available
            ),
            "old_f1_sum": sum(r["old"]["f1"] for r in rows),
            "new_f1_sum": sum(r["new"]["f1"] for r in rows),
        }
    return {
        "groups": groups,
        "planned_new_finals": len(plan["episodes"]),
        "source_exclusions": plan["source_exclusions"],
        "completed_pairs": len(pairs),
        "new_physical_cost": measured_cost(calls),
        "physical_start_receipts": len(starts),
        "starts_without_return_receipt": sum(
            not (output / "calls" / (read(p)["call_id"] + ".json")).exists() for p in starts
        ),
        "extra_start_attempts": max(0, len(starts) - len({read(p)["call_id"] for p in starts})),
        "cost_note": "New call receipts only; start-only/repeated attempts have unmeasured usage. "
        "Per-pair hypothetical deployment includes reused checkpoint/helper and new final once. "
        "Do not sum hypothetical deployment across pairs as physical work.",
        "updated": time.time(),
    }


def run(args):
    import psutil
    from transformers import AutoTokenizer

    source, output = args.source_output.resolve(), args.output.resolve()
    if source == output:
        raise ValueError("followup output must differ from source")
    if not list(source.glob("TERMINAL-*.json")):
        raise ValueError("source owner has not terminated; wait for pilot completion")
    source_plan = read(source / "PLAN.json")
    if campaign.sha(args.cases) != source_plan["cases_sha256"]:
        raise ValueError("source cases changed")
    cases = {c["id"]: c for c in (json.loads(line) for line in args.cases.open())}
    jobs, exclusions, source_hashes = [], Counter(), {}
    for path in sorted((source / "episodes").glob("*.json")):
        episode = read(path)
        source_hashes[str(path)] = campaign.sha(path)
        if not episode["available"]:
            exclusions["source_episode_unavailable"] += 1
            continue
        job = prepare_episode(path, cases[episode["case_id"]])
        jobs.append(job)
        source_hashes.update(job["source_hashes"])
    expected = len(source_plan["case_ids"]) * source_plan["repeats"] * len(probe.ARMS)
    exclusions["source_episode_missing"] = (
        expected - len(jobs) - exclusions["source_episode_unavailable"]
    )
    if not jobs:
        raise ValueError("no completed source episodes")
    plan = {
        "source_output": str(source),
        "source_plan_sha256": campaign.sha(source / "PLAN.json"),
        "cases_sha256": campaign.sha(args.cases),
        "source_hashes": source_hashes,
        "dependencies": {
            str(p): campaign.sha(p)
            for p in (
                Path(__file__),
                Path(probe.__file__),
                probe.BREADTH / "runner.py",
                probe.BREADTH / "runner_v2.py",
                probe.BREADTH / "campaign.py",
                probe.MUSIQUE / "metrics/answer.py",
            )
        },
        "model": source_plan["model"],
        "phrase_instruction": probe.PHRASE_INSTRUCTION,
        "episodes": {
            j["episode"]["episode_id"]: {
                "prompt_digest": runtime.digest(j["prompt"]),
                "seed": j["seed"],
                "temperature": j["temperature"],
                "max_tokens": j["max_tokens"],
                "source_request_digests": j["source_request_digests"],
            }
            for j in jobs
        },
        "source_exclusions": dict(exclusions),
        "new_work": "one final call per source episode",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PLAN.json").exists():
        if read(output / "PLAN.json") != plan:
            raise ValueError("immutable resume plan mismatch")
    else:
        runtime.save(output / "PLAN.json", plan)
    pending = [
        j for j in jobs if not (output / "pairs" / (j["episode"]["episode_id"] + ".json")).exists()
    ]
    if not pending:
        campaign.snapshot(output / "SUMMARY.json", summary(output, plan))
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
        # The native client has a 90-second timeout; verify an actual new final immediately.
        first = collect(client, pending[0], cases[pending[0]["episode"]["case_id"]])
        if not first["available"]:
            raise RuntimeError("first scientific final request failed")
        with ThreadPoolExecutor(max_workers=4) as pool:
            for position in range(1, len(pending), 4):
                if campaign.STOP.is_set() or client.stop.is_set() or time.time() >= deadline - 120:
                    break
                futures = [
                    pool.submit(collect, client, j, cases[j["episode"]["case_id"]])
                    for j in pending[position : position + 4]
                ]
                for future in futures:
                    future.result()
                campaign.snapshot(output / "SUMMARY.json", summary(output, plan))
                campaign.snapshot(
                    output / "STATUS.json",
                    {
                        "state": "collecting",
                        "returned": client.returned,
                        "transport_errors": client.errors,
                        "last_return": client.last_return,
                        "deadline": deadline,
                        "updated": time.time(),
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
            campaign.snapshot(output / "SUMMARY.json", summary(output, plan))
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
    parser.add_argument("--hours", type=float, default=2)
    run(parser.parse_args())
