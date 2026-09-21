"""Final-only removal of provisional answer or whole checkpoint, holding helpers fixed."""

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
import execution_probe
import probe

campaign, runtime = probe.campaign, probe.runtime
CONDITIONS = ("remove_answer", "remove_checkpoint")
read = answer_format.read


def changed_prompt(original, condition):
    prefix, raw = original.rsplit("\n", 1)
    body = json.loads(raw)
    if condition == "remove_answer":
        del body["initial_attempt"]["answer"]
    elif condition == "remove_checkpoint":
        del body["initial_attempt"]
    else:
        raise ValueError("unknown checkpoint condition")
    return prefix + "\n" + json.dumps(body, ensure_ascii=False)


def saved_report(episode, records):
    if episode["arm"].endswith("isolated"):
        trace = [
            {
                "question": json.loads(r["prompt"].split("\n", 1)[1])["question"],
                "answer": execution_probe.parse_helper(r["text"]),
            }
            for r in records[:-1]
        ]
        if trace != episode["helper_trace"]:
            raise ValueError("saved helper trace differs from actual helper requests/responses")
        return json.dumps(trace, ensure_ascii=False)
    return records[0]["text"]


def prepare_episode(path, case, source_plan):
    path = Path(path)
    episode = read(path)
    if not episode["available"] or episode["case_id"] != case["id"]:
        raise ValueError("source episode unavailable or identity differs")
    if episode["arm"] not in execution_probe.ARMS:
        raise ValueError("unknown source execution arm")
    key = episode["episode_id"]
    helpers = (
        [key + "-helper1", key + "-helper2"]
        if episode["arm"].endswith("isolated")
        else [key + "-helper"]
    )
    if episode["new_call_ids"] != helpers + [key + "-final"]:
        raise ValueError("source call-role mismatch")
    initial_path = (
        Path(source_plan["source_output"])
        / "calls"
        / (episode["reused_checkpoint_call_id"] + ".json")
    )
    paths = [initial_path] + [
        path.parent.parent / "calls" / (cid + ".json") for cid in episode["new_call_ids"]
    ]
    all_records, hashes = [], {str(path): campaign.sha(path)}
    for p in paths:
        record = read(p)
        if not record["available"] or record["request_digest"] != runtime.digest(record["request"]):
            raise ValueError("source native call unavailable or request changed")
        if record["request"]["model"] != source_plan["model"]:
            raise ValueError("source native model changed")
        all_records.append(record)
        hashes[str(p)] = campaign.sha(p)
    initial, records = all_records[0], all_records[1:]
    if initial["request_digest"] != episode["source_checkpoint_request_digest"]:
        raise ValueError("original checkpoint request differs")
    state = probe.checkpoint(initial["text"])
    if runtime.digest(state) != episode["original_checkpoint_digest"]:
        raise ValueError("original checkpoint state differs")
    if initial["prompt"] != probe.initial_prompt(case):
        raise ValueError("original checkpoint public input differs")
    for p, expected in source_plan["source_checkpoints"][case["id"]]["source_hashes"].items():
        if campaign.sha(p) != expected:
            raise ValueError("original checkpoint source hash changed")
        hashes[p] = expected
    report = saved_report(episode, records)
    prompt = probe.final_prompt(case, state, report, answer_style="span")
    if prompt != records[-1]["prompt"]:
        raise ValueError("saved final cannot be reconstructed exactly")
    sampling = records[-1]["request"]["sampling_params"]
    if (
        sampling["seed"] != episode["seed"]
        or sampling["temperature"] != 0.5
        or sampling["max_tokens"] != 128
    ):
        raise ValueError("source final sampling contract differs")
    return {
        "episode": episode,
        "initial": initial,
        "records": records,
        "prompt": prompt,
        "seed": sampling["seed"],
        "temperature": sampling["temperature"],
        "source_hashes": hashes,
        "source_request_digests": {r["call_id"]: r["request_digest"] for r in all_records},
    }


def collect(client, job, case, condition):
    episode = job["episode"]
    key = episode["episode_id"] + "-" + condition
    path = client.output / "pairs" / (key + ".json")
    if path.exists():
        return read(path)
    new = client.call(
        key + "-final",
        changed_prompt(job["prompt"], condition),
        job["seed"],
        128,
        temperature=job["temperature"],
    )
    old_grade = probe.grade(job["records"][-1]["text"], case)
    new_grade = (
        probe.grade(new["text"], case)
        if new["available"]
        else {"valid": False, "correct": False, "f1": 0.0, "parsed": None}
    )
    reused = [job["initial"], *job["records"][:-1]]
    result = {
        "pair_id": key,
        "source_episode_id": episode["episode_id"],
        "case_id": case["id"],
        "arm": episode["arm"],
        "repeat": episode["repeat"],
        "condition": condition,
        "seed": job["seed"],
        "available": new["available"],
        "old": old_grade,
        "new": new_grade,
        "old_answer": old_grade["parsed"],
        "new_answer": new_grade["parsed"],
        "reused_call_ids": [r["call_id"] for r in reused],
        "new_call_id": new["call_id"],
        "source_hashes": job["source_hashes"],
        "source_request_digests": job["source_request_digests"],
        "new_physical_cost": answer_format.measured_cost([new]),
        "hypothetical_deployed_cost": answer_format.measured_cost([*reused, new]),
        "annotation_privileged": episode["arm"].startswith("reference"),
        "error": new.get("error"),
        "ended": time.time(),
    }
    runtime.save(path, result)
    return result


def summarize(output, plan):
    rows = [read(p) for p in (output / "pairs").glob("*.json")]
    calls = [read(p) for p in (output / "calls").glob("*.json")]
    starts = [read(p) for p in (output / "starts").glob("*.json")]
    groups = {}
    for arm in execution_probe.ARMS:
        for condition in CONDITIONS:
            subset = [r for r in rows if r["arm"] == arm and r["condition"] == condition]
            paired = [r for r in subset if r["available"]]
            groups[arm + "|" + condition] = {
                "planned_source_parents_x_repeats": plan["planned_per_arm"],
                "eligible_source_episodes": plan["eligible_by_arm"].get(arm, 0),
                "completed": len(subset),
                "available": len(paired),
                "old_correct": sum(r["old"]["correct"] for r in subset),
                "new_correct": sum(r["new"]["correct"] for r in subset),
                "old_f1_sum": sum(r["old"]["f1"] for r in subset),
                "new_f1_sum": sum(r["new"]["f1"] for r in subset),
                "paired_wins": sum(not r["old"]["correct"] and r["new"]["correct"] for r in paired),
                "paired_losses": sum(
                    r["old"]["correct"] and not r["new"]["correct"] for r in paired
                ),
            }
    return {
        "groups": groups,
        "planned_new_calls_if_all_sources_available": plan["planned_all_sources"],
        "eligible_new_calls": len(plan["episodes"]) * len(CONDITIONS),
        "source_exclusions": plan["source_exclusions"],
        "new_physical_cost": answer_format.measured_cost(calls),
        "physical_start_receipts": len(starts),
        "unresolved_start_receipts": sum(
            not (output / "calls" / (s["call_id"] + ".json")).exists() for s in starts
        ),
        "extra_start_attempts": max(0, len(starts) - len({s["call_id"] for s in starts})),
        "interpretation": "Only original initial_attempt.answer or entire initial_attempt removed "
        "from final input. Actual helpers and span-final instruction unchanged. Original finals "
        "are reused baselines. Source failures excluded explicitly, not repaired. Physical new "
        "cost counts only replay finals; hypothetical deployment includes source checkpoint and "
        "helpers once. Reference-plan privilege persists; unknown usage is unmeasured, not free.",
        "updated": time.time(),
    }


def run(args):
    import psutil
    from transformers import AutoTokenizer

    source, output = args.source_output.resolve(), args.output.resolve()
    if source == output or not list(source.glob("TERMINAL-*.json")):
        raise ValueError("separate output and completed source owner required")
    source_plan = read(source / "PLAN.json")
    if campaign.sha(args.cases) != source_plan["cases_sha256"]:
        raise ValueError("source cases checksum differs")
    with args.cases.open() as stream:
        cases = {c["id"]: c for c in (json.loads(line) for line in stream)}
    jobs, exclusions, hashes, eligible = [], Counter(), {}, Counter()
    for path in sorted((source / "episodes").glob("*.json")):
        episode = read(path)
        hashes[str(path)] = campaign.sha(path)
        if not episode["available"]:
            exclusions[episode["arm"] + ":source_unavailable"] += 1
            continue
        job = prepare_episode(path, cases[episode["case_id"]], source_plan)
        jobs.append(job)
        hashes.update(job["source_hashes"])
        eligible[episode["arm"]] += 1
    planned_per_arm = len(source_plan["case_ids"]) * source_plan["repeats"]
    for arm in execution_probe.ARMS:
        exclusions[arm + ":source_missing"] = (
            planned_per_arm - eligible[arm] - exclusions[arm + ":source_unavailable"]
        )
    if not jobs:
        raise ValueError("no available source final episodes")
    plan = {
        "source_output": str(source),
        "source_plan_sha256": campaign.sha(source / "PLAN.json"),
        "cases_sha256": campaign.sha(args.cases),
        "source_hashes": hashes,
        "model": source_plan["model"],
        "conditions": list(CONDITIONS),
        "planned_per_arm": planned_per_arm,
        "eligible_by_arm": dict(eligible),
        "planned_all_sources": planned_per_arm * len(execution_probe.ARMS) * len(CONDITIONS),
        "source_exclusions": dict(exclusions),
        "episodes": {
            j["episode"]["episode_id"]: {
                "source_request_digests": j["source_request_digests"],
                "seed": j["seed"],
                "temperature": j["temperature"],
                "max_tokens": 128,
                "prompt_digests": {
                    c: runtime.digest(changed_prompt(j["prompt"], c)) for c in CONDITIONS
                },
            }
            for j in jobs
        },
        "dependencies": {
            str(p): campaign.sha(p)
            for p in (
                Path(__file__),
                Path(probe.__file__),
                Path(answer_format.__file__),
                Path(execution_probe.__file__),
                Path(execution_probe.plan_probe.__file__),
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
    pending = [
        (j, c)
        for j in jobs
        for c in CONDITIONS
        if not (output / "pairs" / (j["episode"]["episode_id"] + "-" + c + ".json")).exists()
    ]
    if not pending:
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
        job, condition = pending[0]
        first = collect(client, job, cases[job["episode"]["case_id"]], condition)
        if not first["available"]:
            raise RuntimeError("first scientific final request failed")
        with ThreadPoolExecutor(max_workers=4) as pool:
            for position in range(1, len(pending), 4):
                if campaign.STOP.is_set() or client.stop.is_set() or time.time() >= deadline - 120:
                    break
                futures = [
                    pool.submit(collect, client, j, cases[j["episode"]["case_id"]], c)
                    for j, c in pending[position : position + 4]
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
    parser.add_argument("--hours", type=float, default=2)
    run(parser.parse_args())
