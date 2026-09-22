"""CPU-only queue007 health watcher and one terminal native fresh16 analysis."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
QUEUE = ROOT / "independent-training-queue-007"
RECEIPT = ROOT / "INDEPENDENT-TRAINING-QUEUE-007.json"
DEST = ROOT / "textcraft-fresh-watch-001"
SOURCE = ROOT / "source-textcraft-fresh-readout-001"
PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/"
    "a100-lora-roundtrip/gpu/training/.venv/bin/python"
)
QUEUE_PID, QUEUE_CREATED = 137706, 1790073871.8
PINS = {
    RECEIPT: "b6358104f373a8795f830170e360ced237e1819a2513ed94215f19a7acadfbfd",
    QUEUE / "INVOCATION.json": "56472cf24d8c955b35e9ba3ea247b86f14119d498e10cc4b0dd76e43660c74a1",
    ROOT
    / "source-054-independent-queue/run_independent_queue.py": (
        "35d1c12376dd29513960303df97fee109aa353c7f757175d948f680b00fcd6c8"
    ),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def live(pid, created):
    try:
        process = psutil.Process(pid)
        return (
            abs(process.create_time() - created) < 0.01 and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def release_state(output, is_live=live):
    owners = list(output.glob("OWNER-*.json"))
    if not owners:
        return "no_owner"
    for path in owners:
        owner = read(path)
        terminal = read(path.with_name(path.name.replace("OWNER-", "TERMINAL-")))
        if not owner or not terminal or is_live(owner["pid"], owner["create_time"]):
            return "owner_unresolved"
    return "released"


def queue_resolution(states, queue_live):
    if all(state == "released" for state in states):
        return "analyze"
    return "waiting" if queue_live else "unknown_queue_exited"


def response_health(start, call):
    bound = (
        start["call_id"] == call["call_id"]
        and start["request_digest"] == call["request_digest"]
        and start["started"] == call["started"]
    )
    seconds = call["ended"] - call["started"]
    return dict(
        call_id=call["call_id"],
        request_bound=bound,
        available=call.get("available"),
        healthy=bool(
            bound and call.get("available") and call.get("text") and call.get("output_token_ids")
        ),
        seconds=seconds,
        returned_within_90_seconds=0 <= seconds <= 90,
        output_tokens=len(call.get("output_token_ids") or []),
    )


def main():
    started = time.time()
    DEST.mkdir(exist_ok=False)
    result = dict(
        started=started,
        failure=None,
        analysis_returncode=None,
        scientific_comparison_available=False,
    )
    seen, firsts = set(), {}

    def event(name, **fields):
        if name not in seen:
            seen.add(name)
            value = dict(event=name, observed=time.time(), **fields)
            save(DEST / (name + ".json"), value)
            print(json.dumps(value), flush=True)

    try:
        verified, large = {}, {}
        for path, digest in PINS.items():
            if sha(path) != digest:
                raise ValueError("watcher identity changed: " + str(path))
            verified[str(path)] = digest
        receipt, invocation = read(RECEIPT), read(QUEUE / "INVOCATION.json")
        if (
            receipt["status"] != "accepted"
            or invocation["pid"] != QUEUE_PID
            or invocation["receipt_sha256"] != PINS[RECEIPT]
        ):
            raise ValueError("accepted queue identity mismatch")
        jobs = receipt["jobs"]
        if [job["name"] for job in jobs] != [
            "textcraft-fresh-privileged-001",
            "textcraft-fresh-public-001",
        ]:
            raise ValueError("unexpected queue jobs")
        for job in jobs:
            for filename, digest in job["pins"].items():
                path = Path(filename)
                if filename in verified:
                    continue
                if path.stat().st_size > 16 * 1024 * 1024:
                    large[filename] = digest
                elif sha(path) != digest:
                    raise ValueError("sealed small input changed: " + filename)
                else:
                    verified[filename] = digest
        command = [
            PYTHON,
            str(SOURCE / "analyze_textcraft_fresh.py"),
            "--privileged-output",
            jobs[0]["output"],
            "--public-output",
            jobs[1]["output"],
            "--report",
            str(ROOT / "analysis-textcraft-fresh-001.json"),
        ]
        save(
            DEST / "INVOCATION.json",
            dict(
                pid=os.getpid(),
                create_time=psutil.Process().create_time(),
                started=started,
                source=str(Path(__file__).resolve()),
                source_sha256=sha(Path(__file__)),
                queue_pid=QUEUE_PID,
                queue_create_time=QUEUE_CREATED,
                queue_live_at_start=live(QUEUE_PID, QUEUE_CREATED),
                queue_receipt_sha256=PINS[RECEIPT],
                verified_small_pins=verified,
                large_ancestry_not_rehashed=large,
                command=command,
                maximum_seconds=18000,
                poll_seconds=10,
                analysis_timeout_seconds=1200,
                policy="No GPU actions. Both authenticated owners must release before analysis. "
                "Missing or preflight-failed runs unknown; no optimizer SUMMARY requirement.",
            ),
        )
        while True:
            if time.time() - started >= 18000:
                raise TimeoutError("five-hour CPU watcher expired; no GPU process modified")
            for job in jobs:
                name, output = job["name"], Path(job["output"])
                if name + "-FIRST-RETURN" in seen:
                    continue
                if name not in firsts:
                    starts = [(p, read(p)) for p in (output / "starts").glob("*.json")]
                    starts = [(p, row) for p, row in starts if row]
                    if starts:
                        firsts[name] = min(starts, key=lambda item: item[1]["started"])
                if name in firsts:
                    path, first = firsts[name]
                    event(
                        name + "-FIRST-REQUEST",
                        call_id=first["call_id"],
                        started=first["started"],
                        receipt_sha256=sha(path),
                    )
                    call_path = output / "calls" / path.name
                    call = read(call_path)
                    if call:
                        health = response_health(first, call)
                        event(
                            name + "-FIRST-RETURN",
                            **health,
                            receipt_sha256=sha(call_path),
                            observed_seconds_after_request=time.time() - first["started"],
                        )
                        if not health["healthy"] or not health["returned_within_90_seconds"]:
                            event(name + "-ALERT-NATIVE-HEALTH", **health)
                    elif time.time() - first["started"] > 90:
                        event(name + "-ALERT-FIRST-RETURN-LATE", call_id=first["call_id"])
                job_record = read(QUEUE / (name + ".json"))
                if job_record and not list(output.glob("OWNER-*.json")):
                    event(
                        name + "-PREFLIGHT-UNAVAILABLE",
                        job_record=job_record,
                        interpretation="Unknown, not zero success",
                    )
            states = [release_state(Path(job["output"])) for job in jobs]
            resolution = queue_resolution(states, live(QUEUE_PID, QUEUE_CREATED))
            if resolution != "waiting":
                result["queue_resolution"] = dict(
                    state=resolution,
                    output_states=states,
                    jobs={job["name"]: read(QUEUE / (job["name"] + ".json")) for job in jobs},
                    job_receipt_sha256={
                        str(p): sha(p)
                        for job in jobs
                        if (p := QUEUE / (job["name"] + ".json")).exists()
                    },
                    owner_terminal_sha256={
                        str(p): sha(p)
                        for job in jobs
                        for pattern in ("OWNER-*.json", "TERMINAL-*.json")
                        for p in Path(job["output"]).glob(pattern)
                    },
                )
                if resolution != "analyze":
                    raise RuntimeError(
                        "authenticated queue exited before both owner releases; "
                        "unavailable comparison, not zero outcome"
                    )
                break
            time.sleep(10)
        event("ANALYSIS-STARTING", command=command)
        with (DEST / "analysis.log").open("x") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=1200,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "NVIDIA_VISIBLE_DEVICES": "none",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OMP_NUM_THREADS": "4",
                },
            )
        result["analysis_returncode"] = completed.returncode
        report = Path(command[-1])
        result["report_sha256"] = sha(report) if report.exists() else None
        if completed.returncode:
            raise RuntimeError("sealed native analyzer failed; log/attempt preserved")
        result["scientific_comparison_available"] = True
    except BaseException as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["ended"] = time.time()
        save(DEST / "RESULT.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
