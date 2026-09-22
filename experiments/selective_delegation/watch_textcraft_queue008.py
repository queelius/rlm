"""CPU-only accepted008 health and terminal analysis; never launches/stops GPU jobs."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import psutil
import watch_textcraft_fresh as h

ROOT = h.ROOT
QUEUE = ROOT / "independent-training-queue-008"
RECEIPT = ROOT / "INDEPENDENT-TRAINING-QUEUE-008.json"
GATE = QUEUE / "GATE-INVOCATION.json"
PID, CREATED = 159369, 1790076396.65
SOURCE = ROOT / "source-textcraft-endpoint-fresh-001"
PINS = {
    RECEIPT: "3194d991caed7746cd39e80a8fa026dc4760606c24ccb99df75ad3361121bc49",
    GATE: "b6a04b195493a6864c00bc36c27d83935312f2b7ce60c25f8d06deb2284a702c",
    SOURCE / "SOURCE.json": "809f4978c39db87819d2e8a04315146ccfff97feab47013b67d746b305e9fa18",
}
READOUTS = [
    ROOT / name
    for name in (
        "textcraft-fresh-public-001",
        "textcraft-terminal-rl-fresh-001",
        "textcraft-matched-sft-fresh-001",
    )
]


def resolution(queue_live, owner_states, readout_states):
    if queue_live or "owner_unresolved" in owner_states + readout_states:
        return "waiting"
    return "analyze" if all(s == "released" for s in readout_states) else "unavailable"


def inference_scopes(output, name):
    if name == "textcraft-matched-sft-001":
        return []
    if name == "textcraft-terminal-rl-002":
        return [output / "qualification", *sorted(output.glob("batches/sample-*/rollout"))]
    return [output]


def validate():
    for path, digest in PINS.items():
        if h.sha(path) != digest:
            raise ValueError("fixed queue/analyzer identity changed: " + str(path))
    receipt, gate = h.read(RECEIPT), h.read(GATE)
    if (
        receipt["status"] != "accepted"
        or gate["pid"] != PID
        or gate["create_time"] != CREATED
        or gate["receipt_sha256"] != PINS[RECEIPT]
    ):
        raise ValueError("authenticated accepted008 gate required")
    expected = [
        "textcraft-terminal-rl-002",
        "textcraft-matched-sft-001",
        "textcraft-terminal-rl-fresh-001",
        "textcraft-matched-sft-fresh-001",
    ]
    if [j["name"] for j in receipt["jobs"]] != expected:
        raise ValueError("unexpected008 job inventory")
    pins = {str(p): d for p, d in PINS.items()}
    for job in receipt["jobs"]:
        for filename, digest in job["pins"].items():
            if filename in pins or Path(filename).stat().st_size > 16 * 1024 * 1024:
                continue
            if h.sha(Path(filename)) != digest:
                raise ValueError("small sealed input changed: " + filename)
            pins[filename] = digest
    command = [
        h.PYTHON,
        str(SOURCE / "analyze_textcraft_endpoint_fresh.py"),
        "--warm-output",
        str(READOUTS[0]),
        "--rl-output",
        str(READOUTS[1]),
        "--sft-output",
        str(READOUTS[2]),
        "--report",
        str(ROOT / "analysis-textcraft-terminal-fresh-001.json"),
    ]
    return receipt["jobs"], pins, command


def main(args):
    jobs, pins, command = validate()
    if args.validate_only:
        print(json.dumps(dict(validated=True, queue_live=h.live(PID, CREATED), GPU_loaded=False)))
        return
    started = time.time()
    deadline = min(started + 36000, int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    args.output.mkdir(exist_ok=False)
    seen, checked_calls = set(), set()
    result = dict(started=started, failure=None, analysis_returncode=None)

    def event(name, **fields):
        if name in seen:
            return
        seen.add(name)
        row = dict(event=name, observed=time.time(), **fields)
        h.save(args.output / (name + ".json"), row)
        with (args.output / "STATUS.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    h.save(
        args.output / "INVOCATION.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
            queue_pid=PID,
            queue_create_time=CREATED,
            source_sha256={
                str(Path(__file__).resolve()): h.sha(Path(__file__)),
                str(Path(h.__file__).resolve()): h.sha(Path(h.__file__)),
            },
            pins=pins,
            command=command,
            poll_seconds=10,
            analysis_max_seconds=1200,
            no_gpu_mutations=True,
            large_ancestry_rehash=False,
        ),
    )
    try:
        while time.time() < deadline:
            for job in jobs:
                name, output = job["name"], Path(job["output"])
                for scope in inference_scopes(output, name):
                    label = name + "-" + str(scope.relative_to(output)).replace("/", "-")
                    starts = [(p, h.read(p)) for p in (scope / "starts").glob("*.json")]
                    starts = [(p, row) for p, row in starts if row]
                    if starts:
                        path, start = min(starts, key=lambda pair: pair[1]["started"])
                        event(
                            label + "-FIRST-REQUEST",
                            path=str(path),
                            sha256=h.sha(path),
                            started=start["started"],
                        )
                        returned = scope / "calls" / path.name
                        call = h.read(returned)
                        if call:
                            health = h.response_health(start, call)
                            event(
                                label + "-FIRST-RETURN",
                                **health,
                                sha256=h.sha(returned),
                                observed_seconds_after_request=time.time() - start["started"],
                            )
                            if not health["healthy"] or not health["returned_within_90_seconds"]:
                                event(label + "-ALERT-HEALTH", **health)
                        elif time.time() - start["started"] > 90:
                            event(label + "-ALERT-NO-REPLY-90S", call_id=start["call_id"])
                    for path in (scope / "calls").glob("*.json"):
                        if path in checked_calls:
                            continue
                        call = h.read(path)
                        if not call:
                            continue
                        checked_calls.add(path)
                        if not call.get("available"):
                            event(
                                label + "-ALERT-FAILED-" + path.stem,
                                call_id=call["call_id"],
                                failure=call.get("error", call.get("failure")),
                                path=str(path),
                                sha256=h.sha(path),
                            )
                # Optimizer jobs legitimately have no inference replies while training.
                for pattern in ("updates/*.json", "boundaries/*/BOUNDARY.json"):
                    for path in output.glob(pattern):
                        row = h.read(path)
                        if row:
                            event(
                                name + "-PROGRESS-" + path.parent.name + "-" + path.stem,
                                path=str(path),
                                sha256=h.sha(path),
                                record=row,
                            )
                record = h.read(QUEUE / (name + ".json"))
                if record and not list(output.glob("OWNER-*.json")):
                    event(name + "-NO-OWNER", queue_job=record, outcome="unavailable, not zero")
            states = [h.release_state(Path(j["output"])) for j in jobs]
            readouts = [h.release_state(path) for path in READOUTS]
            state = resolution(h.live(PID, CREATED), states, readouts)
            if state != "waiting":
                result.update(resolution=state, owner_states=states, readout_states=readouts)
                result["completion_receipt_sha256"] = {
                    str(path): h.sha(path)
                    for directory in {QUEUE, *READOUTS, *(Path(j["output"]) for j in jobs)}
                    for pattern in ("OWNER-*.json", "TERMINAL-*.json", "RESULT.json")
                    for path in directory.glob(pattern)
                }
                if state == "analyze":
                    remaining = min(1200, deadline - time.time())
                    if remaining <= 0:
                        raise TimeoutError("no CPU analysis time remains")
                    event("ANALYSIS-START", command=command)
                    with (args.output / "ANALYSIS.log").open("x") as stream:
                        process = subprocess.run(
                            command,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            timeout=remaining,
                            check=False,
                            env={
                                **os.environ,
                                "CUDA_VISIBLE_DEVICES": "",
                                "HF_HUB_OFFLINE": "1",
                                "PYTHONDONTWRITEBYTECODE": "1",
                            },
                        )
                    result["analysis_returncode"] = process.returncode
                    if process.returncode:
                        raise RuntimeError("sealed analysis failed; log preserved, no retry")
                    report = Path(command[-1])
                    result["report_sha256"] = h.sha(report)
                break
            time.sleep(min(10, max(0, deadline - time.time())))
        else:
            raise TimeoutError("ten-hour/lease-minus600 CPU watcher cap")
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["ended"] = time.time()
        h.save(args.output / "RESULT.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    main(parser.parse_args())
