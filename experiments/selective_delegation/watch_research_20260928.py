"""Token-free first-response and completion journal for accepted research queues."""

import argparse
import json
import os
import time
from pathlib import Path

import watch_textcraft_fresh as h


def health(start, call, now):
    if call is None:
        return dict(
            healthy=None,
            late=now - start["started"] > 90,
            call_id=start["call_id"],
            observed_seconds=now - start["started"],
        )
    result = h.response_health(start, call)
    result["late"] = not result["returned_within_90_seconds"]
    return result


def operation_paths(root):
    # Discover only this session's explicit operation naming convention.
    return sorted(root.glob("*-20260928-*/ACCEPTED.json"))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = int(os.environ["SLURM_JOB_END_TIME"]) - 600
    invocation = dict(
        pid=os.getpid(),
        create_time=h.psutil.Process().create_time(),
        started=time.time(),
        deadline=deadline,
        no_gpu_actions=True,
        source=str(Path(__file__).resolve()),
        source_sha256=h.sha(Path(__file__)),
        helper_sha256=h.sha(Path(h.__file__)),
        poll_seconds=args.interval,
    )
    h.save(args.output / "INVOCATION.json", invocation)
    receipts, firsts, seen = {}, {}, set()

    def event(key, kind, **fields):
        if (key, kind) in seen:
            return
        seen.add((key, kind))
        record = dict(job=key, event=kind, observed=time.time(), **fields)
        with (args.output / "EVENTS.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    while time.time() < deadline:
        now, snapshot = time.time(), []
        for path in operation_paths(args.root):
            if path not in receipts:
                receipt = h.read(path)
                if not receipt or receipt.get("status") != "accepted":
                    continue
                receipts[path] = receipt
                event(str(path), "ACCEPTED-QUEUE", sha256=h.sha(path), jobs=len(receipt["jobs"]))
            receipt = receipts[path]
            for job in receipt["jobs"]:
                key = path.parent.name + "/" + job["name"]
                execution = h.read(path.parent / "queue" / (job["name"] + ".json"))
                if execution:
                    event(key, "STAGE-EXIT", execution=execution)
                if not job.get("output"):
                    continue
                output = Path(job["output"])
                status = h.read(output / "STATUS.json")
                summary = h.read(output / "SUMMARY.json")
                skipped = h.read(output / "CONDITIONAL-SKIP.json")
                owners = list(output.glob("OWNER-*.json"))
                terminals = list(output.glob("TERMINAL-*.json"))
                state = h.release_state(output) if owners else "not_started"
                if skipped:
                    state = "skipped"
                    event(key, "SCIENTIFIC-SKIP", reason=skipped)
                if state == "released":
                    event(
                        key,
                        "SCIENTIFIC-RELEASE",
                        summary=summary,
                        terminals=[str(p) for p in terminals],
                    )
                if (key, "FIRST-RETURN") not in seen:
                    if key not in firsts:
                        starts = [(p, h.read(p)) for p in (output / "starts").glob("*.json")]
                        starts = [(p, row) for p, row in starts if row]
                        if starts:
                            firsts[key] = min(starts, key=lambda pair: pair[1]["started"])
                    if key in firsts:
                        start_path, start = firsts[key]
                        call_path = output / "calls" / start_path.name
                        call = h.read(call_path)
                        check = health(start, call, now)
                        if call:
                            event(
                                key,
                                "FIRST-RETURN",
                                **check,
                                start_path=str(start_path),
                                call_path=str(call_path),
                                call_sha256=h.sha(call_path),
                            )
                        if check["late"] or check["healthy"] is False:
                            event(key, "ALERT-FIRST-RESPONSE", **check)
                snapshot.append(
                    dict(
                        queue=path.parent.name,
                        job=job["name"],
                        output=str(output),
                        state=state,
                        status=status,
                        summary=summary,
                        skip=skipped,
                        stage_exit=execution,
                    )
                )
        with (args.output / "LATEST.json").open("w") as stream:
            json.dump(dict(updated=now, deadline=deadline, jobs=snapshot), stream, indent=2)
        if args.once:
            break
        time.sleep(min(args.interval, max(0, deadline - time.time())))
    h.save(args.output / "TERMINAL.json", dict(ended=time.time(), events=len(seen)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=h.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("poll no faster than every five seconds")
    main(args)
