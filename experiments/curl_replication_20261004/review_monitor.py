"""Queue bounded CURL reviews in an existing Codex session, never GPU jobs.

Adapted from experiments/selective_delegation/research_review_20260929/monitor.py
in the selective-delegation-20260921 worktree. Original SHA256:
bbab0b043f180900e0e4177cedc4f9c616f7ce1b8f1b7f33887efadf45b1001b.
The pending/acknowledgment, quota, exact-thread and stop/deadline engine is retained.
"""

import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path

OBSERVER_ORIGIN = {
    "path": "experiments/selective_delegation/research_review_20260929/monitor.py",
    "worktree": "selective-delegation-20260921",
    "sha256": "bbab0b043f180900e0e4177cedc4f9c616f7ce1b8f1b7f33887efadf45b1001b",
}


def read(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def observe(config: dict, now: float) -> list[str]:
    """Observe terminal native CURL records; live metrics do not trigger reviews."""
    root = Path(config["campaign_root"])
    events, receipted = [], set()
    for terminal in sorted(root.glob("*/result.json")):
        record = read(terminal)
        if record is None:
            continue
        if type(record.get("exit_code")) is not int or type(record.get("complete")) is not bool:
            continue
        digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        events.append(f"finished:{terminal}:{digest}")
        receipted.add(terminal.parent)
    for path in sorted(root.glob("*/run/metrics.jsonl")):
        if path.parent.parent in receipted:
            continue
        terminal_record = None
        try:
            with path.open() as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue  # A live writer may leave its final JSON line incomplete.
                    if isinstance(record, dict) and record.get("type") in ("end", "failure"):
                        terminal_record = record
        except OSError:
            continue
        if terminal_record is not None:
            digest = hashlib.sha256(
                json.dumps(terminal_record, sort_keys=True).encode()
            ).hexdigest()
            events.append(f"finished:{path}:{digest}")
    return sorted(events)


def reason(state: dict, events: list[str], now: float) -> str | None:
    if state.get("pending"):
        return None
    new = set(events) - set(state.get("seen", []))
    elapsed = now - state["reviewed_at"]
    if new and elapsed >= 300:
        return "new terminal results"
    if elapsed >= 1200:
        return "periodic review"
    return None


def quota_allows(report: dict | None, now: float) -> bool:
    report = report or {}
    remaining, checked = report.get("remaining_percent"), report.get("checked_epoch")
    return (
        type(remaining) in (float, int)
        and math.isfinite(remaining)
        and 12 < remaining <= 100
        and type(checked) in (float, int)
        and 0 <= now - checked <= 900
    )


def message(config: dict, request_id: str, why: str, events: list[str]) -> str:
    output = Path(config["output"])
    return (
        "[AUTOMATED CURL RESEARCH REVIEW under the user's standing request; "
        "not a new human message]\n"
        f"Review {request_id}: {why}. Continue this exact CURL reproduction session.\n"
        f"Read {config['checkpoint']} and "
        f"{output / 'requests' / (request_id + '.json')}.\n"
        f"Read native events in {config['campaign_root']}: result.json receipts and "
        "run/metrics.jsonl episode, update, end and failure records. Inspect real simulator "
        "rewards and learning updates, not merely GPU memory or process presence. "
        "Distinguish fixed endpoints from learning curves, pilot diagnostics from fresh "
        "training seeds, and incomplete/failed attempts from scientific scores. Compare "
        "matched CURL and same-crops/no-contrastive runs without favorable checkpoint "
        "selection or treating evaluation episodes as training replicates. "
        "Interpret what changed, challenge the explanation, and choose the smallest useful "
        "follow-up. Continue or prepare useful queued work under existing owners and caps; "
        "do not modify sealed live sources. Update the learning guide and evidence links "
        "with actual results, record limits and ranked follow-ups, and push verified "
        "milestones. If nothing changed, record that briefly; avoid empty review loops.\n"
        "Read the live main account quota; preserve 10% with 12% dispatch headroom. "
        f"Current allocation review deadline is epoch {config['deadline']}. "
        "Do not ask blocking questions. This observer never performs the reasoning itself.\n"
        "After completing the review, acknowledge it with:\n"
        f"{config.get('python', 'python')} {Path(__file__).resolve()} "
        f"--config {config.get('config_path', 'CONFIG.json')} --ack {request_id} "
        '--decision "Interpretation; next research decision" --evidence PATH [--evidence PATH]\n'
        "Then yield so future queued reviews can run. If work must pause, record why and create "
        "STOP in the observer directory. Do not acknowledge without actually reviewing.\n"
        f"Observed terminal event count: {len(events)}. Full event IDs are in the request receipt."
    )


def native_queue(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=45, check=True)
    if "Queued message " not in result.stdout:
        raise RuntimeError("native queue did not confirm admission")
    return result.stdout.strip()


def submit(
    config: dict, state: dict, events: list[str], now: float, why: str, queue=native_queue
) -> None:
    output = Path(config["output"])
    request_id = f"review-{int(now)}-{uuid.uuid4().hex[:8]}"
    pending = dict(
        id=request_id, requested_at=now, events=events, status="submission_uncertain", reason=why
    )
    state["pending"] = pending
    save(output / "STATE.json", state)  # Persist BEFORE queue admission, including ambiguous exits.
    record = dict(
        pending, thread=config["thread"], message=message(config, request_id, why, events)
    )
    request_path = output / "requests" / (request_id + ".json")
    save(request_path, record)
    argv = [config["codex"], "queue", "--thread", config["thread"], "--message", record["message"]]
    try:
        record["native_admission"] = queue(argv)
        pending["status"] = "queued_not_acknowledged"
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["status"] = pending["status"]
    save(request_path, record)
    save(output / "STATE.json", state)
    print(json.dumps({"review": request_id, "status": pending["status"]}), flush=True)


def acknowledge(
    config: dict, request_id: str, decision: str, evidence: list[str], now: float | None = None
) -> None:
    if not decision.strip():
        raise ValueError("a reviewed decision is required")
    output = Path(config["output"])
    pending = (read(output / "STATE.json") or {}).get("pending") or {}
    if pending.get("id") != request_id:
        raise ValueError("acknowledgement must match the pending request")
    if not evidence or not all(pointer.strip() for pointer in evidence):
        raise ValueError("at least one evidence pointer is required")
    save(
        output / "acknowledgements" / (request_id + ".json"),
        dict(
            id=request_id,
            reviewed_at=time.time() if now is None else now,
            decision=decision,
            evidence=evidence,
            thread=config["thread"],
        ),
    )


def consume_ack(config: dict, state: dict) -> bool:
    pending = state.get("pending")
    if not pending:
        return False
    ack = read(Path(config["output"]) / "acknowledgements" / (pending["id"] + ".json"))
    if not ack or ack.get("id") != pending["id"] or not ack.get("decision", "").strip():
        return False
    state.update(
        pending=None,
        reviewed_at=ack["reviewed_at"],
        seen=pending["events"],
        last_acknowledgement=pending["id"],
    )
    save(Path(config["output"]) / "STATE.json", state)
    return True


def load_quota(config: dict) -> dict:
    spec = importlib.util.spec_from_file_location("quota_sampler", config["quota_sampler"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return asyncio.run(module.sample())


def run(config: dict) -> None:
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "DISPATCHER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        now = time.time()
        state = read(output / "STATE.json") or dict(
            reviewed_at=now,
            seen=observe(config, now),
            pending=None,
        )
        save(
            output / "LAUNCH.json",
            dict(
                pid=os.getpid(),
                started=now,
                thread=config["thread"],
                deadline=config["deadline"],
                source=str(Path(__file__).resolve()),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                no_gpu_actions=True,
                adapted_from=OBSERVER_ORIGIN,
                review_interval_seconds=1200,
            ),
        )
        quota = read(output / "QUOTA.json")
        last_quota_attempt = 0
        while time.time() < config["deadline"] and not (output / "STOP").exists():
            now = time.time()
            consume_ack(config, state)
            events = observe(config, now)
            why = reason(state, events, now)
            if (
                not quota
                or now - quota.get("checked_epoch", 0) >= 900
                or quota.get("remaining_percent") is None
            ) and now - last_quota_attempt >= 60:
                last_quota_attempt = now
                try:
                    quota = load_quota(config)
                except Exception as error:
                    quota = dict(
                        checked_epoch=now, remaining_percent=None, error_type=type(error).__name__
                    )
                save(output / "QUOTA.json", quota)
            if (
                why
                and time.time() < config["deadline"]
                and not (output / "STOP").exists()
                and quota_allows(quota, time.time())
            ):
                submit(config, state, events, time.time(), why)
            save(
                output / "STATUS.json",
                dict(
                    checked_at=time.time(),
                    pending=state.get("pending"),
                    reviewed_at=state["reviewed_at"],
                    due=why,
                    quota_allows_submission=quota_allows(quota, time.time()),
                    event_count=len(events),
                    delivery_verified=bool(state.get("last_acknowledgement")),
                ),
            )
            time.sleep(min(30, max(0, config["deadline"] - time.time())))
        save(
            output / "TERMINAL.json",
            dict(
                ended=time.time(),
                reason="stop_requested" if (output / "STOP").exists() else "allocation_deadline",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ack")
    parser.add_argument("--decision", default="")
    parser.add_argument("--evidence", action="append", default=[])
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    uuid.UUID(cfg["thread"])  # Exact UUID: never an ambiguous session name or --last.
    cfg["config_path"] = str(args.config.resolve())
    if args.ack:
        acknowledge(cfg, args.ack, args.decision, args.evidence)
    else:
        run(cfg)
