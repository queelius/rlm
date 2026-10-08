"""Observe SPO reproduction using the established bounded Codex queue engine."""

import importlib.util
import json
import shlex
from pathlib import Path

ORIGIN = Path("/project/alex_phd/runs/curl-replication-20261004/reviews/code-v2/review_monitor.py")
spec = importlib.util.spec_from_file_location("spo_queue_engine", ORIGIN)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_observe, original_save = base.observe, base.save


def observe(config: dict, now: float) -> list[str]:
    """Read owner receipts, terminal events, and completed native evaluation metrics."""
    events = original_observe(config, now)
    receipted = set()
    event_paths = {Path(value) for value in config.get("native_event_paths", [])}
    for root in base.campaign_roots(config):
        for path in root.glob("*/result.json"):
            record = base.read(path)
            if (
                record
                and type(record.get("exit_code")) is int
                and type(record.get("complete")) is bool
            ):
                receipted.add(path.parent)
        event_paths.update(root.glob("*/events.jsonl"))
        event_paths.update(root.glob("*/run/events.jsonl"))
        for path in sorted(
            root.glob(
                "*/native/evaluation/iteration__*/analysis/TaskPerformanceAnalyzer/**/log.json"
            )
        ):
            record = base.read(path)
            metrics = record.get("metrics", record) if record else None
            if isinstance(metrics, dict) and metrics:
                digest = base.hashlib.sha256(
                    json.dumps(metrics, sort_keys=True).encode()
                ).hexdigest()
                events.append(f"evaluation:{path}:{digest}")
    for path in sorted(event_paths):
        if path.parent in receipted or path.parent.parent in receipted:
            continue
        terminal = None
        try:
            with path.open() as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict) and record.get("type") in ("end", "failure"):
                        terminal = record
        except OSError:
            continue
        if terminal is not None:
            digest = base.hashlib.sha256(json.dumps(terminal, sort_keys=True).encode()).hexdigest()
            events.append(f"finished:{path}:{digest}")
    return sorted(set(events))


def reason(state: dict, events: list[str], now: float) -> str | None:
    if state.get("pending"):
        return None
    elapsed = now - state["reviewed_at"]
    if set(events) - set(state.get("seen", [])) and elapsed >= 600:
        return "new terminal results"
    if elapsed >= 3600:
        return "periodic review"
    return None


def message(config: dict, request_id: str, why: str, events: list[str]) -> str:
    ack = shlex.join(
        [
            config.get("python", "python"),
            str(Path(__file__).resolve()),
            "--config",
            config.get("config_path", "CONFIG.json"),
            "--ack",
            request_id,
        ]
    )
    request = Path(config["output"]) / "requests" / f"{request_id}.json"
    queue = config.get("queue", str(Path(config["campaign_root"]) / "QUEUE.json"))
    return (
        "[AUTOMATED SPO REPRODUCTION REVIEW under the user's standing request; "
        "not a new human message]\n"
        f"Review {request_id}: {why}. Continue this exact SPO session.\n"
        f"Read {config['checkpoint']}, the live queue {queue}, and {request}.\n"
        f"Required reproduction protocol: {config['protocol_summary']}\n"
        "Inspect actual native "
        "results/events, rewards, parameter updates, evaluations, failures, and checkpoint "
        "status. Use the audited endpoint and evaluation rule; no best-test checkpoint "
        "selection. Distinguish "
        "protocol deviations, pipeline validation, incomplete attempts, and measured learning. "
        "Preserve all live owners, sealed sources, failed attempts and checkpoints. Analyze "
        "what changed and update the ranked queue and SESSION_CHECKPOINT.md with evidence "
        "and the next decision. The user is asleep and approved periodic documentation "
        "updates and GitHub pushes after these reviews; push verified completed milestones. "
        "Read the live main account quota; preserve 10% with 12% dispatch headroom. "
        f"The bounded review deadline is epoch {config['deadline']}. "
        "If nothing changed, record that briefly. Acknowledge only after reviewing:\n"
        f'{ack} --decision "Interpretation; next research decision" --evidence PATH\n'
        "Then yield. Create STOP in the observer directory if paused. "
        f"Observed result/event count: {len(events)}; full IDs are in the request receipt."
    )


def save(path: Path, value: dict) -> None:
    """Correct inherited launch metadata without changing the sealed queue engine."""
    if path.name == "LAUNCH.json":
        value = dict(value, review_interval_seconds=3600, terminal_min_interval_seconds=600)
    original_save(path, value)


base.observe, base.reason, base.message, base.save = observe, reason, message, save
base.__file__ = __file__
base.OBSERVER_ORIGIN = dict(
    path=str(ORIGIN), sha256=base.hashlib.sha256(ORIGIN.read_bytes()).hexdigest()
)

if __name__ == "__main__":
    parser = base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ack")
    parser.add_argument("--decision", default="")
    parser.add_argument("--evidence", action="append", default=[])
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    base.uuid.UUID(cfg["thread"])
    if not isinstance(cfg.get("protocol_summary"), str) or not cfg["protocol_summary"].strip():
        raise ValueError("a nonempty protocol_summary is required")
    cfg.setdefault("deadline", 1791621274)
    cfg["config_path"] = str(args.config.resolve())
    if args.ack:
        base.acknowledge(cfg, args.ack, args.decision, args.evidence)
    else:
        base.run(cfg)
