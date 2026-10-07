"""Reuse the sealed observer, with the current protocol supplied by its new config."""

import importlib.util
import json
import shlex
from pathlib import Path

ORIGIN = Path(__file__).with_name("review_monitor_longer.py")
spec = importlib.util.spec_from_file_location("r1_longer_observer_origin", ORIGIN)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base, observe, reason = previous.base, previous.observe, previous.reason


def validate_config(config: dict) -> None:
    protocol = config.get("protocol_summary")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("A nonempty protocol_summary must describe this owner and fixed endpoint")


def message(config: dict, request_id: str, why: str, events: list[str]) -> str:
    validate_config(config)
    original = previous.message(config, request_id, why, events)
    before, marker, tail = original.partition("Protocol: ")
    _, end, after = tail.partition("Inspect actual native model responses")
    if not marker or not end:
        raise ValueError("Sealed predecessor message boundaries changed")
    payload = before + "Protocol: " + config["protocol_summary"].strip() + "\n" + end + after
    return payload.replace(
        shlex.quote(str(ORIGIN.resolve())), shlex.quote(str(Path(__file__).resolve()))
    )


base.message = message
base.__file__ = __file__
base.OBSERVER_ORIGIN = {
    "path": str(ORIGIN.resolve()),
    "sha256": base.hashlib.sha256(ORIGIN.read_bytes()).hexdigest(),
    "engine_origin": base.OBSERVER_ORIGIN,
}

if __name__ == "__main__":
    parser = base.argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "ack", "decision"):
        parser.add_argument(f"--{flag}", required=flag == "config")
    parser.add_argument("--evidence", action="append", default=[])
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    validate_config(cfg)
    base.uuid.UUID(cfg["thread"])
    cfg["config_path"] = str(Path(args.config).resolve())
    if args.ack:
        base.acknowledge(cfg, args.ack, args.decision or "", args.evidence)
    else:
        base.run(cfg)
