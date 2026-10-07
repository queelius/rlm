"""Review the renewed multi-hour Dr. GRPO campaign through the existing observer engine."""

import importlib.util
import json
import shlex
from pathlib import Path

ORIGIN = Path(__file__).with_name("review_monitor.py")
spec = importlib.util.spec_from_file_location("r1_original_observer", ORIGIN)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base, observe, reason = previous.base, previous.observe, previous.reason


def message(config: dict, request_id: str, why: str, events: list[str]) -> str:
    output = Path(config["output"])
    request = output / "requests" / f"{request_id}.json"
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
    roots = ", ".join(str(root) for root in base.campaign_roots(config))
    return (
        "[AUTOMATED LONGER LLM Dr. GRPO REVIEW; not a new human message]\n"
        f"Review {request_id}: {why}. Continue this exact research session.\n"
        f"Read {config['checkpoint']} and {request}; native campaign roots: {roots}.\n"
        "Protocol: fresh original base; 4096 questions selected with Random(42) from official "
        "math_lvl3to5_8k; full verifier math_verify; qwen_math prompt. Each collection has "
        "128 questions with 8 responses each. 32 collections = 256 optimizer updates and "
        "32768 responses; prescribed final checkpoint step_00033 adds no optimizer update. "
        "The 10-hour training cap and automatic full500 chat/raw final evaluations belong "
        "to the same owner. A departed training PID can mean evaluation handoff: verify "
        "the whole owner before admitting another GPU job.\n"
        "Inspect actual native model responses, terminal effective rewards, verifier behavior, "
        "finite gradients/logprobs, optimizer-update counts, throughput and saved checkpoints. "
        "Process presence or occupied GPU memory alone is insufficient. Preserve failures "
        "and sealed live source; do not turn an interrupted run into a completed endpoint. "
        "Compare final tests against our matched base and earlier runs, with no best-checkpoint "
        "or favorable test-score selection. The paper's 74.2% is an author reference, not our "
        "training gain or a stopping/selection target. State the remaining reproduction limits.\n"
        "Analyze what changed and rank the next informative iteration; keep useful follow-on "
        "jobs ready within existing authorization. If nothing changed, record that briefly. "
        "Update evidence, handoff and material presentation claims together; push verified "
        "milestones. Preserve 10% account reserve with 12% dispatch headroom. Actual allocation "
        f"end is epoch 1791621574; this observer's configured cutoff is {config['deadline']}.\n"
        f"Use only this observer's state at {output}; do not reuse an older stopped state. "
        "After completing the review, acknowledge it with:\n"
        f'{ack} --decision "Interpretation; next research decision" --evidence PATH\n'
        "Then yield. If work must pause, record why and create STOP in this observer directory. "
        f"Do not acknowledge without reviewing. Observed event count: {len(events)}."
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
    base.uuid.UUID(cfg["thread"])
    cfg["config_path"] = str(Path(args.config).resolve())
    if args.ack:
        base.acknowledge(cfg, args.ack, args.decision or "", args.evidence)
    else:
        base.run(cfg)
