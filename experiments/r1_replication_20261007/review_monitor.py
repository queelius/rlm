"""Thin native LLM Dr. GRPO observer; retain the established queue and quota engine."""
import importlib.util
import json
import re
import shlex
from pathlib import Path

ORIGIN = Path("/project/alex_phd/runs/curl-replication-20261004/reviews/code-v2/review_monitor.py")
spec = importlib.util.spec_from_file_location("original_review_monitor", ORIGIN)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_reason = base.reason
def observe(config: dict, now: float) -> list[str]:
    events = []
    for root in base.campaign_roots(config):
        paths = set(root.glob("*/debug_*/eval_results/*_math.json"))
        paths.update([*root.glob("eval128*/*.json"), *root.glob("eval500*/*.json")])
        for path in sorted(paths):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(data, list) and len(data) in (64, 128, 500):
                sha = base.hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
                events.append(f"{'eval' if len(data) == 64 else 'final'}{len(data)}:{path}:{sha}")
    if config.get("training_log"):
        try:
            lines = Path(config["training_log"]).read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            if len(line) < 1000 and re.search(r"Traceback|\bERROR\b|(Error|Exception):", line):
                events.append("training-error:" + base.hashlib.sha256(line.encode()).hexdigest())
    if config.get("training_pid"):
        try:
            proc = Path(f"/proc/{int(config['training_pid'])}/stat").read_text()
            stat = proc.rsplit(")", 1)[1].split()
            alive = stat[0] != "Z" and str(config.get("training_start_ticks", stat[19])) == stat[19]
        except (OSError, ValueError, IndexError):
            alive = False
        if not alive:
            events.append(f"training-gone:{config['training_pid']}")
    return sorted(set(events))


def reason(state: dict, events: list[str], now: float) -> str | None:
    new = set(events) - set(state.get("seen", []))
    if not state.get("pending") and any(e.startswith(("final", "training-gone:")) for e in new):
        return "training ended or fixed evaluation ready"
    return original_reason(state, events, now)


def message(config: dict, request_id: str, why: str, events: list[str]) -> str:
    ack = shlex.join([config.get("python", "python"), str(Path(__file__).resolve()),
                      "--config", config.get("config_path", "CONFIG.json"), "--ack", request_id])
    request = Path(config["output"]) / "requests" / f"{request_id}.json"
    return (f"[AUTOMATED LLM Dr. GRPO REVIEW; not a new human message]\n{request_id}: {why}.\n"
            f"Continue this exact LLM session. Read {config['checkpoint']} and {request}.\n"
            "Prioritize the advisor deck of at most five slides, ready by 13:00 UTC. "
            "Inspect native evaluations, actual rewards, updates, failures and checkpoint status. "
            "Use the fixed 128/500-question test when ready; no best-checkpoint selection. "
            "Preserve live owners and source; distinguish pipeline validation from learning gains. "
            "Update evidence and handoff; push verified milestones. Preserve 10% account reserve "
            "with 12% dispatch headroom. Acknowledge only after reviewing:\n"
            f'{ack} --decision "Interpretation; next decision" --evidence PATH\n'
            f"Then yield; create STOP in the observer directory if paused. Events: {len(events)}.")
base.observe, base.reason, base.message = observe, reason, message
base.__file__ = __file__
origin_hash = base.hashlib.sha256(ORIGIN.read_bytes()).hexdigest()
base.OBSERVER_ORIGIN = dict(path=str(ORIGIN), sha256=origin_hash)
if __name__ == "__main__":
    parser = base.argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "ack", "decision"):
        parser.add_argument(f"--{flag}", required=flag == "config")
    parser.add_argument("--evidence", action="append", default=[])
    a = parser.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    base.uuid.UUID(cfg["thread"])
    cfg["config_path"] = str(Path(a.config).resolve())
    base.acknowledge(cfg, a.ack, a.decision or "", a.evidence) if a.ack else base.run(cfg)
