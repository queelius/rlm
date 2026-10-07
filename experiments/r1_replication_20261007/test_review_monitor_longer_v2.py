"""Focused CPU-only fixtures for configurable longer-phase review text."""

import importlib.util
import shlex
from pathlib import Path

import pytest


def monitor():
    path = Path(__file__).with_name("review_monitor_longer_v2.py")
    spec = importlib.util.spec_from_file_location("longer_review_monitor_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path):
    return {
        "campaign_root": str(tmp_path),
        "output": str(tmp_path / "reviews-longer-v2"),
        "checkpoint": "SESSION_CHECKPOINT.md",
        "deadline": 1791621274,
        "python": "/test/python",
        "config_path": str(tmp_path / "v2 config.json"),
        "protocol_summary": (
            "Attempt2: fresh original base; Random(42) selected 4096 math_lvl3to5_8k questions; "
            "2 overlength filtered, 126 dropped by the loader: 3968 effective questions. "
            "qwen_math; full math_verify; 128 questions x 8 responses per collection; "
            "31 collections = 248 optimizer updates and 31744 responses; final step_00032. "
            "10-hour training cap; allocator-only expandable_segments adjustment. "
            "The authenticated attempt2 owner performs both fixed500 final tests. "
            "Attempt1 OOM is preserved: zero completed collections, unknown inner SGD count; "
            "no endpoint or successor evaluation from that attempt."
        ),
    }


def test_configured_protocol_replaces_stale_counts_and_ack_targets_v2(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    payload = module.message(cfg, "review-fixture", "native error", [])
    assert cfg["protocol_summary"] in payload
    for stale in ("32 collections", "256 optimizer", "32768", "step_00033", "13:00", "CURL"):
        assert stale not in payload
    for retained in ("author reference", "74.2%", "no best-checkpoint", "10%", "12%"):
        assert retained in payload
    command = next(line for line in payload.splitlines() if line.startswith("/test/python "))
    argv = shlex.split(command)
    assert argv[1] == str(Path(module.__file__).resolve())
    assert argv[argv.index("--config") + 1] == cfg["config_path"]
    assert module.observe is module.previous.observe
    assert module.reason(dict(reviewed_at=0, seen=[], pending={"id": "pending"}), [], 2000) is None
    cfg["protocol_summary"] = "A later authorized protocol, recorded in its immutable manifest."
    assert cfg["protocol_summary"] in module.message(cfg, "review-next", "periodic", [])
    assert "3968" not in module.message(cfg, "review-next", "periodic", [])


@pytest.mark.parametrize("protocol", [None, "", "   ", 31])
def test_missing_or_invalid_protocol_fails_before_dispatch(tmp_path, protocol):
    module, cfg = monitor(), config(tmp_path)
    cfg["protocol_summary"] = protocol
    with pytest.raises(ValueError, match="protocol_summary"):
        module.validate_config(cfg)
