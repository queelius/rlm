"""Focused CPU fixtures; never admit a real queued review or start GPU work."""

import importlib.util
import json
import shlex
from pathlib import Path


def monitor():
    path = Path(__file__).with_name("review_monitor_longer.py")
    assert path.exists(), "Longer-phase observer wrapper is missing"
    spec = importlib.util.spec_from_file_location("longer_review_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path):
    return {
        "campaign_root": str(tmp_path),
        "output": str(tmp_path / "reviews-longer"),
        "checkpoint": "SESSION_CHECKPOINT.md",
        "deadline": 1791621574,
        "thread": "01a049e7-2027-7253-97c4-c70906238762",
        "codex": "/test/codex",
        "python": "/test/python",
        "config_path": str(tmp_path / "longer config.json"),
    }


def test_message_uses_longer_protocol_without_expired_slide_deadline(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    payload = module.message(cfg, "review-fixture", "periodic review", [])
    for required in (
        "4096",
        "Random(42)",
        "math_lvl3to5_8k",
        "math_verify",
        "qwen_math",
        "128 questions",
        "8 responses",
        "32 collections",
        "256 optimizer updates",
        "step_00033",
        "10-hour",
        "1791621574",
        "74.2",
        "author reference",
        "10%",
        "12%",
        "no best-checkpoint",
        "same owner",
    ):
        assert required in payload
    for expired in ("13:00", "14:00", "five slides", "CURL"):
        assert expired not in payload
    assert cfg["checkpoint"] in payload


def test_ack_command_targets_new_wrapper_and_quotes_config_path(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    payload = module.message(cfg, "review-fixture", "new result", [])
    command = next(line for line in payload.splitlines() if line.startswith("/test/python "))
    argv = shlex.split(command)
    assert argv[:2] == ["/test/python", str(Path(module.__file__).resolve())]
    assert argv[argv.index("--config") + 1] == cfg["config_path"]
    assert argv[argv.index("--ack") + 1] == "review-fixture"


def test_native_events_ignore_partial_json_and_trigger_final_immediately(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    native = tmp_path / "larger4096" / "debug_example" / "eval_results"
    native.mkdir(parents=True)
    (native / "8_math.json").write_text(json.dumps([{"score": 0}] * 64))
    (native / "16_math.json").write_text('[{"score":')
    final = tmp_path / "eval500-larger4096-final"
    final.mkdir()
    (final / "results.json").write_text(json.dumps([{"score": 0}] * 500))
    events = module.observe(cfg, 101)
    assert len(events) == 2
    assert any(event.startswith("eval64:") for event in events)
    assert any(event.startswith("final500:") for event in events)
    assert module.reason(dict(reviewed_at=100, seen=[], pending=None), events, 101)


def test_submission_uses_new_context_and_state_then_ack_unblocks(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    old = tmp_path / "reviews"
    old.mkdir()
    (old / "STOP").touch()
    (old / "STATE.json").write_text('{"pending":{"id":"old-review"}}')
    state = dict(reviewed_at=100, seen=[], pending=None)
    module.base.submit(
        cfg,
        state,
        ["eval64:a"],
        500,
        "new native result",
        queue=lambda _argv: "Queued message fixture",
    )
    persisted = module.base.read(Path(cfg["output"]) / "STATE.json")
    pending = persisted["pending"]
    request = module.base.read(Path(cfg["output"]) / "requests" / f"{pending['id']}.json")
    assert "256 optimizer updates" in request["message"]
    assert module.reason(persisted, ["final500:b", "training-gone:123"], 2000) is None
    module.base.acknowledge(cfg, pending["id"], "Continue fixed endpoint", ["receipt.json"], 510)
    assert module.base.consume_ack(cfg, persisted)
    assert persisted["seen"] == ["eval64:a"] and persisted["pending"] is None
    assert json.loads((old / "STATE.json").read_text())["pending"]["id"] == "old-review"
    assert (old / "STOP").exists()


def test_inherited_quota_headroom_and_periodic_review_remain_bounded():
    module = monitor()
    state = dict(reviewed_at=100, seen=[], pending=None)
    assert module.reason(state, ["eval64:a"], 399) is None
    assert module.reason(state, ["eval64:a"], 400)
    assert module.reason(state, [], 1299) is None
    assert module.reason(state, [], 1300) == "periodic review"
    assert not module.base.quota_allows(dict(remaining_percent=12, checked_epoch=100), 100)
    assert module.base.quota_allows(dict(remaining_percent=12.1, checked_epoch=100), 100)
