"""CPU-only observer contract fixtures; no real queue submissions."""

import importlib.util
import json
import os
from pathlib import Path


def monitor():
    path = Path(__file__).with_name("review_monitor.py")
    assert path.exists(), "R1 observer wrapper is missing"
    spec = importlib.util.spec_from_file_location("r1_review_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path):
    return dict(campaign_root=str(tmp_path), output=str(tmp_path / "reviews"),
                checkpoint="checkpoint.md", deadline=2000, thread="fixture-thread",
                codex="/test/codex", python="/test/python", config_path="config.json")


def test_complete_evaluations_ignore_partial_json_and_wrong_counts(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    native = tmp_path / "train" / "debug_a" / "eval_results"
    native.mkdir(parents=True)
    (native / "0_math.json").write_text(json.dumps([{"score": 0}] * 64))
    (native / "8_math.json").write_text('[{"score":')
    final = tmp_path / "eval128-final"
    final.mkdir()
    (final / "result.json").write_text(json.dumps([{"score": 0}] * 128))
    (final / "partial.json").write_text(json.dumps([{"score": 0}] * 127))
    larger = tmp_path / "eval500-final"
    larger.mkdir()
    (larger / "result.json").write_text(json.dumps([{"score": 0}] * 500))
    events = module.observe(cfg, 100)
    assert len(events) == 3
    assert any(event.startswith("eval64:") for event in events)
    assert any(event.startswith("final128:") for event in events)
    assert any(event.startswith("final500:") for event in events)
    assert module.observe(cfg, 101) == events


def test_live_training_pid_and_errors_have_stable_events(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    cfg.update(training_pid=os.getpid(), training_log=str(tmp_path / "training.log"))
    Path(cfg["training_log"]).write_text("step ok\nTraceback (most recent call last):\n")
    events = module.observe(cfg, 100)
    assert len(events) == 1 and events[0].startswith("training-error:")
    Path(cfg["training_log"]).write_text("step ok\nRuntimeError: stopped\n")
    assert any(e.startswith("training-error:") for e in module.observe(cfg, 101))
    Path(cfg["training_log"]).write_text("warning: expected error count is zero\n")
    assert module.observe(cfg, 102) == []
    cfg["training_pid"] = 999999999
    assert any(event.startswith("training-gone:") for event in module.observe(cfg, 101))


def test_event_debounce_urgent_completion_and_periodic_fallback():
    module = monitor()
    state = dict(reviewed_at=100, seen=[], pending=None)
    assert module.reason(state, ["eval64:a"], 399) is None
    assert module.reason(state, ["eval64:a"], 400)
    assert module.reason(state, [], 1299) is None
    assert module.reason(state, [], 1300) == "periodic review"
    assert module.reason(state, ["final128:a"], 101)
    assert module.reason(state, ["final500:a"], 101)
    assert module.reason(state, ["training-gone:123"], 101)


def test_review_message_targets_llm_slides_and_wrapper_ack(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    message = module.message(cfg, "review-fixture", "periodic review", [])
    assert "CURL" not in message and "Dr. GRPO" in message
    assert "13:00 UTC" in message and "five" in message and "128" in message
    ack = next(line for line in message.splitlines() if line.startswith("/test/python "))
    assert str(Path(module.__file__).resolve()) in ack
    assert "--ack review-fixture" in ack and "--config config.json" in ack


def test_pending_submission_blocks_duplicate_even_for_urgent_event(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    state = dict(reviewed_at=100, seen=[], pending=None)
    module.base.submit(cfg, state, ["eval64:a"], 500, "new results",
                       queue=lambda _argv: "Queued message fixture")
    persisted = module.base.read(Path(cfg["output"]) / "STATE.json")
    assert persisted["pending"]["status"] == "queued_not_acknowledged"
    assert module.reason(persisted, ["final128:b", "training-gone:123"], 2000) is None
    assert not module.base.quota_allows(dict(remaining_percent=12, checked_epoch=500), 500)
