"""Observer fixtures only; never submit a real Codex message or start GPU work."""

import importlib.util
import json
from pathlib import Path

import pytest


def monitor():
    path = Path(__file__).with_name("review_monitor.py")
    assert path.exists(), "CURL review observer is not implemented"
    spec = importlib.util.spec_from_file_location("curl_review_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path):
    return {
        "thread": "01a049e7-2027-7253-97c4-c70906238762",
        "output": str(tmp_path / "reviews"),
        "campaign_root": str(tmp_path / "campaign"),
        "deadline": 10000,
        "checkpoint": str(tmp_path / "SESSION_CHECKPOINT.md"),
        "codex": "/test/codex",
        "python": "/test/python",
        "config_path": str(tmp_path / "CONFIG.json"),
    }


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_only_terminal_queue_receipts_or_manual_records_trigger_events(tmp_path):
    # Treating live metrics/partial JSON as completed evidence would flood reviews.
    module, cfg = monitor(), config(tmp_path)
    root = Path(cfg["campaign_root"])
    save(
        root / "queued" / "result.json",
        {
            "id": "queued",
            "pid": 12,
            "exit_code": 0,
            "complete": True,
            "reason": None,
            "scientific_records": 20,
        },
    )
    manual = root / "manual" / "run" / "metrics.jsonl"
    manual.parent.mkdir(parents=True)
    manual.write_text(json.dumps({"type": "train_metric", "value": 1}) + "\n")
    queued_events = module.observe(cfg, 100)
    assert len(queued_events) == 1
    assert str(root / "queued" / "result.json") in queued_events[0]
    manual.write_text(
        manual.read_text()
        + json.dumps(
            {
                "type": "end",
                "reason": "completed",
                "step": 12500,
                "env_steps": 100000,
            }
        )
        + '\n{"type":'
    )
    events = module.observe(cfg, 110)
    assert len(events) == 2
    assert any(str(manual) in event for event in events)
    assert module.observe(cfg, 120) == events
    failed = root / "failure" / "run" / "metrics.jsonl"
    failed.parent.mkdir(parents=True)
    failed.write_text(
        json.dumps(
            {
                "type": "failure",
                "step": 1001,
                "error": "RuntimeError",
                "message": "fixture",
            }
        )
        + "\n"
    )
    partial = root / "partial" / "result.json"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"exit_code":')
    assert len(module.observe(cfg, 130)) == 3
    # Prefer the authoritative queue receipt to duplicate end records for that job.
    queued_metrics = root / "queued" / "run" / "metrics.jsonl"
    queued_metrics.parent.mkdir(parents=True)
    queued_metrics.write_text(json.dumps({"type": "end", "reason": "completed"}) + "\n")
    assert len(module.observe(cfg, 140)) == 3


def test_additional_campaign_roots_collect_once_and_appear_in_review_payload(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    original, extension = Path(cfg["campaign_root"]), tmp_path / "extension-500k"
    cfg["additional_campaign_roots"] = [str(extension), str(original), str(extension / ".")]
    save(original / "reference" / "result.json", {"exit_code": 0, "complete": True})
    original_end = original / "reference" / "run" / "metrics.jsonl"
    original_end.parent.mkdir(parents=True)
    original_end.write_text(json.dumps({"type": "end", "reason": "completed"}) + "\n")
    extension_end = extension / "extended" / "run" / "metrics.jsonl"
    extension_end.parent.mkdir(parents=True)
    extension_end.write_text(json.dumps({"type": "end", "reason": "completed"}) + "\n")
    events = module.observe(cfg, 100)
    assert len(events) == 2
    assert sum(str(original / "reference" / "result.json") in event for event in events) == 1
    assert sum(str(extension_end) in event for event in events) == 1
    payload = module.message(cfg, "review-fixture", "new terminal results", events)
    assert str(original) in payload
    assert str(extension) in payload
    assert payload.count(str(extension)) == 1


def test_periodic_and_new_terminal_reviews_coalesce_while_pending():
    module = monitor()
    state = {"reviewed_at": 100, "seen": [], "pending": None}
    assert module.reason(state, [], 1299) is None
    assert module.reason(state, [], 1300) == "periodic review"
    assert module.reason(state, ["finished:a"], 399) is None
    assert module.reason(state, ["finished:a"], 400) == "new terminal results"
    state["seen"] = ["finished:a"]
    assert module.reason(state, ["finished:a"], 500) is None
    state["pending"] = {"id": "pending"}
    assert module.reason(state, ["finished:b"], 9000) is None


@pytest.mark.parametrize(
    "report",
    [
        None,
        {},
        {"remaining_percent": 90},
        {"remaining_percent": 12, "checked_epoch": 100},
        {"remaining_percent": 90, "checked_epoch": -900},
        {"remaining_percent": 90, "checked_epoch": 101},
        {"remaining_percent": float("nan"), "checked_epoch": 100},
    ],
)
def test_unknown_stale_future_or_reserve_quota_blocks_dispatch(report):
    assert not monitor().quota_allows(report, 100)


def test_fresh_quota_above_dispatch_headroom_allows_review():
    assert monitor().quota_allows({"remaining_percent": 12.1, "checked_epoch": 90}, 100)


def test_exact_thread_submission_waits_for_evidence_backed_ack(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    state = {"reviewed_at": 100, "seen": [], "pending": None}
    calls = []

    def queue(argv):
        calls.append(argv)
        return "Queued message fixture"

    module.submit(cfg, state, ["finished:a"], 500, "new terminal results", queue)
    assert calls[0][:4] == ["/test/codex", "queue", "--thread", cfg["thread"]]
    assert "--last" not in calls[0]
    request_id = state["pending"]["id"]
    assert state["pending"]["status"] == "queued_not_acknowledged"
    with pytest.raises(ValueError, match="decision"):
        module.acknowledge(cfg, request_id, "", ["result.json"], now=510)
    with pytest.raises(ValueError, match="evidence"):
        module.acknowledge(cfg, request_id, "Endpoint is valid.", [], now=510)
    module.acknowledge(
        cfg, request_id, "Endpoint is valid; run matched control.", ["RESULTS.md"], now=510
    )
    assert module.consume_ack(cfg, state)
    assert state["pending"] is None
    assert state["reviewed_at"] == 510
    assert state["seen"] == ["finished:a"]
    assert module.reason(state, ["finished:a", "finished:b"], 810)


def test_uncertain_admission_remains_pending_across_restart(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    state = {"reviewed_at": 100, "seen": [], "pending": None}

    def queue(_argv):
        raise TimeoutError("fixture: admission unknown")

    module.submit(cfg, state, [], 1400, "periodic review", queue)
    persisted = module.read(Path(cfg["output"]) / "STATE.json")
    assert persisted["pending"]["status"] == "submission_uncertain"
    assert module.reason(persisted, ["finished:b"], 3000) is None


def test_review_payload_has_real_lines_and_executable_ack_command(tmp_path):
    # Literal backslash-n separators can corrupt the command copied into a review.
    module, cfg = monitor(), config(tmp_path)
    payload = module.message(cfg, "review-fixture", "periodic review", [])
    assert len(payload.splitlines()) >= 6
    assert "\\n" not in payload
    ack = next(line for line in payload.splitlines() if line.startswith("/test/python "))
    assert "--ack review-fixture" in ack
    assert f"--config {cfg['config_path']}" in ack


@pytest.mark.parametrize("deadline_passes", [False, True])
def test_stop_or_deadline_during_quota_read_prevents_queue_admission(
    tmp_path, monkeypatch, deadline_passes
):
    module, cfg = monitor(), config(tmp_path)
    cfg["deadline"] = 105
    now, submitted = [100], []
    save(
        Path(cfg["output"]) / "STATE.json",
        {
            "reviewed_at": -2000,
            "seen": [],
            "pending": None,
        },
    )

    def quota(_cfg):
        if deadline_passes:
            now[0] = 106
        else:
            (Path(cfg["output"]) / "STOP").touch()
        return {"remaining_percent": 90, "checked_epoch": now[0]}

    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module, "load_quota", quota)
    monkeypatch.setattr(module, "submit", lambda *args: submitted.append(args))
    module.run(cfg)
    assert submitted == []
    assert module.read(Path(cfg["output"]) / "TERMINAL.json")["reason"] == (
        "allocation_deadline" if deadline_passes else "stop_requested"
    )
