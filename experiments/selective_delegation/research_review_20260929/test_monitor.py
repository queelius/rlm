"""Focused operational tests; no real Codex submissions or GPU calls."""

import json
from pathlib import Path

import pytest

from experiments.selective_delegation.research_review_20260929 import monitor


def config(tmp_path):
    return {
        "thread": "01a049e7-2027-7253-97c4-c70906238762",
        "output": str(tmp_path / "review"),
        "queues": [str(tmp_path / "queue" / "ACCEPTED.json")],
        "deadline": 10000,
        "checkpoint": str(tmp_path / "SESSION_CHECKPOINT.md"),
        "codex": "/test/codex",
    }


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_timer_and_events_coalesce_and_pending_waits():
    state = {"reviewed_at": 100, "seen": [], "pending": None}
    assert monitor.reason(state, [], 1299) is None
    assert monitor.reason(state, [], 1300) == "periodic review"
    assert monitor.reason(state, ["finished:a"], 399) is None
    assert monitor.reason(state, ["finished:a"], 400) == "new results or service alerts"
    state["pending"] = {"id": "request-1"}
    assert monitor.reason(state, ["finished:a"], 9000) is None


def test_quota_unknown_stale_or_low_never_submits():
    for report in [
        None,
        {},
        {"remaining_percent": 90},
        {"remaining_percent": 12, "checked_epoch": 100},
        {"remaining_percent": 90, "checked_epoch": -900},
    ]:
        assert not monitor.quota_allows(report, 100)
    assert monitor.quota_allows({"remaining_percent": 73, "checked_epoch": 90}, 100)


def test_stage_events_ignore_incomplete_json_and_detect_service_failures(tmp_path):
    cfg = config(tmp_path)
    receipt = Path(cfg["queues"][0])
    output = tmp_path / "science"
    save(receipt, {"status": "accepted", "jobs": [{"name": "collect", "output": str(output)}]})
    save(output / "STATUS.json", {"returned": 5, "failed": 0, "updated": 100})
    assert monitor.observe(cfg, 110) == []
    save(output / "STATUS.json", {"returned": 0, "failed": 20, "updated": 110})
    assert any("service" in e for e in monitor.observe(cfg, 120))
    terminal = receipt.parent / "queue" / "collect.json"
    save(terminal, {"returncode": 0, "ended": 130})
    assert any("finished" in e for e in monitor.observe(cfg, 140))
    terminal.write_text('{"returncode":')
    assert not any("finished" in e for e in monitor.observe(cfg, 140))


def test_submission_records_exact_thread_and_waits_for_model_ack(tmp_path):
    cfg = config(tmp_path)
    state = {"reviewed_at": 100, "seen": [], "pending": None}
    calls = []

    def queue(argv):
        calls.append(argv)
        return "Queued message native-id"

    monitor.submit(cfg, state, ["finished:a"], 500, "new results", queue)
    assert calls[0][:4] == ["/test/codex", "queue", "--thread", cfg["thread"]]
    assert "--last" not in calls[0]
    assert state["pending"]["status"] == "queued_not_acknowledged"
    request_id = state["pending"]["id"]
    with pytest.raises(ValueError, match="decision"):
        monitor.acknowledge(cfg, request_id, "", [], now=510)
    monitor.acknowledge(
        cfg,
        request_id,
        "Transfer is limited; inspect first fresh-A batch.",
        ["COMPARISON.json"],
        now=510,
    )
    assert monitor.consume_ack(cfg, state)
    assert state["pending"] is None
    assert state["reviewed_at"] == 510
    assert state["seen"] == ["finished:a"]
    assert monitor.reason(state, ["finished:a", "finished:b"], 811)


def test_ambiguous_queue_failure_remains_pending_across_restart(tmp_path):
    cfg = config(tmp_path)
    state = {"reviewed_at": 100, "seen": [], "pending": None}

    def fail(_argv):
        raise TimeoutError("acceptance unknown")

    monitor.submit(cfg, state, [], 1400, "periodic", fail)
    persisted = monitor.read(Path(cfg["output"]) / "STATE.json")
    assert persisted["pending"]["status"] == "submission_uncertain"
    assert monitor.reason(persisted, ["new"], 3000) is None


def test_acknowledgement_must_match_pending_request(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="pending"):
        monitor.acknowledge(cfg, "not-the-request", "A real decision.", [], now=100)


def test_service_alert_is_an_explicit_urgent_exception_to_coalescing():
    state = {"reviewed_at": 100, "seen": [], "pending": None}
    assert monitor.reason(state, ["service:no_returns"], 101).startswith("service alert")
    state["seen"] = ["service:no_returns"]
    assert monitor.reason(state, ["service:no_returns"], 102) is None


@pytest.mark.parametrize("deadline_passes", [False, True])
def test_stop_or_deadline_during_quota_sampling_prevents_admission(
    tmp_path, monkeypatch, deadline_passes
):
    cfg = config(tmp_path)
    cfg["deadline"] = 105
    now = [100]
    submitted = []
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
        return {"remaining_percent": 73, "checked_epoch": now[0]}

    monkeypatch.setattr(monitor.time, "time", lambda: now[0])
    monkeypatch.setattr(monitor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(monitor, "load_quota", quota)
    monkeypatch.setattr(monitor, "submit", lambda *args: submitted.append(args))
    monitor.run(cfg)
    assert submitted == []


def test_model_review_requires_evidence_pointer(tmp_path):
    cfg = config(tmp_path)
    save(Path(cfg["output"]) / "STATE.json", {"pending": {"id": "pending-1"}})
    with pytest.raises(ValueError, match="evidence"):
        monitor.acknowledge(
            cfg, "pending-1", "A review without evidence is incomplete.", [], now=100
        )
