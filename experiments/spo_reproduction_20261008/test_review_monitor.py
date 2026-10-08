"""CPU-only SPO observer fixtures; never submit to the real Codex queue."""

import importlib.util
import json
from pathlib import Path

import pytest


def monitor():
    path = Path(__file__).with_name("review_monitor.py")
    assert path.exists(), "SPO observer adapter is missing"
    spec = importlib.util.spec_from_file_location("spo_review_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path):
    return dict(
        campaign_root=str(tmp_path),
        output=str(tmp_path / "reviews"),
        checkpoint=str(tmp_path / "SESSION_CHECKPOINT.md"),
        queue=str(tmp_path / "QUEUE.json"),
        deadline=1791621274,
        thread="01a049e7-2027-7253-97c4-c70906238762",
        codex="/test/codex",
        python="/test/python",
        config_path=str(tmp_path / "config with spaces.json"),
        protocol_summary=(
            "Exact SPO-chain Rho-1.1B/GSM8K protocol audit: author 56.7 at checkpoint 690 of "
            "1000 scheduled iterations; selection unclear."
        ),
    )


def test_terminal_receipts_accept_failures_and_ignore_partial_or_invalid_records(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    for name, record in (
        ("success", dict(exit_code=0, complete=True)),
        ("failed", dict(exit_code=1, complete=False)),
        ("live", dict(complete=False)),
        ("invalid", dict(exit_code=True, complete=True)),
    ):
        run = tmp_path / name
        run.mkdir()
        (run / "result.json").write_text(json.dumps(record))
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "result.json").write_text('{"exit_code":')
    events = module.observe(cfg, 100)
    assert len(events) == 2
    assert any(f"finished:{tmp_path / 'success' / 'result.json'}:" in e for e in events)
    assert any(f"finished:{tmp_path / 'failed' / 'result.json'}:" in e for e in events)
    assert module.observe(cfg, 101) == events


def test_native_terminal_events_skip_live_updates_and_receipted_duplicates(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    native = tmp_path / "train" / "run"
    native.mkdir(parents=True)
    events_path = native / "events.jsonl"
    events_path.write_text(
        '{"type":"update","iteration":1,"reward":0.3}\n{"type":"end","iteration":1000}\n{"type":'
    )
    events = module.observe(cfg, 100)
    assert len(events) == 1 and events[0].startswith(f"finished:{events_path}:")
    (native.parent / "result.json").write_text('{"exit_code":0,"complete":true}')
    events = module.observe(cfg, 101)
    assert len(events) == 1 and events[0].startswith(f"finished:{native.parent / 'result.json'}:")
    events_path.write_text('{"type":"update","iteration":1,"reward":0.3}\n')
    (native.parent / "result.json").unlink()
    assert module.observe(cfg, 102) == []


@pytest.mark.parametrize("record", ['{"metrics":{"correct_frac":0.567}}', '{"correct_frac":0.567}'])
def test_native_evaluation_metrics_trigger_without_terminal_and_skip_log_copies(tmp_path, record):
    module, cfg = monitor(), config(tmp_path)
    analyzer = (
        tmp_path
        / "attempt"
        / "native"
        / "evaluation"
        / "iteration__690"
        / "analysis"
        / "TaskPerformanceAnalyzer"
        / "gsm8k"
    )
    analyzer.mkdir(parents=True)
    (analyzer / "log.json").write_text(record)
    (analyzer / "log_1.json").write_text('{"metrics":{"correct_frac":0.567}}')
    empty = analyzer.parent / "empty"
    empty.mkdir()
    (empty / "log.json").write_text('{"metrics":{}}')
    partial = analyzer.parent / "partial"
    partial.mkdir()
    (partial / "log.json").write_text('{"metrics":')
    events = module.observe(cfg, 100)
    assert len(events) == 1
    assert events[0].startswith(f"evaluation:{analyzer / 'log.json'}:")
    assert module.observe(cfg, 101) == events
    (analyzer / "log.json").write_text('{"metrics":{"correct_frac":0.57}}')
    assert module.observe(cfg, 102) != events


@pytest.mark.parametrize(
    "events,now,want",
    [
        (["finished:new"], 699, None),
        (["finished:new"], 700, "new terminal results"),
        ([], 3699, None),
        ([], 3700, "periodic review"),
        (["finished:seen"], 700, None),
    ],
)
def test_hourly_periodic_and_ten_minute_terminal_debounce(events, now, want):
    module = monitor()
    state = dict(reviewed_at=100, seen=["finished:seen"], pending=None)
    assert module.reason(state, events, now) == want


def test_exact_thread_request_pending_ack_and_new_results(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    state = dict(reviewed_at=100, seen=[], pending=None)
    captured = []

    def queue(argv):
        captured.append(argv)
        persisted = module.base.read(Path(cfg["output"]) / "STATE.json")
        assert persisted["pending"]["status"] == "submission_uncertain"
        return "Queued message fixture"

    module.base.submit(cfg, state, ["finished:one"], 700, "new terminal results", queue=queue)
    assert captured[0][:4] == [
        "/test/codex",
        "queue",
        "--thread",
        "01a049e7-2027-7253-97c4-c70906238762",
    ]
    pending = module.base.read(Path(cfg["output"]) / "STATE.json")
    assert pending["pending"]["status"] == "queued_not_acknowledged"
    assert module.reason(pending, ["finished:one", "finished:two"], 5000) is None
    request_id = pending["pending"]["id"]
    with pytest.raises(ValueError, match="match the pending"):
        module.base.acknowledge(cfg, "wrong", "reviewed", ["evidence"], now=750)
    module.base.acknowledge(cfg, request_id, "Observed failure; repair", ["evidence"], now=750)
    assert module.base.consume_ack(cfg, pending)
    assert pending["seen"] == ["finished:one"]
    assert module.reason(pending, ["finished:one"], 1350) is None
    assert module.reason(pending, ["finished:one", "finished:two"], 1350)


def test_ambiguous_queue_exit_keeps_pending_and_quota_blocks_reserve(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    state = dict(reviewed_at=100, seen=[], pending=None)

    def uncertain(_argv):
        raise RuntimeError("admission uncertain")

    module.base.submit(cfg, state, [], 700, "periodic review", queue=uncertain)
    persisted = module.base.read(Path(cfg["output"]) / "STATE.json")
    assert persisted["pending"]["status"] == "submission_uncertain"
    assert module.reason(persisted, ["finished:two"], 5000) is None
    assert not module.base.quota_allows(dict(remaining_percent=12, checked_epoch=700), 700)
    assert module.base.quota_allows(dict(remaining_percent=13, checked_epoch=700), 700)
    assert not module.base.quota_allows(dict(remaining_percent=13, checked_epoch=700), 1601)


def test_request_message_carries_spo_protocol_queue_and_quoted_ack_command(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    message = module.message(cfg, "review-fixture", "periodic review", [])
    assert "SPO" in message and "Rho-1.1B" in message and "GSM8K" in message
    assert cfg["protocol_summary"] in message
    assert cfg["checkpoint"] in message and cfg["queue"] in message
    assert "best-test checkpoint" in message
    assert "10%" in message and "12%" in message
    assert "1791621274" in message
    assert "CURL" not in message and "Dr. GRPO" not in message
    ack = next(line for line in message.splitlines() if line.startswith("/test/python "))
    import shlex

    argv = shlex.split(ack)
    assert argv[:6] == [
        "/test/python",
        str(Path(module.__file__).resolve()),
        "--config",
        cfg["config_path"],
        "--ack",
        "review-fixture",
    ]


def test_review_message_requires_protocol_summary(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    cfg.pop("protocol_summary")
    with pytest.raises(KeyError, match="protocol_summary"):
        module.message(cfg, "review-fixture", "periodic review", [])


def test_expired_deadline_exits_without_dispatch_and_records_actual_schedule(tmp_path):
    module, cfg = monitor(), config(tmp_path)
    cfg["deadline"] = 1
    module.base.run(cfg)
    output = Path(cfg["output"])
    assert module.base.read(output / "TERMINAL.json")["reason"] == "allocation_deadline"
    assert not (output / "requests").exists()
    launch = module.base.read(output / "LAUNCH.json")
    assert launch["review_interval_seconds"] == 3600
    assert launch["terminal_min_interval_seconds"] == 600
