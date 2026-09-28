"""Health receipts must bind real returned calls, not occupied GPU memory."""

import importlib.util
from pathlib import Path


def watcher():
    path = Path(__file__).with_name("watch_research_20260928.py")
    assert path.exists(), "research first-response watcher is not implemented"
    spec = importlib.util.spec_from_file_location("research_watch_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bound_return_is_healthy_but_mismatched_request_is_not():
    w = watcher()
    start = dict(call_id="c0", request_digest="a", started=100)
    call = dict(start, ended=102, available=True, text="{}", output_token_ids=[1, 2])
    assert w.health(start, call, 103)["healthy"] is True
    call["request_digest"] = "other"
    assert w.health(start, call, 103)["healthy"] is False


def test_missing_return_remains_unknown_and_lateness_is_explicit():
    w = watcher()
    start = dict(call_id="c0", request_digest="a", started=100)
    early = w.health(start, None, 140)
    assert early["healthy"] is None and early["late"] is False
    late = w.health(start, None, 191)
    assert late["healthy"] is None and late["late"] is True
