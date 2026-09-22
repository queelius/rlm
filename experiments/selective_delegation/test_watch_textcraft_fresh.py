import json

import watch_textcraft_fresh as w


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_owner_requires_terminal_and_authenticated_process_exit(tmp_path):
    put(tmp_path / "OWNER-a.json", {"pid": 17, "create_time": 10})
    assert w.release_state(tmp_path, lambda *_: False) == "owner_unresolved"
    put(tmp_path / "TERMINAL-a.json", {"failure": None})
    assert w.release_state(tmp_path, lambda *_: True) == "owner_unresolved"
    assert w.release_state(tmp_path, lambda *_: False) == "released"


def test_queue_exit_without_owner_is_unknown_not_zero(tmp_path):
    assert w.release_state(tmp_path, lambda *_: False) == "no_owner"
    assert w.queue_resolution(["released", "no_owner"], False) == "unknown_queue_exited"
    assert w.queue_resolution(["released", "no_owner"], True) == "waiting"
    assert w.queue_resolution(["released", "released"], True) == "analyze"


def test_first_response_binds_request_and_records_late_native_return():
    start = {"call_id": "x", "request_digest": "same", "started": 20}
    call = {**start, "ended": 111, "available": True, "text": "{}", "output_token_ids": [123]}
    result = w.response_health(start, call)
    assert result["healthy"] is True
    assert result["returned_within_90_seconds"] is False
    call["request_digest"] = "different"
    assert w.response_health(start, call)["healthy"] is False
