"""A new receipt needs monitoring without modifying the live September28 watcher."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def implementation():
    path = Path(__file__).with_name("watch.py")
    assert path.exists(), "follow-on watcher wrapper not implemented"
    spec = importlib.util.spec_from_file_location("followon_watch", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_explicit_accepted_receipt_is_selected_without_old_date_glob(tmp_path):
    path = tmp_path / "evidence-followups-20260929-001.json"
    path.write_text(json.dumps(dict(status="accepted", jobs=[dict(name="x")])))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert implementation().checked_receipt(path, digest) == path.resolve()


def test_unreviewed_receipt_is_rejected(tmp_path):
    path = tmp_path / "ACCEPTED.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="receipt"):
        implementation().checked_receipt(path, "0" * 64)
