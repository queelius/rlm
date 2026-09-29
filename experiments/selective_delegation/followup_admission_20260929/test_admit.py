"""Only the new composition boundaries; the existing executor is unchanged."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def implementation():
    path = Path(__file__).with_name("admit.py")
    assert path.exists(), "follow-on composer not implemented"
    spec = importlib.util.spec_from_file_location("followon_composer", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def job(name="a", output="/science/a", pins=None):
    return dict(
        name=name,
        argv=["python", "experiment.py"],
        output=output,
        cap_seconds=60,
        pins=pins or {"source": "abc"},
    )


def test_reviewed_descriptor_changes_only_by_adding_receipt_pin(tmp_path):
    original = job()
    path = tmp_path / "PREPARED.json"
    path.write_text(json.dumps({"jobs": [original]}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = implementation().load_jobs(path, digest)
    assert result == [dict(original, pins={**original["pins"], str(path): digest})]
    assert json.loads(path.read_text())["jobs"] == [original]


def test_rejects_unreviewed_receipt_bytes(tmp_path):
    path = tmp_path / "PREPARED.json"
    path.write_text(json.dumps({"jobs": [job()]}))
    with pytest.raises(ValueError, match="receipt"):
        implementation().load_jobs(path, "0" * 64)


def test_rejects_duplicate_scientific_output():
    with pytest.raises(ValueError, match="output"):
        implementation().combine([[job()], [job("b")]])


def test_rejects_duplicate_job_name_even_with_another_output():
    with pytest.raises(ValueError, match="name"):
        implementation().combine([[job()], [job(output="/science/b")]])


def test_rejects_conflicting_pins_instead_of_silently_overwriting():
    with pytest.raises(ValueError, match="pin"):
        implementation().combine([[job()], [job("b", "/science/b", {"source": "other"})]])


def test_cpu_jobs_without_output_remain_valid():
    audit = job("audit")
    del audit["output"]
    assert implementation().combine([[job()], [audit]]) == [job(), audit]
