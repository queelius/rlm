"""Follow-up insertion must conserve accepted jobs and precede dependent readouts."""

import copy
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("extend_research_tail_20260928.py")
    spec = importlib.util.spec_from_file_location("extended_research_tail", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def job(name, output=True):
    result = dict(name=name, argv=["python", name], cap_seconds=120, pins={"original": "hash"})
    if output:
        result["output"] = "/runs/" + name
    return result


def fixture():
    return dict(
        inherited=[
            job("phi-readout"),
            job("alf-audit", False),
            job("fresh-raw-rl-collect-0001"),
            job("fresh-train"),
            job("fresh-raw-warm-readout-0001"),
            job("fresh-binder-warm-readout-0001"),
            job("fresh-trained-readout"),
            job("compact-compare", False),
        ],
        mask=[job("payload-train"), job("payload-compare", False)],
        teaching=[job("replicate-train"), job("replicate-readout")],
        transfer=[job("transfer-readout"), job("transfer-compare", False)],
    )


def test_inserted_transfer_has_warm_controls_without_duplicate_original_runs():
    values = fixture()
    original = copy.deepcopy(values)
    jobs = module().extend_tail(**values)
    assert [j["name"] for j in jobs] == [
        "phi-readout",
        "alf-audit",
        "replicate-train",
        "replicate-readout",
        "fresh-raw-warm-readout-0001",
        "fresh-binder-warm-readout-0001",
        "transfer-readout",
        "transfer-compare",
        "fresh-raw-rl-collect-0001",
        "fresh-train",
        "fresh-trained-readout",
        "compact-compare",
        "payload-train",
        "payload-compare",
    ]
    assert values == original
    for group in original.values():
        for expected in group:
            assert [j for j in jobs if j["name"] == expected["name"]] == [expected]


def test_duplicate_new_scientific_output_is_rejected():
    values = fixture()
    values["transfer"][0]["output"] = values["inherited"][0]["output"]
    with pytest.raises(ValueError, match="duplicate scientific output"):
        module().extend_tail(**values)


def test_missing_warm_control_is_not_silently_invented():
    values = fixture()
    values["inherited"].pop(4)
    with pytest.raises(ValueError, match="warm controls"):
        module().extend_tail(**values)


def test_duplicate_job_name_is_rejected_even_without_scientific_output():
    values = fixture()
    values["transfer"][1]["name"] = "alf-audit"
    with pytest.raises(ValueError, match="duplicate job name"):
        module().extend_tail(**values)


def test_new_provenance_cannot_replace_an_original_source_pin():
    with pytest.raises(ValueError, match="changed inherited pin"):
        module().augment_pins([job("original-job")], {"original": "different-hash"})
    original = job("original-job")
    combined = module().augment_pins([original], {"orchestration": "new-hash"})
    assert combined[0]["pins"] == {"original": "hash", "orchestration": "new-hash"}
    assert original["pins"] == {"original": "hash"}
