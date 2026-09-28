"""Changing waiting order must not change, duplicate or drop scientific jobs."""

import copy
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("reprioritize_tail_20260928.py")
    spec = importlib.util.spec_from_file_location("priority_tail", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def job(name, output=True):
    result = dict(name=name, argv=["python", name], cap_seconds=120, pins={"source": "hash"})
    if output:
        result["output"] = "/runs/" + name
    return result


def blocks():
    screens = [
        job("flexible-decomposition-" + mode + suffix, output=not suffix)
        for mode in ("flat", "fixed", "adaptive")
        for suffix in ("", "-audit")
    ]
    return dict(
        compact=screens + [job("compact-collect"), job("compact-train")],
        transfer=[job("phi-train"), job("phi-readout")],
        alf=[job("alf-train"), job("alf-readout")],
        fresh=[job("fresh-collect"), job("fresh-train")],
        reward=[job("cost-train")],
    )


def test_priority_preserves_jobs_and_block_dependencies():
    values = blocks()
    before = copy.deepcopy(values)
    ordered = module().prioritize(**values)
    assert values == before
    expected = (
        values["compact"][:6]
        + values["transfer"]
        + values["alf"]
        + values["fresh"]
        + values["reward"]
        + values["compact"][6:]
    )
    assert ordered == expected
    assert len(ordered) == sum(map(len, values.values()))


def test_duplicate_scientific_output_is_rejected():
    values = blocks()
    values["alf"][0]["output"] = values["transfer"][0]["output"]
    with pytest.raises(ValueError, match="duplicate scientific output"):
        module().prioritize(**values)


def test_missing_screen_is_rejected():
    values = blocks()
    values["compact"].pop(0)
    with pytest.raises(ValueError, match="six screen descriptors"):
        module().prioritize(**values)
