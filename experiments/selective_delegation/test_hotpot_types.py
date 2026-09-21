"""Exact official-type join and paired protocol accounting."""

import importlib.util
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "hotpot_types", Path(__file__).with_name("analyze_hotpot_types.py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_type_join_requires_exact_source_id_question_and_unique_inventory():
    m = module()
    cases = [{"id": "p", "question": "Exactly?", "metadata": {"source_id": "native"}}]
    mirror = [{"id": "native", "question": "Exactly?", "type": "comparison"}]
    assert m.join_types(cases, mirror) == {"p": "comparison"}
    with pytest.raises(ValueError, match="question"):
        m.join_types(cases, [{**mirror[0], "question": "exactly?"}])
    with pytest.raises(ValueError, match="duplicate"):
        m.join_types(cases, mirror + mirror)


def test_official_yes_no_and_protocol_only_pairs():
    m = module()
    a = m.outcome({"available": True, "text": '{"answer":"yes indeed"}'}, "yes", "scored")
    assert a["valid"] and a["em"] == a["f1"] == 0
    wrong = dict(em=0.0, f1=0.0, valid=False, unobserved=False, status="invalid_helper")
    correct = dict(em=1.0, f1=1.0, valid=True, unobserved=False, status="scored")
    values = {
        (p, r, c): (wrong if c == "sft" else correct)
        for p in ("p", "q")
        for r in range(2)
        for c in m.POLICIES
    }
    result = m.summarize(["p", "q"], values, draws=25)
    pair = result["comparisons"]["base_minus_sft"]
    assert pair["em"]["estimate"] == 1
    assert pair["wins"]["categories"] == {"protocol_involved": 4}
    assert pair["losses"]["episodes"] == 0
    assert result["groups"]["sft"]["planned"] == 4


def test_unavailable_is_not_a_protocol_win():
    m = module()
    missing = dict(em=0.0, f1=0.0, valid=False, unobserved=True, status="final_unavailable")
    correct = dict(em=1.0, f1=1.0, valid=True, unobserved=False, status="scored")
    values = {
        ("p", r, c): (missing if c == "sft" else correct) for r in range(2) for c in m.POLICIES
    }
    result = m.summarize(["p"], values, draws=10)
    assert result["comparisons"]["base_minus_sft"]["wins"]["categories"] == {
        "unobserved_involved": 2
    }
    assert result["groups"]["sft"]["unobserved"] == 2
