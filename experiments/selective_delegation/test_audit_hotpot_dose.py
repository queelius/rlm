"""Dose readout must use strict official Hotpot scoring, not receipt scores."""

import importlib.util
from pathlib import Path


def test_official_boolean_scoring_and_missing_vs_invalid_are_distinct():
    path = Path(__file__).with_name("audit_hotpot_dose.py")
    assert path.exists(), "dose audit missing"
    spec = importlib.util.spec_from_file_location("dose_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    valid = module.outcome({"available": True, "text": '{"answer":"yes perhaps"}'}, "yes", "scored")
    assert valid == {
        "em": 0.0,
        "f1": 0.0,
        "valid": True,
        "unobserved": False,
        "answer": "yes perhaps",
        "status": "scored",
    }
    invalid = module.outcome({"available": True, "text": "yes"}, "yes", "scored")
    assert invalid["valid"] is False and invalid["unobserved"] is False
    missing = module.outcome(None, "yes", "missing_episode")
    assert missing["valid"] is False and missing["unobserved"] is True
