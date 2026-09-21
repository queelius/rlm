"""Static dependency checks use exactly the executor's reference syntax."""

import importlib.util
from pathlib import Path


def test_static_scan_finds_self_forward_and_nonpositive_even_in_later_steps():
    path = Path(__file__).with_name("audit_transfer_structure.py")
    assert path.exists(), "transfer structure audit missing"
    spec = importlib.util.spec_from_file_location("transfer_structure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    questions = ["Who is #1?", "Where is #1?", "Compare #4 with #0 and #3; not #12abc."]
    assert module.reference_defects(questions) == [
        {"step": 1, "reference": 1, "kind": "self"},
        {"step": 3, "reference": 4, "kind": "forward"},
        {"step": 3, "reference": 0, "kind": "nonpositive"},
        {"step": 3, "reference": 3, "kind": "self"},
    ]
    assert module.reference_defects(["Who?", "Where is #1?", "Compare #1 and #2."]) == []
