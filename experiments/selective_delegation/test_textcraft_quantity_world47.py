import json
import sys
from types import SimpleNamespace

import textcraft_quantity_world47 as subject


def test_analyze_unpacks_plan_and_reports_public_minus_corrected(monkeypatch, tmp_path) -> None:
    plans = {}
    for arm, score in (("quantity_corrected_original", 0), ("public", 1)):
        directory = tmp_path / arm
        (directory / "episodes").mkdir(parents=True)
        plan = {
            "jobs": [{"task_id": "t", "repeat": 0, "condition": arm}],
            "caveat": "test",
            "model": "fake",
        }
        (directory / "PLAN.json").write_text(json.dumps(plan))
        (directory / "episodes" / "one.json").write_text(
            json.dumps({"task_id": "t", "repeat": 0, "native_score": score})
        )
        plans[arm] = plan
    monkeypatch.setattr(subject, "output", lambda arm: tmp_path / arm)
    multi = SimpleNamespace(checked_world=lambda plan: "world")
    monkeypatch.setattr(subject, "build", lambda arm: (multi, plans[arm], [], {}))
    compared = {}

    class Audit:
        @staticmethod
        def analyze(directory, tokenizer, world):
            return {"paired": 1, "depth_strata": 1, "directory": str(directory), "world": world}

    def compare(jobs, left, right):
        compared.update(left=left, right=right)
        return {"delta": right[("t", 0)]["native_score"] - left[("t", 0)]["native_score"]}

    profiles = SimpleNamespace(audit=Audit, compare=compare)
    monkeypatch.setitem(sys.modules, "analyze_textcraft_profiles", profiles)
    tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: "tokenizer")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    report = tmp_path / "report.json"
    subject.analyze(report)
    result = json.loads(report.read_text())
    assert result["public_minus_quantity_corrected_original"]["delta"] == 1
    assert compared["left"][("t", 0)]["native_score"] == 0
    assert compared["right"][("t", 0)]["native_score"] == 1
