"""CPU-only whole-output audit/contrast regression; no model weights loaded."""

from pathlib import Path

import eval_textcraft_memory as m
from test_eval_textcraft_memory import (
    test_actual_five_call_output_layout_replays_native_with_memory_hook as saved_episode,
)


def test_whole_audit_preserves_missing_and_single_condition(tmp_path, monkeypatch):
    saved_episode(tmp_path)
    plan = m.read(tmp_path / "PLAN.json")
    _, tasks = m.fixed.template_inputs()
    # Explicit temporary dead-owner fixture, not a claimed scientific collection.
    m.c.save(
        tmp_path / "OWNER-fixture.json",
        dict(pid=1073741823, create_time=0, source=str(Path(m.c.__file__).resolve())),
    )
    m.c.save(tmp_path / "TERMINAL-fixture.json", dict(elapsed_seconds=123, failure=None))
    report = tmp_path / "AUDIT.json"
    monkeypatch.setattr(m, "output", lambda policy, mode: tmp_path)
    monkeypatch.setattr(m, "report_path", lambda policy, mode: report)
    monkeypatch.setattr(m, "prepare", lambda policy, mode: (plan, tasks, plan["adapter"]))
    m.audit_arm("public1", "ledger_recent4")
    result = m.read(report)
    group = next(iter(result["groups"].values()))
    assert group["planned"] == 32 and group["observed"] == 1 and group["missing"] == 31
    assert group["won"] == 1 and result["physical_cost"]["calls"] == 5
    assert result["owner_wall_seconds"] == 123
    assert "paired" not in result and "depth_strata" not in result
    assert result["sha256"][str(tmp_path / "PLAN.json")] == m.sha(tmp_path / "PLAN.json")


def test_whole_factorial_reports_recent_minus_full_ledger_benefit(tmp_path, monkeypatch):
    template, _ = m.fixed.template_inputs()
    binding = template["adapter"]
    monkeypatch.setattr(m, "output", lambda policy, mode: tmp_path / mode)
    monkeypatch.setattr(m, "report_path", lambda policy, mode: tmp_path / (mode + ".json"))
    monkeypatch.setattr(m, "endpoint", lambda policy: binding)
    monkeypatch.setattr(m.c, "ROOT", tmp_path)
    # Full history: ledger gives -1. Recent history: ledger gives +1. Interaction +2.
    scores = dict(history_full=1, ledger_full=0, recent4=0, ledger_recent4=1)
    for mode, score in scores.items():
        directory = m.output("public1", mode)
        plan = m.build_plan(template, "public1", mode, binding)
        m.c.save(directory / "PLAN.json", plan)
        hashes = {str(directory / "PLAN.json"): m.sha(directory / "PLAN.json")}
        for job in plan["jobs"]:
            path = directory / "episodes" / (job["episode_id"] + ".json")
            m.c.save(path, {**job, "observed": True, "native_score": score})
            hashes[str(path)] = m.sha(path)
        # Synthetic already-audited outcomes exercise only aggregate arithmetic/provenance.
        m.c.save(
            m.report_path("public1", mode),
            dict(sha256=hashes, groups={}, physical_cost={}, owner_wall_seconds=1),
        )
    m.compare_policy("public1")
    result = m.read(tmp_path / "analysis-textcraft-memory-public1-factorial-001.json")
    assert (
        result["paired_comparisons"]["ledger_full_minus_history_full"]["complete_panel_difference"]
        == -1
    )
    effect = result["ledger_benefit_recent_minus_full"]
    assert effect["complete_panel_difference"] == 2 and effect["ci95"] == [2, 2]
    assert effect["known_pairs"] == 32 and effect["wins"] == 32
