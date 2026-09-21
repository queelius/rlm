import copy

import pytest


def test_actual_baseline_request_accepts_only_declared_sampling_package():
    import json
    from pathlib import Path

    import analyze_qampari_sampling as m
    from transformers import AutoTokenizer

    root = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
    plan = json.loads((root / "qampari-reading-001/PLAN.json").read_text())
    case = json.loads(Path(plan["cases"]).read_text().splitlines()[0])
    job = next(
        j for j in plan["jobs"] if j["case_id"] == case["id"] and j["condition"] == "direct200"
    )
    spec = m.original.requests(case, job)[0]
    row = json.loads((root / "qampari-reading-001/calls" / (spec["call_id"] + ".json")).read_text())
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True)
    # CPU fixture substitutes only the declared package; not a new scientific response.
    changed = copy.deepcopy(row)
    changed["request"]["sampling"].update(m.control.SAMPLING)
    changed["request_digest"] = m.original.probe.runtime.digest(changed["request"])
    result = m.audit(changed, spec, plan, row["plan_sha256"], row["input_token_ids"], tokenizer)
    assert result["stop"] in ("eos", "at_output_cap")
    with pytest.raises(ValueError, match="request"):
        m.audit(row, spec, plan, row["plan_sha256"], row["input_token_ids"], tokenizer)
    changed["request"]["seed"] += 1
    with pytest.raises(ValueError, match="request"):
        m.audit(changed, spec, plan, row["plan_sha256"], row["input_token_ids"], tokenizer)


def test_pair_categories_do_not_turn_missing_into_scientific_losses():
    import analyze_qampari_sampling as m

    good = {"observed": True, "valid": True, "metrics": {"f1": 1}}
    bad = {"observed": True, "valid": False, "metrics": {"f1": 0}}
    missing = {"observed": False, "valid": False, "metrics": {"f1": 0}}
    assert m.change(good, bad) == {"category": "protocol_involved", "f1_delta": -1}
    assert m.change(good, missing) == {"category": "unobserved_involved", "f1_delta": None}


def test_unresolved_start_preserves_unknown_latency_and_token_accounting():
    import analyze_qampari_sampling as m

    cost = m.measured_including_unresolved([{"available": False, "started": 10, "usage": {}}])
    assert cost["calls"] == 1 and cost["unknown_usage_calls"] == 1
    assert cost["unknown_latency_calls"] == 1
