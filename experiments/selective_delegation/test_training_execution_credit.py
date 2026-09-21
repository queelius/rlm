"""CPU source-schema binding and no-generation reconstruction fixtures."""

import pytest


@pytest.mark.parametrize("kind", ["rl", "frozen"])
def test_both_native_sampling_schemas_require_exact_base_final(kind):
    import training_execution_credit as mod

    sampling = {
        "temperature": 0.5,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 128,
        "max_time": 90.0,
    }
    request = {
        "prompt": "PUBLIC",
        "input_token_ids": [1, 2],
        "seed": 77,
        "role": "final",
        "model": "base",
        "adapter_enabled": False,
        "adapter_sha256": None,
    }
    request.update(sampling if kind == "rl" else {"sampling": {"do_sample": True, **sampling}})
    call = {
        "request": request,
        "request_digest": mod.probe.runtime.digest(request),
        "input_token_ids": [1, 2],
        "output_token_ids": [3],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        "available": True,
    }
    mod.validate_source_call(call, "PUBLIC", "final", 77, 128, None, "base", kind)
    with pytest.raises(ValueError, match="source"):
        mod.validate_source_call(call, "PUBLIC", "final", 78, 128, None, "base", kind)
    with pytest.raises(ValueError, match="source"):
        mod.validate_source_call(call, "GOLD", "final", 77, 128, None, "base", kind)


def test_actual_saved_rl_and_frozen_final_requests_reconstruct_without_generation():
    import training_execution_credit as mod

    root = mod.ROOT
    if not (root / "frozen-execution-001/PLAN.json").exists():
        pytest.skip("external completed native receipts not present")
    cases = {
        c["id"]: c
        for c in map(mod.json.loads, (root / "inputs-001/cases.jsonl").read_text().splitlines())
    }
    source_plan = mod.runtime.read(root / "rl-fullpass-001/PLAN.json")
    case_id = sorted(source_plan["case_ids_by_update"][0])[0]
    for setting in (0, 1):
        source = root / ("rl-fullpass-001/batch-0001" if setting == 0 else "frozen-execution-001")
        identity = f"u01-{case_id}-c0" if setting == 0 else f"{case_id}-c0-e0"
        episode = mod.runtime.read(source / "episodes" / (identity + ".json"))
        calls = {
            cid: mod.runtime.read(source / "calls" / (cid + ".json")) for cid in episode["call_ids"]
        }
        final, prompt, seed = mod.reconstruct_source(
            episode,
            calls,
            cases[case_id],
            source_plan,
            source_plan["case_ids_by_update"][0].index(case_id),
            setting,
        )
        assert final["request"]["prompt"] == prompt and final["request"]["seed"] == seed
        assert mod.runtime.plan_only_prompt(cases[case_id], episode["plan"]) != prompt


def test_saved_native_final_replay_routes_only_base_final(tmp_path):
    import copy

    import training_execution_credit as mod

    if not (mod.ROOT / "frozen-execution-001/PLAN.json").exists():
        pytest.skip("external completed receipts unavailable")
    _, cases, jobs = mod.prepare(mod.ROOT, tmp_path, 1 / 3)
    job = next(j for j in jobs if j["control"])

    class SavedClient:
        output = tmp_path

        def call(self, identity, prompt, condition, role, seed, *, max_new_tokens):
            assert (condition, role, max_new_tokens) == ("base", "final", 128)
            assert prompt == job["factual"]["request"]["prompt"]
            assert seed == job["factual"]["request"]["seed"]
            call = copy.deepcopy(job["factual"])
            call["call_id"] = identity
            return call

    result = mod.runtime.collect(SavedClient(), job, cases[job["case_id"]], "factual_replay")
    assert result["same_output_tokens"] and result["same_text"] and result["same_grade"]
