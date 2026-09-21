"""Final-only checkpoint ablation changes precisely the named state fields."""

import importlib.util
import json
from pathlib import Path

import probe
from test_plan_probe import case_and_state


def implementation():
    path = Path(__file__).with_name("checkpoint_replay.py")
    assert path.exists(), "checkpoint replay implementation missing"
    spec = importlib.util.spec_from_file_location("checkpoint_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_requested_checkpoint_fields_change_and_helper_is_identical():
    mod = implementation()
    case, state = case_and_state()
    report = '[{"question":"actual subquestion","answer":"1954"}]'
    original = probe.final_prompt(case, state, report, answer_style="span")
    prefix, text = original.rsplit("\n", 1)
    body = json.loads(text)
    for condition in mod.CONDITIONS:
        changed = mod.changed_prompt(original, condition)
        new_prefix, new_text = changed.rsplit("\n", 1)
        assert new_prefix == prefix
        new = json.loads(new_text)
        expected = dict(body)
        if condition == "remove_answer":
            expected["initial_attempt"] = {k: v for k, v in state.items() if k != "answer"}
        else:
            del expected["initial_attempt"]
        assert new == expected
        assert new["helper_report"] == report
        assert all(
            secret not in changed
            for secret in ["SECRET_GOLD", "SECRET_ALIAS", "SECRET_SOURCE", "SECRET_STEP_ONE"]
        )


def test_isolated_report_reconstructed_from_actual_calls_not_host_reference():
    mod = implementation()
    episode = {
        "arm": "reference_isolated",
        "helper_trace": [
            {"question": "Find company", "answer": "PREDICTED"},
            {"question": "When did PREDICTED end?", "answer": "1954"},
        ],
    }
    records = [
        {"text": '{"answer":"PREDICTED"}', "prompt": 'Instruction\n{"question":"Find company"}'},
        {
            "text": '{"answer":"1954"}',
            "prompt": 'Instruction\n{"question":"When did PREDICTED end?"}',
        },
        {"text": '{"answer":"1932"}'},
    ]
    assert json.loads(mod.saved_report(episode, records)) == episode["helper_trace"]


def test_new_final_only_cost_excludes_old_final_and_preserves_sampling(tmp_path):
    mod = implementation()
    case, state = case_and_state()
    initial = {
        "call_id": "checkpoint",
        "available": True,
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    helper = {
        "call_id": "helper",
        "available": True,
        "usage": {"prompt_tokens": 20, "completion_tokens": 3},
    }
    old = {
        "call_id": "old-final",
        "available": True,
        "text": '{"answer":"Old"}',
        "usage": {"prompt_tokens": 999, "completion_tokens": 999},
    }
    job = {
        "episode": {
            "episode_id": "episode",
            "case_id": case["id"],
            "arm": "model_bundled",
            "repeat": 0,
        },
        "initial": initial,
        "records": [helper, old],
        "source_hashes": {},
        "source_request_digests": {},
        "seed": 700,
        "temperature": 0.5,
        "prompt": probe.final_prompt(case, state, "helper", answer_style="span"),
    }

    class Client:
        output = tmp_path
        calls = []

        def call(self, cid, prompt, seed, cap, temperature):
            self.calls.append((cid, seed, cap, temperature))
            return {
                "call_id": cid,
                "available": True,
                "text": '{"answer":"New"}',
                "usage": {"prompt_tokens": 30, "completion_tokens": 4},
            }

    client = Client()
    result = mod.collect(client, job, case, "remove_answer")
    assert client.calls == [("episode-remove_answer-final", 700, 128, 0.5)]
    assert result["new_physical_cost"] == {
        "calls": 1,
        "prompt_tokens": 30,
        "completion_tokens": 4,
        "unknown_usage_calls": 0,
    }
    assert result["hypothetical_deployed_cost"] == {
        "calls": 3,
        "prompt_tokens": 60,
        "completion_tokens": 9,
        "unknown_usage_calls": 0,
    }
