"""Question-only annotation intervention with paired fresh computation."""

import importlib.util
import json
from pathlib import Path

import probe


def implementation():
    path = Path(__file__).with_name("plan_probe.py")
    assert path.exists(), "reference-plan implementation missing"
    spec = importlib.util.spec_from_file_location("plan_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_and_state():
    case = {
        "id": "parent",
        "split": "train",
        "question": "Public question",
        "documents": [{"id": "4", "title": "Public title", "text": "Source text"}],
        "answer": "SECRET_GOLD",
        "metadata": {
            "hops": 2,
            "source_id": "SECRET_SOURCE",
            "answer_aliases": ["SECRET_ALIAS"],
            "question_decomposition": [
                {
                    "question": "Find manufacturer",
                    "answer": "SECRET_STEP_ONE",
                    "id": "SECRET_COMPONENT",
                    "paragraph_support_idx": "SECRET_SUPPORT",
                },
                {"question": "When did #1 end?", "answer": "SECRET_STEP_TWO"},
            ],
        },
    }
    state = {
        "answer": "Original provisional",
        "evidence_question": "Original evidence question",
        "subquestions": ["MODEL QUESTION ONE", "MODEL QUESTION TWO"],
    }
    return case, state


def test_reference_helper_projects_questions_only_and_changes_only_subquestions():
    mod = implementation()
    case, state = case_and_state()
    generated = mod.helper_prompt(case, state, "model_plan")
    reference = mod.helper_prompt(case, state, "reference_questions")
    assert generated.split("\n", 1)[0] == reference.split("\n", 1)[0]
    original_body, reference_body = [
        json.loads(p.split("\n", 1)[1]) for p in (generated, reference)
    ]
    assert original_body["initial_attempt"] == state
    expected = {**state, "subquestions": ["Find manufacturer", "When did #1 end?"]}
    assert reference_body["initial_attempt"] == expected
    assert original_body["documents"] == reference_body["documents"] == case["documents"]
    assert all(
        secret not in reference
        for secret in [
            "SECRET_GOLD",
            "SECRET_ALIAS",
            "SECRET_SOURCE",
            "SECRET_STEP_ONE",
            "SECRET_STEP_TWO",
            "SECRET_COMPONENT",
            "SECRET_SUPPORT",
        ]
    )
    assert state["subquestions"] == ["MODEL QUESTION ONE", "MODEL QUESTION TWO"]


class Client:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def call(self, call_id, prompt, seed, max_tokens, temperature=0.5):
        self.calls.append((call_id, prompt, seed, max_tokens, temperature))
        return {
            "call_id": call_id,
            "available": True,
            "text": "Helper evidence" if call_id.endswith("-helper") else '{"answer":"answer"}',
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }


def source_fixture():
    case, state = case_and_state()
    source = {
        "state": state,
        "record": {
            "call_id": "parent-checkpoint",
            "available": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        },
        "source_hashes": {"source.json": "hash"},
        "request_digest": "request_hash",
    }
    return case, source


def test_both_final_arms_keep_original_checkpoint_and_actual_helper_report(tmp_path):
    mod = implementation()
    case, source = source_fixture()
    clients = []
    for arm in mod.ARMS:
        client = Client(tmp_path / arm)
        mod.collect(client, case, source, arm, 0)
        clients.append(client)
    expected = probe.final_prompt(case, source["state"], "Helper evidence", answer_style="span")
    assert all(client.calls[-1][1] == expected for client in clients)
    assert "Find manufacturer" not in expected
    assert all(len(client.calls) == 2 for client in clients)


def test_paired_new_seed_caps_and_costs_distinguish_reused_checkpoint(tmp_path):
    mod = implementation()
    case, source = source_fixture()
    results, settings = [], []
    for arm in mod.ARMS:
        client = Client(tmp_path / arm)
        results.append(mod.collect(client, case, source, arm, 1))
        settings.append([call[2:] for call in client.calls])
    assert (
        settings[0]
        == settings[1]
        == [(probe.SEED + 10101, 384, 0.5), (probe.SEED + 10100, 128, 0.5)]
    )
    assert results[0]["new_physical_cost"] == {
        "calls": 2,
        "prompt_tokens": 40,
        "completion_tokens": 10,
        "unknown_usage_calls": 0,
    }
    assert results[0]["hypothetical_deployed_cost"] == {
        "calls": 3,
        "prompt_tokens": 50,
        "completion_tokens": 13,
        "unknown_usage_calls": 0,
    }
    assert results[1]["annotation_privileged"] is True
