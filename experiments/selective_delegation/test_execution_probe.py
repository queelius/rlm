"""Isolated plan execution must bind actual predictions and hide future/root context."""

import importlib.util
import json
from pathlib import Path

import probe
from test_plan_probe import case_and_state, source_fixture


def implementation():
    path = Path(__file__).with_name("execution_probe.py")
    assert path.exists(), "execution diagnostic implementation missing"
    spec = importlib.util.spec_from_file_location("execution_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Client:
    def __init__(self, output, first='{"answer":"PREDICTED_MANUFACTURER"}'):
        self.output, self.first, self.calls = output, first, []

    def call(self, call_id, prompt, seed, max_tokens, temperature=0.5):
        self.calls.append((call_id, prompt, seed, max_tokens, temperature))
        text = self.first if call_id.endswith("-helper1") else '{"answer":"result"}'
        return {
            "call_id": call_id,
            "available": True,
            "text": text,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }


def test_isolation_binds_only_actual_prediction_and_never_exposes_gold(tmp_path):
    mod = implementation()
    case, source = source_fixture()
    client = Client(tmp_path)
    result = mod.collect(client, case, source, "reference_isolated", 0)
    assert result["available"]
    first, second, final = [call[1] for call in client.calls]
    assert "Find manufacturer" in first and "When did" not in first
    assert "When did PREDICTED_MANUFACTURER end?" in second
    assert "#1" not in json.loads(second.split("\n", 1)[1])["question"]
    for prompt in (first, second):
        assert all(
            secret not in prompt
            for secret in [
                "SECRET_GOLD",
                "SECRET_ALIAS",
                "SECRET_SOURCE",
                "SECRET_STEP_ONE",
                "SECRET_STEP_TWO",
                "Public question",
                "Original provisional",
                "Original evidence question",
                "MODEL QUESTION",
            ]
        )
    assert "Original provisional" in final
    assert [call[3] for call in client.calls] == [192, 192, 128]


def test_invalid_first_helper_json_stops_without_fallback_or_final(tmp_path):
    mod = implementation()
    case, source = source_fixture()
    client = Client(tmp_path, first="unstructured answer")
    result = mod.collect(client, case, source, "reference_isolated", 0)
    assert not result["available"] and result["helper_parse_failure"]
    assert len(client.calls) == 1
    assert result["new_physical_cost"]["calls"] == 1


def test_bundled_prompt_has_plan_without_checkpoint_and_matched_final_seed(tmp_path):
    mod = implementation()
    case, state = case_and_state()
    prompt = mod.bundled_prompt(case, state, "reference_bundled")
    body = json.loads(prompt.split("\n", 1)[1])
    assert body["question"] == case["question"]
    assert body["questions"] == ["Find manufacturer", "When did #1 end?"]
    assert set(body) == {"question", "documents", "questions"}
    case, source = source_fixture()
    finals = []
    for arm in mod.ARMS:
        client = Client(tmp_path / arm)
        result = mod.collect(client, case, source, arm, 1)
        finals.append(client.calls[-1][2:])
        expected = 3 if arm.endswith("isolated") else 2
        assert result["new_physical_cost"]["calls"] == expected
        assert result["hypothetical_deployed_cost"]["calls"] == expected + 1
    assert finals == [(probe.SEED + 20100, 128, 0.5)] * 4
