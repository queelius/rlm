import json
import time

import pytest


def test_ordered_reason_action_parser_rejects_reversed_extra_and_invalid_values():
    import alfworld_local_reason as m

    assert m.parse_output('{"reason":"Inspect first","action_index":1}', ["look", "open"]) == (
        "Inspect first",
        "open",
    )
    for text in (
        '{"action_index":0,"reason":"later"}',
        '{"reason":"","action_index":0}',
        '{"reason":"   ","action_index":0}',
        '{"reason":"x","action_index":true}',
        '{"reason":"x","action_index":-1}',
        '{"reason":"x","action_index":1}',
        '{"reason":"x","action_index":0,"extra":1}',
        '{"reason":"x","reason":"y","action_index":0}',
        json.dumps({"reason": "x" * 241, "action_index": 0}),
    ):
        with pytest.raises(ValueError):
            m.parse_output(text, ["look"])


def test_saved_public_context_matches_026_and_excludes_host_secrets():
    import alfworld_local_reason as m

    game = json.loads(m.base.MANIFEST.read_text())["proposed_screen"]["games"][0]
    public = {**game["public"], "won": "SECRET", "game_path": "SECRET"}
    history = [{"action": "look", "feedback": "visible", "reason": "SECRET"}]
    actual = m.prompt(public["feedback"], public, history)
    expected = m.base.prompts(public["feedback"], public, history, "")["flat"]
    assert actual.split("\n", 1)[1] == expected.split("\n", 1)[1]
    assert "SECRET" not in actual


def test_reason_not_persisted_but_rejection_and_all_generated_tokens_are(tmp_path):
    import alfworld_local_reason as m

    class Env:
        actions = 0

        def reset(self):
            return {
                "public": {"feedback": "Task: look twice", "admissible_commands": ["look"]},
                "host": {"won": False, "done": False},
            }

        def step(self, action):
            assert action == "look"
            self.actions += 1
            return {
                "public": {"feedback": "visible", "admissible_commands": ["look"]},
                "host": {"won": self.actions == 2, "done": self.actions == 2},
            }

    class Client:
        prompts = []

        def token_count(self, text):
            return len(text)

        def call(self, cid, prompt, policy, role, seed, cap, trimming):
            self.prompts.append(prompt)
            assert policy == role == "local_reason"
            assert seed == 2026092178 + len(self.prompts) - 1 and cap == 128
            text = (
                '{"action_index":0,"reason":"rejected"}'
                if len(self.prompts) == 1
                else '{"reason":"PRIVATE_ACCEPTED_REASON","action_index":0}'
            )
            return {
                "call_id": cid,
                "available": True,
                "text": text,
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": 23},
            }

    client = Client()
    row = m.play_episode(
        client,
        Env(),
        {"episode_id": "fixture", "policy": "local_reason", "seed": 2026092178},
        tmp_path,
    )
    assert row["won"] and row["actions"] == 2 and row["generated_tokens"] == 69
    assert row["invalid_outputs"] == 1
    third = client.prompts[2]
    assert "PRIVATE_ACCEPTED_REASON" not in third and "rejected" in third
    assert "Environment state unchanged" in third


def test_reason_tokens_reach_exact_episode_budget_without_extra_call(tmp_path):
    import alfworld_local_reason as m

    class Env:
        def reset(self):
            return {
                "public": {"feedback": "do task", "admissible_commands": ["look"]},
                "host": {"won": False, "done": False},
            }

        def step(self, action):
            return self.reset()

    class Client:
        def token_count(self, text):
            return 50

        def call(self, cid, prompt, policy, role, seed, cap, trimming):
            return {
                "call_id": cid,
                "available": True,
                "text": '{"reason":"next","action_index":0}',
                "usage": {"prompt_tokens": 50, "completion_tokens": cap},
            }

    row = m.play_episode(
        Client(),
        Env(),
        {"episode_id": "budget", "policy": "local_reason", "seed": 2026092178},
        tmp_path,
    )
    assert row["termination"] == "token_budget"
    assert row["generated_tokens"] == 2048 and row["actions"] == 16
    assert len(row["call_ids"]) == 16


def test_three_invalid_responses_stop_without_environment_action(tmp_path):
    import alfworld_local_reason as m

    class Env:
        def reset(self):
            return {
                "public": {"feedback": "do task", "admissible_commands": ["look"]},
                "host": {"won": False, "done": False},
            }

        def step(self, action):
            raise AssertionError("invalid response must not execute")

    class Client:
        def token_count(self, text):
            return 50

        def call(self, cid, prompt, policy, role, seed, cap, trimming):
            return {
                "call_id": cid,
                "available": True,
                "text": '{"action_index":0}',
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            }

    row = m.play_episode(
        Client(),
        Env(),
        {"episode_id": "invalid", "policy": "local_reason", "seed": 2026092178},
        tmp_path,
    )
    assert row["observed"] and not row["won"] and row["actions"] == 0
    assert row["termination"] == "three_consecutive_invalid" and row["generated_tokens"] == 15


def test_new_instruction_cannot_silently_trim_extra_history():
    import alfworld_local_reason as m

    class Client:
        def token_count(self, text):
            return 8100 if text.startswith(m.INSTRUCTION) else 100

    with pytest.raises(ValueError, match="no extra trim"):
        m.bounded_prompt(
            Client(),
            "Task",
            {"feedback": "now", "admissible_commands": ["look"]},
            [{"action": "look", "feedback": "visible"}],
            128,
        )


def test_actual_native_client_records_reason_response_and_decodes_index(tmp_path):
    import alfworld_local_reason as m

    torch = pytest.importorskip("torch")

    class Tokenizer:
        eos_token_id = pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert "reason" in messages[0]["content"]
            assert kwargs["enable_thinking"] is False
            return [1, 2]

        def decode(self, ids, **kwargs):
            assert ids == [9, 0]
            return '{"reason":"Inspect this place","action_index":0}'

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            assert kwargs["temperature"] == 0.5 and kwargs["max_new_tokens"] == 128
            return torch.tensor([[1, 2, 9, 0]])

    game = json.loads(m.base.MANIFEST.read_text())["proposed_screen"]["games"][0]
    prompt = m.prompt(game["public"]["feedback"], game["public"], [])
    client = m.base.BaseClient(Model(), Tokenizer(), tmp_path, time.time() + 100)
    row = client.call("native", prompt, "local_reason", "local_reason", m.base.SEEDS[0], 128, {})
    assert row["available"] and not row["request"]["adapter_enabled"]
    assert row["usage"]["completion_tokens"] == 2
    assert (
        m.parse_output(row["text"], game["public"]["admissible_commands"])[1]
        == (game["public"]["admissible_commands"][0])
    )
