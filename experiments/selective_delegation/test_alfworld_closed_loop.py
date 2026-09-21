import json

import pytest


def test_indices_bind_actual_saved_admissible_list_and_reject_bool():
    import alfworld_closed_loop as mod

    manifest = json.loads(mod.MANIFEST.read_text())
    commands = manifest["proposed_screen"]["games"][0]["public"]["admissible_commands"]
    assert mod.parse_output('{"action_index":0}', "flat", commands) == commands[0]
    assert (
        mod.parse_output(json.dumps({"action_index": len(commands) - 1}), "worker", commands)
        == commands[-1]
    )
    for text in (
        '{"action_index":true}',
        '{"action_index":-1}',
        '{"action_index":999}',
        '{"action_index":1.0}',
        '{"action_index":"1"}',
        '{"action_index":0,"action_index":0}',
        '{"action_index":0,"action":"look"}',
    ):
        with pytest.raises(ValueError):
            mod.parse_output(text, "flat", commands)


@pytest.mark.parametrize("policy", ["flat", "manager_worker"])
def test_rejection_feedback_roundtrip_and_token_accounting(tmp_path, policy):
    import alfworld_closed_loop as mod

    class Env:
        def reset(self):
            return {
                "public": {"feedback": "Task: look", "admissible_commands": ["look"]},
                "host": {"won": False, "done": False},
            }

        def step(self, action):
            assert action == "look"
            return {
                "action": action,
                "public": {"feedback": "done", "admissible_commands": []},
                "host": {"won": True, "done": True},
            }

    class Client:
        def __init__(self):
            self.prompts = []

        def token_count(self, text):
            return len(text)

        def call(self, identity, prompt, policy, role, seed, cap, trimming):
            self.prompts.append(prompt)
            text = (
                '{"action_index":true}'
                if len(self.prompts) == 1
                else '{"goal":"look"}'
                if role == "manager"
                else '{"action_index":0}'
            )
            return dict(
                call_id=identity,
                available=True,
                text=text,
                usage={"prompt_tokens": len(prompt), "completion_tokens": 10},
            )

    client = Client()
    row = mod.play_episode(
        client, Env(), {"episode_id": "fixture", "policy": policy, "seed": 3}, tmp_path
    )
    assert row["won"] and row["actions"] == 1 and row["invalid_outputs"] == 1
    assert row["generated_tokens"] == (20 if policy == "flat" else 30)
    second = json.loads(client.prompts[1].rsplit("\n", 1)[1])["public_context"]
    assert second["current_feedback"] == "Task: look"
    assert second["history"][0]["event"] == "controller_rejection"
    assert "unchanged" in second["history"][0]["feedback"]
    assert second["admissible_commands"] == [{"action_index": 0, "command": "look"}]


def test_matched_public_context_and_first_native_prompt_fixture():
    import alfworld_closed_loop as mod

    game = json.loads(mod.MANIFEST.read_text())["proposed_screen"]["games"][0]
    texts = mod.prompts(game["public"]["feedback"], game["public"], [], "find target")
    context = [json.loads(t.rsplit("\n", 1)[1])["public_context"] for t in texts.values()]
    assert context[0] == context[1] == context[2]
    assert "action_index" in texts["flat"] and "game.tw-pddl" not in texts["flat"]
    assert "won" not in context[0] and "scene" not in context[0]
    assert mod.parse_output('{"goal":"find target"}', "manager", []) == "find target"


def test_actual_public_prompt_uses_unchanged_native_client_and_decodes_index(tmp_path):
    import time

    import alfworld_closed_loop as mod

    torch = pytest.importorskip("torch")

    class Tokenizer:
        eos_token_id = pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert "action_index" in messages[0]["content"]
            assert kwargs["enable_thinking"] is False
            return [1, 2]

        def decode(self, ids, **kwargs):
            assert ids == [9, 0]
            return '{"action_index":0}'

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            assert kwargs["temperature"] == 0.5 and kwargs["max_new_tokens"] == 128
            return torch.tensor([[1, 2, 9, 0]])

    game = json.loads(mod.MANIFEST.read_text())["proposed_screen"]["games"][0]
    prompt = mod.prompts(game["public"]["feedback"], game["public"], [], "")["flat"]
    client = mod.BaseClient(Model(), Tokenizer(), tmp_path, time.time() + 100)
    row = client.call("native-fixture", prompt, "flat", "flat", mod.SEEDS[0], 128, {})
    assert row["available"] and not row["request"]["adapter_enabled"]
    assert (
        mod.parse_output(row["text"], "flat", game["public"]["admissible_commands"])
        == game["public"]["admissible_commands"][0]
    )
