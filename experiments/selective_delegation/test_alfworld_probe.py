import json

import pytest


def test_strict_objects_reject_extra_keys_duplicate_keys_and_inadmissible_action():
    from alfworld_probe import parse_output

    assert parse_output('{"action":"look"}', "worker", ["look"]) == "look"
    for text in (
        '{"action":"look","won":true}',
        '{"action":"look","action":"look"}',
        '{"action":"invented"}',
        '```{"action":"look"}```',
    ):
        with pytest.raises(ValueError):
            parse_output(text, "flat", ["look"])


def test_prompts_never_project_host_fields_and_share_public_context():
    from alfworld_probe import prompts

    current = {
        "feedback": "public room",
        "admissible_commands": ["look"],
        "won": "WON_SECRET",
        "facts": "FACT_SECRET",
        "game": "PATH_SECRET",
    }
    history = [{"action": "look", "feedback": "old room", "gold": "GOLD_SECRET"}]
    values = prompts("initial task", current, history, "find target")
    for text in values.values():
        assert all(
            s not in text for s in ("WON_SECRET", "FACT_SECRET", "PATH_SECRET", "GOLD_SECRET")
        )
    payloads = [json.loads(text.rsplit("\n", 1)[1]) for text in values.values()]
    assert (
        payloads[0]["public_context"]
        == payloads[1]["public_context"]
        == payloads[2]["public_context"]
    )
    assert payloads[2]["current_goal"] == "find target"


class FakeEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return {
            "public": {"feedback": "initial task", "admissible_commands": ["look"]},
            "host": {"won": False, "done": False, "reward": 0},
        }

    def step(self, action):
        assert action == "look"
        self.steps += 1
        row = self.reset()
        row["host"].update(won=self.steps == 5, done=self.steps == 5)
        return row


class FakeClient:
    def __init__(self, invalid=False):
        self.roles, self.caps, self.seeds = [], [], []
        self.invalid = invalid

    def token_count(self, text):
        return len(text)

    def call(self, identity, prompt, policy, role, seed, cap, trimming):
        self.roles.append(role)
        self.caps.append(cap)
        self.seeds.append(seed)
        text = (
            "bad"
            if self.invalid
            else '{"goal":"find target"}'
            if role == "manager"
            else '{"action":"look"}'
        )
        return {
            "call_id": identity,
            "available": True,
            "text": text,
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 10},
        }


def test_manager_refresh_costs_and_common_action_seeds(tmp_path):
    from alfworld_probe import play_episode

    clients = []
    for policy in ("flat", "manager_worker"):
        client, env = FakeClient(), FakeEnv()
        result = play_episode(
            client, env, {"episode_id": policy, "policy": policy, "seed": 17}, tmp_path
        )
        assert result["won"] and result["observed"] and result["actions"] == 5
        assert result["generated_tokens"] == (50 if policy == "flat" else 70)
        clients.append(client)
    assert clients[1].roles == [
        "manager",
        "worker",
        "worker",
        "worker",
        "worker",
        "manager",
        "worker",
    ]
    assert clients[0].seeds == [
        s for r, s in zip(clients[1].roles, clients[1].seeds, strict=True) if r == "worker"
    ]


def test_three_invalid_outputs_stop_without_environment_fallback(tmp_path):
    from alfworld_probe import play_episode

    client, env = FakeClient(invalid=True), FakeEnv()
    result = play_episode(
        client, env, {"episode_id": "invalid", "policy": "flat", "seed": 17}, tmp_path
    )
    assert result["termination"] == "three_consecutive_invalid"
    assert result["generated_tokens"] == 30 and result["actions"] == env.steps == 0
    assert result["observed"] and not result["won"]


def test_every_manager_token_counts_and_last_call_cannot_exceed_budget(tmp_path):
    from alfworld_probe import play_episode

    class NeverDone(FakeEnv):
        def step(self, action):
            self.steps += 1
            return self.reset()

    class Expensive(FakeClient):
        def call(self, *args):
            row = super().call(*args)
            row["usage"]["completion_tokens"] = min(100, args[5])
            return row

    client = Expensive()
    result = play_episode(
        client,
        NeverDone(),
        {"episode_id": "budget", "policy": "manager_worker", "seed": 17},
        tmp_path,
    )
    assert result["termination"] == "token_budget" and result["generated_tokens"] == 2048
    assert len(client.roles) == 21 and client.caps[-1] == 48
    assert client.roles.count("manager") == 5 and result["actions"] == 16


def test_oldest_history_trimming_is_identical_despite_role_goal_and_remaining_budget():
    from alfworld_probe import bounded_prompts

    client = FakeClient()
    current = {"feedback": "CURRENT", "admissible_commands": ["look"]}
    history = [{"action": "look", "feedback": str(i) * 1000} for i in range(10)]
    a, metadata_a = bounded_prompts(client, "INITIAL", current, history, "goal", 128)
    b, metadata_b = bounded_prompts(client, "INITIAL", current, history, "other goal", 1)
    assert metadata_a["dropped_history"] > 0
    assert metadata_a["dropped_history"] == metadata_b["dropped_history"]
    context_a = json.loads(a["flat"].rsplit("\n", 1)[1])["public_context"]
    context_b = json.loads(b["worker"].rsplit("\n", 1)[1])["public_context"]
    assert context_a == context_b
    assert context_a["history"][-1]["feedback"] == "9" * 1000
    assert context_a["initial_observation"] == "INITIAL"


def test_native_generation_receipts_are_base_only_for_all_three_roles(tmp_path):
    import time

    torch = pytest.importorskip("torch")
    from alfworld_probe import BaseClient

    class Tokenizer:
        eos_token_id, pad_token_id = 0, 0

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return [11, 12, 13]

        def decode(self, values, **kwargs):
            assert values == [31, 0]
            return '{"action":"look"}'

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            assert not torch.is_grad_enabled()
            assert kwargs["max_new_tokens"] == 128 and kwargs["temperature"] == 0.5
            return torch.cat([kwargs["input_ids"], torch.tensor([[31, 0]])], dim=1)

    client = BaseClient(Model(), Tokenizer(), tmp_path, time.time() + 60)
    for role in ("flat", "manager", "worker"):
        row = client.call(
            role, "public prompt", "flat" if role == "flat" else "manager_worker", role, 17, 128, {}
        )
        assert row["available"] and row["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
        assert row["request"]["adapter_enabled"] is False
        assert row["request"]["adapter_sha256"] is None
        assert json.loads((tmp_path / "calls" / (role + ".json")).read_text()) == row
