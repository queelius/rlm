import json
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "policy,expected_calls", [("flat", 1), ("manager_worker", 2), ("local_reason", 1)]
)
def test_three_routes_keep_frozen_public_projection_and_charge_manager(
    tmp_path, policy, expected_calls
):
    import alfworld_unseen as m

    selected = json.loads(m.INPUT.read_text())["games"][0]

    class Env:
        def reset(self):
            return {"public": selected["public"], "host": {"won": False, "done": False}}

        def step(self, action):
            assert action in selected["public"]["admissible_commands"]
            return {
                "action": action,
                "public": {"feedback": "done", "admissible_commands": []},
                "host": {"won": True, "done": True},
            }

    class Client:
        def token_count(self, text):
            return len(text) // 4

        def call(self, cid, text, condition, role, seed, cap, trimming):
            context = json.loads(text.split("\n", 1)[1])["public_context"]
            assert context["initial_observation"] == selected["public"]["feedback"]
            assert "game" not in context and "won" not in context and condition == policy
            assert selected["game"] not in text and cap == 128
            response = (
                '{"goal":"Inspect"}'
                if role == "manager"
                else '{"reason":"Inspect","action_index":0}'
                if role == "local_reason"
                else '{"action_index":0}'
            )
            return {
                "call_id": cid,
                "available": True,
                "text": response,
                "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 10},
            }

    job = {"episode_id": "fixture", "policy": policy, "seed": 2026092195}
    result = m.play_episode(Client(), Env(), job, tmp_path)
    assert result["won"] and result["actions"] == 1
    assert len(result["call_ids"]) == expected_calls
    assert result["generated_tokens"] == 10 * expected_calls


def test_actual_unseen_isolated_bridge_reset_matches_frozen_public(tmp_path):
    import alfworld_unseen as m

    panel = json.loads(m.INPUT.read_text())
    original = json.loads(m.local.base.MANIFEST.read_text())
    game = panel["games"][0]
    env = m.CheckedBridge(
        game,
        str(Path(original["environment"]["path"]) / ".venv/bin/python"),
        str(m.local.base.MANIFEST.parent.parent),
        time.time() + 30,
        tmp_path / "bridge.stderr",
    )
    env.expected_public = game["public"]
    try:
        reset = env.reset()
        assert reset["public"] == game["public"]
        assert not reset["host"]["won"]
        stepped = env.step("look")
        assert stepped["action"] == "look"
        assert set(stepped["public"]) == {"feedback", "admissible_commands"}
    finally:
        env.close()
