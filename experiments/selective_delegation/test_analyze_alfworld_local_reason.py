import copy
import json
import time

import pytest


def test_actual_saved_public_native_replay_detects_reason_and_prompt_tampering(tmp_path):
    import alfworld_local_reason as collector
    import analyze_alfworld_local_reason as analyzer

    torch = pytest.importorskip("torch")
    game = json.loads(collector.base.MANIFEST.read_text())["proposed_screen"]["games"][0]
    job = {
        "episode_id": "native",
        "policy": "local_reason",
        "seed": 2026092178,
        "game_index": 0,
        "game": game,
    }

    class Tokenizer:
        eos_token_id = pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return [ord(c) + 1 for c in messages[0]["content"]]

        def decode(self, values, **kwargs):
            return '{"reason":"Inspect this place","action_index":0}'

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return torch.cat((kwargs["input_ids"], torch.tensor([[9, 0]])), dim=1)

    class Env:
        def reset(self):
            return {"public": game["public"], "host": {"won": False, "done": False}}

        def step(self, action):
            return {
                "action": action,
                "public": {"feedback": "done", "admissible_commands": []},
                "host": {"won": True, "done": True},
            }

    tokenizer = Tokenizer()
    client = collector.base.BaseClient(Model(), tokenizer, tmp_path, time.time() + 90)
    row = collector.play_episode(client, Env(), job, tmp_path)
    calls = {p.stem: json.loads(p.read_text()) for p in (tmp_path / "calls").glob("*.json")}
    plan = {"model": str(collector.base.evaluation.planner.BASE)}

    def read(path):
        return json.loads(path.read_text())

    audit = analyzer.audit_episode(job, row, calls, read, tmp_path, plan, tokenizer)
    assert audit["actions"] == 1 and audit["reason_characters"] == [18]
    assert audit["generated_tokens"] == 2
    wrong = copy.deepcopy(calls)
    wrong[row["call_ids"][0]]["request"]["prompt"] += " leaked"
    with pytest.raises(ValueError, match="request"):
        analyzer.audit_episode(job, row, wrong, read, tmp_path, plan, tokenizer)


def test_game_bootstrap_keeps_both_seeds_and_missing_not_loss():
    import analyze_alfworld_local_reason as analyzer

    rows = [
        {
            "game_index": g,
            "seed": seed,
            "policy": policy,
            "observed": True,
            "won": (seed == 1 if policy == "local_reason" else seed == 2),
        }
        for g in range(8)
        for seed in (1, 2)
        for policy in ("flat", "local_reason")
    ]
    result = analyzer.pair(rows, "local_reason", "flat", [1, 2], draws=100)
    assert result["wins"] == result["losses"] == 8
    assert result["game_bootstrap"]["estimate"] == 0
    assert result["game_bootstrap"]["ci95"] == [0, 0]
    rows[0]["observed"] = False
    result = analyzer.pair(rows, "local_reason", "flat", [1, 2], draws=100)
    assert result["unobserved_pairs"] == 1 and result["game_bootstrap"] is None
