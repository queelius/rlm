"""Small planned-denominator and public receipt reconstruction fixtures."""

import analyze_alfworld_screen as analysis


def panel():
    return {
        "cases": [
            dict(
                episode_id=f"{game}-{seed}-{policy}",
                game_index=game,
                seed=seed,
                policy=policy,
                game={"game": "/data/pick-X/trial/game"},
            )
            for game in range(8)
            for seed in (1, 2)
            for policy in ("flat", "manager_worker")
        ],
        "policies": ["flat", "manager_worker"],
        "seeds": [1, 2],
    }


def test_missing_and_unobserved_are_not_observed_losses_or_wins():
    plan = panel()
    base = dict(actions=1, generated_tokens=10, invalid_outputs=0, termination="native_done")
    rows = [
        dict(plan["cases"][0], **base, observed=True, won=True),
        dict(plan["cases"][1], **base, observed=False, won=False),
    ]
    result = analysis.summarize(plan, rows, [])
    assert result["policies"]["flat"]["won"] == 1
    assert result["policies"]["flat"]["planned"] == 16
    assert result["policies"]["flat"]["unknown"] == 15
    assert result["policies"]["manager_worker"]["unknown"] == 16
    assert result["paired"]["complete_pairs"] == 0
    assert result["paired"]["game_bootstrap"] is None


def test_two_seeds_are_clustered_as_eight_games():
    plan = panel()
    rows = [
        dict(
            j,
            observed=True,
            won=j["policy"] == "manager_worker",
            actions=1,
            generated_tokens=10,
            invalid_outputs=0,
            termination="native_done",
        )
        for j in plan["cases"]
    ]
    result = analysis.summarize(plan, rows, [], draws=100)
    assert result["paired"]["complete_pairs"] == 16
    assert result["paired"]["complete_games"] == 8
    assert result["paired"]["game_bootstrap"]["estimate"] == 1
    assert result["paired"]["game_bootstrap"]["ci95"] == [1, 1]


def test_completed_native_smoke_public_requests_reconstruct():
    import json
    from pathlib import Path

    import pytest

    output = Path(
        "/project/alex_phd/runs/rlm-research-r4/sidecars/"
        "selective-delegation-20260921/alfworld-screen-001"
    )
    if not (output / "SMOKE.json").exists():
        pytest.skip("completed external smoke unavailable")

    def read(path):
        return json.loads(path.read_text())

    plan = read(output / "PLAN.json")
    for job in plan["cases"][:2]:
        row = read(output / "episodes" / (job["episode_id"] + ".json"))
        calls = {cid: read(output / "calls" / (cid + ".json")) for cid in row["call_ids"]}
        audit = analysis.audit_episode(job, row, calls, read, output, plan)
        assert audit["actions"] == row["actions"]
        assert audit["invalid"] == row["invalid_outputs"]


def test_indexed_native_rejection_history_is_replayed(tmp_path):
    import time

    import alfworld_closed_loop as closed
    import pytest

    torch = pytest.importorskip("torch")
    current = {"feedback": "Task: look", "admissible_commands": ["look"]}
    job = dict(episode_id="indexed", policy="flat", seed=3, game={"public": current})

    class Env:
        def reset(self):
            return {"public": current, "host": {"won": False, "done": False}}

        def step(self, action):
            return {
                "action": action,
                "public": {"feedback": "done", "admissible_commands": []},
                "host": {"won": True, "done": True},
            }

    class Tokenizer:
        eos_token_id = pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return '{"action_index":true}' if ids[0] == 99 else '{"action_index":0}'

    class Model:
        device = torch.device("cpu")
        count = 0

        def generate(self, **kwargs):
            self.count += 1
            return torch.tensor([[1, 2, 99 if self.count == 1 else 9, 0]])

    client = closed.BaseClient(Model(), Tokenizer(), tmp_path, time.time() + 100)
    row = closed.play_episode(client, Env(), job, tmp_path)

    def read(path):
        return analysis.json.loads(path.read_text())

    calls = {cid: read(tmp_path / "calls" / (cid + ".json")) for cid in row["call_ids"]}
    plan = dict(
        schema="alfworld-closed-loop-indexed-v1",
        request_cap=128,
        token_limit=2048,
        action_limit=50,
        model=str(closed.evaluation.planner.BASE),
    )
    audit = analysis.audit_episode(job, row, calls, read, tmp_path, plan)
    assert audit["actions"] == 1 and audit["invalid"] == 1
    assert audit["invalid_reasons"] == {"invalid_index": 1}
