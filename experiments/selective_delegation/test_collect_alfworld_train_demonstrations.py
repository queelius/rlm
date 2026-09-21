import hashlib

import collect_alfworld_train_demonstrations as demonstrations


def test_label_blind_selection_picks_one_supported_family_and_public_rows_hide_target(tmp_path):
    games = []
    for family in demonstrations.SUPPORTED_FAMILIES:
        for trial in ("trial_b", "trial_a"):
            game = tmp_path / family / trial / "game.tw-pddl"
            game.parent.mkdir(parents=True, exist_ok=True)
            game.write_text(f"{family}/{trial}")
            games.append(game)
    chosen = demonstrations.select_games(games)
    assert set(chosen) == set(demonstrations.SUPPORTED_FAMILIES)
    assert all(
        chosen[family].name == "game.tw-pddl"
        and hashlib.sha256(chosen[family].read_bytes()).hexdigest()
        for family in chosen
    )
    row = demonstrations.public_row("feedback", ["look", "inventory"])
    assert row == {"feedback": "feedback", "admissible_commands": ["look", "inventory"]}
    assert "expert" not in repr(row).lower()


def test_incomplete_expert_step_is_not_reported_as_a_win():
    assert not demonstrations.completed_won([{"public": {"feedback": "x"}}])
