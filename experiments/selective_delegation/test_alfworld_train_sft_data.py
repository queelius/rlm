import json

import alfworld_train_sft_data as data


def test_selection_excludes_prior_qualification_games_and_prompt_has_strict_index_target(tmp_path):
    prior = set()
    games = []
    for family in data.FAMILIES:
        for trial in range(7):
            game = tmp_path / family / f"trial-{trial}" / "game.tw-pddl"
            game.parent.mkdir(parents=True, exist_ok=True)
            game.write_text(f"{family}/{trial}")
            games.append(game)
        prior.add(games[-1])
    selected = data.select_games(games, prior)
    assert all(len(value) == 6 for value in selected.values())
    assert not set().union(*map(set, selected.values())) & prior
    prompt = data.flat_prompt(
        "initial", {"feedback": "current", "admissible_commands": ["look"]}, []
    )
    assert json.loads(data.target(0)) == {"action_index": 0}
    assert "initial_observation" in prompt and "admissible_commands" in prompt


def test_collect_cli_records_selected_subcommand(tmp_path):
    args = data.make_parser().parse_args(
        [
            "collect",
            "--data-root",
            str(tmp_path),
            "--prior-traces",
            str(tmp_path),
            "--output",
            str(tmp_path),
        ]
    )
    assert args.command == "collect"
