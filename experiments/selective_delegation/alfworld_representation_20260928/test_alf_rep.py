import json

import pytest


def test_command_decoder_matches_exact_current_command_without_repair():
    import alf_rep as r

    commands = ["look", "take soapbar 1 from countertop 1"]
    assert r.parse_command('{"command":"look"}', "flat", commands) == "look"
    for text in (
        '{"command":"LOOK"}',
        '{"command":" look"}',
        '{"command":"look "}',
        '{"command":"take soapbar 1"}',
        '{"command":0}',
        '{"command":"look","command":"look"}',
        '{"command":"look","reason":"x"}',
        '{"action_index":0}',
    ):
        with pytest.raises(ValueError):
            r.parse_command(text, "flat", commands)


def test_interface_prompts_have_byte_identical_public_context_and_full_history():
    import alf_rep as r

    current = dict(feedback="room", admissible_commands=["look", "inventory"])
    history = [
        dict(action="look", feedback="room"),
        dict(event="controller_rejection", action="bad", feedback="rejected"),
    ]
    index = r.controller("index").prompts("task", current, history, "")["flat"]
    command = r.controller("command").prompts("task", current, history, "")["flat"]
    assert index.split("\n", 1)[1] == command.split("\n", 1)[1]
    context = json.loads(command.split("\n", 1)[1])["public_context"]
    assert context["history"] == history
    assert context["admissible_commands"] == [
        {"action_index": 0, "command": "look"},
        {"action_index": 1, "command": "inventory"},
    ]


def test_command_training_projection_keeps_original_native_actions_and_order():
    import alf_rep_data as data

    result = data.project_rows()
    assert len(result) == 524
    for row in result:
        context = json.loads(row["prompt"].split("\n", 1)[1])["public_context"]
        expected = context["admissible_commands"][row["original_action_index"]]["command"]
        assert json.loads(row["target_json"]) == {"command": expected}
        assert row["dropped_history"] == 0
    assert [row["id"] for row in result] == [row["id"] for row in data.original_rows()]


def test_fresh_selection_is_outcome_blind_and_excludes_prior_games():
    import alf_rep_data as data

    selection = data.select_panel()
    assert len(selection["games"]) == len(set(selection["games"])) == 12
    assert not set(selection["games"]) & set(selection["excluded_games"])
    assert all("/valid_unseen/" in path for path in selection["games"])
    assert selection["family_counts"] == dict.fromkeys(data.FAMILIES, 2)


def test_context_overflow_does_not_trim_history():
    import alf_rep as r

    class Client:
        def token_count(self, text):
            return 8192

    with pytest.raises(ValueError, match="full public history"):
        r.controller("command").bounded_prompts(
            Client(), "task", dict(feedback="x", admissible_commands=["look"]), [], "", 128
        )


def test_command_trainer_rejects_index_targets_without_relabeling():
    import alf_rep_data as data
    import alf_rep_train as train

    rows = data.project_rows()
    manifest = dict(examples=524, successful_games_only=True)
    assert train.validate_rows(rows, manifest) == 524
    rows[0] = dict(rows[0], target_json='{"action_index":0}')
    with pytest.raises(ValueError):
        train.validate_rows(rows, manifest)


def test_saved_native_command_fixture_preserves_transitions_and_charges_invalids(tmp_path):
    import alf_rep_fixture as fixture

    result = fixture.check(tmp_path / "native")
    assert result["passed"] and result["scientific_model_calls"] == 0
    assert all(row["won"] and row["native_replayed"] for row in result["cases"])
    command = next(row for row in result["cases"] if row["mode"] == "command")
    assert command["invalid_outputs"] == 1
    assert command["calls"] == command["actions"] + 1
    assert result["same_native_action_sequence"] is True


def test_learning_interaction_keeps_missing_cells_unknown_and_subtracts_base_gain():
    import alf_rep_compare as compare

    cells = {"index-base": 0, "index-trained": 1, "command-base": 1, "command-trained": 1}
    assert compare.interaction(cells) == -1
    del cells["command-trained"]
    assert compare.interaction(cells) is None
