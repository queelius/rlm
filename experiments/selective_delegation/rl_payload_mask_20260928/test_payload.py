"""Keep boundary/EOS loss and original sampled-prefix checks intact."""

import payload as m
import pytest


def test_payload_mask_preserves_boundary_tokens_and_fixed_schema_key():
    text = '{"action":"craft","ingredients":{"r":2},"target_item":"x","output_count":2}'
    a, b = text.index('{"r"'), text.index(',"target_item"')
    offsets = [(0, 2), (a - 2, a + 1), (a + 1, a + 4), (a + 4, b - 1), (b - 1, b + 2)]
    assert m.payload_offsets_mask(text, offsets) == [1, 1, 0, 0, 1]


def test_payload_location_ignores_ingredients_string_inside_note():
    text = '{"note":"ingredients: pretend","ingredients":{"r":2},"action":"craft"}'
    a = text.index('{"r"')
    assert m.payload_offsets_mask(text, [(9, 20), (a + 2, a + 3)]) == [1, 0]


def test_masked_objective_keeps_episode_denominator_and_signed_credit():
    import torch

    logps = torch.tensor([[-2.0, -3.0, -1.0]], requires_grad=True)
    loss = m.masked_objective(logps, -2 / 3, [1, 0, 1])
    assert float(loss.detach()) == pytest.approx(-0.0625)
    loss.backward()
    assert logps.grad.tolist()[0] == pytest.approx([1 / 48, 0, 1 / 48])
    assert m.masked_batch_objective(
        [m.SimpleNamespace(call_id="a", advantage=-2 / 3)],
        {"a": [-2.0, -3.0, -1.0]},
        {"a": [1, 0, 1]},
    ) == pytest.approx(-0.0625)


def test_eligibility_needs_observed_recipe_and_divisible_quantity():
    action = {"action": "craft", "ingredients": {"r": 7}, "target_item": "x", "output_count": 2}
    history = [
        {
            "action": {"action": "get_info", "items": ["x"]},
            "feedback": [{"item": "x", "recipes": [{"ingredients": {"r": 1}, "result_count": 2}]}],
        }
    ]
    assert not m.eligibility(action, [], None)[0]
    assert m.eligibility(action, history, None)[0]
    assert not m.eligibility({**action, "output_count": 3}, history, None)[0]
    wrong = {
        "requested_action": action,
        "executed_action": action,
        "binding_reason": "bound_observed_single_recipe",
    }
    with pytest.raises(ValueError, match="saved binding"):
        m.eligibility(action, history, wrong)


def test_private_rewrite_fails_closed_instead_of_silently_missing_loss_site():
    with pytest.raises(ValueError, match="exactly once"):
        m.rewrite_run("def run(args):\n    pass\n")


def test_first_actual_replay_records_and_rejects_a_large_numerical_gap(tmp_path):
    import torch

    with pytest.raises(ValueError, match="first saved-call numerical replay failed"):
        m.first_replay(tmp_path, {"call_id": "fixture"}, torch.tensor([[-1.0, -2.0]]), [-1.0, -3.0])
    receipt = m.p.read(tmp_path / "FIRST-REPLAY.json")
    assert receipt["original_tokens"] == 2
    assert receipt["maximum_generation_replay_gap"] == 1
    assert receipt["mean_generation_replay_gap"] == 0.5
    assert receipt["passed"] is False
