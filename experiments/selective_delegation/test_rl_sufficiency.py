import copy
import time
from itertools import permutations

import pytest


def test_paired_reward_and_zero_cursor_not_optimizer():
    import rl_sufficiency as run

    positive = {"answer": "London", "answer_aliases": []}
    good = [{"answerable": True, "answer": "London"}, {"answerable": False, "answer": ""}]
    assert run.pair_reward(positive, good) == 1
    wrong = copy.deepcopy(good)
    wrong[0] = {"answerable": False, "answer": ""}
    assert run.pair_reward(positive, wrong) == 0
    wrong[0] = {"answerable": True, "answer": "Madrid"}
    assert run.pair_reward(positive, wrong) == 0
    with pytest.raises(ValueError, match="answer:string"):
        run.baseline.parse_output('{"answerable":true,"answer":80}')
    assert run.pair_reward(positive, [None, good[1]]) == 0
    state = {"step": 0, "sample_cursor": 0, "zero_streak": 0}
    for i in range(4):
        state = run.advance(state, [[0, 0, 0, 0]] * 16)
        assert state["sample_cursor"] == i + 1 and state["step"] == 0
    assert run.finished(state)
    state = run.advance(
        {"step": 0, "sample_cursor": 0, "zero_streak": 0}, [[0, 0, 0, 1]] + [[0, 0, 0, 0]] * 15
    )
    assert state["step"] == 1 and state["effective_groups"] == 1


def test_pairing_mean_credits_mispaired_marginal_successes_and_matches_all_pairings():
    import rl_sufficiency as run

    group = [
        {"positive_success": 1, "negative_success": 0, "reward": 0},
        {"positive_success": 0, "negative_success": 1, "reward": 0},
        {"positive_success": 0, "negative_success": 0, "reward": 0},
        {"positive_success": 0, "negative_success": 0, "reward": 0},
    ]
    credits = run.response_advantages([group], "pairing_mean")[0]
    assert any(value for pair in credits for value in pair)
    assert run.response_advantages([group], "diagonal") == [[(0, 0)] * 4]
    diagonal = run.response_advantages([[{"reward": value} for value in (0, 0, 0, 1)]], "diagonal")[
        0
    ]
    assert diagonal == [(-1 / 3, -1 / 3)] * 3 + [(1, 1)]

    pos_scores = [1.5, -2.0, 0.25, 4.0]
    neg_scores = [-1.0, 3.0, 2.5, -0.5]
    enumerated = 0.0
    for perm in permutations(range(4)):
        rewards = [
            group[i]["positive_success"] * group[perm[i]]["negative_success"] for i in range(4)
        ]
        advantages = run.rl.rloo(rewards)
        enumerated += sum(advantages[i] * (pos_scores[i] + neg_scores[perm[i]]) for i in range(4))
    closed = sum(p * s for (p, _), s in zip(credits, pos_scores, strict=True)) + sum(
        n * s for (_, n), s in zip(credits, neg_scores, strict=True)
    )
    assert closed == pytest.approx(enumerated / 24)


def test_pairing_mean_preserves_marginal_edge_cases_and_rejects_unknown_estimator():
    import rl_sufficiency as run

    all_zero = [{"positive_success": 0, "negative_success": 0, "reward": 0}] * 4
    assert run.response_advantages([all_zero], "pairing_mean") == [[(0, 0)] * 4]
    uniform_negative = [
        {"positive_success": value, "negative_success": 1, "reward": value}
        for value in (1, 0, 0, 0)
    ]
    credits = run.response_advantages([uniform_negative], "pairing_mean")[0]
    assert any(pos for pos, _ in credits)
    assert all(neg == 0 for _, neg in credits)
    with pytest.raises(ValueError, match="unknown estimator"):
        run.response_advantages([all_zero], "not-an-estimator")


def test_additive_reward_has_literal_diagonal_rloo_and_can_mix_when_product_is_flat():
    import rl_sufficiency as run

    literal = [
        {"positive_success": 0, "negative_success": 0, "reward": 0},
        {"positive_success": 1, "negative_success": 0, "reward": 0},
        {"positive_success": 0, "negative_success": 1, "reward": 0},
        {"positive_success": 1, "negative_success": 1, "reward": 1},
    ]
    assert [run.training_reward(pair, "additive") for pair in literal] == [0, 0.5, 0.5, 1]
    assert run.response_advantages([literal]) == run.response_advantages(
        [literal], "diagonal", "product"
    )
    advantages = run.response_advantages([literal], "diagonal", "additive")[0]
    assert [value for pair in advantages for value in pair] == pytest.approx(
        [-2 / 3, -2 / 3, 0, 0, 0, 0, 2 / 3, 2 / 3]
    )
    flat_product = [
        {"positive_success": 1, "negative_success": 0, "reward": 0},
        {"positive_success": 0, "negative_success": 0, "reward": 0},
        {"positive_success": 0, "negative_success": 1, "reward": 0},
        {"positive_success": 0, "negative_success": 0, "reward": 0},
    ]
    assert run.response_advantages([flat_product], "diagonal", "product") == [[(0, 0)] * 4]
    assert any(
        value
        for pair in run.response_advantages([flat_product], "diagonal", "additive")[0]
        for value in pair
    )


def test_additive_uniform_rewards_skip_adam_and_pairing_mean_combination_is_rejected():
    import rl_sufficiency as run

    uniform = [{"positive_success": 1, "negative_success": 0, "reward": 0, "records": []}] * 4
    assert run.response_advantages([uniform], "diagonal", "additive") == [[(0, 0)] * 4]
    assert run.optimize_rl(None, None, [uniform], time.time() + 90, "diagonal", "additive") == {
        "optimizer_called": False
    }
    with pytest.raises(ValueError, match="pairing_mean requires product"):
        run.response_advantages([uniform], "pairing_mean", "additive")


def test_pairing_mean_nonzero_credit_advances_step_when_diagonal_rewards_are_zero():
    import rl_sufficiency as run

    mispaired = [
        {"positive_success": 1, "negative_success": 0, "reward": 0},
        {"positive_success": 0, "negative_success": 1, "reward": 0},
        {"positive_success": 0, "negative_success": 0, "reward": 0},
        {"positive_success": 0, "negative_success": 0, "reward": 0},
    ]
    zero = [{"positive_success": 0, "negative_success": 0, "reward": 0}] * 4
    state = {"step": 0, "sample_cursor": 0, "zero_streak": 0}
    state = run.advance(state, run.response_advantages([mispaired] + [zero] * 15, "pairing_mean"))
    assert state == {"step": 1, "sample_cursor": 1, "zero_streak": 0, "effective_groups": 1}
    assert all(pair["reward"] == 0 for pair in mispaired)
    state = run.advance(state, run.response_advantages([zero] * 16, "pairing_mean"))
    assert state == {"step": 1, "sample_cursor": 2, "zero_streak": 1, "effective_groups": 0}


def test_native_paired_logp_gradient_includes_only_emitted_eos():
    import rl_sufficiency as run
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(9)
    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            )
        ),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    a = {"input_token_ids": [1, 2], "output_token_ids": [3, 4]}
    b = {"input_token_ids": [1, 5], "output_token_ids": [6]}  # capped: no fake EOS
    loss = run.pair_loss(model, [a, b], 1)
    expected = 0
    for record in (a, b):
        prefix, emitted = record["input_token_ids"], record["output_token_ids"]
        logits = model(input_ids=torch.tensor([prefix + emitted[:-1]]), use_cache=False).logits
        selected = torch.log_softmax(logits[:, len(prefix) - 1 :, :].float() / 0.8, dim=-1)
        expected -= selected.gather(-1, torch.tensor([emitted]).unsqueeze(-1)).sum() / 64
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "lora_" in n
    )
    assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
    with pytest.raises(ValueError):
        run.pair_reward(positive={"answer": "London"}, predictions=[None, None], available=False)


def test_native_sampling_replay_and_zero_adam_preserve_state(tmp_path):
    import rl_sufficiency as run
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            )
        ),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )

    class Tokenizer:
        eos_token_id = 15
        pad_token_id = 15

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return "decoded-native-ids"

    states = []
    layer = next(m for m in model.modules() if hasattr(m, "lora_A"))
    layer.register_forward_pre_hook(lambda m, args: states.append(not m.disable_adapters))
    client = run.NativeClient(
        model, Tokenizer(), tmp_path, time.time() + 90, {"fixture": "adapter"}
    )
    record = client.call("sample", {"public": {"question": "Where?", "documents": []}}, 42)
    assert states and all(states)
    assert record["request"]["sampling"]["temperature"] == 0.8
    assert record["available"] and len(record["generation_logps"]) == len(
        record["output_token_ids"]
    )
    model.eval()
    with torch.no_grad():
        replay = run.rl.root_logps(model, record).flatten()
    torch.testing.assert_close(
        replay, torch.tensor(record["generation_logps"]), atol=1e-5, rtol=1e-5
    )
    optimizer = torch.optim.AdamW(run.rl.planner_parameters(model), lr=2e-5)
    (-run.rl.root_logps(model, record).sum()).backward()
    optimizer.step()
    before = [p.detach().clone() for p in model.parameters()]
    adam_before = copy.deepcopy(optimizer.state_dict())
    result = run.optimize_rl(model, optimizer, [[{"reward": 0}] * 4] * 16, time.time() + 90)
    assert result == {"optimizer_called": False}
    for a, b in zip(before, model.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key, value in adam_before["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(
                tensor, optimizer.state_dict()["state"][key][field], atol=0, rtol=0
            )
    state = {"step": 0, "sample_cursor": 0, "zero_streak": 0}
    run.save_boundary(model, optimizer, tmp_path / "owner", state)
    state = run.advance(state, [[0, 0, 0, 0]] * 16)
    saved = run.save_boundary(model, optimizer, tmp_path / "owner", state)
    restored = run.boundary_inventory(tmp_path / "owner")
    assert restored[-1]["state"]["step"] == 0 and restored[-1]["state"]["sample_cursor"] == 1
    other = torch.optim.AdamW(run.rl.planner_parameters(model), lr=2e-5)
    other.load_state_dict(torch.load(saved / "optimizer.pt", weights_only=True))
    assert len(other.state_dict()["state"]) == len(adam_before["state"])


def test_pairing_mean_moves_tiny_peft_for_mispaired_group_then_skips_zero_block():
    import rl_sufficiency as run
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(41)
    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            )
        ),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    optimizer = torch.optim.AdamW(run.rl.planner_parameters(model), lr=2e-5, weight_decay=0)
    group = []
    for index, (positive, negative) in enumerate(((1, 0), (0, 1), (0, 0), (0, 0))):
        records = [
            {
                "call_id": f"p{index}",
                "input_token_ids": [1, 2 + index],
                "output_token_ids": [6 + index],
                "generation_logps": [0.0],
            },
            {
                "call_id": f"n{index}",
                "input_token_ids": [1, 12 + index],
                "output_token_ids": [16 + index],
                "generation_logps": [0.0],
            },
        ]
        group.append(
            {
                "positive_success": positive,
                "negative_success": negative,
                "reward": 0,
                "records": records,
            }
        )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = run.optimize_rl(model, optimizer, [group], time.time() + 180, "pairing_mean")
    assert result["parameter_delta_l2"] > 0
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
    after = [parameter.detach().clone() for parameter in model.parameters()]
    adam_after = copy.deepcopy(optimizer.state_dict())
    zero = [{"positive_success": 0, "negative_success": 0, "reward": 0}] * 4
    assert run.optimize_rl(model, optimizer, [zero], time.time() + 180, "pairing_mean") == {
        "optimizer_called": False
    }
    for old, new in zip(after, model.parameters(), strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    for key, value in adam_after["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(
                tensor, optimizer.state_dict()["state"][key][field], atol=0, rtol=0
            )


def test_boundary_cursors_reject_skipped_updates_and_out_of_range():
    import rl_sufficiency as run

    initial = {"step": 0, "sample_cursor": 0, "zero_streak": 0}
    run.validate_boundary_state(initial)
    run.validate_boundary_state({"step": 1, "sample_cursor": 1, "zero_streak": 0}, initial)
    run.validate_boundary_state({"step": 0, "sample_cursor": 1, "zero_streak": 1}, initial)
    for state in (
        {"step": 2, "sample_cursor": 1, "zero_streak": 0},
        {"step": 0, "sample_cursor": 1, "zero_streak": 0},
        {"step": 0, "sample_cursor": 9, "zero_streak": 1},
    ):
        with pytest.raises(ValueError):
            run.validate_boundary_state(state, initial)


def test_control_has_four_copies_per_variant_and_no_host_fields_in_prompt():
    import rl_sufficiency as run

    cases = [
        {
            "parent_id": "p",
            "answerable": label,
            "answer": "SECRET",
            "public": {"question": "Where?", "documents": []},
        }
        for label in (True, False)
    ]
    rows = run.control_examples(cases, ["p"])
    assert len(rows) == 8 and len({r["id"] for r in rows}) == 8
    assert sum('"answerable":true' in r["target"] for r in rows) == 4
    assert sum('"answer":""' in r["target"] for r in rows) == 4
    assert all("SECRET" not in r["prompt"] for r in rows)
