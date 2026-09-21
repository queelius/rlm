import copy
import time

import pytest


def test_paired_reward_and_zero_cursor_not_optimizer():
    import rl_sufficiency as run

    positive = {"answer": "London", "answer_aliases": []}
    good = [{"answerable": True, "answer": "London"}, {"answerable": False, "answer": ""}]
    assert run.pair_reward(positive, good) == 1
    wrong = copy.deepcopy(good)
    wrong[0] = {"answerable": False, "answer": ""}
    assert run.pair_reward(positive, wrong) == 0
    state = {"step": 0, "sample_cursor": 0, "zero_streak": 0}
    for i in range(4):
        state = run.advance(state, [[0, 0, 0, 0]] * 16)
        assert state["sample_cursor"] == i + 1 and state["step"] == 0
    assert run.finished(state)
    state = run.advance(
        {"step": 0, "sample_cursor": 0, "zero_streak": 0}, [[0, 0, 0, 1]] + [[0, 0, 0, 0]] * 15
    )
    assert state["step"] == 1 and state["effective_groups"] == 1


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
