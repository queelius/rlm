import pytest
import train_textcraft_matched_sft as m


def test_whole_rows_cross_cycle_and_report_overshoot():
    rows = [{"id": "a", "target_ids": [1, 2, 3]}, {"id": "b", "target_ids": [4, 5]}]
    schedule = m.row_schedule(rows, [0, 1], [4, 6])
    assert [s["row_ids"] for s in schedule] == [["a", "b"], ["a", "b", "a"]]
    assert [s["actual_target_tokens"] for s in schedule] == [5, 8]
    assert [s["overshoot_tokens"] for s in schedule] == [1, 2]


def test_budget_counts_invalid_actions_but_not_flat_groups_or_synthetic_eos():
    episodes, calls = [], {}
    for task in range(8):
        for repeat in range(4):
            cid = f"{task}-{repeat}"
            episodes.append(
                dict(
                    task_id=str(task),
                    episode_id=cid,
                    repeat=repeat,
                    observed=True,
                    native_score=int(task == 0 and repeat == 0),
                    call_ids=[cid],
                )
            )
            calls[cid] = dict(
                call_id=cid,
                episode_id=cid,
                available=True,
                input_token_ids=[9],
                output_token_ids=[1, 2],
                text="invalid",
                request=dict(
                    task_id=str(task),
                    input_token_ids=[9],
                    sampling=dict(temperature=0.5, top_p=1.0, top_k=0),
                    model_manifest_sha256="m",
                    adapter_sha256="a",
                    adapter_commit_sha256="c",
                ),
            )
    update = dict(optimizer_called=True, nonzero_action_calls=4, train_eval_replay_token_count=8)
    assert m.credited_budget(episodes, calls, update) == 8
    update["train_eval_replay_token_count"] = 9
    with pytest.raises(ValueError, match="committed"):
        m.credited_budget(episodes, calls, update)


def test_failed_or_mismatched_endpoint_never_becomes_partial_dose():
    terminal = dict(
        complete=True,
        endpoint_usable=True,
        failure=None,
        stopped=False,
        actual_optimizer_steps=2,
        committed_optimizer_steps=2,
    )
    summary = dict(
        complete=True, failure=None, actual_optimizer_steps=2, committed_optimizer_steps=2
    )
    assert m.endpoint_steps(terminal, summary) == 2
    assert (
        m.endpoint_steps(
            {**terminal, "actual_optimizer_steps": 0, "committed_optimizer_steps": 0},
            {**summary, "actual_optimizer_steps": 0, "committed_optimizer_steps": 0},
        )
        == 0
    )
    with pytest.raises(ValueError):
        m.endpoint_steps({**terminal, "endpoint_usable": False}, summary)
    with pytest.raises(ValueError):
        m.endpoint_steps(terminal, {**summary, "committed_optimizer_steps": 1})


def test_tiny_native_gold_loss_and_committed_actual_adam_step(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                attention_dropout=0,
                eos_token_id=2,
            )
        ),
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0, target_modules=["q_proj"]),
    )
    row = dict(input_ids=[3, 4, 5, 6], target_ids=[6, 2])
    logits = model(
        input_ids=torch.tensor([[3, 4, 5, 6]]), logits_to_keep=2, use_cache=False
    ).logits.float()
    expected = -(logits.log_softmax(-1)[0, 0, 6] + logits.log_softmax(-1)[0, 1, 2]) / 2
    loss = m.teacher_loss(model, row, 2)
    assert torch.allclose(loss, expected)
    optimizer = torch.optim.AdamW(
        [v for v in model.parameters() if v.requires_grad], lr=2e-5, weight_decay=0
    )
    loss.backward()
    assert all(v.grad is None for n, v in model.named_parameters() if "lora_" not in n)
    optimizer.step()
    path = m.checkpoint.save_checkpoint(model, optimizer, tmp_path, dict(step=1))
    saved = torch.load(path / "optimizer.pt", weights_only=True)
    assert {int(v["step"]) for v in saved["state"].values()} == {1}
    assert m.read(path / "STATE.json")["step"] == 1
