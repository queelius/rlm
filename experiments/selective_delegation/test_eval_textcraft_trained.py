import time

import eval_textcraft as collector


def test_frozen_adapter_native_receipt_and_condition_are_honest(tmp_path):
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
        adapter_name="textcraft_action",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    class Tokenizer:
        eos_token_id = pad_token_id = 15

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return "native:" + str(ids)

    binding = {"path": "fixed-checkpoint", "sha256": "weights", "commit_sha256": "commit"}
    client = collector.NativeClient(
        model, Tokenizer(), tmp_path, time.time() + 60, "model", "plan", adapter=binding
    )
    row = client.call(
        dict(
            call_id="one",
            prompt="public",
            policy="flat",
            condition="trained_original",
            role="root",
            depth=0,
            node_id="n0",
            parent_node_id=None,
            episode_id="one",
            task_id="task",
            global_call_index=0,
            seed=7,
            cap=8,
        )
    )
    assert row["available"] and row["condition"] == "trained_original"
    assert row["request"]["adapter_enabled"] is True
    assert row["request"]["adapter_sha256"] == "weights"
    assert row["request"]["adapter_commit_sha256"] == "commit"
    assert not any(p.requires_grad for p in model.parameters())


def test_trained_jobs_preserve_profiles_and_fixed_common_seeds():
    import eval_textcraft_trained as trained

    base = [
        dict(
            episode_id=f"t{i:02}-r{rep}-flat",
            task_id=f"task{i}",
            repeat=rep,
            seed=collector.SEEDS[rep],
            policy="flat",
        )
        for i in range(8)
        for rep in range(2)
    ]
    jobs = trained.paired_jobs(base)
    assert len(jobs) == len({j["episode_id"] for j in jobs}) == 32
    for left, right in zip(jobs[::2], jobs[1::2], strict=True):
        assert left["seed"] == right["seed"] and left["task_id"] == right["task_id"]
        assert left["policy"] == right["policy"] == "flat"
        assert (left["condition"], right["condition"]) == (
            "trained_original",
            "trained_instruction_control",
        )
        assert (left["prompt_profile"], right["prompt_profile"]) == (
            "original",
            "instruction_control",
        )


def test_endpoint_dose_requires_all23_updates_not_just_directory():
    import eval_textcraft_trained as trained
    import pytest

    plan = dict(
        schema="textcraft-public-flat-action-sft-v1",
        model=str(collector.BASE),
        rows=366,
        tasks=32,
        planned_updates=23,
        epochs=1,
        effective_batch=16,
        last_update_rows=14,
        target_only_json_eos=True,
    )
    state = dict(step=23, epoch=1, cursor=0)
    steps = [dict(step=i + 1, rows=16 if i < 22 else 14) for i in range(23)]
    trained.validate_dose(plan, state, steps)
    with pytest.raises(ValueError):
        trained.validate_dose(plan, {**state, "step": 22}, steps)
    with pytest.raises(ValueError):
        trained.validate_dose(plan, state, steps[:-1])
