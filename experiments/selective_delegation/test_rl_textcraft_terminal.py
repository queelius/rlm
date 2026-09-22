import importlib.util
import json
import time

import pytest
import torch


def trainer():
    assert importlib.util.find_spec("rl_textcraft_terminal") is not None
    import rl_textcraft_terminal

    return rl_textcraft_terminal


def tiny_model():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    return get_peft_model(
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
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0, target_modules=["q_proj"], task_type="CAUSAL_LM"
        ),
        adapter_name="textcraft_action",
    )


def test_native_scores_replay_and_mode_switch_targets_only_lora(tmp_path):
    r = trainer()
    model = tiny_model()
    r.set_mode(model, training=False)
    assert not any(p.requires_grad for p in model.parameters()) and model.config.use_cache

    class Tokenizer:
        eos_token_id = pad_token_id = 15

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return "native:" + str(ids)

    client = r.ScoredClient(
        model,
        Tokenizer(),
        tmp_path,
        time.time() + 60,
        "base",
        "plan",
        adapter=dict(path="warm", sha256="weights", commit_sha256="commit"),
    )
    call = client.call(
        dict(
            call_id="one",
            prompt="public",
            policy="flat",
            role="root",
            depth=0,
            node_id="n0",
            parent_node_id=None,
            episode_id="ep",
            task_id="task",
            global_call_index=0,
            seed=42,
            cap=5,
        )
    )
    assert call["available"]
    capture = json.loads((tmp_path / "generation-logps/one.json").read_text())
    before = r.action_logps(model, call)
    assert before.detach().flatten().tolist() == pytest.approx(capture["logps"], abs=1e-5)
    assert len(capture["logps"]) == len(call["output_token_ids"])
    params = r.set_mode(model, training=True)
    train_logps = r.action_logps(model, call)
    assert train_logps.detach().flatten().tolist() == pytest.approx(
        before.detach().flatten().tolist(), abs=1e-5
    )
    (-train_logps.sum() / 32).backward()
    assert not model.config.use_cache
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
    r.set_mode(model, training=False)
    assert not any(p.requires_grad for p in model.parameters())


def test_flat_boundary_is_committed_without_advancing_adam_or_stopping_first(tmp_path):
    r = trainer()
    model = tiny_model()
    params = r.set_mode(model, training=True)
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0)
    initial = dict(step=0, sample_cursor=0, zero_streak=0, plan_sha256="plan")
    a = r.save_boundary(model, optimizer, tmp_path, initial)
    skipped = r.advance(initial, updated=False)
    assert skipped["step"] == 0 and skipped["sample_cursor"] == 1
    assert not r.finished(skipped)
    b = r.save_boundary(model, optimizer, tmp_path, skipped)
    assert a != b and a.name == b.name == "checkpoint-0000"
    assert json.loads((b / "STATE.json").read_text())["zero_streak"] == 1
    assert torch.load(b / "optimizer.pt", weights_only=True)["state"] == {}
    assert r.finished(r.advance(skipped, updated=False))
    assert r.advance(skipped, updated=True)["step"] == 1
    with pytest.raises(ValueError, match="Adam"):
        r.save_boundary(model, optimizer, tmp_path, {**skipped, "sample_cursor": 2, "step": 1})
    sum(p.sum() for p in params).backward()
    optimizer.step()
    committed = r.save_boundary(model, optimizer, tmp_path, r.advance(skipped, updated=True))
    assert (committed / "adapter_model.safetensors").is_file()
    assert json.loads((committed / "COMMIT.json").read_text())["step"] == 1
    restored_adam = torch.load(committed / "optimizer.pt", weights_only=True)
    assert {int(v["step"]) for v in restored_adam["state"].values()} == {1}


def test_prepare_only_binds_warm_and_all063_slots_without_reading_outcomes(tmp_path):
    r = trainer()
    from argparse import Namespace

    plan = r.prepare(Namespace(root=r.c.ROOT, output=tmp_path, hours=3, prepare_only=True))
    assert plan["planned_tasks"] == 8 and plan["episodes_per_batch"] == 32
    assert plan["maximum_optimizer_updates"] == 2 and plan["maximum_sampled_batches"] == 4
    assert plan["warm"]["training_plan_sha256"] == r.reader.public.PLAN_SHA
    assert plan["first_batch"]["generation_logps"] == "not_recorded; recompute_under_exact056"
    assert len(plan["fresh_jobs"]["2"]) == 32


@pytest.mark.parametrize("maximum,mean", [(0.26, 0.01), (0.10, 0.026)])
def test_replay_guard_saves_failure_before_any_optimizer_action(tmp_path, maximum, mean):
    r = trainer()

    class Optimizer:
        steps = 0

        def step(self):
            self.steps += 1

    optimizer = Optimizer()
    assert hasattr(r, "checked_optimizer_step")
    with pytest.raises(ValueError, match="train/eval"):
        r.checked_optimizer_step(
            optimizer,
            tmp_path,
            previous_step=0,
            maximum=maximum,
            mean=mean,
            count=20,
            max_tolerance=0.25,
            mean_tolerance=0.025,
        )
    evidence = json.loads((tmp_path / "TRAIN-EVAL-REPLAY.json").read_text())
    assert evidence["passed"] is False and evidence["max_abs"] == maximum
    assert evidence["mean_abs"] == mean and optimizer.steps == 0
    assert not (tmp_path / "OPTIMIZER-STEP-STARTED.json").exists()
