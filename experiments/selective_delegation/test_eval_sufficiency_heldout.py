import hashlib
import json

import pytest


def fixture_endpoint(tmp_path, *, step=1, mode="rl"):
    import eval_sufficiency_heldout as m

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    output = tmp_path / mode
    plan = {"mode": mode, "warmstart": {"fixed": "joint32"}, "cases_sha256": "TRAIN"}
    save(output / "PLAN.json", plan)
    for cursor in range(step + 1):
        checkpoint = output / f"boundaries/sample-{cursor:04d}" / f"checkpoint-{cursor:04d}"
        state = {
            "step": cursor,
            "sample_cursor": cursor,
            "zero_streak": 0,
            "plan_sha256": m.sha(output / "PLAN.json"),
        }
        save(checkpoint / "STATE.json", state)
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "rng.pt"):
            (checkpoint / name).write_text(name)
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}
        save(checkpoint / "COMMIT.json", {"step": cursor, "files": files})
        save(
            checkpoint.parent / "BOUNDARY.json",
            {
                "checkpoint": str(checkpoint),
                "state": state,
                "commit_sha256": m.sha(checkpoint / "COMMIT.json"),
            },
        )
    save(output / "OWNER-fixture.json", {"pid": 999999999, "create_time": 0})
    save(
        output / "TERMINAL-fixture.json",
        {
            **state,
            "failure": None,
            "stopped": False,
            "endpoint": str(checkpoint),
            "endpoint_selection": "last committed boundary, never held score",
        },
    )
    save(
        output / "SUMMARY.json",
        {
            "failure": None,
            "endpoint": str(checkpoint),
            "actual_optimizer_steps": step,
            "committed_sampled_blocks": step,
            "matched_control_complete": mode == "sft_control",
        },
    )
    return output, plan


def test_terminal_endpoint_and_commit_are_authoritative(tmp_path):
    import eval_sufficiency_heldout as m

    output, plan = fixture_endpoint(tmp_path)
    identity = m.endpoint_identity(output, "rl", plan["warmstart"])
    assert identity["step"] == 1
    checkpoint = m.Path(identity["path"])
    (checkpoint / "adapter_model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="checkpoint file"):
        m.endpoint_identity(output, "rl", plan["warmstart"])


def test_failed_or_wrong_warmstart_endpoint_rejected(tmp_path):
    import eval_sufficiency_heldout as m

    output, plan = fixture_endpoint(tmp_path)
    with pytest.raises(ValueError, match="warmstart"):
        m.endpoint_identity(output, "rl", {})
    terminal = output / "TERMINAL-fixture.json"
    row = json.loads(terminal.read_text())
    row["failure"] = "observed inference failure"
    terminal.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="failed"):
        m.endpoint_identity(output, "rl", plan["warmstart"])


def test_zero_updates_skip_and_fixed_pairing():
    import eval_sufficiency_heldout as m

    assert m.skip_reason({"step": 0}) == "zero RL optimizer updates; no duplicate triple readout"
    assert m.skip_reason({"step": 2}) is None
    cases = [{"id": "opaque", "parent_id": "host-secret"}]
    assert m.jobs(cases) == [
        {
            "episode_id": f"opaque-{seed}",
            "case_id": "opaque",
            "parent_id": "host-secret",
            "seed": seed,
        }
        for seed in (2026092181, 2026092182)
    ]


def test_prompt_identical_and_host_secrets_excluded():
    import eval_sufficiency_heldout as m

    case = {
        "public": {"question": "Which?", "documents": []},
        "answer": "GOLD-SECRET",
        "parent_id": "HOST-SECRET",
    }
    assert m.baseline.prompt(case) == m.adapter_runtime.prompt(case)
    assert "SECRET" not in m.baseline.prompt(case)


def test_three_named_adapters_route_native_cpu_calls_and_stay_frozen(tmp_path):
    import time

    import eval_sufficiency_heldout as m
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM")
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
        config,
        adapter_name=m.CONDITIONS[0],
    )
    for condition in m.CONDITIONS[1:]:
        model.add_adapter(condition, config)
    model.eval()
    seen = []
    layer = next(module for module in model.modules() if hasattr(module, "lora_A"))
    layer.register_forward_pre_hook(lambda module, args: seen.append(tuple(module.active_adapters)))

    class Tokenizer:
        eos_token_id = 31
        pad_token_id = 31

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            return [1, 2, 3]

        def decode(self, ids, **kwargs):
            return '{"answerable":false,"answer":""}'

    for condition in m.CONDITIONS:
        m.activate(model, condition)
        seen.clear()
        identity = {"files": {"adapter_model.safetensors": condition}}
        client = m.adapter_runtime.AdapterClient(
            model, Tokenizer(), tmp_path / condition, time.time() + 90, identity
        )
        row = client.call(
            "fixture",
            "public prompt",
            condition,
            "sufficiency",
            2026092181,
            2,
            {"truncation": False},
        )
        assert row["available"] and seen and set(seen) == {(condition,)}
        assert row["request"]["adapter_sha256"] == condition
        assert row["request"]["sampling"]["temperature"] == 0.5
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not torch.cuda.is_initialized()
