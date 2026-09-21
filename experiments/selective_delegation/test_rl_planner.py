"""CPU contracts for on-policy root-only RLOO, sampling, and frozen downstream weights."""

import importlib.util
import random
from pathlib import Path

import pytest
import torch


def implementation():
    path = Path(__file__).with_name("rl_planner.py")
    assert path.exists(), "planner RL implementation missing"
    spec = importlib.util.spec_from_file_location("rl_planner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rloo_advantages_and_root_sequence_loss_have_correct_gradient_direction():
    mod = implementation()
    assert mod.rloo([0, 1, 0, 1]) == pytest.approx([-2 / 3, 2 / 3, -2 / 3, 2 / 3])
    assert mod.rloo([1, 1, 1, 1]) == [0, 0, 0, 0]
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    tokens = torch.tensor([[1, 1]])
    logps = mod.token_logps(logits, tokens)
    loss = mod.policy_loss(logps, 2 / 3)
    loss.backward()
    # Both emitted positions (including a final EOS when present) have equal credit.
    assert torch.allclose(logits.grad[0, 0], logits.grad[0, 1])
    assert logits.grad[0, 0, 1] < 0 and logits.grad[0, 0, 0] > 0
    assert float(loss.detach()) == pytest.approx(
        (2 / 3) * 2 * torch.log(torch.tensor(2.0)).item() / 64
    )


def test_seeds_vary_only_root_candidate_and_keep_downstream_common():
    mod = implementation()
    rows = [mod.seed_schedule(1, 3, candidate) for candidate in range(4)]
    assert len({r["root"] for r in rows}) == 4
    assert len({r["downstream"] for r in rows}) == 1
    assert mod.seed_schedule(2, 3, 0) != rows[0]


def test_lora_only_update_changes_planner_but_disabled_helpers_stay_frozen():
    mod = implementation()
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(13)
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_dropout=0.0,
        )
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.eval()
    params = mod.planner_parameters(model)
    ids = torch.tensor([[1, 2, 3]])
    with model.disable_adapter(), torch.no_grad():
        before_helper = model(input_ids=ids, use_cache=False).logits.clone()
    before_planner = model(input_ids=ids, use_cache=False).logits.detach().clone()
    optimizer = torch.optim.SGD(params, lr=0.5)
    loss = mod.policy_loss(
        mod.token_logps(model(input_ids=ids, use_cache=False).logits[:, -1:], torch.tensor([[4]])),
        1.0,
    )
    loss.backward()
    assert all(p.grad is None for name, p in model.named_parameters() if "lora_" not in name)
    optimizer.step()
    with model.disable_adapter(), torch.no_grad():
        after_helper = model(input_ids=ids, use_cache=False).logits
    after_planner = model(input_ids=ids, use_cache=False).logits.detach()
    assert torch.equal(before_helper, after_helper)
    assert not torch.equal(before_planner, after_planner)


def test_admission_requires_two_diverse_mixed_groups_without_dropping_other_groups():
    mod = implementation()
    groups = [
        [
            {"reward": 0, "plan_valid": True, "plan": {"subquestions": ["A"]}},
            {"reward": 1, "plan_valid": True, "plan": {"subquestions": ["B"]}},
            {"reward": 0, "plan_valid": False},
            {"reward": 0, "plan_valid": False},
        ]
    ]
    assert not mod.batch_diagnostics(groups + [groups[0][:1] * 4] * 15)["admitted"]
    result = mod.batch_diagnostics(groups * 2 + [groups[0][:1] * 4] * 14)
    assert result["admitted"] and result["qualifying_groups"] == 2
    assert result["episodes"] == 64 and len(result["groups"]) == 16


def test_valid_vs_invalid_plan_variation_alone_does_not_admit_update():
    mod = implementation()
    group = [
        {"reward": 1, "plan_valid": True, "plan": {"subquestions": ["A"]}},
        {"reward": 1, "plan_valid": True, "plan": {"subquestions": ["B"]}},
        {"reward": 0, "plan_valid": False},
        {"reward": 0, "plan_valid": False},
    ]
    result = mod.batch_diagnostics([group] * 16)
    assert not result["admitted"]
    assert result["groups"][0]["root_protocol_only_variation"]


def test_root_logprob_alignment_includes_every_emitted_token_including_eos():
    mod = implementation()
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_dropout=0.0,
        )
    ).eval()
    record = {"input_token_ids": [1, 5], "output_token_ids": [7, 9, 2]}
    selected = mod.root_logps(model, record)
    full = model(input_ids=torch.tensor([[1, 5, 7, 9, 2]]), use_cache=False).logits[:, 1:4]
    expected = (
        torch.log_softmax(full.float() / 0.8, -1)
        .gather(-1, torch.tensor([[[7], [9], [2]]]))
        .squeeze(-1)
    )
    assert selected.shape == (1, 3)
    assert torch.allclose(selected, expected, atol=1e-6)


def test_fixed_parent_blocks_preserve_first_panel_and_repeat_default():
    mod = implementation()
    training = [{"id": f"p{i:03}", "split": "train"} for i in range(256)]
    repeated = mod.parent_schedule(training, 4)
    blocks = mod.parent_schedule(training, 4, "consecutive")
    assert all(batch == repeated[0] for batch in repeated)
    assert blocks[0] == repeated[0]
    assert len({c["id"] for batch in blocks for c in batch}) == 64
    assert all(len(batch) == 16 for batch in blocks)
    assert blocks == mod.parent_schedule(list(reversed(training)), 4, "consecutive")


def test_consecutive_full_pass_covers_all_training_parents_once_with_fixed_order():
    mod = implementation()
    training = [{"id": f"p{i:03}", "split": "train"} for i in range(256)]
    blocks = mod.parent_schedule(training, 16, "consecutive")
    assert len(blocks) == 16 and all(len(batch) == 16 for batch in blocks)
    flattened = [case["id"] for batch in blocks for case in batch]
    assert len(set(flattened)) == 256
    assert set(flattened) == {case["id"] for case in training}
    assert blocks[:4] == mod.parent_schedule(training, 4, "consecutive")
    assert blocks == mod.parent_schedule(list(reversed(training)), 16, "consecutive")
    for count in (0, 17):
        with pytest.raises(ValueError, match="bounded"):
            mod.parent_schedule(training, count, "consecutive")
    with pytest.raises(ValueError, match="bounded"):
        mod.parent_schedule(training, 5, "repeated")


def test_continuation_schedule_preserves_prefix_and_uses_second_shuffle_global_updates():
    mod = implementation()
    training = [{"id": f"p{i:03}", "split": "train"} for i in range(256)]
    blocks = mod.continuation_schedule(training)
    assert blocks[:16] == mod.parent_schedule(training, 16, "consecutive")
    expected = list(training)
    random.Random(2026092109).shuffle(expected)
    assert [c for b in blocks[16:] for c in b] == expected[:128]
    assert len(blocks) == 24 and len({c["id"] for b in blocks[16:] for c in b}) == 128
    assert mod.seed_schedule(17, 0, 0) != mod.seed_schedule(1, 0, 0)


def test_continuation_bounds_cannot_expand_fresh_run_or_dose():
    mod = implementation()
    mod.validate_run_bounds(4, 3, "repeated", False)
    mod.validate_run_bounds(16, 3, "consecutive", False)
    mod.validate_run_bounds(24, 1.5, "consecutive", True)
    for args in (
        (24, 3, "consecutive", False),
        (25, 1, "consecutive", True),
        (24, 1.6, "consecutive", True),
        (24, 1, "repeated", True),
    ):
        with pytest.raises(ValueError):
            mod.validate_run_bounds(*args)


def test_restore_retains_adam_moments_steps_and_python_torch_rng(tmp_path):
    mod = implementation()
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    old = torch.optim.AdamW([parameter], lr=2e-5, weight_decay=0.0)
    for _ in range(16):
        old.zero_grad()
        parameter.square().sum().backward()
        old.step()
    torch.save(old.state_dict(), tmp_path / "optimizer.pt")
    random.seed(17)
    torch.manual_seed(17)
    rng = {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": []}
    torch.save(rng, tmp_path / "rng.pt")
    expected_random, expected_torch = random.random(), torch.rand(3)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = mod.restore_optimizer_rng([restored_parameter], tmp_path, 16)
    assert random.random() == expected_random
    assert torch.equal(torch.rand(3), expected_torch)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(restored.state[restored_parameter][key], old.state[parameter][key])
    for optimizer, param in ((old, parameter), (restored, restored_parameter)):
        optimizer.zero_grad()
        param.square().sum().backward()
        optimizer.step()
    assert torch.equal(parameter, restored_parameter)
    with pytest.raises(ValueError, match="optimizer step"):
        mod.restore_optimizer_rng([restored_parameter], tmp_path, 17)
    bad = old.state_dict()
    bad["param_groups"][0]["lr"] = 1e-3
    torch.save(bad, tmp_path / "optimizer.pt")
    with pytest.raises(ValueError, match="hyperparameters"):
        mod.restore_optimizer_rng([restored_parameter], tmp_path, 17)


def test_continuation_ancestry_contract_and_local_resume_precedence(tmp_path):
    import copy

    mod = implementation()
    training = [{"id": f"p{i:03}", "split": "train"} for i in range(256)]
    schedule = [[c["id"] for c in b] for b in mod.continuation_schedule(training)]
    old = {
        "schema": "fresh-planner-rloo-v1",
        "parent_schedule": "consecutive",
        "case_ids_by_update": schedule[:16],
        "updates": 16,
        "helper_contract": {"mode": "trained_helper"},
        "cases_sha256": "cases",
        "adapter": "sft48",
        "adapter_binding": {},
        "model": "base",
        "base_manifest_sha256": "base-sha",
        "seed": 2026092108,
        "parents_per_update": 16,
        "candidates_per_parent": 4,
        "denominator": 64,
        "root_temperature": 0.8,
        "helper_final_temperature": 0.5,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "clip": 1.0,
        "caps": {"root": 128, "helper_total": 384, "final": 128},
        "objective": "RLOO",
        "admission": "fixed",
        "frozen_helpers": "fixed",
        "dependencies": {},
        "environment": {"torch": "T", "transformers": "F", "peft": "P"},
    }
    ancestor = tmp_path / "ancestor"
    checkpoint = ancestor / "checkpoint-0016"
    checkpoint.mkdir(parents=True)
    state = {
        "step": 16,
        "cursor": 0,
        "next_update": 17,
        "case_ids": schedule[15],
        "helper_contract": old["helper_contract"],
        "component_identity": mod.component_identity(old),
    }

    def commit(path, value):
        mod.probe.runtime.save(path / "STATE.json", value)
        for name in ("optimizer.pt", "rng.pt", "adapter_model.safetensors", "adapter_config.json"):
            (path / name).write_text("fixture")
        mod.probe.runtime.save(
            path / "COMMIT.json",
            {
                "step": value["step"],
                "files": {
                    p.name: mod.probe.campaign.sha(p)
                    for p in path.iterdir()
                    if p.name != "COMMIT.json"
                },
            },
        )

    commit(checkpoint, state)
    mod.probe.runtime.save(ancestor / "PLAN.json", old)
    mod.probe.runtime.save(ancestor / "OWNER-a.json", {})
    mod.probe.runtime.save(
        ancestor / "TERMINAL-a.json",
        {"failure": None, "state": "completed_updates", "optimizer_steps": 16},
    )
    new = {**old, "updates": 24, "case_ids_by_update": schedule}
    ancestry, restored_state = mod.validate_continuation(checkpoint, new)
    assert restored_state == state
    assert ancestry["starting_step"] == 16 and ancestry["additional_updates"] == 8
    assert ancestry["checkpoint_files_sha256"]["optimizer.pt"]
    for key, value in (
        ("learning_rate", 0.01),
        ("helper_contract", {"mode": "base"}),
        ("base_manifest_sha256", "other"),
    ):
        with pytest.raises(ValueError, match="continuation contract"):
            mod.validate_continuation(checkpoint, {**new, key: value})
    bad = copy.deepcopy(new)
    bad["case_ids_by_update"][0].reverse()
    with pytest.raises(ValueError, match="prefix"):
        mod.validate_continuation(checkpoint, bad)
    plan = {**new, "continuation": ancestry}
    output = tmp_path / "fork"
    output.mkdir()
    restored, s = mod.select_restore(output, plan, False, restored_state)
    assert restored == checkpoint and s["step"] == 16
    local = output / "checkpoint-0017"
    local.mkdir()
    commit(
        local,
        {
            **state,
            "step": 17,
            "next_update": 18,
            "case_ids": schedule[16],
            "component_identity": mod.component_identity(plan),
            "continuation_identity": mod.probe.runtime.digest(ancestry),
        },
    )
    restored, s = mod.select_restore(output, plan, True, restored_state)
    assert restored == local and s["step"] == 17
    with pytest.raises(ValueError, match="resume"):
        mod.select_restore(output, plan, False, restored_state)
    mod.probe.runtime.save(ancestor / "OWNER-unresolved.json", {})
    with pytest.raises(ValueError, match="completed ancestor"):
        mod.validate_continuation(checkpoint, new)


def test_reminder_is_exact_eval_helper_contract_and_checkpoint_identity_is_bound():
    import eval_helper

    mod = implementation()
    case = {"question": "Original?", "documents": [{"id": "p", "title": "T", "text": "DOC"}]}
    base = mod.rollout_helper_prompt(case, "Step?", {"mode": "base"})
    reminder = mod.build_helper_contract("format_reminder")
    assert reminder["format_reminder"] == eval_helper.FORMAT_REMINDER
    assert mod.rollout_helper_prompt(case, "Step?", reminder) == base + eval_helper.FORMAT_REMINDER
    plan = {
        "helper_contract": reminder,
        "parent_schedule": "consecutive",
        "case_ids_by_update": [["a"], ["b"]],
    }
    binding = mod.component_identity(plan)
    mod.validate_component_resume({"component_identity": binding}, plan)
    changed = {**plan, "helper_contract": {**reminder, "format_reminder": "other"}}
    with pytest.raises(ValueError, match="component"):
        mod.validate_component_resume({"component_identity": binding}, changed)
    with pytest.raises(ValueError, match="adapter"):
        mod.build_helper_contract("base", Path("unused"))


def test_two_tiny_models_route_freeze_helper_and_keep_final_base_unchanged():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    mod = implementation()

    def tiny():
        return get_peft_model(
            Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    attention_dropout=0.0,
                )
            ),
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        ).eval()

    root, helper = tiny(), tiny()
    parameters = mod.planner_parameters(root)
    mod.freeze_helper_model(helper, root)
    mod.assert_frozen_helper(helper, parameters)
    assert mod.model_for_role(root, helper, "root", "trained_helper") == (root, True, "root")
    assert mod.model_for_role(root, helper, "helper", "trained_helper") == (helper, True, "helper")
    assert mod.model_for_role(root, helper, "final", "trained_helper") == (root, False, "root")
    assert mod.model_for_role(root, None, "helper", "format_reminder") == (root, False, "root")
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad(), root.disable_adapter():
        final_before = root(input_ids=ids).logits.clone()
    with torch.no_grad():
        helper_before = helper(input_ids=ids).logits.clone()
    optimizer = torch.optim.SGD(parameters, lr=0.5)
    mod.policy_loss(
        mod.root_logps(root, {"input_token_ids": [1, 2], "output_token_ids": [3]}), 1.0
    ).backward()
    optimizer.step()
    mod.assert_frozen_helper(helper, parameters)
    with torch.no_grad(), root.disable_adapter():
        assert torch.equal(final_before, root(input_ids=ids).logits)
    with torch.no_grad():
        assert torch.equal(helper_before, helper(input_ids=ids).logits)
    with pytest.raises(ValueError, match="separate"):
        mod.freeze_helper_model(root, root)


def test_native_call_receipts_keep_root_temperature_and_separate_frozen_helper_identity(tmp_path):
    import time
    from contextlib import contextmanager
    from types import SimpleNamespace

    mod = implementation()

    class Model:
        device = "cpu"

        def __init__(self):
            self.config = SimpleNamespace(use_cache=False)
            self.disabled = False
            self.calls = []

        def eval(self):
            return self

        def gradient_checkpointing_disable(self):
            pass

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def generate(self, input_ids, generation_config, **kwargs):
            self.calls.append(
                (self.disabled, generation_config.temperature, generation_config.output_scores)
            )
            return SimpleNamespace(
                sequences=torch.cat((input_ids, torch.tensor([[3, 2]])), dim=1),
                scores=[torch.zeros((1, 4))] * 2,
            )

    class Tokenizer:
        eos_token_id = pad_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return '{"answer":"actual"}'

    root, helper = Model(), Model()
    client = mod.Client(
        root,
        Tokenizer(),
        tmp_path,
        time.time() + 100,
        "root-sha",
        helper_model=helper,
        helper_contract={
            "mode": "trained_helper",
            "adapter_binding": {"adapter_model.safetensors": "helper-sha"},
        },
    )
    rows = [client.call(role, "prompt", role, 1, 128) for role in ("root", "helper", "final")]
    assert root.calls == [(False, 0.8, True), (True, 0.5, False)]
    assert helper.calls == [(False, 0.5, False)]
    assert [row["adapter_sha256"] for row in rows] == ["root-sha", "helper-sha", None]
    assert [row["model_instance"] for row in rows] == ["root", "helper", "root"]
    assert ["generation_logps" in row for row in rows] == [True, False, False]
    assert all(row["available"] and row["usage"]["completion_tokens"] == 2 for row in rows)


def test_trained_helper_contract_requires_committed_helper_role_and_fixed_dose(tmp_path):
    import json

    mod = implementation()
    adapter = tmp_path / "checkpoint-0036"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture-only-not-a-model")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_dropout": 0}))
    (adapter / "STATE.json").write_text(json.dumps({"step": 36, "epoch": 1}))
    (adapter / "COMMIT.json").write_text(
        json.dumps(
            {
                "step": 36,
                "files": {
                    name: mod.probe.campaign.sha(adapter / name)
                    for name in ("adapter_model.safetensors", "adapter_config.json", "STATE.json")
                },
            }
        )
    )
    training = {
        "role": "helper",
        "model": str(mod.evaluation.planner.BASE),
        "model_manifest_sha256": mod.build_helper_contract()["base_manifest_sha256"],
    }
    (tmp_path / "PLAN.json").write_text(json.dumps(training))
    contract = mod.build_helper_contract("trained_helper", adapter)
    assert contract["weights_frozen"] and contract["separate_model"]
    assert contract["adapter"] == str(adapter)
    assert contract["training_plan_sha256"] == mod.probe.campaign.sha(tmp_path / "PLAN.json")
    (tmp_path / "PLAN.json").write_text(json.dumps({**training, "role": "planner"}))
    with pytest.raises(ValueError, match="role"):
        mod.build_helper_contract("trained_helper", adapter)
    (tmp_path / "PLAN.json").write_text(json.dumps(training))
    (adapter / "STATE.json").write_text(json.dumps({"step": 35, "epoch": 0}))
    with pytest.raises(ValueError, match="hash"):
        mod.build_helper_contract("trained_helper", adapter)
