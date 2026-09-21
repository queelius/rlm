"""Independent CPU qualification for the sealed trained-actor adapter routing contract."""

import alfworld_trained_actor as subject
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel


def test_real_peft_manager_is_base_and_action_roles_are_adapter_enabled_and_restored():
    base = GPT2LMHeadModel(
        GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2)
    ).eval()
    model = get_peft_model(
        base,
        LoraConfig(r=2, lora_alpha=32, target_modules=["c_attn"], task_type="CAUSAL_LM"),
        autocast_adapter_dtype=True,
    ).eval()
    for name, parameter in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.data.fill_(1.0)
        parameter.requires_grad_(False)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        with subject.role_adapter_context(model, "flat"):
            flat_before = model(input_ids=ids).logits
        with subject.role_adapter_context(model, "manager"):
            manager = model(input_ids=ids).logits
        with model.disable_adapter():
            base_logits = model(input_ids=ids).logits
        # This raw PEFT context intentionally re-enables its adapter's trainability; the
        # production role context must restore the frozen readout contract afterwards.
        with subject.role_adapter_context(model, "manager"):
            pass
        with subject.role_adapter_context(model, "worker"):
            worker = model(input_ids=ids).logits
        with subject.role_adapter_context(model, "flat"):
            flat_after = model(input_ids=ids).logits
    assert torch.allclose(manager, base_logits)
    assert not torch.allclose(flat_before, base_logits)
    assert torch.allclose(worker, flat_before)
    assert torch.allclose(flat_after, flat_before)
    assert not any(parameter.requires_grad for parameter in model.parameters())
