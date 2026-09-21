import contextlib


def test_role_adapter_context_keeps_manager_base_and_enables_flat_worker():
    import alfworld_trained_actor as subject

    class Model:
        def __init__(self):
            self.events = []

        def disable_adapter(self):
            self.events.append("disable")
            return contextlib.nullcontext()

        def parameters(self):
            return []

    model = Model()
    with subject.role_adapter_context(model, "manager"):
        model.events.append("manager")
    with subject.role_adapter_context(model, "flat"):
        model.events.append("flat")
    with subject.role_adapter_context(model, "worker"):
        model.events.append("worker")
    assert model.events == ["disable", "manager", "flat", "worker"]


def test_role_identity_is_explicit_before_generation():
    import alfworld_trained_actor as subject

    assert subject.role_identity("manager", "adapter-digest") == {
        "adapter_enabled": False,
        "adapter_sha256": None,
    }
    assert subject.role_identity("flat", "adapter-digest") == {
        "adapter_enabled": True,
        "adapter_sha256": "adapter-digest",
    }
    assert subject.role_identity("worker", "adapter-digest")["adapter_enabled"] is True


def test_native_request_records_adapter_role_before_manager_generation(tmp_path):
    import alfworld_trained_actor as subject
    import torch

    class Tokenizer:
        eos_token_id = pad_token_id = 0

        def apply_chat_template(self, *_args, **_kwargs):
            return [1, 2]

        def decode(self, _ids, **_kwargs):
            return '{"goal":"find it"}'

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.disabled = 0

        def disable_adapter(self):
            self.disabled += 1
            return contextlib.nullcontext()

        def parameters(self):
            return []

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 8, 0]])

    model = Model()
    client = subject.ActionAdapterClient(model, Tokenizer(), tmp_path, 10**10, "adapter-digest")
    row = client.call("manager", "public", "manager_worker", "manager", 3, 128, {})
    assert row["available"] and row["request"]["adapter_enabled"] is False
    assert row["request"]["adapter_sha256"] is None and model.disabled == 1
