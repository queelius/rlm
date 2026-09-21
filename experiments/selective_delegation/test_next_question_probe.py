"""Small CPU contracts for the frozen-prefix feedback screen."""

import importlib.util
import json
import time
from pathlib import Path

import pytest
from test_eval_planner import CASE


def implementation():
    path = Path(__file__).with_name("next_question_probe.py")
    assert path.exists(), "next-question collector missing"
    spec = importlib.util.spec_from_file_location("next_question_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACE = [
    dict(step=1, question="First?", resolved_question="First?", answer="PREDICTED_ONE"),
    dict(
        step=2,
        question="Where is #1?",
        resolved_question="Where is PREDICTED_ONE?",
        answer="PREDICTED_TWO",
    ),
]


def test_visibility_changes_only_answer_values_and_excludes_host_secrets():
    m = implementation()
    hidden = m.next_prompt(CASE, TRACE, "hidden")
    visible = m.next_prompt(CASE, TRACE, "feedback")
    h = json.loads(hidden.split("\n", 1)[1])
    v = json.loads(visible.split("\n", 1)[1])
    assert h["history"] == [
        {"step": t["step"], "question": t["question"], "answer": None} for t in TRACE
    ]
    for t in v["history"]:
        t["answer"] = None
    assert h == v
    for secret in (
        "GOLD_SECRET",
        "SOURCE_SECRET",
        "REFERENCE_SECRET",
        "STEP_SECRET",
        "PUBLIC_SOURCE",
        "PREDICTED_ONE",
        "PREDICTED_TWO",
        "resolved_question",
    ):
        assert secret not in hidden


@pytest.mark.parametrize(
    "text",
    [
        '{"subquestions":[]}',
        '{"subquestions":["a","b"]}',
        '{"subquestions":[""]}',
        '{"subquestions":["a"],"subquestions":["b"]}',
    ],
)
def test_singleton_has_no_stop_or_repair(text):
    with pytest.raises(ValueError):
        implementation().parse_next(text)


def test_inventory_is_outcome_independent_and_has_192_call_ceiling():
    m = implementation()
    cases = [
        {**CASE, "id": str(i), "split": "transfer", "metadata": {"hops": 4}} for i in range(64)
    ]
    selected = m.select_parents(cases)
    assert len(selected) == len(set(selected)) == 16
    assert selected == m.select_parents(list(reversed(cases)))
    assert m.CAPS == {"root": 64, "helper": 48, "final": 128}
    assert len(selected) * 2 * len(m.ARMS) * 3 == 192
    assert m.seeds("parent", 0) == m.seeds("parent", 0)
    assert m.seeds("parent", 0) != m.seeds("parent", 1)


def test_actual_binding_same_prefix_final_and_unavailable_slot(tmp_path):
    m = implementation()

    class Client:
        output = tmp_path

        def __init__(self):
            self.requests = []

        def call(self, identity, prompt, condition, role, seed, *, max_new_tokens):
            self.requests.append((prompt, condition, role, seed, max_new_tokens))
            text = {
                "root": '{"subquestions":["What follows #2?"]}',
                "helper": '{"answer":"PREDICTED_THREE"}',
                "final": '{"answer":"GOLD_SECRET"}',
            }[role]
            return {
                "call_id": identity,
                "available": True,
                "text": text,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    prefix = {"eligible": True, "trace": TRACE, "dependencies": [], "source_hashes": {}}
    client = Client()
    for arm in m.ARMS:
        result = m.collect(client, CASE, prefix, arm, 0)
        assert result["status"] == "scored" and result["correct"]
    assert "What follows PREDICTED_TWO?" in client.requests[1][0]
    assert client.requests[1][2] == "helper"
    assert client.requests[2][0] == client.requests[5][0]
    assert [r[3:] for r in client.requests[:3]] == [r[3:] for r in client.requests[3:]]
    before = len(client.requests)
    result = m.collect(client, CASE, {**prefix, "eligible": False}, "hidden", 1)
    assert result["status"] == "source_prefix_unavailable"
    assert result["outcome_observed"] is False and len(client.requests) == before


def test_tiny_native_peft_routes_root_helper_and_base_final(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    m = implementation()
    torch.set_num_threads(1)
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )

    def tiny():
        model = peft.get_peft_model(
            transformers.Qwen3ForCausalLM(config),
            peft.LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
        )
        model.eval()
        return model

    class Tokenizer:
        eos_token_id = pad_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            return [3, 4]

        def decode(self, *args, **kwargs):
            return '{"answer":"real native token continuation"}'

    root, helper = tiny(), tiny()
    observed = []
    root_layer = root.base_model.model.model.layers[0].self_attn.q_proj
    helper_layer = helper.base_model.model.model.layers[0].self_attn.q_proj
    root_layer.register_forward_pre_hook(
        lambda *_: observed.append(("root", root_layer.disable_adapters))
    )
    helper_layer.register_forward_pre_hook(
        lambda *_: observed.append(("helper", helper_layer.disable_adapters))
    )
    contract = {
        "mode": "trained_helper",
        "model": "fixture-base",
        "adapter_binding": {"adapter_model.safetensors": "helper-sha"},
    }
    client = m.ScreenClient(
        root,
        Tokenizer(),
        tmp_path,
        time.time() + 100,
        "root-sha",
        helper_contract=contract,
        helper_model=helper,
    )
    receipts = [
        client.call(role, "prompt", "sft", role, 1, max_new_tokens=1)
        for role in ("root", "helper", "final")
    ]
    assert all(r["available"] and len(r["output_token_ids"]) == 1 for r in receipts)
    assert [r["adapter_sha256"] for r in receipts] == ["root-sha", "helper-sha", None]
    assert [r["model_instance"] for r in receipts] == ["root", "helper", "root"]
    assert observed == [("root", False), ("helper", False), ("root", True)]
    assert not any(p.requires_grad for model in (root, helper) for p in model.parameters())


def test_saved_prefix_rebuilds_native_prompts_and_rejects_answer_edit(tmp_path):
    m = implementation()
    source, donor = tmp_path / "source", tmp_path / "donor"
    source.mkdir()
    root_binding = {"adapter_model.safetensors": "root-sha"}
    helper_binding = {"adapter_model.safetensors": "helper-sha"}
    case = {**CASE, "split": "transfer"}
    selected = {"case_ids": [case["id"]], "cases_sha256": "cases-sha"}
    plan = dict(
        cases_sha256="cases-sha",
        split="transfer",
        case_ids=[case["id"]],
        repeats=2,
        conditions=["trained_helper"],
        root_adapter_binding=root_binding,
        helper_adapter_binding=helper_binding,
        model=str(m.evaluation.planner.BASE),
        source_output=str(donor),
        dependencies={},
    )
    m.probe.runtime.save(source / "PLAN.json", plan)
    m.probe.runtime.save(source / "SUMMARY.json", {"all_planned_complete": True})
    m.probe.runtime.save(source / "OWNER-owner.json", {})
    m.probe.runtime.save(source / "TERMINAL-owner.json", {})
    generated = {"subquestions": [t["question"] for t in TRACE]}
    request = dict(
        prompt=m.evaluation.planner_prompt(case),
        model=plan["model"],
        adapter_enabled=True,
        adapter_sha256="root-sha",
    )
    root = dict(
        call_id="root",
        request=request,
        request_digest=m.probe.runtime.digest(request),
        role="root",
        condition="sft",
        adapter_enabled=True,
        adapter_sha256="root-sha",
        available=True,
        text=json.dumps(generated),
    )
    m.probe.runtime.save(donor / "calls/root.json", root)
    episode = dict(
        episode_id="opaque-r0-trained_helper",
        case_id="opaque",
        repeat=0,
        condition="trained_helper",
        reused_root_call_id="root",
        plan=generated,
        root_request_digest=root["request_digest"],
        seed=100,
        helper_trace=TRACE,
        status="scored",
        call_ids=[],
    )
    for i, step in enumerate(TRACE):
        identity = episode["episode_id"] + f"-helper-{i + 1}"
        request = dict(
            prompt=m.eval_helper.helper_prompt(case, step["resolved_question"], "trained_helper"),
            role="helper",
            model=plan["model"],
            condition="trained_helper",
            adapter_enabled=True,
            adapter_sha256="helper-sha",
            seed=101 + i,
            sampling={"max_new_tokens": 192, "temperature": 0.5},
        )
        call = dict(
            call_id=identity,
            request=request,
            request_digest=m.probe.runtime.digest(request),
            available=True,
            text=json.dumps({"answer": step["answer"]}),
        )
        m.probe.runtime.save(source / "calls" / (identity + ".json"), call)
        episode["call_ids"].append(identity)
    path = source / "episodes/opaque-r0-trained_helper.json"
    m.probe.runtime.save(path, episode)
    prefixes, hashes = m.prepare_prefixes(
        source, selected, {"opaque": case}, root_binding, helper_binding
    )
    assert prefixes["opaque"]["trace"] == TRACE and prefixes["opaque"]["eligible"]
    assert len(prefixes["opaque"]["dependencies"]) == 3 and str(path) in hashes
    episode = json.loads(json.dumps(episode))
    episode["helper_trace"][0]["answer"] = "EDITED_NOT_NATIVE"
    path.write_text(json.dumps(episode))
    with pytest.raises(ValueError, match="native execution"):
        m.prepare_prefixes(source, selected, {"opaque": case}, root_binding, helper_binding)
