import json
import time
from pathlib import Path


def test_saved_direct_native_encoding_sampling_and_strict_grade(tmp_path):
    import qampari_sampling_control as control
    import torch
    from transformers import AutoTokenizer

    root = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
    case = json.loads((root / "qampari-inputs-001/cases.jsonl").read_text().splitlines()[0])
    plan = json.loads((root / "qampari-reading-001/PLAN.json").read_text())
    job = next(
        j for j in plan["jobs"] if j["condition"] == "direct200" and j["case_id"] == case["id"]
    )
    spec = control.original.requests(case, job)[0]
    saved = json.loads((root / "qampari-reading-001/calls" / f"{spec['call_id']}.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        control.panel.MODEL, local_files_only=True, trust_remote_code=False
    )
    answer = json.dumps({"answers": [case["answer_list"][0]["aliases"][0]]})
    output_ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]

    class Model:
        device = torch.device("cpu")

        def parameters(self):
            return []

        def generate(self, **kwargs):
            assert kwargs["input_ids"][0].tolist() == saved["input_token_ids"]
            assert kwargs["temperature"] == 0.7 and kwargs["top_p"] == 0.8 and kwargs["top_k"] == 20
            assert kwargs["max_new_tokens"] == 1024 and kwargs["do_sample"] is True
            assert "repetition_penalty" not in kwargs and "presence_penalty" not in kwargs
            return torch.tensor([saved["input_token_ids"] + output_ids])

    call = control.NativeClient(Model(), tokenizer, tmp_path, time.time() + 90).call(spec)
    assert call["available"] and call["request"]["prompt"] == saved["request"]["prompt"]
    assert call["request"]["seed"] == saved["request"]["seed"]
    assert call["request"]["sampling"] == {
        **saved["request"]["sampling"],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    assert control.original.parse_answers(call["text"]) == [case["answer_list"][0]["aliases"][0]]
    assert (
        control.original.grade(control.original.parse_answers(call["text"]), case["answer_list"])[
            "recall"
        ]
        > 0
    )


def test_only_complete_fixed32_readout_passes_gate():
    import qampari_sampling_control as control

    summary = {
        "planned_episodes": 32,
        "recorded_episodes": 32,
        "planned_calls": 32,
        "observed": 32,
        "missing_or_unavailable": 0,
        "physical_cost": {"calls": 32, "failed_calls": 0},
        "unresolved_starts": [],
    }
    assert control.complete(summary)
    summary["observed"] = 31
    assert not control.complete(summary)
