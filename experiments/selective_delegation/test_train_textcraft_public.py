import copy
from pathlib import Path

import pytest

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")


def test_exact_public_discovery_source_is_qualified_not_old_privileged_input():
    import train_textcraft_public as wrapper

    contract = wrapper.validate_prepared(ROOT / "textcraft-public-discovery-prototype-001")
    assert contract["rows"] == 366 and contract["tasks"] == 32
    assert contract["prompt_tokens"] == 438065 and contract["supervised_tokens"] == 8820
    assert contract["teaching_policy"] == "public_observation_prerequisite_discovery"
    with pytest.raises(ValueError, match="manifest"):
        wrapper.validate_prepared(ROOT / "textcraft-train-inputs-001")


def test_teacher_qualification_cannot_drop_tasks_or_claim_unchecked_history():
    import train_textcraft_public as wrapper

    manifest = wrapper.read(ROOT / "textcraft-public-discovery-prototype-001/MANIFEST.json")
    audit = wrapper.read(ROOT / "textcraft-public-discovery-prototype-001/PUBLIC-REPLAY-AUDIT.json")
    wrapper.validate_qualification(manifest, audit)
    bad = copy.deepcopy(manifest)
    bad["eligible_task_count"] = 31
    with pytest.raises(ValueError):
        wrapper.validate_qualification(bad, audit)
    with pytest.raises(ValueError):
        wrapper.validate_qualification(manifest, {**audit, "past_query_provenance_checked": False})


def test_extra_host_fields_are_not_tokenized_or_supervised():
    import train_textcraft_public as wrapper

    class Tokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "public only"}]
            return [1, 2]

        def encode(self, text, **kwargs):
            assert text == '{"action":"finish","message":"done"}'
            return [3, 4]

    row = {
        "task_id": "host-secret",
        "step": 0,
        "prompt": "public only",
        "target": '{"action":"finish","message":"done"}',
        "gold_future": "SECRET",
        "input_ids": [999],
        "labels": [999],
        "feedback": "SECRET",
    }
    tokenized = wrapper.recipe.tokenize_rows([row], Tokenizer())[0]
    assert tokenized["input_ids"] == [1, 2, 3, 4] and tokenized["target_ids"] == [3, 4, 9]
