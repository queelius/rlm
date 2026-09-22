import importlib.util
import json
import time
from collections import Counter

import pytest


def reader():
    assert importlib.util.find_spec("eval_textcraft_train_readiness") is not None
    import eval_textcraft_train_readiness

    return eval_textcraft_train_readiness


def test_selection_ignores_order_and_gold_outcome_fields_and_refuses_val():
    r = reader()
    tasks = [
        {"id": f"textcraft_synth.train.{d}{i}", "misc": {"max_depth": d}}
        for d in (2, 3, 4)
        for i in range(5)
    ]
    selected = r.select_tasks(tasks)
    assert Counter(t["misc"]["max_depth"] for t in selected) == {2: 2, 3: 2, 4: 4}
    for t in tasks:
        t["native_score"] = 1
        t["misc"]["gold_trajectory"] = ["irrelevant"]
    assert [t["id"] for t in r.select_tasks(tasks[::-1])] == [t["id"] for t in selected]
    tasks[0]["id"] = "textcraft_synth.val.123"
    with pytest.raises(ValueError, match="TRAIN"):
        r.select_tasks(tasks)


def test_four_repeats_interleave_tasks_and_preserve_missing_denominator(tmp_path):
    r = reader()
    tasks = [{"id": f"textcraft_synth.train.{i}"} for i in range(8)]
    jobs = r.jobs(tasks)
    assert len(jobs) == len({j["episode_id"] for j in jobs}) == 32
    assert [j["task_id"] for j in jobs[:8]] == [t["id"] for t in tasks]
    assert [j["repeat"] for j in jobs] == [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8
    assert [j["seed"] for j in jobs[::8]] == [2026092213, 2026092214, 2026092215, 2026092216]
    summary = r.c.summarize(tmp_path, {"jobs": jobs})
    assert summary["groups"]["train_public056"]["planned"] == 32
    assert summary["groups"]["train_public056"]["missing_or_unknown"] == 32


def test_real_training_public_prompt_and_native_return_receipt(tmp_path):
    r = reader()
    import torch
    from transformers import AutoTokenizer

    task = json.loads((r.c.ROOT / r.SOURCE_TASKS).read_text().splitlines()[0])
    row = json.loads(
        (r.c.ROOT / "textcraft-public-discovery-prototype-001/rows.jsonl")
        .read_text()
        .splitlines()[0]
    )
    tokenizer = AutoTokenizer.from_pretrained(r.c.BASE, local_files_only=True)
    prompt = r.c.bridge.initial_prompt(task, "flat")
    assert prompt == row["prompt"]
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert ids == row["input_ids"][: row["prompt_tokens"]]
    assert task["id"] not in prompt and "gold_trajectory" not in prompt
    assert "num_craft_steps" not in prompt

    class Model:
        device = torch.device("cpu")

        def parameters(self):
            return []

        def generate(self, **kwargs):
            answer = tokenizer.encode(
                '{"action":"finish","message":"done"}', add_special_tokens=False
            ) + [tokenizer.eos_token_id]
            return torch.cat([kwargs["input_ids"], torch.tensor([answer])], dim=1)

    client = r.c.NativeClient(Model(), tokenizer, tmp_path, time.time() + 60, "model", "plan")
    job = r.jobs([task] * 8)[0]
    result = r.c.episode(
        task,
        job,
        client,
        r.c.bridge.load_world(),
        tmp_path,
        time.time() + 60,
        profile="original",
    )
    call = json.loads(next((tmp_path / "calls").glob("*.json")).read_text())
    assert call["available"] and call["input_token_ids"] == ids
    assert call["request"]["prompt"] == row["prompt"]
    assert result["observed"] and result["native_score"] == 0
    assert result["global_calls"] == 1 and result["status"] == "finished"
