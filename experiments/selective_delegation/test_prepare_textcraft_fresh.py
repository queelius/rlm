import importlib

import pytest


def row(name, root, depth=2):
    return {"id": name, "misc": {"target_items": {root: 1}, "max_depth": depth}}


def test_selection_excludes_exposed_and_training_roots_without_reading_outcomes():
    m = importlib.import_module("prepare_textcraft_fresh")
    rows = [row("seen", "seen"), row("train", "train"), row("a", "a"), row("a2", "a")]
    selected, audit = m.select(rows, [row("seen", "seen")], [row("other", "train")], {2: 1})
    assert len(selected) == 1
    assert selected[0]["misc"]["target_items"] == {"a": 1}
    assert audit["excluded_ids_or_roots"] == 2
    assert (
        m.select(list(reversed(rows)), [row("seen", "seen")], [row("other", "train")], {2: 1})[0]
        == selected
    )
    with pytest.raises(ValueError, match="insufficient"):
        m.select(rows, [row("seen", "seen")], [row("other", "train")], {2: 2})


def test_dependencies_count_shared_intermediates_not_only_root():
    m = importlib.import_module("prepare_textcraft_fresh")
    from types import SimpleNamespace

    world = SimpleNamespace(
        recipes={
            "root": [SimpleNamespace(ingredients={"middle": 2})],
            "middle": [SimpleNamespace(ingredients={"ore": 1})],
        }
    )
    assert m.dependencies({"root": 1}, world) == {"root", "middle"}


def test_real_official_task_native_and_public_token_trace():
    m = importlib.import_module("prepare_textcraft_fresh")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(m.training.BASE, local_files_only=True)
    task = m.rows(m.original.TASKS)[0]
    world = m.bridge.load_world()
    assert m.bridge.replay_gold(task, world)["native_score"] == 1
    result, trace = m.public.trajectory(task, world, tokenizer)
    assert result["native_score"] == 1
    assert result["finish_included"]
    assert max(r["prompt_tokens"] + 256 for r in trace) <= 8192
