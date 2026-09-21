"""Label-blind ordering/exclusions and public projection for the fixed fresh panel."""

import copy
import hashlib

import pytest


def row(identity, question=None):
    return {
        "id": identity,
        "question": question or f"Question {identity}?",
        "answer": "GOLD_SECRET",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": ["T"], "sent_id": [0]},
        "context": {"title": ["T", "D"], "sentences": [["Shared source."], ["Other source."]]},
    }


def test_fixed_hash_selection_ignores_gold_and_excludes_all_exposed_questions_and_ids():
    import prepare_hotpot_fresh as mod

    rows = [row(str(i)) for i in range(12)] + [row("dup"), row("dup")]
    explorer = [{"_id": "0", "question": "Question 0?"}]
    prior = [{"id": "old", "question": " QUESTION 1? ", "metadata": {"source_id": "2"}}]
    selected, excluded, eligible = mod.select_rows(rows, explorer, prior, count=3)
    expected = sorted(
        [str(i) for i in range(3, 12)],
        key=lambda i: hashlib.sha256(f"2026092175:{i}".encode()).hexdigest(),
    )[:3]
    assert [r["id"] for r in selected] == expected
    assert {r["source_id"] for r in excluded} == {"0", "1", "2", "dup"}
    assert len(eligible) == 9
    changed = copy.deepcopy(rows)
    for r in changed:
        r.update(answer="OTHER_GOLD", type="comparison", supporting_facts={})
    assert [r["id"] for r in mod.select_rows(changed, explorer, prior, count=3)[0]] == expected
    assert [
        r["id"] for r in mod.select_rows(list(reversed(rows)), explorer, prior, count=3)[0]
    ] == expected


def test_projection_keeps_gold_host_only_and_document_exposure_does_not_filter():
    import prepare_hotpot_fresh as mod
    import probe

    cases = [mod.project(row("a")), mod.project(row("b"))]
    public = probe.public(cases[0])
    assert "GOLD_SECRET" not in str(public) and "supporting" not in str(public)
    assert cases[0]["answer"] == "GOLD_SECRET" and cases[0]["dataset"] == "hotpotqa"
    assert cases[0]["metadata"]["source_split"] == "validation"
    prior = [{**cases[0], "id": "train", "split": "train"}]
    audit = mod.document_audit(cases, prior)
    assert audit["train_parents"] == 1
    assert audit["exact_title_text_exposed_documents"] == 4
    assert audit["document_clusters"] == [[cases[0]["id"], cases[1]["id"]]]
    assert audit["exposure_filtered"] is False
    assert len(cases) == 2


def test_duplicate_new_questions_choose_hash_first_without_replacement_bias():
    import prepare_hotpot_fresh as mod

    rows = [row("a", "Same?"), row("b", " SAME? "), row("c", "Other?")]
    selected, excluded, _ = mod.select_rows(rows, [], [], count=2)
    assert len({mod.normalized_question(r["question"]) for r in selected}) == 2
    assert any("duplicate_normalized_question_in_pool" in r["reasons"] for r in excluded)
    with pytest.raises(ValueError, match="eligible"):
        mod.select_rows(rows, [], [], count=3)
