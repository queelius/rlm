import json


def test_public_projection_excludes_native_type_id_and_gold_fields():
    import prepare_qampari_panel as panel

    row = {
        "id": "SECRET_wikidata",
        "question": "Which entities?",
        "answers": ["SECRET"],
        "target": "SECRET",
        "positive_ctxs": [{"text": "SECRET"}],
        "ctxs": [
            {
                "id": "SECRET",
                "title": "Public title",
                "text": "Public text",
                "score": 13,
                "has_answer": True,
            }
        ],
    }
    public = panel.public_context(row)
    assert public == {
        "question": "Which entities?",
        "documents": [{"docid": "d0", "title": "Public title", "text": "Public text"}],
    }
    assert "SECRET" not in panel.make_prompt(public)
    assert json.loads(panel.make_prompt(public).split("\n", 1)[1]) == public


def test_hash_selection_uses_id_without_type_filter():
    import prepare_qampari_panel as panel

    rows = [
        {"id": f"{n}__{kind}__dev"}
        for n, kind in enumerate(["wikidata_simple", "wikidata_comp", "wikidata_intersection"])
    ]
    assert panel.select(rows, 3) == sorted(rows, key=lambda r: panel.rank_key(r["id"]))
