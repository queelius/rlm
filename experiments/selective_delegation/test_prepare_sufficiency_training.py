import json

import prepare_sufficiency_training as prep


def raw(parent, answerable, *, question="public question", atom="42"):
    return dict(
        id=parent,
        question=question,
        answerable=answerable,
        answer="HOST ANSWER",
        paragraphs=[
            dict(idx=0, title="public title", paragraph_text="public text", is_supporting=True)
        ],
        question_decomposition=[dict(id=atom, answer="HOST STEP", question="HOST Q")],
        host_secret="SECRET",
    )


def test_pair_projection_and_positive_repeat_control():
    pair = [raw("parent", True), raw("parent", False)]
    joint, positive = prep.build_examples({"parent": pair}, ["parent"])
    assert len(joint) == len(positive) == 2
    assert joint[0]["prompt"] == positive[0]["prompt"] == positive[1]["prompt"]
    assert len({r["id"] for r in positive}) == 2
    assert json.loads(joint[1]["target"]) == {"answerable": False, "answer": ""}
    assert all(
        json.loads(r["target"]) == {"answerable": True, "answer": "HOST ANSWER"} for r in positive
    )
    assert all("SECRET" not in r["prompt"] and "HOST" not in r["prompt"] for r in joint + positive)


def test_excludes_parent_question_or_either_variant_atomic_overlap():
    exclusions = prep.empty_exclusions()
    prep.add_exclusion(raw("held", True, question="Q  DUP", atom="17"), exclusions)
    assert prep.exclusion_reasons([raw("held", True)], exclusions) == ["parent"]
    assert "question" in prep.exclusion_reasons([raw("other", True, question="q dup")], exclusions)
    assert "component" in prep.exclusion_reasons(
        [raw("other", True), raw("other", False, atom="17")], exclusions
    )
    assert prep.exclusion_reasons([raw("other", True)], exclusions) == []
