"""Prepare a frozen, unevaluated HotpotQA explorer-sample transfer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SEED = 2026092109
CACHE = Path("/project/alex_phd/research-cache/datasets/") / (
    "musique-v1.0-922ac98f19a201998dbdae6d7f2887a5258dbdeb"
)
SOURCE = CACHE / "hotpot_official_explorer_sample100.json"
MUSIQUE_CASES = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/") / (
    "selective-delegation-20260921/inputs-001/cases.jsonl"
)
SOURCE_URL = "https://hotpotqa.github.io/js/camera_ready_nd8_dev_sample100.json"
OFFICIAL_EVALUATOR = Path("/project/alex_phd/research-cache/repos/") / (
    "hotpot-3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py"
)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalize_answer(value: str) -> str:
    """The official HotpotQA answer normalization, copied as a small auditable contract."""
    value = value.lower()
    value = "".join(char for char in value if char not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def source_documents(row: dict, seed: int) -> tuple[list[dict], list[str]]:
    """Project all documents without leaking their support/distractor role or source order."""
    raw = [
        {"title": row["title_a"], "text": " ".join(row["para_a"]), "support": True},
        {"title": row["title_b"], "text": " ".join(row["para_b"]), "support": True},
    ]
    raw.extend(
        {"title": distractor[0], "text": " ".join(distractor[1]), "support": False}
        for distractor in row["distractors"]
    )
    if any(not item["title"].strip() or not item["text"].strip() for item in raw):
        raise ValueError("source contains an empty document title or text")
    ordered = sorted(
        raw,
        key=lambda item: hashlib.sha256(
            f"{seed}:{row['_id']}:{item['title']}:{item['text']}".encode()
        ).hexdigest(),
    )
    documents = [
        {"id": f"d{index}", "title": item["title"], "text": item["text"]}
        for index, item in enumerate(ordered)
    ]
    support_ids = [f"d{index}" for index, item in enumerate(ordered) if item["support"]]
    return documents, support_ids


def _musique_train(cases_path: Path) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    questions, titles, title_texts = set(), set(), set()
    with cases_path.open() as stream:
        for line in stream:
            case = json.loads(line)
            if case["split"] != "train":
                continue
            questions.add(normalize_answer(case["question"]))
            for document in case["documents"]:
                title = normalize_answer(document["title"])
                text = normalize_answer(document["text"])
                titles.add(title)
                title_texts.add((title, text))
    return questions, titles, title_texts


def _case(row: dict, seed: int) -> dict:
    documents, support_ids = source_documents(row, seed)
    return {
        "id": hashlib.sha256(f"hotpotqa-explorer-sample100:{row['_id']}".encode()).hexdigest()[:24],
        "split": "transfer",
        "dataset": "hotpotqa",
        "question": row["question"],
        "documents": documents,
        "answer": row["answer"],
        "metadata": {
            "source_id": row["_id"],
            "source_split": "official_camera_ready_explorer_sample100",
            "answer_aliases": [],
            "supporting_document_ids": support_ids,
            "supporting_facts": row["supporting_facts"],
        },
    }


def prepare(
    source: Path, musique_cases: Path, output: Path, *, count: int = 32, seed: int = SEED
) -> dict:
    """Write one new immutable transfer panel; labels remain host-only in case records."""
    source, musique_cases, output = map(Path, (source, musique_cases, output))
    if output.exists():
        raise FileExistsError(output)
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rows = json.loads(source.read_text())
    if not isinstance(rows, list) or len({row["_id"] for row in rows}) != len(rows):
        raise ValueError("source requires unique row IDs")
    train_questions, musique_titles, musique_title_texts = _musique_train(musique_cases)
    reasons = Counter()
    eligible = []
    for row in rows:
        documents, _ = source_documents(row, seed)
        if len(documents) != 10:
            reasons["not_ten_documents"] += 1
        elif normalize_answer(row["question"]) in train_questions:
            reasons["normalized_question_overlap_muSiQue_train"] += 1
        else:
            eligible.append(row)
    eligible.sort(key=lambda row: row["_id"])
    if len(eligible) < count:
        raise ValueError(f"requested {count} cases, only {len(eligible)} eligible")
    chosen = random.Random(seed).sample(eligible, count)
    chosen.sort(key=lambda row: row["_id"])
    cases = [_case(row, seed) for row in chosen]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("opaque ID collision")
    selected_documents = [document for case in cases for document in case["documents"]]
    title_overlap = sum(
        normalize_answer(doc["title"]) in musique_titles for doc in selected_documents
    )
    title_text_overlap = sum(
        (normalize_answer(doc["title"]), normalize_answer(doc["text"])) in musique_title_texts
        for doc in selected_documents
    )
    serialized = "".join(
        json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases
    ).encode()
    manifest = {
        "schema": "selective-delegation-hotpot-explorer-transfer-inputs-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_unevaluated_no_model_calls",
        "counts": {"transfer": len(cases)},
        "selection": {
            "seed": seed,
            "rule": "Filter ten public documents and normalized MuSiQue-train question overlap; "
            "sort source IDs; random.Random(seed).sample; sort selected source IDs for output.",
            "source_rows": len(rows),
            "eligible_after_question_exclusion": len(eligible),
            "selected": len(cases),
            "question_overlap_excluded": reasons["normalized_question_overlap_muSiQue_train"],
            "not_ten_documents_excluded": reasons["not_ten_documents"],
        },
        "source": {
            "cached_file": str(source.resolve()),
            "sha256": sha256(source),
            "url": SOURCE_URL,
            "retrieved_utc": "2026-09-12",
            "license": "CC BY-SA 4.0",
            "revision": "official_camera_ready_explorer_sample100",
            "canonical_full_hotpot_dev": False,
        },
        "musique_train_overlap_audit": {
            "cases_path": str(musique_cases.resolve()),
            "cases_sha256": sha256(musique_cases),
            "normalization": "Official Hotpot lower/collapse whitespace/remove ASCII "
            "punctuation/articles.",
            "train_normalized_questions": len(train_questions),
            "selected_documents": len(selected_documents),
            "selected_documents_title_overlap": title_overlap,
            "selected_documents_exact_normalized_title_text_overlap": title_text_overlap,
            "title_or_paragraph_overlap_excluded": False,
        },
        "observation": "Only question and public document id/title/text are model-visible.",
        "host_only": "Answer, aliases, source ID, supporting facts, and support document IDs "
        "remain outside prompt construction.",
        "scoring": {
            "official_evaluator": str(OFFICIAL_EVALUATOR),
            "official_evaluator_sha256": sha256(OFFICIAL_EVALUATOR),
            "metric": "HotpotQA answer EM/F1: lowercase, remove ASCII punctuation and articles, "
            "collapse whitespace; no aliases in this source.",
            "probe_grade_compatible": False,
            "probe_difference": "probe.grade uses the imported MuSiQue alias-max EM/F1 metric; "
            "Hotpot outputs require a separately sealed official regrade at execution time.",
        },
        "limitations": [
            "Official explorer/camera-ready sample diagnostic, not canonical HotpotQA dev.",
            "Not a clean unseen/OOD or pretraining-contamination-free claim.",
            "MuSiQue four-hop transfer remains unused and unchanged.",
        ],
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("prepare_hotpot.py", "test_prepare_hotpot.py")
        },
        "cases_sha256": hashlib.sha256(serialized).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cases.jsonl").open("xb") as stream:
        stream.write(serialized)
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--musique-cases", type=Path, default=MUSIQUE_CASES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    receipt = prepare(
        arguments.source,
        arguments.musique_cases,
        arguments.output,
        count=arguments.count,
        seed=arguments.seed,
    )
    print(json.dumps({"counts": receipt["counts"], "cases_sha256": receipt["cases_sha256"]}))
