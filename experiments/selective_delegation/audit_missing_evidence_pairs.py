#!/usr/bin/env python3
"""Audit official MuSiQue full-dev pairs without selecting a model panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def source_components(source_id: str) -> set[str]:
    if "__" not in source_id:
        return set()
    return set(source_id.split("__", 1)[1].split("_"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def public_paragraphs(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: paragraph[key] for key in ("idx", "title", "paragraph_text")}
        for paragraph in row["paragraphs"]
    ]


def summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--fresh003", type=Path, required=True)
    parser.add_argument("--breadth", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {args.output}")

    selected_rows = read_jsonl(args.inputs) + read_jsonl(args.fresh003)
    selected_parents: set[str] = set()
    selected_questions: set[str] = set()
    selected_components: set[str] = set()
    for row in selected_rows:
        metadata = row.get("metadata", {})
        selected_parents.add(metadata.get("source_id", row["id"]))
        selected_questions.add(normalized_question(row["question"]))
        selected_components.update(map(str, metadata.get("component_ids", [])))

    breadth_rows = read_jsonl(args.breadth)
    breadth_rows = [row for row in breadth_rows if row.get("dataset") == "musique"]
    breadth_parents = {row["metadata"]["source_id"] for row in breadth_rows}
    breadth_questions = {normalized_question(row["question"]) for row in breadth_rows}
    breadth_components = set().union(*(source_components(item) for item in breadth_parents))

    with zipfile.ZipFile(args.archive) as archive:
        member = "data/musique_full_v1.0_dev.jsonl"
        rows = [json.loads(line) for line in archive.read(member).splitlines()]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)

    equality = Counter()
    public = Counter()
    answerable_docs: list[int] = []
    missing_docs: list[int] = []
    answerable_chars: list[int] = []
    missing_chars: list[int] = []
    document_deltas: Counter[int] = Counter()
    character_deltas: list[int] = []
    hidden_supports = Counter()
    exclusions = Counter()
    clean_ids: list[str] = []
    hop_counts = Counter()
    answerability_patterns = Counter()

    for source_id, pair in grouped.items():
        answerability_patterns[tuple(sorted(row["answerable"] for row in pair))] += 1
        if len(pair) != 2 or {row["answerable"] for row in pair} != {False, True}:
            raise ValueError(f"not a paired official group: {source_id}")
        answerable = next(row for row in pair if row["answerable"])
        missing = next(row for row in pair if not row["answerable"])
        equality["question_same"] += answerable["question"] == missing["question"]
        equality["answer_same"] += answerable["answer"] == missing["answer"]
        equality["aliases_same"] += (
            answerable.get("answer_aliases") == missing.get("answer_aliases")
        )
        equality["decomposition_same"] += (
            answerable.get("question_decomposition") == missing.get("question_decomposition")
        )
        supported_public = public_paragraphs(answerable)
        missing_public = public_paragraphs(missing)
        public["exact_same"] += supported_public == missing_public
        public["same_document_count"] += len(supported_public) == len(missing_public)
        public["same_index_sequence"] += (
            [item["idx"] for item in supported_public] == [item["idx"] for item in missing_public]
        )
        public["same_title_sequence"] += (
            [item["title"] for item in supported_public]
            == [item["title"] for item in missing_public]
        )
        public["same_title_set"] += (
            {item["title"] for item in supported_public}
            == {item["title"] for item in missing_public}
        )
        supported_chars = sum(
            len(item["title"]) + len(item["paragraph_text"]) for item in supported_public
        )
        missing_chars_count = sum(
            len(item["title"]) + len(item["paragraph_text"]) for item in missing_public
        )
        answerable_docs.append(len(supported_public))
        missing_docs.append(len(missing_public))
        answerable_chars.append(supported_chars)
        missing_chars.append(missing_chars_count)
        document_deltas[len(supported_public) - len(missing_public)] += 1
        character_deltas.append(supported_chars - missing_chars_count)
        hidden_supports["answerable"] += sum(
            bool(paragraph["is_supporting"]) for paragraph in answerable["paragraphs"]
        )
        hidden_supports["missing"] += sum(
            bool(paragraph["is_supporting"]) for paragraph in missing["paragraphs"]
        )

        components = source_components(source_id)
        reasons: list[str] = []
        if source_id in selected_parents:
            reasons.append("selected_parent")
        if source_id in breadth_parents:
            reasons.append("breadth_parent")
        if normalized_question(answerable["question"]) in selected_questions:
            reasons.append("selected_normalized_question")
        if normalized_question(answerable["question"]) in breadth_questions:
            reasons.append("breadth_normalized_question")
        if components & selected_components:
            reasons.append("selected_atomic_component")
        if components & breadth_components:
            reasons.append("breadth_atomic_component")
        if reasons:
            exclusions.update(reasons)
        else:
            clean_ids.append(source_id)
            hop_counts[source_id.split("__", 1)[0]] += 1

    ordered = sorted(
        clean_ids,
        key=lambda source_id: hashlib.sha256(f"{args.seed}:{source_id}".encode()).hexdigest(),
    )
    result = {
        "purpose": (
            "Schema/provenance audit only: no panel selected, no model outputs inspected, "
            "and no causal support-deletion claim."
        ),
        "official_objective": (
            "MuSiQue full pairs evaluate the supplied-document answerability label; the official "
            "group answer-sufficiency metric retains the answerable-member answer score only when "
            "the two predicted answerability labels exactly equal the official pair labels."
        ),
        "inputs": {
            "archive": str(args.archive),
            "archive_sha256": sha256(args.archive),
            "archive_member": member,
            "inputs_001": {"path": str(args.inputs), "sha256": sha256(args.inputs)},
            "fresh_dev_inputs_003": {"path": str(args.fresh003), "sha256": sha256(args.fresh003)},
            "breadth_cases_v2": {"path": str(args.breadth), "sha256": sha256(args.breadth)},
        },
        "full_dev": {
            "rows": len(rows),
            "parent_groups": len(grouped),
            "group_sizes": dict(Counter(map(len, grouped.values()))),
            "answerability_patterns": {
                str(key): value for key, value in answerability_patterns.items()
            },
            "same_fields_per_parent_group": dict(equality),
            "hidden_support_paragraphs": dict(hidden_supports),
        },
        "public_document_confound_audit": {
            "projection_allowlist": ["idx", "title", "paragraph_text"],
            "must_not_project": [
                "answerable",
                "is_supporting",
                "question_decomposition",
                "answer",
                "answer_aliases",
            ],
            "pair_counts": dict(public),
            "document_count_delta_answerable_minus_missing": dict(document_deltas),
            "answerable_document_count": summary(answerable_docs),
            "missing_document_count": summary(missing_docs),
            "answerable_title_text_characters": summary(answerable_chars),
            "missing_title_text_characters": summary(missing_chars),
            "character_delta_answerable_minus_missing": summary(character_deltas),
        },
        "exact_exposure_exclusions": {
            "selected_rows": len(selected_rows),
            "selected_parents": len(selected_parents),
            "selected_normalized_questions": len(selected_questions),
            "selected_atomic_components": len(selected_components),
            "breadth_rows": len(breadth_rows),
            "breadth_parents": len(breadth_parents),
            "breadth_normalized_questions": len(breadth_questions),
            "breadth_atomic_components_resolved_from_source_ids": len(breadth_components),
            "nonexclusive_exclusion_counts": dict(exclusions),
            "remaining_component_disjoint_parent_pairs": len(clean_ids),
            "remaining_hop_counts": dict(hop_counts),
            "selection_seed_if_later_accepted": args.seed,
            "candidate_order_sha256_no_cases_selected": hashlib.sha256(
                "\n".join(ordered).encode()
            ).hexdigest(),
        },
        "limitations": [
            "Component and normalized-question exclusions establish only known "
            "dataset-overlap disjointness, not semantic, document, or pretraining disjointness.",
            "Pairs rewrite/swap visible context, so they test official supplied-context "
            "answerability rather than an isolated causal removal of one support paragraph.",
            "Designated supports can have lexical/grounding gaps; official labels are the "
            "canonical benchmark target, not proof that the answer is globally false or that "
            "a real retriever would need more evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
