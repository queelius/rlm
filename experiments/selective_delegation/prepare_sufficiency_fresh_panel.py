"""Freeze a label-blind paired DEV sufficiency panel outside official TRAIN/past panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import prepare_sufficiency_panel as panel

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
SEED = 2026092192
TRAIN_MEMBER = "data/musique_full_v1.0_train.jsonl"


def selection_key(source_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{source_id}".encode()).hexdigest()


def select_pairs(pairs: dict[str, list[dict]], count: int = 32) -> list[str]:
    if len(pairs) < count:
        raise ValueError("insufficient eligible complete pairs; no replacement")
    return sorted(pairs, key=selection_key)[:count]


def document_overlap(selected: list[dict], train_documents: list[tuple[str, str]]) -> dict:
    """Descriptive exact/title exposure counts; never an eligibility filter."""
    exact_train, title_train = set(train_documents), {title for title, _ in train_documents}
    documents = {
        (document["title"], document["text"])
        for row in selected
        for document in row["public"]["documents"]
    }
    exact_variants = title_variants = 0
    for row in selected:
        values = {(doc["title"], doc["text"]) for doc in row["public"]["documents"]}
        exact_variants += bool(values & exact_train)
        title_variants += bool({title for title, _ in values} & title_train)
    return {
        "selected_unique_documents": len(documents),
        "exact_title_text_shared_unique_documents": len(documents & exact_train),
        "title_shared_unique_documents": len({title for title, _ in documents} & title_train),
        "selected_variants_with_exact_overlap": exact_variants,
        "selected_variants_with_title_overlap": title_variants,
    }


def _components(parent: str, rows: list[dict]) -> set[str]:
    return panel.source_components(parent) | {
        str(step["id"]) for row in rows for step in row.get("question_decomposition", [])
    }


def _public_context(row: dict) -> dict:
    documents = [{"title": p["title"], "text": p["paragraph_text"]} for p in row["paragraphs"]]
    documents.sort(
        key=lambda d: hashlib.sha256(
            json.dumps([SEED, row["question"], d], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    )
    return {
        "question": row["question"],
        "documents": [{"docid": f"d{i}", **doc} for i, doc in enumerate(documents)],
    }


def _previous_features(cases_path: Path) -> tuple[set[str], set[str], set[str]]:
    rows = panel.read_jsonl(cases_path)
    parents = {row["parent_id"] for row in rows}
    questions = {panel.normalized_question(row["public"]["question"]) for row in rows}
    components = {component for row in rows for component in row.get("component_ids", [])}
    return parents, questions, components


def prepare(root: Path, output: Path) -> dict:
    """Write labels only in host-side fields after label-blind parent selection."""
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("immutable panel output exists")
    archive = panel.ARCHIVE
    if panel.sha256(archive) != "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd":
        raise ValueError("official archive identity changed")
    previous_path = root / "sufficiency-inputs-001/cases.jsonl"
    previous_parents, previous_questions, previous_components = _previous_features(previous_path)
    members, grouped_dev, train_parents, train_components = {}, defaultdict(list), set(), set()
    with zipfile.ZipFile(archive) as source:
        for member in (panel.MEMBER, TRAIN_MEMBER):
            data = source.read(member)
            members[member] = hashlib.sha256(data).hexdigest()
            for line in data.splitlines():
                row = json.loads(line)
                if member == panel.MEMBER:
                    grouped_dev[row["id"]].append(row)
                else:
                    train_parents.add(row["id"])
                    train_components.update(_components(row["id"], [row]))
    eligible, exclusions = {}, Counter()
    for parent, rows in grouped_dev.items():
        if len(rows) != 2 or {row["answerable"] for row in rows} != {True, False}:
            raise ValueError("official DEV pair inventory malformed")
        if len({row["question"] for row in rows}) != 1:
            raise ValueError("official pair question differs")
        question = panel.normalized_question(rows[0]["question"])
        components = _components(parent, rows)
        reasons = []
        if parent in previous_parents:
            reasons.append("prior_panel_parent")
        if question in previous_questions:
            reasons.append("prior_panel_normalized_question")
        if components & previous_components:
            reasons.append("prior_panel_component")
        if parent in train_parents:
            reasons.append("official_train_parent")
        if components & train_components:
            reasons.append("official_train_component")
        if reasons:
            exclusions.update(reasons)
        else:
            eligible[parent] = rows
    selected_parents = select_pairs(eligible)
    cases = []
    for parent in selected_parents:
        for row in eligible[parent]:
            public = _public_context(row)
            cases.append(
                {
                    "id": hashlib.sha256(
                        json.dumps(public, ensure_ascii=False, sort_keys=True).encode()
                    ).hexdigest()[:24],
                    "parent_id": parent,
                    "public": public,
                    "answerable": row["answerable"],
                    "answer": row["answer"],
                    "answer_aliases": row["answer_aliases"],
                    "component_ids": sorted(_components(parent, eligible[parent])),
                    "hops": len(row["question_decomposition"]),
                }
            )
    if len(cases) != 64 or len({row["id"] for row in cases}) != 64:
        raise ValueError("expected64 unique public variants")
    cases.sort(key=lambda row: row["id"])
    selected_exact = {
        (doc["title"], doc["text"]) for row in cases for doc in row["public"]["documents"]
    }
    selected_titles = {title for title, _ in selected_exact}
    train_documents = []
    with zipfile.ZipFile(archive) as source:
        for line in source.read(TRAIN_MEMBER).splitlines():
            row = json.loads(line)
            for paragraph in row["paragraphs"]:
                value = paragraph["title"], paragraph["paragraph_text"]
                if value in selected_exact or value[0] in selected_titles:
                    train_documents.append(value)
    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    with cases_path.open("x", encoding="utf-8") as stream:
        for row in cases:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema": "musique-fresh-paired-sufficiency-panel-v1",
        "selection_seed": SEED,
        "selection_rule": "first32 SHA256('2026092192:'+originalid), no label/hop/length filter",
        "parent_count": 32,
        "variant_count": 64,
        "parents": selected_parents,
        "eligible_parent_count": len(eligible),
        "nonexclusive_exclusions": dict(exclusions),
        "natural_hops": dict(Counter(row["hops"] for row in cases if row["answerable"])),
        "prior_panel": {"path": str(previous_path), "sha256": panel.sha256(previous_path)},
        "official_train_exposure": {
            "train_parent_count": len(train_parents),
            "train_component_count": len(train_components),
            "selected_parent_overlap": len(set(selected_parents) & train_parents),
            "selected_component_overlap": len(
                {component for row in cases for component in row["component_ids"]}
                & train_components
            ),
            "document_overlap_descriptive_only": document_overlap(cases, train_documents),
        },
        "archive": str(archive),
        "archive_sha256": panel.sha256(archive),
        "members_sha256": members,
        "cases_sha256": panel.sha256(cases_path),
        "source_sha256": {
            str(Path(__file__).resolve()): panel.sha256(Path(__file__)),
            str(Path(panel.__file__).resolve()): panel.sha256(Path(panel.__file__)),
        },
        "official_metric_sha256": {
            str(path): panel.sha256(path) for path in (panel.OFFICIAL / "metrics").glob("*.py")
        },
        "license": "CC BY4.0; official MuSiQue full DEV pairs",
        "limitations": (
            "Known parent/component-disjoint from official TRAIN and the prior32 panel, not "
            "semantic, document, or pretraining clean. Exact document overlap establishes shared "
            "documents; title-only overlap does not establish identical facts. Official variants "
            "rewrite/swap natural context and are not causal support-deletion pairs."
        ),
    }
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
