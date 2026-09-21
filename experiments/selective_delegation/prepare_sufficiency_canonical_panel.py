"""Freeze a label-blind paired DEV panel excluding all known non-TRAIN study inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import prepare_sufficiency_fresh_panel as fresh
import prepare_sufficiency_panel as panel

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
SEED = 2026092192
TRAIN_MEMBER = "data/musique_full_v1.0_train.jsonl"


def selection_key(source_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{source_id}".encode()).hexdigest()


def study_input_rows(root: Path) -> dict[str, list[dict]]:
    """Enumerate every retained input panel; TRAIN rows do not count as evaluation exposure."""
    result = {}
    for path in sorted(root.glob("*inputs*/cases.jsonl")):
        rows = panel.read_jsonl(path)
        rows = [row for row in rows if row.get("split") != "train"]
        result[str(path.relative_to(root))] = rows
    return result


def _row_features(row: dict) -> tuple[str, str, set[str]]:
    metadata = row.get("metadata", {})
    parent = str(row.get("parent_id") or metadata.get("source_id") or row.get("id", ""))
    question = row.get("question") or row.get("public", {}).get("question", "")
    components = set(map(str, row.get("component_ids", metadata.get("component_ids", []))))
    components.update(panel.source_components(parent))
    return parent, panel.normalized_question(question), components


def _public(row: dict) -> dict:
    docs = [{"title": p["title"], "text": p["paragraph_text"]} for p in row["paragraphs"]]
    docs.sort(
        key=lambda doc: hashlib.sha256(
            json.dumps([SEED, row["question"], doc], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    )
    return {
        "question": row["question"],
        "documents": [{"docid": f"d{i}", **doc} for i, doc in enumerate(docs)],
    }


def _plan_inventory(root: Path) -> dict[str, int]:
    """Record named output selections that resolve to input panels; they add no new labels."""
    names = ("next-question-screen-001/SELECTION.json", "validation-span-001/PLAN.json")
    result = {}
    for name in names:
        path = root / name
        if path.exists():
            value = json.loads(path.read_text())
            result[name] = len(value.get("case_ids", []))
    for path in sorted(root.glob("transfer-musique-fourhop-*/PLAN.json")):
        result[str(path.relative_to(root))] = len(json.loads(path.read_text()).get("case_ids", []))
    return result


def prepare(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("immutable output exists")
    if (
        panel.sha256(panel.ARCHIVE)
        != "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
    ):
        raise ValueError("official archive identity changed")
    studies = study_input_rows(root)
    breadth_path = root.parent / "unattended-breadth-20260914/data/cases-v2.jsonl"
    breadth = [row for row in panel.read_jsonl(breadth_path) if row.get("dataset") == "musique"]
    parents, questions, components = set(), set(), set()
    for rows in [*studies.values(), breadth]:
        for row in rows:
            parent, question, atoms = _row_features(row)
            parents.add(parent)
            questions.add(question)
            components.update(atoms)
    grouped, train_parents, train_components, members = defaultdict(list), set(), set(), {}
    with zipfile.ZipFile(panel.ARCHIVE) as source:
        for member in (panel.MEMBER, TRAIN_MEMBER):
            data = source.read(member)
            members[member] = hashlib.sha256(data).hexdigest()
            for line in data.splitlines():
                row = json.loads(line)
                if member == panel.MEMBER:
                    grouped[row["id"]].append(row)
                else:
                    train_parents.add(row["id"])
                    train_components.update(
                        panel.source_components(row["id"])
                        | {str(step["id"]) for step in row["question_decomposition"]}
                    )
    eligible, exclusions = {}, Counter()
    for parent, rows in grouped.items():
        if len(rows) != 2 or {row["answerable"] for row in rows} != {True, False}:
            raise ValueError("official DEV pair inventory malformed")
        question = panel.normalized_question(rows[0]["question"])
        atoms = panel.source_components(parent) | {
            str(step["id"]) for row in rows for step in row["question_decomposition"]
        }
        reasons = []
        if parent in parents:
            reasons.append("known_nontrain_parent")
        if question in questions:
            reasons.append("known_nontrain_normalized_question")
        if atoms & components:
            reasons.append("known_nontrain_component")
        if parent in train_parents:
            reasons.append("official_train_parent")
        if atoms & train_components:
            reasons.append("official_train_component")
        if reasons:
            exclusions.update(reasons)
        else:
            eligible[parent] = rows
    chosen = sorted(eligible, key=selection_key)[:32]
    if len(chosen) != 32:
        raise ValueError("fewer than32 canonical eligible pairs")
    cases = []
    for parent in chosen:
        for row in eligible[parent]:
            public = _public(row)
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
                    "component_ids": sorted(
                        panel.source_components(parent)
                        | {
                            str(step["id"])
                            for pair in eligible[parent]
                            for step in pair["question_decomposition"]
                        }
                    ),
                    "hops": len(row["question_decomposition"]),
                }
            )
    cases.sort(key=lambda row: row["id"])
    if len(cases) != 64 or len({row["id"] for row in cases}) != 64:
        raise ValueError("expected64 unique variants")
    selected_docs = {
        (doc["title"], doc["text"]) for row in cases for doc in row["public"]["documents"]
    }
    selected_titles = {title for title, _ in selected_docs}
    train_docs = []
    with zipfile.ZipFile(panel.ARCHIVE) as source:
        for line in source.read(TRAIN_MEMBER).splitlines():
            for paragraph in json.loads(line)["paragraphs"]:
                value = paragraph["title"], paragraph["paragraph_text"]
                if value in selected_docs or value[0] in selected_titles:
                    train_docs.append(value)
    output.mkdir(parents=True)
    cases_path = output / "cases.jsonl"
    with cases_path.open("x", encoding="utf-8") as stream:
        for row in cases:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema": "musique-canonical-paired-sufficiency-panel-v1",
        "selection_seed": SEED,
        "selection_rule": (
            "first32 SHA256('2026092192:'+originalid), no label/hop/length/output filter"
        ),
        "parent_count": 32,
        "variant_count": 64,
        "parents": chosen,
        "eligible_parent_count": len(eligible),
        "natural_hops": dict(Counter(row["hops"] for row in cases if row["answerable"])),
        "nonexclusive_exclusions": dict(exclusions),
        "known_nontrain_inputs": {
            name: {"rows": len(rows), "sha256": panel.sha256(root / name)}
            for name, rows in studies.items()
        },
        "breadth": {
            "path": str(breadth_path),
            "rows": len(breadth),
            "sha256": panel.sha256(breadth_path),
        },
        "resolved_named_output_selections": _plan_inventory(root),
        "official_train_exposure": {
            "train_parent_count": len(train_parents),
            "train_component_count": len(train_components),
            "selected_parent_overlap": len(set(chosen) & train_parents),
            "selected_component_overlap": len(
                {component for row in cases for component in row["component_ids"]}
                & train_components
            ),
            "document_overlap_descriptive_only": fresh.document_overlap(cases, train_docs),
        },
        "archive": str(panel.ARCHIVE),
        "archive_sha256": panel.sha256(panel.ARCHIVE),
        "members_sha256": members,
        "cases_sha256": panel.sha256(cases_path),
        "source_sha256": {
            str(Path(__file__).resolve()): panel.sha256(Path(__file__)),
            str(Path(panel.__file__).resolve()): panel.sha256(Path(panel.__file__)),
            str(Path(fresh.__file__).resolve()): panel.sha256(Path(fresh.__file__)),
        },
        "official_metric_sha256": {
            str(path): panel.sha256(path) for path in (panel.OFFICIAL / "metrics").glob("*.py")
        },
        "limitations": " ".join(
            [
                "Known study-input identity-disjoint and official-TRAIN component-disjoint,",
                "not semantic, document, or pretraining clean.",
                "Exact document overlap is descriptive, not an exclusion.",
                "Official variants retain natural context/length changes",
                "and are not causal deletion pairs.",
            ]
        ),
    }
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output), indent=2, sort_keys=True))
