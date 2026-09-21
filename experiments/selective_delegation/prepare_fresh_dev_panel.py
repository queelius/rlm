"""Freeze a label-blind, disjoint 64-parent MuSiQue development panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import data

SEED = 2026092113
CURRENT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
BREADTH = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914")


def normalized_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def json_values(path: Path):
    """Parse concatenated JSON values, including records with physical newlines."""
    decoder, text, index = json.JSONDecoder(), path.read_text(), 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index == len(text):
            return
        value, index = decoder.raw_decode(text, index)
        yield value


def selection(output: Path, *, archive: Path = data.ARCHIVE) -> dict:
    output, archive = Path(output).resolve(), Path(archive).resolve()
    if output.exists():
        raise FileExistsError(output)
    current_cases = CURRENT / "inputs-001/cases.jsonl"
    breadth_cases = BREADTH / "data/cases-v2.jsonl"
    current = [json.loads(line) for line in current_cases.read_text().splitlines()]
    current_ids = {row["metadata"]["source_id"] for row in current}
    current_questions = {normalized_question(row["question"]) for row in current}
    current_components = {
        str(component) for row in current for component in row["metadata"]["component_ids"]
    }
    breadth_ids, breadth_questions = set(), set()
    for row in json_values(breadth_cases):
        if row.get("dataset") == "musique":
            breadth_ids.add(row["metadata"]["source_id"])
            breadth_questions.add(normalized_question(row["question"]))
    breadth_components, resolved_breadth_ids, raw = set(), set(), []
    with zipfile.ZipFile(archive) as zipped:
        for source_split in ("train", "dev"):
            with zipped.open(f"data/musique_ans_v1.0_{source_split}.jsonl") as member:
                rows = [json.loads(line) for line in member]
            if source_split == "dev":
                raw = rows
            for row in rows:
                if row["id"] in breadth_ids:
                    resolved_breadth_ids.add(row["id"])
                    breadth_components.update(data.components(row))
    if resolved_breadth_ids != breadth_ids:
        raise ValueError("breadth MuSiQue source IDs are not fully resolvable in official archive")
    eligible, reasons = {2: [], 3: []}, Counter()
    for row in raw:
        hops, component_ids = len(row["question_decomposition"]), data.components(row)
        if not row["answerable"]:
            reasons["unanswerable"] += 1
        elif hops not in eligible:
            reasons["outside_hop_panel"] += 1
        elif row["id"] in current_ids:
            reasons["existing_selected_parent"] += 1
        elif row["id"] in breadth_ids:
            reasons["breadth_parent"] += 1
        elif normalized_question(row["question"]) in current_questions:
            reasons["existing_selected_normalized_question"] += 1
        elif normalized_question(row["question"]) in breadth_questions:
            reasons["breadth_normalized_question"] += 1
        elif component_ids & current_components:
            reasons["existing_selected_atomic_component"] += 1
        elif component_ids & breadth_components:
            reasons["breadth_atomic_component"] += 1
        else:
            eligible[hops].append(row)
    selected, split_report = [], {}
    for hops in (2, 3):
        ordered = sorted(
            eligible[hops],
            key=lambda row: hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest(),
        )
        chosen = ordered[:32]
        if len(chosen) != 32:
            raise ValueError(f"only {len(chosen)} eligible {hops}-hop parents")
        selected.extend(data._case(row, "development", "dev") for row in chosen)
        split_report[str(hops)] = {
            "available_after_external_exclusions": len(eligible[hops]),
            "selected": len(chosen),
        }
    source_ids = [row["metadata"]["source_id"] for row in selected]
    normalized = [normalized_question(row["question"]) for row in selected]
    components = [component for row in selected for component in row["metadata"]["component_ids"]]
    if len(set(source_ids)) != 64 or len(set(normalized)) != 64:
        raise ValueError("new parent or normalized-question collision")
    if set(components) & current_components or set(components) & breadth_components:
        raise ValueError("prior atomic component overlap")
    output.mkdir(parents=True, exist_ok=False)
    cases_path = output / "cases.jsonl"
    with cases_path.open("x") as stream:
        for case in selected:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema": "selective-delegation-fresh-dev-inputs-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "development",
        "counts": {"parents": 64, "two_hop": 32, "three_hop": 32},
        "seed": SEED,
        "selection": "Label-blind SHA256(seed:source_id) order within hop count, "
        "then the first 32 each of 2-hop and 3-hop after external exclusions.",
        "normalized_question": "NFKC, casefold, and collapse Unicode whitespace",
        "selection_did_not_inspect": ["answer", "answer_aliases", "step answers", "model outputs"],
        "inventory": {
            "raw_dev_parents": len(raw),
            "existing_selected_parents": len(current_ids),
            "existing_selected_atomic_components": len(current_components),
            "breadth_musique_parents": len(breadth_ids),
            "breadth_musique_normalized_questions": len(breadth_questions),
            "breadth_atomic_components_resolved_from_official_source": len(breadth_components),
            "eligible_before_new_cross_component_filter": {
                str(key): len(value) for key, value in eligible.items()
            },
            "exclusions": dict(reasons),
            "new_panel": split_report,
        },
        "source": {
            "dataset": "MuSiQue v1.0",
            "archive_path": str(archive),
            "archive_sha256": sha256(archive),
            "url": "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view",
            "repository": "https://github.com/StonyBrookNLP/musique",
            "repository_revision": data.REVISION,
            "license": "CC BY 4.0",
            "source_member": "data/musique_ans_v1.0_dev.jsonl",
        },
        "exposure_sources": {
            "existing_inputs": {"path": str(current_cases), "sha256": sha256(current_cases)},
            "breadth_cases": {"path": str(breadth_cases), "sha256": sha256(breadth_cases)},
        },
        "code_hashes": {
            "prepare_fresh_dev_panel.py": sha256(Path(__file__)),
            "data.py": sha256(Path(data.__file__)),
        },
        "constraints": [
            "All labels/decompositions are host-only; data.public_projection is the model "
            "allowlist.",
            "Disjoint from inputs-001 parents, normalized questions, and atomic components.",
            "Breadth atomic components are resolved label-blind from official source IDs; "
            "disjoint from breadth parents, normalized questions, and components.",
            "Atomic components may recur within this single development split, as in data.py.",
            "This development panel is not untouched confirmation data or evidence of pretraining "
            "noncontamination.",
        ],
        "cases_sha256": sha256(cases_path),
    }
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = selection(args.output)
    print(json.dumps({"counts": result["counts"], "cases_sha256": result["cases_sha256"]}))
