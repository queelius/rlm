"""Freeze public MuSiQue contexts and host-only labels for selective delegation."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
CACHE = Path("/project/alex_phd/research-cache")
ARCHIVE = CACHE / f"datasets/musique-v1.0-{REVISION}/musique_data_v1.0.zip"
PRIOR_CASES = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/") / (
    "unattended-breadth-20260914/data/cases-v2.jsonl"
)
SEED = 20260921


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def public_projection(case: dict) -> dict:
    """Allowlist the complete model-visible input, including paragraph identifiers."""
    return {
        "question": case["question"],
        "documents": [
            {key: document[key] for key in ("id", "title", "text")}
            for document in case["documents"]
        ],
    }


def components(row: dict) -> set[str]:
    return {str(step["id"]) for step in row["question_decomposition"]}


def _case(row: dict, split: str, source_split: str) -> dict:
    return {
        "id": hashlib.sha256(f"musique-v1.0:{row['id']}".encode()).hexdigest()[:24],
        "split": split,
        "dataset": "musique",
        "question": row["question"],
        "documents": [
            {"id": str(p["idx"]), "title": p.get("title", ""), "text": p["paragraph_text"]}
            for p in row["paragraphs"]
        ],
        "answer": row["answer"],
        "answer_type": "string",
        "metadata": {
            "source_id": row["id"],
            "source_split": source_split,
            "answer_aliases": row.get("answer_aliases", []),
            "hops": len(row["question_decomposition"]),
            "component_ids": sorted(components(row)),
            "supporting_paragraph_ids": [
                str(p["idx"]) for p in row["paragraphs"] if p["is_supporting"]
            ],
            "question_decomposition": row["question_decomposition"],
        },
    }


def prepare(
    output: Path,
    train_count: int = 256,
    validation_count: int = 64,
    transfer_count: int = 64,
    *,
    archive: Path = ARCHIVE,
    prior_cases: Path = PRIOR_CASES,
    seed: int = SEED,
) -> dict:
    """Prepare a new immutable directory; never overwrite an earlier preparation."""
    output, archive, prior_cases = Path(output), Path(archive), Path(prior_cases)
    if output.exists():
        raise FileExistsError(output)
    counts = {"train": train_count, "validation": validation_count, "transfer": transfer_count}
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError("Counts must be nonnegative integers")
    with prior_cases.open() as stream:
        previous = [json.loads(line) for line in stream]
    excluded_ids = {row["metadata"]["source_id"] for row in previous if row["dataset"] == "musique"}
    with zipfile.ZipFile(archive) as stream:
        raw = {}
        for split in ("train", "dev"):
            # Binary file iteration separates physical newlines only; U+2028 is data.
            with stream.open(f"data/musique_ans_v1.0_{split}.jsonl") as member:
                raw[split] = [json.loads(line) for line in member]
    selected, seen_parents, earlier_components, reports = [], set(), set(), {}
    for split, source_split, hops in (
        ("train", "train", {2, 3}),
        ("validation", "dev", {2, 3}),
        ("transfer", "dev", {4}),
    ):
        candidates = sorted(
            raw[source_split],
            key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
        )
        reasons = Counter()
        eligible = []
        for row in candidates:
            if not row["answerable"]:
                reasons["unanswerable"] += 1
            elif len(row["question_decomposition"]) not in hops:
                reasons["outside_hop_panel"] += 1
            elif row["id"] in excluded_ids:
                reasons["prior_breadth_parent"] += 1
            elif row["id"] in seen_parents:
                reasons["parent_in_prior_split"] += 1
            elif components(row) & earlier_components:
                reasons["atomic_component_in_prior_split"] += 1
            else:
                eligible.append(row)
        chosen = eligible[: counts[split]]
        if len(chosen) != counts[split]:
            raise ValueError(f"{split}: requested {counts[split]}, only {len(eligible)} eligible")
        selected.extend(_case(row, split, source_split) for row in chosen)
        seen_parents.update(row["id"] for row in chosen)
        earlier_components.update(c for row in chosen for c in components(row))
        reports[split] = {
            "source_member": f"data/musique_ans_v1.0_{source_split}.jsonl",
            "allowed_hops": sorted(hops),
            "raw_count": len(candidates),
            "eligible_count": len(eligible),
            "selected_count": len(chosen),
            "exclusions": dict(reasons),
            "selected_hops": dict(
                Counter(str(len(row["question_decomposition"])) for row in chosen)
            ),
        }
    if len({case["id"] for case in selected}) != len(selected):
        raise ValueError("Duplicate or colliding parent IDs")
    archive_hash, prior_hash = sha256(archive), sha256(prior_cases)
    manifest = {
        "schema": "selective-delegation-inputs-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "selection": "SHA256(seed:source_id) ordering; train, validation, transfer priority",
        "counts": counts,
        "splits": reports,
        "source": {
            "archive_path": str(archive),
            "archive_sha256": archive_hash,
            "url": "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view",
            "repository": "https://github.com/StonyBrookNLP/musique",
            "repository_revision": REVISION,
            "license": "CC BY 4.0",
            "retrieval_date": "unknown; existing archive reused; previously recorded by 2026-09-14",
        },
        "prior_exposure_exclusion": {
            "path": str(prior_cases),
            "sha256": prior_hash,
            "parent_ids": sorted(excluded_ids),
        },
        "code_hashes": {
            name: sha256(Path(__file__).with_name(name)) for name in ("data.py", "test_data.py")
        },
        "constraints": [
            "All parent variants and seeds inherit the one parent split.",
            "Atomic component IDs are disjoint between selected splits, not within splits.",
            "Only question and paragraph id/title/text are model-visible.",
            "Prior breadth parents excluded; no claim of pretraining noncontamination.",
            "Transfer labels must remain unopened until training choices are fixed.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    case_path = output / "cases.jsonl"
    with case_path.open("x") as stream:
        for case in selected:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["cases_sha256"] = sha256(case_path)
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--transfer-count", type=int, default=64)
    args = parser.parse_args()
    receipt = prepare(args.output, args.train_count, args.validation_count, args.transfer_count)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": receipt["counts"],
                "cases_sha256": receipt["cases_sha256"],
            }
        )
    )
