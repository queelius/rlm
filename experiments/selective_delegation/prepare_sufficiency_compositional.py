"""Freeze an exploratory balanced 3/4-hop paired DEV panel without atomic filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import prepare_sufficiency_heldout as held

panel = held.panel
canonical = held.canonical
SEED = 2026092205


def select(pairs):
    selected = []
    for hop in (3, 4):
        candidates = [
            p for p, rows in pairs.items() if len(rows[0]["question_decomposition"]) == hop
        ]
        if len(candidates) < 16:
            raise ValueError("insufficient eligible parents; no replacement rule")
        selected.extend(
            sorted(candidates, key=lambda p: hashlib.sha256(f"{SEED}:{p}".encode()).hexdigest())[
                :16
            ]
        )
    return selected


def excluded(parent, normalized_question, parents, questions):
    return parent in parents or normalized_question in questions


def prepare(root, output):
    if output.exists():
        raise FileExistsError("immutable panel already exists")
    known = {k: set() for k in ("parents", "questions", "components")}
    prior_dev_atoms = set()
    paths = sorted(root.glob("*inputs*/cases.jsonl"))
    paths += [root / "sufficiency-rl-input-proposal-001/cases.jsonl"]
    paths += [
        root.parent / "unattended-breadth-20260914/data" / n
        for n in ("cases.jsonl", "cases-v2.jsonl")
    ]
    inventory = {}
    for path in paths:
        rows = panel.read_jsonl(path)
        for row in rows:
            parent, question, atoms = canonical._row_features(row)
            known["parents"].add(parent)
            known["questions"].add(question)
            known["components"].update(atoms)
            if row.get("split") != "train":
                prior_dev_atoms.update(atoms)
        inventory[str(path)] = {"sha256": panel.sha256(path), "rows": len(rows)}
    archive_sha = panel.sha256(panel.ARCHIVE)
    if archive_sha != "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd":
        raise ValueError("official archive changed")
    train = {k: set() for k in known}
    grouped, member_hashes = defaultdict(list), {}
    with zipfile.ZipFile(panel.ARCHIVE) as source:
        for member in (panel.MEMBER, canonical.TRAIN_MEMBER):
            digest = hashlib.sha256()
            with source.open(member) as stream:
                for line in stream:
                    digest.update(line)
                    row = json.loads(line)
                    if member == panel.MEMBER:
                        grouped[row["id"]].append(row)
                    else:
                        parent, question, atoms = held.features([row])
                        train["parents"].add(parent)
                        train["questions"].add(question)
                        train["components"].update(atoms)
            member_hashes[member] = digest.hexdigest()
    eligible, counts = {}, Counter()
    for parent, rows in grouped.items():
        if (
            len(rows) != 2
            or {r["answerable"] for r in rows} != {True, False}
            or len({r["question"] for r in rows}) != 1
        ):
            raise ValueError("malformed official DEV pair")
        p, q, _ = held.features(rows)
        why = []
        if excluded(p, q, known["parents"], known["questions"]):
            why.append("known_parent_or_question")
        if excluded(p, q, train["parents"], train["questions"]):
            why.append("official_train_parent_or_question")
        if why:
            counts.update(why)
        else:
            eligible[parent] = rows
    selected = select(eligible)
    cases, overlap = [], {}
    for parent in selected:
        atoms = held.features(eligible[parent])[2]
        overlap[parent] = {
            "official_train_atomic_ids": sorted(atoms & train["components"]),
            "prior_nontrain_panel_atomic_ids": sorted(atoms & prior_dev_atoms),
        }
        for row in eligible[parent]:
            public = canonical._public(row)
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
                    "component_ids": sorted(atoms),
                    "hops": len(row["question_decomposition"]),
                    "split": "development",
                }
            )
    cases.sort(key=lambda c: c["id"])
    if len({c["id"] for c in cases}) != 64:
        raise ValueError("not 64 distinct public variants")
    docs = {(d["title"], d["text"]) for c in cases for d in c["public"]["documents"]}
    titles = {t for t, _ in docs}
    train_docs = set()
    with zipfile.ZipFile(panel.ARCHIVE) as source, source.open(canonical.TRAIN_MEMBER) as stream:
        for line in stream:
            for d in json.loads(line)["paragraphs"]:
                value = d["title"], d["paragraph_text"]
                if value in docs or value[0] in titles:
                    train_docs.add(value)
    from transformers import AutoTokenizer

    model = held.baseline.native.evaluation.planner.BASE
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    lengths = held.prompt_lengths(tokenizer, cases)
    clusters = held.analyze_helper.component_clusters(
        [
            {"id": p, "metadata": {"component_ids": sorted(held.features(eligible[p])[2])}}
            for p in selected
        ]
    )
    output.mkdir(parents=True)
    with (output / "cases.jsonl").open("x") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema": "musique-compositional-paired-sufficiency-v1",
        "selection_seed": SEED,
        "selection_rule": "First16 SHA256('2026092205:'+officialDEVparent) within each of "
        "3-hop and 4-hop; no atomic/document/label/outcome filtering",
        "parent_count": 32,
        "variant_count": 64,
        "parents": selected,
        "hops": dict(Counter(c["hops"] for c in cases if c["answerable"])),
        "eligible_parents_by_hop": dict(
            Counter(len(rows[0]["question_decomposition"]) for rows in eligible.values())
        ),
        "nonexclusive_exclusion_counts": dict(counts),
        "excluded_inputs": inventory,
        "exclusion_policy": "Exact parent or normalized question in inventoried panels/all "
        "splits or official TRAIN; includes retained CPU-only panels "
        "without implying outcome exposure. No atomic/document filter.",
        "official_train_identity_counts": {k: len(v) for k, v in train.items()},
        "parent_atomic_overlap": overlap,
        "parents_with_official_train_atomic_overlap": sum(
            bool(o["official_train_atomic_ids"]) for o in overlap.values()
        ),
        "parents_with_prior_nontrain_panel_atomic_overlap": sum(
            bool(o["prior_nontrain_panel_atomic_ids"]) for o in overlap.values()
        ),
        "component_clusters": clusters,
        "component_cluster_count": len(clusters),
        "official_train_document_overlap": canonical.fresh.document_overlap(
            cases, list(train_docs)
        ),
        "public_document_order_seed": canonical.SEED,
        "evaluation_seeds": [2026092181, 2026092182],
        "planned_calls_per_model": 128,
        "planned_three_model_calls": 384,
        "prompt_token_counts": lengths,
        "max_prompt_plus_128": max(lengths.values()) + 128,
        "ready_without_truncation": max(lengths.values()) + 128 <= 8192,
        "cases_sha256": panel.sha256(output / "cases.jsonl"),
        "archive": str(panel.ARCHIVE),
        "archive_sha256": archive_sha,
        "members_sha256": member_hashes,
        "source_sha256": {
            **held.source_hashes(),
            str(Path(held.__file__).resolve()): panel.sha256(Path(held.__file__)),
        },
        "preparer_sha256": panel.sha256(Path(__file__)),
        "model_manifest_sha256": panel.sha256(model / "local-research-manifest.json"),
        "status": "CPU frozen; conditional046 readout not GPU accepted",
        "limitations": "Deliberately balanced depth panel, not natural DEV. Prior study DEV "
        "atomic overlap and TRAIN document overlap are not unseen facts. "
        "Zero official TRAIN atomic overlap does not imply no document reuse. "
        "Official sufficiency labels remain unchanged; no semantic relabeling.",
    }
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return {
        k: manifest[k]
        for k in (
            "cases_sha256",
            "hops",
            "component_cluster_count",
            "max_prompt_plus_128",
            "parents_with_official_train_atomic_overlap",
            "parents_with_prior_nontrain_panel_atomic_overlap",
            "official_train_document_overlap",
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=canonical.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve(), args.output.resolve())))
