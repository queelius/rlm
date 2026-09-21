"""Freeze a separate paired DEV panel before conditional sufficiency RL/SFT updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import analyze_helper
import prepare_sufficiency_canonical_panel as canonical
import sufficiency_probe as baseline

panel = canonical.panel
SEED = 2026092198
TRAIN_PROPOSAL_SHA = "774ceda6bbac38b23fec6f823909e606889e804e7201fd37f338ffb81e9f7f81"


def select(pairs, count=32):
    if len(pairs) < count:
        raise ValueError("insufficient eligible pairs; no outcome-based replacement")
    return sorted(pairs, key=lambda p: hashlib.sha256(f"{SEED}:{p}".encode()).hexdigest())[:count]


def source_hashes():
    return {
        str(Path(m.__file__).resolve()): panel.sha256(Path(m.__file__))
        for m in (canonical, canonical.fresh, panel, baseline, analyze_helper)
    }


def prompt_lengths(tokenizer, cases):
    return {
        c["id"]: len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": baseline.prompt(c)}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        for c in cases
    }


def features(rows):
    parent = rows[0]["id"]
    question = panel.normalized_question(rows[0]["question"])
    atoms = panel.source_components(parent) | {
        str(step["id"]) for row in rows for step in row.get("question_decomposition", [])
    }
    return parent, question, atoms


def reasons(rows, excluded):
    parent, question, atoms = features(rows)
    return (
        (["known_parent"] if parent in excluded["parents"] else [])
        + (["known_question"] if question in excluded["questions"] else [])
        + (["known_component"] if atoms & excluded["components"] else [])
    )


def prepare(root, output):
    if output.exists():
        raise FileExistsError("immutable held-out panel already exists")
    train_proposal = root / "sufficiency-rl-input-proposal-001/cases.jsonl"
    if panel.sha256(train_proposal) != TRAIN_PROPOSAL_SHA:
        raise ValueError("128-parent TRAIN proposal differs")
    excluded = {k: set() for k in ("parents", "questions", "components")}
    inputs = sorted(root.glob("*inputs*/cases.jsonl")) + [train_proposal]
    inputs += [
        root.parent / "unattended-breadth-20260914/data" / name
        for name in ("cases.jsonl", "cases-v2.jsonl")
    ]
    inventory = {}
    for path in inputs:
        rows = panel.read_jsonl(path)
        for row in rows:
            parent, question, atoms = canonical._row_features(row)
            excluded["parents"].add(parent)
            excluded["questions"].add(question)
            excluded["components"].update(atoms)
        inventory[str(path)] = {
            "rows": len(rows),
            "sha256": panel.sha256(path),
            "policy": "exclude all rows/all splits, including conservatively retained CPU panels",
        }
    archive_hash = panel.sha256(panel.ARCHIVE)
    if archive_hash != "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd":
        raise ValueError("official archive changed")
    train = {k: set() for k in excluded}
    grouped, members = defaultdict(list), {}
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
                        parent, question, atoms = features([row])
                        train["parents"].add(parent)
                        train["questions"].add(question)
                        train["components"].update(atoms)
            members[member] = digest.hexdigest()
    eligible, counts = {}, Counter()
    for parent, pair in grouped.items():
        if (
            len(pair) != 2
            or {r["answerable"] for r in pair} != {True, False}
            or len({r["question"] for r in pair}) != 1
        ):
            raise ValueError("malformed official paired DEV inventory")
        why = reasons(pair, excluded) + [
            "official_train_" + x.removeprefix("known_") for x in reasons(pair, train)
        ]
        if why:
            counts.update(why)
        else:
            eligible[parent] = pair
    selected = select(eligible)
    cases = []
    for parent in selected:
        atoms = sorted(features(eligible[parent])[2])
        for row in eligible[parent]:
            public = canonical._public(row)  # Same qualified label-blind document order algorithm.
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
                    "component_ids": atoms,
                    "hops": len(row["question_decomposition"]),
                    "split": "development",
                }
            )
    cases.sort(key=lambda c: c["id"])
    if len(cases) != 64 or len({c["id"] for c in cases}) != 64:
        raise ValueError("not 64 unique retained variants")
    selected_docs = {(d["title"], d["text"]) for c in cases for d in c["public"]["documents"]}
    titles = {title for title, _ in selected_docs}
    train_docs = set()
    with zipfile.ZipFile(panel.ARCHIVE) as source, source.open(canonical.TRAIN_MEMBER) as stream:
        for line in stream:
            for d in json.loads(line)["paragraphs"]:
                value = d["title"], d["paragraph_text"]
                if value in selected_docs or value[0] in titles:
                    train_docs.add(value)
    from transformers import AutoTokenizer

    model = baseline.native.evaluation.planner.BASE
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    lengths = prompt_lengths(tokenizer, cases)
    clusters = analyze_helper.component_clusters(
        [
            {"id": p, "metadata": {"component_ids": sorted(features(eligible[p])[2])}}
            for p in selected
        ]
    )
    output.mkdir(parents=True)
    with (output / "cases.jsonl").open("x") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema": "musique-heldout-paired-sufficiency-v1",
        "selection_seed": SEED,
        "selection_rule": "First32 SHA256('2026092198:'+officialDEVparent); "
        "no hop/label/outcome filter",
        "public_document_order_seed": canonical.SEED,
        "parent_count": 32,
        "variant_count": 64,
        "parents": selected,
        "eligible_parent_count": len(eligible),
        "natural_hops": dict(Counter(c["hops"] for c in cases if c["answerable"])),
        "nonexclusive_exclusion_counts": dict(counts),
        "excluded_inputs": inventory,
        "exclusion_policy": "All inventoried case panels/all splits and both breadth inventories; "
        "128-parent TRAIN proposal and full official TRAIN. Includes CPU-abandoned panels; "
        "this is not a claim that those panels were outcome-exposed.",
        "official_train_identity_counts": {k: len(v) for k, v in train.items()},
        "official_train_parent_question_component_overlap": 0,
        "component_clusters": clusters,
        "component_cluster_count": len(clusters),
        "official_train_document_overlap": canonical.fresh.document_overlap(
            cases, list(train_docs)
        ),
        "evaluation_seeds": [2026092181, 2026092182],
        "planned_calls_per_model": 128,
        "planned_three_model_calls": 384,
        "prompt_token_counts": lengths,
        "max_prompt_plus_128": max(lengths.values()) + 128,
        "ready_without_truncation": max(lengths.values()) + 128 <= 8192,
        "cases_sha256": panel.sha256(output / "cases.jsonl"),
        "archive": str(panel.ARCHIVE),
        "archive_sha256": archive_hash,
        "members_sha256": members,
        "source_sha256": source_hashes(),
        "preparer_sha256": panel.sha256(Path(__file__)),
        "official_metric_sha256": {
            str(p): panel.sha256(p) for p in (panel.OFFICIAL / "metrics").glob("*.py")
        },
        "model_manifest_sha256": panel.sha256(model / "local-research-manifest.json"),
        "status": "Frozen CPU panel before conditional037/038 updates; readout not GPU accepted",
        "limitations": "Not semantic/document/pretraining clean. "
        "Report realized hop mix, document exposure and shared components; all32 remain primary.",
    }
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return {
        k: manifest[k]
        for k in (
            "parent_count",
            "eligible_parent_count",
            "natural_hops",
            "component_cluster_count",
            "max_prompt_plus_128",
            "ready_without_truncation",
            "cases_sha256",
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=canonical.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve(), args.output.resolve())))
