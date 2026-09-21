"""CPU-only128new TRAIN pair proposal; no optimizer, model calls or accepted RL job."""

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import prepare_sufficiency_training as training

SEED = 2026092193


def select(parents, count=128):
    if len(set(parents)) != len(parents) or len(parents) < count:
        raise ValueError("insufficient/duplicate TRAIN inventory")
    return sorted(parents, key=lambda p: hashlib.sha256(f"{SEED}:{p}".encode()).hexdigest())[:count]


def prepare(root, output):
    if output.exists():
        raise FileExistsError("immutable proposal already exists")
    prior_path = root / "sufficiency-sft-inputs-draft-001/MANIFEST.json"
    prior = json.loads(prior_path.read_text())
    exclusions = {k: set(v) for k, v in prior["excluded_identities"].items()}
    exclusions["parents"].update(prior["selected_parents"])
    sources = {str(prior_path): training.panel.sha256(prior_path)}
    paths = sorted(root.glob("*inputs*/cases.jsonl"))
    paths += [
        root.parent / "unattended-breadth-20260914/data" / name
        for name in ("cases.jsonl", "cases-v2.jsonl")
    ]
    counts = {}
    for path in paths:
        rows = training.panel.read_jsonl(path)
        for row in rows:
            training.add_exclusion(row, exclusions)  # Intentionally ALL splits, including TRAIN.
        sources[str(path)] = training.panel.sha256(path)
        counts[str(path)] = len(rows)
    # Explicit prepared-SFT parent inventory; original public metadata is expanded below.
    for path in sorted(root.glob("*sft*inputs*/**/examples.jsonl")):
        rows = training.panel.read_jsonl(path)
        for row in rows:
            exclusions["parents"].add(row.get("parent_id", row["id"]))
        sources[str(path)] = training.panel.sha256(path)
        counts[str(path)] = len(rows)
    archive = training.panel.ARCHIVE
    if training.panel.sha256(archive) != prior["archive_sha256"]:
        raise ValueError("official TRAIN archive differs")
    metadata, digest = defaultdict(list), hashlib.sha256()
    with zipfile.ZipFile(archive) as source:
        with source.open(training.TRAIN_MEMBER) as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                slim = {
                    k: row[k] for k in ("id", "question", "question_decomposition", "answerable")
                }
                metadata[row["id"]].append(slim)
        if digest.hexdigest() != prior["member_sha256"][training.TRAIN_MEMBER]:
            raise ValueError("official TRAIN member differs")
        for parent, pair in metadata.items():
            if parent in exclusions["parents"]:
                for row in pair:
                    training.add_exclusion(row, exclusions)
        rejected, eligible = {}, []
        for parent, pair in metadata.items():
            if len(pair) != 2 or {r["answerable"] for r in pair} != {True, False}:
                raise ValueError("malformed official TRAIN pair")
            reasons = training.exclusion_reasons(pair, exclusions)
            if reasons:
                rejected[parent] = reasons
            else:
                eligible.append(parent)
        selected = select(eligible)
        wanted, pairs = set(selected), defaultdict(list)
        with source.open(training.TRAIN_MEMBER) as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in wanted:
                    pairs[row["id"]].append(row)
    cases = []
    for parent in selected:
        for row in sorted(pairs[parent], key=lambda r: not r["answerable"]):
            public = training.panel.public_context(row)
            cases.append(
                {
                    "id": hashlib.sha256(
                        json.dumps(public, sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest()[:24],
                    "parent_id": parent,
                    "split": "train",
                    "public": public,
                    "answerable": row["answerable"],
                    "answer": row["answer"],
                    "answer_aliases": row["answer_aliases"],
                    "component_ids": sorted(training.features(row)[2]),
                }
            )
    if len({c["id"] for c in cases}) != 256:
        raise ValueError("duplicate public variants; no silent replacement")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        training.sufficiency.native.evaluation.planner.BASE,
        local_files_only=True,
        trust_remote_code=False,
    )
    examples, _ = training.build_examples(pairs, selected)
    audit = training.audit_tokens(examples, tokenizer)
    max_prompt_plus_generation = max(r["prompt_tokens"] + 128 for r in audit["rows"])
    manifest = {
        "schema": "paired-sufficiency-rl-input-proposal-v1",
        "status": "proposed_cpu_only",
        "seed": SEED,
        "parent_count": 128,
        "variant_count": 256,
        "selection": "first128 SHA256('2026092193:'+officialTRAINparent), no outcome filtering",
        "selected_parents": selected,
        "case_ids": [c["id"] for c in cases],
        "fixed_parent_blocks": [selected[i : i + 16] for i in range(0, 128, 16)],
        "eligible_parent_count": len(eligible),
        "rejected_parents": rejected,
        "archive": str(archive),
        "archive_sha256": prior["archive_sha256"],
        "train_member": training.TRAIN_MEMBER,
        "train_member_sha256": digest.hexdigest(),
        "exclusion_source_sha256": sources,
        "exclusion_row_counts": counts,
        "inherited_official_dev_test_exclusions": prior["member_sha256"],
        "excluded_identity_counts": {k: len(v) for k, v in exclusions.items()},
        "scope": "Explicit inventoried September21 inputs/all splits, both historical breadth "
        "case inventories/all splits, all prepared SFT parents, and inherited full official "
        "DEV/TEST; not an assertion of unknowable history or pretraining disjointness.",
        "token_audit": audit,
        "max_prompt_plus_generation128": max_prompt_plus_generation,
        "ready_without_truncation": not audit["exceeds_8192"]
        and not audit["exceeds_target128"]
        and max_prompt_plus_generation <= 8192,
        "source_sha256": {
            str(Path(m.__file__).resolve()): training.panel.sha256(Path(m.__file__))
            for m in (training, training.panel, training.sufficiency)
        },
        "preparer_sha256": training.panel.sha256(Path(__file__)),
        "model_manifest_sha256": training.panel.sha256(
            training.sufficiency.native.evaluation.planner.BASE / "local-research-manifest.json"
        ),
        "scientific_status": "Input proposal only; no optimizer or GPU work accepted. "
        "Future reward/loop acceptance remains conditional on SFT and "
        "exposure-audited replication.",
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cases.jsonl").open("x") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["cases_sha256"] = training.panel.sha256(output / "cases.jsonl")
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return {
        k: manifest[k]
        for k in (
            "parent_count",
            "variant_count",
            "eligible_parent_count",
            "ready_without_truncation",
            "max_prompt_plus_generation128",
            "cases_sha256",
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve(), args.output.resolve())))
