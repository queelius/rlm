"""CPU-only fixed TRAIN pairs and equal-example positive-only SFT control."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import prepare_sufficiency_panel as panel
import sufficiency_probe as sufficiency
import train_planner

SEED = 2026092189
TRAIN_MEMBER = "data/musique_full_v1.0_train.jsonl"


def empty_exclusions():
    return {"parents": set(), "questions": set(), "components": set()}


def features(row):
    meta = row.get("metadata", {})
    parent = meta.get("source_id", row.get("parent_id", row.get("id", "")))
    question = row.get("question", row.get("public", {}).get("question", ""))
    atoms = panel.source_components(parent)
    atoms.update(map(str, row.get("component_ids", [])))
    atoms.update(map(str, meta.get("component_ids", [])))
    atoms.update(str(s["id"]) for s in row.get("question_decomposition", []))
    atoms.update(str(s["id"]) for s in meta.get("question_decomposition", []))
    return parent, panel.normalized_question(question), atoms


def add_exclusion(row, exclusions):
    parent, question, atoms = features(row)
    exclusions["parents"].add(parent)
    if question:
        exclusions["questions"].add(question)
    exclusions["components"].update(atoms)


def exclusion_reasons(pair, exclusions):
    values = [features(r) for r in pair]
    return [
        name
        for name, hit in (
            ("parent", any(p in exclusions["parents"] for p, _, _ in values)),
            ("question", any(q in exclusions["questions"] for _, q, _ in values)),
            ("component", any(a & exclusions["components"] for _, _, a in values)),
        )
        if hit
    ]


def build_examples(pairs, selected):
    joint, positive = [], []
    for parent in selected:
        ordered = sorted(pairs[parent], key=lambda r: not r["answerable"])
        if len(ordered) != 2 or [r["answerable"] for r in ordered] != [True, False]:
            raise ValueError("complete official pair required")
        for index, row in enumerate(ordered):
            case = {"public": panel.public_context(row)}
            joint.append(
                dict(
                    id=f"{parent}-slot{index}",
                    parent_id=parent,
                    split="train",
                    prompt=sufficiency.prompt(case),
                    target=json.dumps(
                        {
                            "answerable": row["answerable"],
                            "answer": row["answer"] if row["answerable"] else "",
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )
        for index in range(2):
            positive.append({**joint[-2], "id": f"{parent}-slot{index}"})
    return joint, positive


def audit_tokens(rows, tokenizer):
    audit = []
    for row in rows:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        target = tokenizer.encode(row["target"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        audit.append(
            dict(
                id=row["id"],
                prompt_tokens=len(prefix),
                target_tokens=len(target),
                total_tokens=len(prefix) + len(target),
            )
        )
    return dict(
        rows=audit,
        prompt_tokens=sum(r["prompt_tokens"] for r in audit),
        target_tokens=sum(r["target_tokens"] for r in audit),
        max_prompt=max(r["prompt_tokens"] for r in audit),
        max_target=max(r["target_tokens"] for r in audit),
        max_total=max(r["total_tokens"] for r in audit),
        exceeds_8192=sum(r["total_tokens"] > 8192 for r in audit),
        exceeds_target128=sum(r["target_tokens"] > 128 for r in audit),
        truncation=False,
    )


def prepare(root, output, *, tokenizer=None):
    if output.exists():
        raise FileExistsError("immutable preparation exists")
    archive = panel.ARCHIVE
    if panel.sha256(archive) != "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd":
        raise ValueError("official archive identity changed")
    exclusions = empty_exclusions()
    sources, members, counts = {}, {}, Counter()
    with zipfile.ZipFile(archive) as source:
        for name in ("data/musique_full_v1.0_dev.jsonl", "data/musique_full_v1.0_test.jsonl"):
            digest = hashlib.sha256()
            with source.open(name) as stream:
                for line in stream:
                    digest.update(line)
                    add_exclusion(json.loads(line), exclusions)
                    counts[name] += 1
            members[name] = digest.hexdigest()
        paths = sorted(root.glob("*inputs*/cases.jsonl"))
        paths += [
            root.parent / "unattended-breadth-20260914/data/cases-v2.jsonl",
            archive.parent / "hotpot_official_explorer_sample100.json",
        ]
        for path in paths:
            if not path.exists():
                raise ValueError("known exclusion input missing: " + str(path))
            sources[str(path)] = panel.sha256(path)
            rows = (
                json.loads(path.read_text()) if path.suffix == ".json" else panel.read_jsonl(path)
            )
            for row in rows:
                if row.get("split") == "train":
                    continue
                add_exclusion(row, exclusions)
                counts[str(path)] += 1
        # First pass stores eligibility metadata only, not all476MB of public documents.
        metadata = defaultdict(list)
        digest = hashlib.sha256()
        with source.open(TRAIN_MEMBER) as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                metadata[row["id"]].append(
                    {k: row[k] for k in ("id", "question", "question_decomposition", "answerable")}
                )
        members[TRAIN_MEMBER] = digest.hexdigest()
        rejected, eligible = {}, []
        for parent, pair in metadata.items():
            if len(pair) != 2 or {r["answerable"] for r in pair} != {True, False}:
                raise ValueError("TRAIN pair inventory malformed")
            reasons = exclusion_reasons(pair, exclusions)
            if reasons:
                rejected[parent] = reasons
            else:
                eligible.append(parent)
        eligible.sort(key=lambda p: hashlib.sha256(f"{SEED}:{p}".encode()).hexdigest())
        selected = eligible[:256]
        if len(selected) != 256:
            raise ValueError("fewer than256 eligible TRAIN parents; no replacement")
        wanted = set(selected)
        pairs = defaultdict(list)
        with source.open(TRAIN_MEMBER) as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in wanted:
                    pairs[row["id"]].append(row)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            sufficiency.native.evaluation.planner.BASE,
            local_files_only=True,
            trust_remote_code=False,
        )
    joint, positive = build_examples(pairs, selected)
    audits = {
        arm: audit_tokens(rows, tokenizer)
        for arm, rows in [("joint", joint), ("positive_only", positive)]
    }
    report = dict(
        schema="sufficiency-train-comparison-inputs-v1",
        status="unsealed_cpu_preparation",
        seed=SEED,
        train_member=TRAIN_MEMBER,
        archive=str(archive),
        archive_sha256=panel.sha256(archive),
        member_sha256=members,
        exclusion_source_sha256=sources,
        exclusion_row_counts=dict(counts),
        excluded_identity_counts={k: len(v) for k, v in exclusions.items()},
        excluded_identities={k: sorted(v) for k, v in exclusions.items()},
        train_parent_count=len(metadata),
        eligible_parent_count=len(eligible),
        rejected_parents=rejected,
        selected_parents=selected,
        selected_hops=dict(Counter(len(pairs[p][0]["question_decomposition"]) for p in selected)),
        token_audits=audits,
        selection="first256 SHA256(seed:TRAINparent), no outcome/hop/length filter",
        comparison="equal512examples and32updates; not equal tokens/information; "
        "positive inputs repeatedtwice",
        training_ready=all(
            not a["exceeds_8192"] and not a["exceeds_target128"] for a in audits.values()
        ),
        source_sha256={
            str(Path(m.__file__).resolve()): panel.sha256(Path(m.__file__))
            for m in (panel, sufficiency, train_planner)
        },
        preparer_sha256=panel.sha256(Path(__file__)),
    )
    output.mkdir(parents=True, exist_ok=False)
    for arm, rows in [("joint", joint), ("positive_only", positive)]:
        destination = output / arm
        destination.mkdir()
        data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        with (destination / "examples.jsonl").open("x") as stream:
            stream.write(data)
        manifest = dict(
            role="sufficiency",
            arm=arm,
            examples=512,
            training_parents=256,
            split="train",
            seed=SEED,
            selected_parents=selected,
            examples_sha256=panel.sha256(destination / "examples.jsonl"),
            model=str(sufficiency.native.evaluation.planner.BASE),
            model_manifest_sha256=panel.sha256(
                sufficiency.native.evaluation.planner.BASE / "local-research-manifest.json"
            ),
            token_audit=audits[arm],
            training_order=[rows[i]["id"] for i in train_planner.epoch_order(512, SEED, 0)],
            target="official TRAIN answerable+answer JSON; negative empty; "
            "privileged labels only in target",
            source_sha256=report["source_sha256"],
            preparer_sha256=report["preparer_sha256"],
        )
        with (destination / "MANIFEST.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=panel.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "train_parent_count",
                    "eligible_parent_count",
                    "selected_hops",
                    "training_ready",
                )
            }
            | {
                "token_audits": {
                    a: {k: v for k, v in r.items() if k != "rows"}
                    for a, r in result["token_audits"].items()
                }
            }
        )
    )
