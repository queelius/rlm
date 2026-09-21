"""Immutable label-blind128-parent canonical Hotpot validation replication panel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import analyze_helper
import eval_planner
import prepare_hotpot
from prepare_fresh_dev_panel import normalized_question

SEED = 2026092175
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SOURCE_SHA = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
SOURCE = Path("/project/alex_phd/research-cache/datasets/hotpotqa-official-dev-20260921/") / (
    "distractor-validation-00000-of-00001.parquet"
)
ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def rank(identity):
    return digest(f"{SEED}:{identity}")


def select_rows(rows, explorer, prior, *, count=128):
    ids = Counter(r["id"] for r in rows)
    explorer_ids = {r["_id"] for r in explorer}
    old_ids = {r["id"] for r in prior} | {
        r.get("metadata", {}).get("source_id")
        for r in prior
        if isinstance(r.get("metadata", {}).get("source_id"), str)
    }
    old_questions = {normalized_question(r["question"]) for r in prior + explorer}
    eligible, excluded, seen = [], [], set()
    for row in sorted(rows, key=lambda r: (rank(r["id"]), r["id"])):
        identity, question = row["id"], normalized_question(row["question"])
        reasons = []
        if identity in explorer_ids:
            reasons.append("explorer100_id")
        if identity in old_ids:
            reasons.append("previous_input_id")
        if question in old_questions:
            reasons.append("previous_normalized_original_question")
        if ids[identity] > 1:
            reasons.append("duplicate_source_id_all_copies_excluded")
        if question in seen:
            reasons.append("duplicate_normalized_question_in_pool")
        if reasons:
            excluded.append(
                {
                    "source_id": identity,
                    "rank_sha256": rank(identity),
                    "question_normalized_sha256": digest(question),
                    "reasons": reasons,
                }
            )
        else:
            seen.add(question)
            eligible.append(row)
    if len(eligible) < count:
        raise ValueError("not enough eligible parents for fixed panel")
    return eligible[:count], excluded, [r["id"] for r in eligible]


def document_key(document, *, normalized=False):
    pair = [document["title"], document["text"]]
    if normalized:
        pair = [normalized_question(value) for value in pair]
    return digest(json.dumps(pair, ensure_ascii=False))


def project(row):
    titles, paragraphs = row["context"]["title"], row["context"]["sentences"]
    documents = [
        {"title": title, "text": " ".join(sentences)}
        for title, sentences in zip(titles, paragraphs, strict=True)
    ]
    documents.sort(key=lambda d: digest(f"{SEED}:doc:{row['id']}:{d['title']}:{d['text']}"))
    documents = [{"id": f"d{i}", **doc} for i, doc in enumerate(documents)]
    return {
        "id": digest("hotpotqa-canonical-validation:" + row["id"])[:24],
        "split": "transfer",
        "dataset": "hotpotqa",
        "question": row["question"],
        "documents": documents,
        "answer": row["answer"],
        "metadata": {
            "source_id": row["id"],
            "source_split": "validation",
            "source_config": "distractor",
            "source_revision": REVISION,
            "answer_aliases": [],
            "type": row["type"],
            "level": row["level"],
            "supporting_facts": row["supporting_facts"],
            "component_ids": [
                "public-document:" + document_key(d, normalized=True) for d in documents
            ],
            "component_definition": "Shared NFKC/casefold/whitespace normalized public title+text; "
            "not gold support or annotated reasoning components.",
        },
    }


def document_audit(cases, prior):
    train = {r["id"]: r for r in prior if r["split"] == "train"}
    exact, normalized, titles = defaultdict(set), defaultdict(set), defaultdict(set)
    for case in train.values():
        for doc in case["documents"]:
            exact[document_key(doc)].add(case["id"])
            normalized[document_key(doc, normalized=True)].add(case["id"])
            titles[normalized_question(doc["title"])].add(case["id"])
    exposure = []
    for case in cases:
        for doc in case["documents"]:
            exposure.append(
                {
                    "case_id": case["id"],
                    "document_id": doc["id"],
                    "exact_title_text_sha256": document_key(doc),
                    "exact_train_parent_ids": sorted(exact[document_key(doc)]),
                    "normalized_train_parent_ids": sorted(
                        normalized[document_key(doc, normalized=True)]
                    ),
                    "title_train_parent_ids": sorted(titles[normalized_question(doc["title"])]),
                }
            )
    title_cases = [
        {
            "id": c["id"],
            "metadata": {
                "component_ids": [normalized_question(d["title"]) for d in c["documents"]]
            },
        }
        for c in cases
    ]
    return {
        "train_parents": len(train),
        "selected_documents": len(exposure),
        "exposure_filtered": False,
        "exact_title_text_exposed_documents": sum(
            bool(r["exact_train_parent_ids"]) for r in exposure
        ),
        "normalized_title_text_exposed_documents": sum(
            bool(r["normalized_train_parent_ids"]) for r in exposure
        ),
        "normalized_title_exposed_documents": sum(
            bool(r["title_train_parent_ids"]) for r in exposure
        ),
        "document_exposure": exposure,
        "document_clusters": analyze_helper.component_clusters(cases),
        "title_only_clusters": analyze_helper.component_clusters(title_cases),
        "caution": "Public-document connectivity is observed, not verified independent "
        "atomic reasoning components.",
    }


def token_audit(cases):
    from transformers import AutoTokenizer

    model = eval_planner.planner.BASE
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    rows = []
    for case in cases:
        prompts = {
            "planner_root": (eval_planner.planner_prompt(case), 128),
            "direct": (eval_planner.direct_prompt(case), 128),
            "original_question_helper_scaffold": (
                eval_planner.isolated_helper_prompt(case, case["question"]),
                384,
            ),
            "one_question_empty_trace_final_scaffold": (
                eval_planner.final_prompt(
                    case,
                    {"subquestions": [case["question"]]},
                    {"execution": "isolated", "steps": []},
                ),
                128,
            ),
        }
        for role, (prompt, cap) in prompts.items():
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
                raise ValueError("expected native input token IDs")
            rows.append(
                {
                    "case_id": case["id"],
                    "role": role,
                    "input_tokens": len(ids),
                    "output_cap": cap,
                    "exceeds_8192": len(ids) + cap > 8192,
                }
            )
    return {
        "model": str(model),
        "tokenizer_sha256": {
            p.name: prepare_hotpot.sha256(p)
            for p in sorted(model.glob("*"))
            if p.name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        },
        "rows": rows,
        "flagged": sum(r["exceeds_8192"] for r in rows),
        "truncated": False,
        "caution": "Root/direct are exact initial prompts. Helper/final are public scaffolds only; "
        "actual generated plans/resolved questions/traces require runtime context checks. "
        "No case is removed.",
    }


def prepare(root, output):
    import pyarrow.parquet as pq

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if prepare_hotpot.sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("pinned canonical Parquet changed")
    rows = pq.read_table(SOURCE).to_pylist()
    if len(rows) != 7405:
        raise ValueError("expected canonical7405 validation rows")
    explorer = json.loads(prepare_hotpot.SOURCE.read_text())
    if len(explorer) != 100 or len({r["_id"] for r in explorer}) != 100:
        raise ValueError("expected complete explorer100 exclusion inventory")
    paths = sorted(root.rglob("cases.jsonl"))
    required = {
        "inputs-001",
        "hotpot-inputs-001",
        "fresh-dev-inputs-001",
        "fresh-dev-inputs-002",
        "fresh-dev-inputs-003",
    }
    if not required <= {p.parent.name for p in paths}:
        raise ValueError("missing September21 input exclusion inventory")
    prior = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    selected, excluded, eligible = select_rows(rows, explorer, prior)
    cases = [project(row) for row in selected]
    if len({c["id"] for c in cases}) != 128:
        raise ValueError("opaque case ID collision")
    audit, tokens = document_audit(cases, prior), token_audit(cases)
    if audit["train_parents"] != 256:
        raise ValueError("expected256 unique September21 training parents for exposure audit")
    serialized = "".join(
        json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in cases
    ).encode()
    manifest = {
        "schema": "canonical-hotpot-fresh-inputs-v1",
        "status": "prepared_no_model_answers",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {"transfer": 128},
        "source": {
            "path": str(SOURCE),
            "sha256": SOURCE_SHA,
            "revision": REVISION,
            "url": f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{REVISION}/distractor/validation-00000-of-00001.parquet",
            "origin": "canonical distractor validation HF mirror; "
            "not byte-identical-JSON assertion",
            "license": "CC-BY-SA-4.0",
            "rows": 7405,
            "acquisition_sha256": prepare_hotpot.sha256(SOURCE.parent / "ACQUISITION.json"),
        },
        "selection": {
            "seed": SEED,
            "count": 128,
            "rule": "Ascending SHA256('2026092175:'+original_id), first128 after "
            "identity/original-question exclusions; no labels/support/outcomes used.",
            "question_normalization": "NFKC, casefold, collapse Unicode whitespace",
            "explorer100_source_ids": sorted(r["_id"] for r in explorer),
            "selected_source_ids": [r["id"] for r in selected],
            "eligible_source_ids_in_rank_order": eligible,
            "exclusions": excluded,
            "exclusion_reason_counts": dict(
                Counter(reason for e in excluded for reason in e["reasons"])
            ),
            "prior_case_ids": sorted({c["id"] for c in prior}),
            "prior_normalized_question_sha256": sorted(
                {digest(normalized_question(c["question"])) for c in prior}
            ),
            "input_sha256": {
                str(p): prepare_hotpot.sha256(p) for p in [*paths, prepare_hotpot.SOURCE]
            },
            "type_counts_descriptive_only": dict(Counter(r["type"] for r in selected)),
        },
        "host_only": "Gold answer/type/level/supporting facts/source identities/connectivity/"
        "exposure; prompts use existing public projection only.",
        "scoring": {
            "metric": "official_hotpotqa_em_f1",
            "evaluator": str(prepare_hotpot.OFFICIAL_EVALUATOR),
            "evaluator_sha256": prepare_hotpot.sha256(prepare_hotpot.OFFICIAL_EVALUATOR),
        },
        "document_audit": audit,
        "token_audit": tokens,
        "environment": {
            p: importlib.metadata.version(p) for p in ("pyarrow", "transformers", "tokenizers")
        },
        "code_sha256": {
            str(p): prepare_hotpot.sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("test_prepare_hotpot_fresh.py"),
                Path(prepare_hotpot.__file__),
                Path(eval_planner.__file__),
                Path(analyze_helper.__file__),
                Path(__file__).with_name("prepare_fresh_dev_panel.py"),
                Path(eval_planner.planner.__file__),
                Path(eval_planner.probe.__file__),
            )
        },
        "cases_sha256": hashlib.sha256(serialized).hexdigest(),
        "limitations": [
            "Within-dataset validation replication, not untouched pretraining or OOD evidence.",
            "Natural hash-uniform panel, not type-balanced; document exposure is recorded, "
            "never filtered.",
            "Selection was not conditioned on model outcomes or gold-derived difficulty.",
        ],
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
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.root, args.output)
    print(
        json.dumps(
            {
                "cases_sha256": receipt["cases_sha256"],
                "counts": receipt["counts"],
                "token_flags": receipt["token_audit"]["flagged"],
            }
        )
    )
