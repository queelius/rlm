"""Freeze an outcome-independent paired official MuSiQue full-dev panel."""

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from audit_missing_evidence_pairs import normalized_question, read_jsonl, sha256, source_components

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
CACHE = Path("/project/alex_phd/research-cache")
ARCHIVE = (
    CACHE / "datasets/musique-v1.0-922ac98f19a201998dbdae6d7f2887a5258dbdeb/musique_data_v1.0.zip"
)
OFFICIAL = CACHE / "repos/musique-922ac98f19a201998dbdae6d7f2887a5258dbdeb"
MEMBER = "data/musique_full_v1.0_dev.jsonl"
SEED = 2026092180


def selection_key(source_id):
    return hashlib.sha256(f"{SEED}:{source_id}".encode()).hexdigest()


def select_pairs(pairs, count=32):
    if len(pairs) < count:
        raise ValueError("insufficient eligible parents; no replacement")
    return sorted(pairs, key=selection_key)[:count]


def public_context(row):
    docs = [{"title": p["title"], "text": p["paragraph_text"]} for p in row["paragraphs"]]
    docs.sort(
        key=lambda d: hashlib.sha256(
            json.dumps([SEED, row["question"], d], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    )
    return {
        "question": row["question"],
        "documents": [{"docid": f"d{i}", **d} for i, d in enumerate(docs)],
    }


def prepare(root, output):
    if output.exists():
        raise FileExistsError("immutable panel already exists")
    inputs = [root / "inputs-001/cases.jsonl", root / "fresh-dev-inputs-003/cases.jsonl"]
    breadth = root.parent / "unattended-breadth-20260914/data/cases-v2.jsonl"
    hotpot = [root / "hotpot-inputs-001/cases.jsonl", root / "hotpot-fresh-inputs-001/cases.jsonl"]
    original100 = ARCHIVE.parent / "hotpot_official_explorer_sample100.json"
    audit = root / "analysis-missing-evidence-pairs-001.json"
    paths = [ARCHIVE, audit, *inputs, breadth, *hotpot, original100]
    if sha256(ARCHIVE) != "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd":
        raise ValueError("official archive changed")
    seen = sum((read_jsonl(p) for p in inputs), [])
    seen += [r for r in read_jsonl(breadth) if r.get("dataset") == "musique"]
    parents, questions, components = set(), set(), set()
    for row in seen:
        meta = row.get("metadata", {})
        sid = meta.get("source_id", row["id"])
        parents.add(sid)
        questions.add(normalized_question(row["question"]))
        components.update(source_components(sid))
        components.update(map(str, meta.get("component_ids", [])))
    hotpot_rows = sum((read_jsonl(p) for p in hotpot if p.exists()), [])
    hotpot_rows += json.loads(original100.read_text())
    hotpot_questions = {normalized_question(r["question"]) for r in hotpot_rows}
    with zipfile.ZipFile(ARCHIVE) as archive:
        member_bytes = archive.read(MEMBER)
    grouped = defaultdict(list)
    for line in member_bytes.splitlines():
        row = json.loads(line)
        grouped[row["id"]].append(row)
    eligible, excluded, old_eligible = {}, Counter(), 0
    for sid, pair in grouped.items():
        if len(pair) != 2 or {r["answerable"] for r in pair} != {True, False}:
            raise ValueError("official pair malformed")
        if len({r["question"] for r in pair}) != 1:
            raise ValueError("pair question differs")
        atoms = source_components(sid) | {
            str(step["id"]) for r in pair for step in r["question_decomposition"]
        }
        qs = {normalized_question(r["question"]) for r in pair}
        reasons = []
        if sid in parents:
            reasons.append("prior_parent")
        if qs & questions:
            reasons.append("prior_normalized_question")
        if atoms & components:
            reasons.append("prior_atomic_component_union_both_variants")
        old_eligible += not reasons
        if qs & hotpot_questions:
            reasons.append("prior_hotpot_normalized_question")
        if reasons:
            excluded.update(reasons)
        else:
            eligible[sid] = pair
    if old_eligible != 438:
        raise ValueError(f"expected438 pre-Hotpot eligible, found{old_eligible}")
    selected, cases = select_pairs(eligible), []
    for sid in selected:
        for row in eligible[sid]:
            public = public_context(row)
            identity = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()[:24]
            cases.append(
                {
                    "id": identity,
                    "parent_id": sid,
                    "public": public,
                    "answerable": row["answerable"],
                    "answer": row["answer"],
                    "answer_aliases": row["answer_aliases"],
                    "component_ids": sorted(
                        {str(s["id"]) for r in eligible[sid] for s in r["question_decomposition"]}
                    ),
                    "hops": len(row["question_decomposition"]),
                }
            )
    if len({r["id"] for r in cases}) != 64:
        raise ValueError("public variant identities not unique")
    cases.sort(key=lambda r: r["id"])
    output.mkdir(parents=True)
    with (output / "cases.jsonl").open("x") as stream:
        for row in cases:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "musique-full-sufficiency-panel-v1",
        "selection_seed": SEED,
        "selection_rule": "first32 SHA256('2026092180:'+originalid), no hop filter",
        "parents": selected,
        "parent_count": 32,
        "variant_count": 64,
        "pre_hotpot_eligible": old_eligible,
        "eligible": len(eligible),
        "nonexclusive_exclusions": dict(excluded),
        "hotpot_rows_checked": len(hotpot_rows),
        "hotpot_unique_questions": len(hotpot_questions),
        "selected_hops": dict(
            Counter(len(eligible[s][0]["question_decomposition"]) for s in selected)
        ),
        "source_sha256": {str(p): sha256(p) for p in paths if p.exists()},
        "member": MEMBER,
        "member_sha256": hashlib.sha256(member_bytes).hexdigest(),
        "cases_sha256": sha256(output / "cases.jsonl"),
        "preparer_sha256": sha256(Path(__file__)),
        "official_metric_sha256": {str(p): sha256(p) for p in (OFFICIAL / "metrics").glob("*.py")},
        "license": "CC BY4.0; official full development split",
        "limitations": "Known-parent/component disjoint, not semantic or pretraining clean. "
        "Both variants retain natural title/content/length confounds; not pure evidence deletion.",
    }
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output), indent=2))
