"""Report public-document overlap from the 256-parent SFT corpus to frozen panels.

This is an exposure audit only: it never changes inputs or selects/excludes cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def document_role(case: dict, document: dict) -> str:
    metadata = case["metadata"]
    support_key = next(
        (key for key in ("supporting_paragraph_ids", "supporting_document_ids") if key in metadata),
        None,
    )
    if support_key is None:
        return "unknown"
    support_ids = set(map(str, metadata[support_key]))
    return "support" if str(document["id"]) in support_ids else "distractor"


def document_key(document: dict) -> tuple[str, str]:
    return document["title"], document["text"]


def collect_panel(cases: list[dict], train_documents: dict, train_titles: dict) -> dict:
    exact, titles = set(), set()
    panel_roles, train_roles = Counter(), Counter()
    parents_exact = parents_title = 0
    per_parent = []
    for case in cases:
        exact_here = title_here = False
        exact_keys, title_keys = set(), set()
        parent_panel_roles, parent_train_roles = Counter(), Counter()
        for document in case["documents"]:
            key = document_key(document)
            if key in train_documents:
                exact.add(key)
                exact_keys.add(key)
                exact_here = True
                panel_role = document_role(case, document)
                panel_roles[panel_role] += 1
                parent_panel_roles[panel_role] += 1
                matched_train_roles = train_documents[key]
                train_roles.update(matched_train_roles)
                parent_train_roles.update(matched_train_roles)
            if document["title"] in train_titles:
                titles.add(document["title"])
                title_keys.add(document["title"])
                title_here = True
        parents_exact += exact_here
        parents_title += title_here
        per_parent.append(
            {
                "parent_id": case["id"],
                "exact_title_text": {
                    "match_occurrences": sum(parent_panel_roles.values()),
                    "shared_unique_documents": len(exact_keys),
                },
                "title": {"shared_unique_titles": len(title_keys)},
                "exact_match_occurrences_by_panel_role": dict(sorted(parent_panel_roles.items())),
                "exact_match_occurrences_by_train_role": dict(sorted(parent_train_roles.items())),
            }
        )
    return {
        "parents": len(cases),
        "documents": sum(len(case["documents"]) for case in cases),
        "unique_documents": len({document_key(d) for case in cases for d in case["documents"]}),
        "unique_titles": len({d["title"] for case in cases for d in case["documents"]}),
        "exact_title_text": {
            "shared_unique_documents": len(exact),
            "parents_with_match": parents_exact,
        },
        "title": {"shared_unique_titles": len(titles), "parents_with_match": parents_title},
        "exact_match_occurrences_by_panel_role": dict(sorted(panel_roles.items())),
        "exact_match_occurrences_by_train_role": dict(sorted(train_roles.items())),
        "parents_detail": per_parent,
    }


def panel_from_plan(path: Path, lookup: dict[str, dict]) -> list[dict]:
    plan = json.loads(path.read_text())
    return [lookup[case_id] for case_id in plan["case_ids"]]


def audit(root: Path) -> dict:
    inputs = root / "inputs-001/cases.jsonl"
    hotpot = root / "hotpot-inputs-001/cases.jsonl"
    fresh = root / "fresh-dev-inputs-003/cases.jsonl"
    plans = {
        "heldvalidation32": root / "held-sft-001/PLAN.json",
        "fourhop64": root / "transfer-musique-fourhop-sft-001/PLAN.json",
        "hotpot32": root / "transfer-hotpot-sft-001/PLAN.json",
    }
    input_rows, hotpot_rows, fresh_rows = map(read_cases, (inputs, hotpot, fresh))
    train = [case for case in input_rows if case["split"] == "train"]
    train_documents, train_titles = defaultdict(set), defaultdict(set)
    for case in train:
        for document in case["documents"]:
            train_documents[document_key(document)].add(document_role(case, document))
            train_titles[document["title"]].add(document_role(case, document))
    input_lookup = {case["id"]: case for case in input_rows}
    hotpot_lookup = {case["id"]: case for case in hotpot_rows}
    panels = {
        "heldvalidation32": panel_from_plan(plans["heldvalidation32"], input_lookup),
        "fourhop64": panel_from_plan(plans["fourhop64"], input_lookup),
        "hotpot32": panel_from_plan(plans["hotpot32"], hotpot_lookup),
        "freshdev64": fresh_rows,
    }
    return {
        "method": {
            "document_identity": "literal tuple (title, text); document IDs deliberately ignored",
            "title_identity": "literal title; inclusive of exact title+text matches",
            "support_mapping": "host-only supporting_*_ids in frozen metadata; no answer text read",
            "interpretation": (
                "title-only overlap is not evidence of identical text or fact exposure"
            ),
        },
        "sources_sha256": {
            "audit_document_exposure.py": sha256(Path(__file__)),
            "inputs-001/cases.jsonl": sha256(inputs),
            "hotpot-inputs-001/cases.jsonl": sha256(hotpot),
            "fresh-dev-inputs-003/cases.jsonl": sha256(fresh),
            **{str(path.relative_to(root)): sha256(path) for path in plans.values()},
        },
        "train": {
            "parents": len(train),
            "unique_documents": len(train_documents),
            "unique_titles": len(train_titles),
        },
        "panels": {
            name: collect_panel(cases, train_documents, train_titles)
            for name, cases in panels.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        help="Write JSON once; fails rather than replacing an existing report.",
    )
    args = parser.parse_args()
    result = audit(args.root)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        with args.report.open("x") as handle:
            handle.write(serialized)
    print(serialized, end="")
