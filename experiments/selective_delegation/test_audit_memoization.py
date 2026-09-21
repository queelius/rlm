"""Exact requests, not matching seeds alone, define eligible replay groups."""

import importlib.util
from pathlib import Path


def test_request_grouping_excludes_roots_and_detects_output_disagreements():
    path = Path(__file__).with_name("audit_memoization.py")
    assert path.exists(), "memoization audit missing"
    spec = importlib.util.spec_from_file_location("memo_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for index, prompt, role, answer in (
        (0, "q", "helper", "a"),
        (1, "q", "helper", "a"),
        (2, "other", "helper", "a"),
        (3, "q", "root", "a"),
        (4, "f", "final", "a"),
        (5, "f", "final", "b"),
    ):
        records.append(
            {
                "call_id": str(index),
                "role": role,
                "available": True,
                "request": {
                    "prompt": prompt,
                    "input_token_ids": [1],
                    "role": role,
                    "model": "base",
                    "adapter_enabled": role == "root",
                    "adapter_sha256": "adapter" if role == "root" else None,
                    "seed": 7,
                },
                "text": answer,
                "output_token_ids": [ord(answer)],
                "finish_reason": "eos",
                "started": index,
                "ended": index + 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
    groups, avoidable = module.duplicate_groups(records)
    assert len(groups) == 2
    assert [r["call_id"] for r in avoidable] == ["1"]
    disagreement = next(g for g in groups if g["role"] == "final")
    assert disagreement["disagreement_fields"] == ["text", "output_token_ids"]
    assert disagreement["safely_replayable_observed"] is False
