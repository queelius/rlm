"""Immutable train-only helper SFT rows; no training or model loading."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
from collections import Counter
from pathlib import Path

import eval_planner as evaluation

SEED = 2026092111
BASE = evaluation.planner.BASE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_examples(cases, *, expected_parents=256, expected_steps=570):
    training = sorted((c for c in cases if c["split"] == "train"), key=lambda c: c["id"])
    if len(training) != expected_parents or len({c["id"] for c in training}) != len(training):
        raise ValueError("training parent count/identity differs")
    random.Random(SEED).shuffle(training)
    result = []
    for case in training:
        previous = []
        steps = case["metadata"]["question_decomposition"]
        if len(steps) not in (2, 3):
            raise ValueError("expected two/three-step train annotations")
        for index, step in enumerate(steps):
            question, answer = step["question"], step["answer"]
            if not all(isinstance(s, str) and s.strip() for s in (question, answer)):
                raise ValueError("nonempty annotated question/answer strings required")
            resolved = evaluation.bind_question(question, previous)
            result.append(
                {
                    "id": f"{case['id']}-step{index + 1:02d}",
                    "parent_id": case["id"],
                    "step_index": index,
                    "split": "train",
                    "prompt": evaluation.isolated_helper_prompt(case, resolved),
                    "target": json.dumps(
                        {"answer": answer}, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
            previous.append(answer)
    if len(result) != expected_steps:
        raise ValueError("training step count differs")
    return result


def token_audit(examples, tokenizer):
    if type(tokenizer.eos_token_id) is not int:
        raise ValueError("one native EOS required")
    rows = []
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        target = tokenizer.encode(example["target"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        if len(prompt) + len(target) > 6144 or len(target) > 48:
            raise ValueError("helper example exceeds total6144/target48; never truncate")
        rows.append(
            {
                "id": example["id"],
                "prompt_tokens": len(prompt),
                "target_tokens_including_eos": len(target),
                "total_tokens": len(prompt) + len(target),
            }
        )
    return {
        "rows": rows,
        "examples": len(rows),
        "truncation_used": False,
        "eos_token_id": tokenizer.eos_token_id,
        "budgets": {"total": 6144, "target_including_eos": 48},
        "maximum_prompt_tokens": max(r["prompt_tokens"] for r in rows),
        "maximum_target_tokens_including_eos": max(r["target_tokens_including_eos"] for r in rows),
        "maximum_total_tokens": max(r["total_tokens"] for r in rows),
        "exceed_total_2048": sum(r["total_tokens"] > 2048 for r in rows),
        "exceed_total_6144": 0,
        "exceed_target_48": 0,
    }


def prepare(cases_path, output, *, tokenizer=None, expected_parents=256, expected_steps=570):
    cases_path, output = Path(cases_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("test_prepare_helper_training.py"),
        Path(evaluation.__file__),
        Path(evaluation.planner.__file__),
        Path(evaluation.probe.__file__),
    ]
    snapshots = {p.resolve(): p.read_bytes() for p in source_paths}
    raw = cases_path.read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    examples = build_examples(
        cases, expected_parents=expected_parents, expected_steps=expected_steps
    )
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            BASE, local_files_only=True, trust_remote_code=False
        )
    audit = token_audit(examples, tokenizer)
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in examples
    ).encode()
    if any(p.read_bytes() != contents for p, contents in snapshots.items()):
        raise ValueError("preparation source changed during tokenization; do not seal mixed source")
    manifest = {
        "schema": "selective-delegation-helper-sft-inputs-v1",
        "role": "helper",
        "cases_path": str(cases_path),
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "model": str(BASE),
        "model_manifest_sha256": sha(BASE / "local-research-manifest.json"),
        "examples": len(examples),
        "examples_count": len(examples),
        "training_parents": len({row["parent_id"] for row in examples}),
        "examples_sha256": hashlib.sha256(serialized).hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256(contents).hexdigest() for p, contents in snapshots.items()
        },
        "source_split_counts": dict(Counter(case["split"] for case in cases)),
        "teacher_binding": "earlier annotated train answers only",
        "question_transformation": "No relation-shorthand rewrite; raw question except #k binding",
        "supervision": "Privileged annotated train questions/answers; no validation or transfer "
        "targets. Inference binds only prior predicted helper answers, creating exposure shift.",
        "prompt_contract": "Exact isolated_helper_prompt: supplied question and all public docs; "
        "no original composed question, support fields, future answers, or final-gold metadata.",
        "order_seed": SEED,
        "order_rule": "Sorted train parent IDs, seeded parent shuffle, "
        "then annotation step order; trainer records any epoch-level reshuffle separately.",
        "ordered_ids": [row["id"] for row in examples],
        "duplicate_prompt_target_pairs": len(examples)
        - len({(r["prompt"], r["target"]) for r in examples}),
        "training_performed": False,
        "token_audit": audit,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "files_sha256": {
                name: sha(BASE / name)
                for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
                if (BASE / name).exists()
            },
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "transformers": importlib.metadata.version("transformers"),
        },
        "artifact_sha256": {
            "examples.jsonl": hashlib.sha256(serialized).hexdigest(),
            **{
                "source/" + p.name: hashlib.sha256(contents).hexdigest()
                for p, contents in snapshots.items()
            },
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    with (output / "examples.jsonl").open("xb") as handle:
        handle.write(serialized)
    for path, contents in snapshots.items():
        with (output / "source" / path.name).open("xb") as handle:
            handle.write(contents)
    # Last-written manifest marks the fully prepared, immutable input package.
    with (output / "MANIFEST.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.cases, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "examples",
                    "training_parents",
                    "examples_sha256",
                    "duplicate_prompt_target_pairs",
                )
            }
            | {
                "token_audit": {
                    key: value for key, value in result["token_audit"].items() if key != "rows"
                }
            },
            indent=2,
        )
    )
