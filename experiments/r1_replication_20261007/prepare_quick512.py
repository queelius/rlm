"""CPU-only, immutable first-eligible subset; invoke with the pinned R1 Python."""

import argparse
import ast
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def content_digest(dataset):
    return hashlib.sha256(json.dumps(dataset.to_dict(), sort_keys=True).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_template(source):
    """Load only the official pure template function, avoiding training imports."""
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_qwen_math_template"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


def first_eligible(dataset, tokenizer, template, count=512, max_length=1024):
    """Retain source order and the native inclusive default-tokenizer predicate."""
    selected, lengths, rejected = [], [], []
    for index, row in enumerate(dataset):
        length = len(tokenizer(template(row["problem"]))["input_ids"])
        if length <= max_length:
            selected.append(index)
            lengths.append(length)
            if len(selected) == count:
                return selected, lengths, rejected
        else:
            rejected.append(index)
    raise ValueError(f"Only {len(selected)} eligible rows; need {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("/project/alex_phd/runs/r1-zero-replication-20261007")
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path("/project/alex_phd/research-cache/repos/understand-r1-zero"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/project/alex_phd/research-cache/models/"
            "Qwen--Qwen2.5-Math-1.5B--4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
        ),
    )
    args = parser.parse_args()
    from datasets import Dataset, DatasetDict, load_from_disk
    from oat.utils.data import PromptDataset, get_tokenizer

    run = args.run_root
    source_path = run / "data-oat/train4096-lvl3to5-chat-drgrpo42"
    output = run / "data-oat/quick512-lvl3to5-v1"
    manifest_path = run / "quick512-data-manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Immutable destination already exists; refusing overwrite")
    source_manifest_path = run / "larger4096-data-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source = load_from_disk(str(source_path))["train"]
    assert len(source) == 4096
    assert content_digest(source) == source_manifest["content_sha256"]
    for relative, expected in source_manifest["files"].items():
        assert file_digest(source_path / relative) == expected, relative
    source_order = list(range(source_manifest["source_rows"]))
    random.Random(42).shuffle(source_order)
    assert source["source_row_index"] == source_order[:4096]

    tokenizer = get_tokenizer(str(args.model))
    template_source = args.source_repo / "train_zero_math.py"
    template = native_template(template_source)
    selected, lengths, rejected = first_eligible(source, tokenizer, template)
    subset = source.select(selected)
    assert all(level in (3, 4, 5) for level in subset["level"])
    assert subset["problem"] == subset["question"]

    templated = Dataset.from_dict(
        {
            "problem": [template(question) for question in subset["problem"]],
            "answer": subset["answer"],
        }
    )
    strategy = SimpleNamespace(
        args=SimpleNamespace(prompt_max_length=1024), is_rank_0=lambda: False
    )
    native = PromptDataset(
        templated,
        tokenizer,
        strategy,
        input_key="problem",
        output_key="answer",
        apply_chat_template=False,
        get_reference=True,
    )
    assert len(native) == 512
    assert native.processed_prompts == templated["problem"]
    assert native.references == subset["answer"]

    math500 = load_from_disk(str(args.source_repo / "datasets/evaluation_suite/math"))
    monitor = load_from_disk(str(run / "data-oat/monitor64"))["math"]
    heldout = load_from_disk(str(run / "data-oat/heldout128"))["math"]
    assert len(math500) == 500 and len(monitor) == 64 and len(heldout) == 128
    eval_order = list(range(500))
    random.Random(142).shuffle(eval_order)
    assert monitor["source_row_index"] == eval_order[:64]
    assert heldout["source_row_index"] == eval_order[64:192]
    for split in (monitor, heldout):
        for row in split:
            official = math500[row["source_row_index"]]
            assert row["problem"] == official["problem"]
            assert row["answer"] == official["answer"]

    def normalized(questions):
        return {" ".join(question.split()) for question in questions}

    questions = normalized(subset["problem"])
    overlap = {
        "training_vs_math500": len(questions & normalized(math500["problem"])),
        "training_vs_heldout128": len(questions & normalized(heldout["problem"])),
        "monitor64_vs_heldout128": len(
            normalized(monitor["problem"]) & normalized(heldout["problem"])
        ),
    }
    assert not any(overlap.values()), overlap
    assert set(monitor["source_row_index"]).isdisjoint(heldout["source_row_index"])

    # Materialize a fresh Arrow dataset, rather than copying transformed cache files.
    DatasetDict(train=Dataset.from_dict(subset.to_dict(), features=subset.features)).save_to_disk(
        str(output)
    )
    saved = load_from_disk(str(output))["train"]
    assert content_digest(saved) == content_digest(subset)
    saved_files = {
        str(path.relative_to(output)): file_digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    assert not any("cache-" in name for name in saved_files)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": source_manifest["source_repo"],
        "source_commit": source_manifest["source_commit"],
        "source_dataset": source_manifest["source_dataset"],
        "source_path": str(source_path),
        "source_rows": len(source),
        "source_content_sha256": content_digest(source),
        "source_manifest_sha256": file_digest(source_manifest_path),
        "source_existing_order": "random.Random(42).shuffle(range(8523)); first4096",
        "selection": "First512 eligible rows in existing source order; no rewards/correctness used",
        "source_subset_row_indices": selected,
        "source_row_indices": subset["source_row_index"],
        "rejected_prefix_source_subset_row_indices": rejected,
        "path": str(output),
        "n": len(saved),
        "native_predicate": "len(tokenizer(apply_qwen_math_template(problem))['input_ids']) <=1024",
        "tokenizer_path": str(args.model),
        "tokenizer_revision": "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
        "tokenizer_files": {
            name: file_digest(args.model / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        "template_source_sha256": file_digest(template_source),
        "preparation_script_sha256": file_digest(Path(__file__)),
        "prompt_token_lengths": lengths,
        "content_digest_method": "sha256(json.dumps(dataset.to_dict(),sort_keys=True).encode())",
        "content_sha256": content_digest(saved),
        "files": saved_files,
        "checks": {
            "native_PromptDataset_used": len(native),
            "rollout_batch_size": 16,
            "drop_last_batches": len(native) // 16,
            "drop_last_discarded": len(native) % 16,
            "all_levels_in_3_4_5": True,
            "no_transformed_caches_copied": True,
            "exact_whitespace_normalized_overlap": overlap,
            "heldout128_rows": 128,
            "monitor64_rows": 64,
            "heldout128_source_order": "random.Random(142) permutation of500, indices64:192",
            "monitor64_source_order": "random.Random(142) permutation of500, indices0:64",
            "heldout128_exact_source_questions_and_answers": True,
            "pretraining_or_paraphrase_contamination_assessed": False,
        },
        "evaluation_content_sha256": {
            "monitor64": content_digest(monitor),
            "heldout128": content_digest(heldout),
        },
        "python": sys.version,
        "license": source_manifest["license"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "path": str(output),
                "manifest": str(manifest_path),
                "n": len(saved),
                "native_used": len(native),
                "batches": len(native) // 16,
                "rejected_prefix": rejected,
                "content_sha256": manifest["content_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
