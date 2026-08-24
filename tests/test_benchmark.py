"""Durability and paired-semantics tests for the optional Oolong runner."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

import benchmarks.oolong as oolong
from benchmarks.oolong import (
    BenchmarkBudget,
    BenchmarkCall,
    BenchmarkComparison,
    BenchmarkCondition,
    BenchmarkExample,
    BenchmarkFailure,
    BenchmarkFailureKind,
    BenchmarkTarget,
    ConditioningMode,
    PromptProfile,
    benchmark_targets,
    build_parser,
    instructions,
    paired_requests,
    run_comparisons,
    run_pair,
    summarize,
    typed_oolong_reference,
    verifier_provenance,
)
from tests.fakes import responses_text


def comparison() -> BenchmarkComparison:
    return BenchmarkComparison.from_options(
        example=BenchmarkExample.from_context(
            model="qwen",
            context="same context",
            task="same task",
            expected_count=1,
            source_id="source-1",
        ),
        mode=ConditioningMode.ORACLE,
        rlm_harness_fingerprint="a" * 64,
        seed=42,
        repetition=3,
        sampling={"temperature": 0},
        budget=BenchmarkBudget(600, 64, 20_000),
    )


def successful_call(value: BenchmarkComparison) -> BenchmarkCall:
    return BenchmarkCall(
        200,
        responses_text("answer", model="qwen", input_tokens=3, output_tokens=2),
        {
            "x-rlm-run-id": "run-test",
            "x-rlm-model-calls": "1",
            "x-rlm-input-tokens": "3",
            "x-rlm-output-tokens": "2",
            "x-rlm-usage-unreported-calls": "0",
            "x-rlm-controller-model": value.example.model,
            "x-rlm-controller-options-sha256": value.model_options_sha256(),
            "x-rlm-harness-fingerprint": value.rlm_harness_fingerprint,
        },
        None,
    )


def test_empty_selection_and_failure_accounting_are_explicit() -> None:
    with pytest.raises(ValueError, match="selected no examples"):
        summarize([])
    summary = summarize([{"score": None, "failure": {"kind": "timeout"}}])
    assert {
        key: summary[key]
        for key in ("examples", "scored", "failures", "mean_score", "failure_rate")
    } == {
        "examples": 1,
        "scored": 0,
        "failures": 1,
        "mean_score": None,
        "failure_rate": 1.0,
    }


def test_checkpoint_resumes_only_missing_arm_and_rejects_drift(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint.json"
    value = comparison()
    targets = benchmark_targets("http://rlm", "http://direct")
    calls: list[BenchmarkCondition] = []

    def interrupted(target: BenchmarkTarget, request: dict[str, Any]) -> BenchmarkCall:
        calls.append(target.condition)
        if target.condition is BenchmarkCondition.DIRECT:
            raise RuntimeError("interruption")
        return successful_call(value)

    with pytest.raises(RuntimeError, match="interruption"):
        run_comparisons([value], targets=targets, output=output, call=interrupted)
    assert calls == [BenchmarkCondition.RLM, BenchmarkCondition.DIRECT]
    assert len(json.loads(output.read_text())["results"]) == 1
    calls.clear()
    run_comparisons(
        [value],
        targets=targets,
        output=output,
        call=lambda target, request: calls.append(target.condition) or successful_call(value),
    )
    assert calls == [BenchmarkCondition.DIRECT]
    with pytest.raises(ValueError, match="provenance"):
        run_comparisons(
            [replace(value, seed=43)],
            targets=targets,
            output=output,
            call=lambda target, request: successful_call(value),
        )


def test_resume_rejects_tampered_prompt_text(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint.json"
    value = comparison()
    run_comparisons(
        [value],
        targets=benchmark_targets("http://rlm", "http://direct"),
        output=output,
        call=lambda target, request: successful_call(value),
    )
    checkpoint = json.loads(output.read_text())
    checkpoint["results"][0]["prompt_text"] += " tampered"
    output.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="checkpoint digest"):
        run_comparisons(
            [value],
            targets=benchmark_targets("http://rlm", "http://direct"),
            output=output,
            call=lambda target, request: successful_call(value),
        )


def test_conditioning_pairing_and_attestation_are_separate() -> None:
    assert "ask_batch" not in instructions("task", 1, mode=ConditioningMode.MINIMAL)
    assert "ask_batch" in instructions("task", 1, mode=ConditioningMode.ORACLE)
    value = comparison()
    rlm_request, direct_request = paired_requests(value)
    assert rlm_request is not direct_request
    assert rlm_request["input"] == direct_request["input"] == "same context"
    assert "ask_batch" in rlm_request["instructions"]
    assert "ask_batch" not in direct_request["instructions"]
    records = run_pair(
        benchmark_targets("http://rlm", "http://direct"),
        comparison=value,
        call=lambda target, request: successful_call(value),
    )
    assert [record["prompt_profile"] for record in records] == [
        PromptProfile.RLM_ORACLE_V1.value,
        PromptProfile.DIRECT_V1.value,
    ]
    bad = run_pair(
        benchmark_targets("http://rlm", "http://direct"),
        comparison=value,
        call=lambda target, request: (
            replace(
                successful_call(value),
                headers={
                    **successful_call(value).headers,
                    "x-rlm-controller-options-sha256": "0" * 64,
                },
            )
            if target.condition is BenchmarkCondition.RLM
            else successful_call(value)
        ),
    )
    assert bad[0]["failure"]["kind"] == BenchmarkFailureKind.PROVENANCE.value


def test_pair_arms_have_distinct_canonical_episode_ids_and_nullable_failed_rlm_run_id() -> None:
    value = comparison()
    records = run_pair(
        benchmark_targets("http://rlm", "http://direct"),
        comparison=value,
        call=lambda target, request: (
            BenchmarkCall(
                None,
                None,
                {},
                BenchmarkFailure(BenchmarkFailureKind.TIMEOUT, "TimeoutError", "timed out"),
            )
            if target.condition is BenchmarkCondition.RLM
            else successful_call(value)
        ),
    )

    rlm, direct = records
    assert rlm["pair_id"] == direct["pair_id"] == value.pair_id()
    assert rlm["episode_id"] != direct["episode_id"]
    assert rlm["episode_id"] == oolong.episode_id(value.pair_id(), BenchmarkCondition.RLM)
    assert direct["episode_id"] == oolong.episode_id(value.pair_id(), BenchmarkCondition.DIRECT)
    assert rlm["run_id"] is None
    assert direct["run_id"] is None

    successful_rlm = run_pair(
        benchmark_targets("http://rlm", "http://direct"),
        comparison=value,
        call=lambda target, request: successful_call(value),
    )[0]
    assert successful_rlm["run_id"] == "run-test"


@pytest.mark.parametrize(
    "field", ["model", "input", "instructions", "stream", "background", "seed"]
)
def test_sampling_cannot_override_protocol_fields(field: str) -> None:
    value = comparison()
    with pytest.raises(ValueError, match="sampling"):
        BenchmarkComparison.from_options(
            example=value.example,
            mode=value.mode,
            rlm_harness_fingerprint=value.rlm_harness_fingerprint,
            seed=value.seed,
            repetition=value.repetition,
            sampling={field: True},
            budget=value.budget,
        )


def test_pair_identity_binds_prompt_text_and_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    value = comparison()
    assert replace(value, seed=99).pair_id() != value.pair_id()
    original = oolong.direct_instructions
    monkeypatch.setattr(
        oolong, "direct_instructions", lambda task, count: original(task, count) + " drift"
    )
    assert comparison().pair_id() != value.pair_id()


def test_required_baseline_and_harness_flags() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--url", "http://rlm", "--model", "qwen", "--rlm-harness-fingerprint", "a" * 64]
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--url", "http://rlm", "--direct-url", "http://direct", "--model", "qwen"]
        )


def test_failure_is_structured() -> None:
    assert BenchmarkFailure(
        BenchmarkFailureKind.TIMEOUT, "TimeoutError", "timed out"
    ).to_dict() == {"kind": "timeout", "exception_type": "TimeoutError", "message": "timed out"}


def test_dataclass_verifier_scores_are_durable_strict_json(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Score:
        score: float
        exact: bool

    value = comparison()
    output = tmp_path / "checkpoint.json"
    run_comparisons(
        [value],
        targets=benchmark_targets("http://rlm", "http://direct"),
        output=output,
        call=lambda target, request: successful_call(value),
        score=lambda current, answer: Score(1.0, True),
    )
    assert json.loads(output.read_text())["results"][0]["score"] == {
        "exact": True,
        "score": 1.0,
    }


def test_reversed_targets_are_normalized_and_invalid_target_sets_do_not_call() -> None:
    value = comparison()
    normal = benchmark_targets("http://rlm", "http://direct")
    seen: list[tuple[BenchmarkCondition, str, str]] = []

    def call(target: BenchmarkTarget, request: dict[str, Any]) -> BenchmarkCall:
        seen.append((target.condition, target.url, request["instructions"]))
        return successful_call(value)

    run_comparisons([value], targets=tuple(reversed(normal)), output=None, call=call)
    assert seen[0][0:2] == (BenchmarkCondition.RLM, "http://rlm")
    assert "ask_batch" in seen[0][2]
    assert seen[1][0:2] == (BenchmarkCondition.DIRECT, "http://direct")
    assert "ask_batch" not in seen[1][2]

    for invalid in ((normal[0], normal[0]), (normal[0],), (*normal, normal[0])):
        calls: list[BenchmarkTarget] = []
        with pytest.raises(ValueError, match="targets"):
            run_comparisons(
                [value],
                targets=invalid,
                output=None,
                call=lambda target, request, observed=calls: (
                    observed.append(target) or successful_call(value)
                ),
            )
        assert calls == []


def test_checkpoint_digest_rejects_deletion_reordering_and_schema_tampering(tmp_path: Path) -> None:
    value = comparison()
    output = tmp_path / "checkpoint.json"
    targets = benchmark_targets("http://rlm", "http://direct")
    run_comparisons(
        [value], targets=targets, output=output, call=lambda target, request: successful_call(value)
    )
    original = json.loads(output.read_text())
    for mutation in (
        lambda payload: payload["results"].pop(),
        lambda payload: payload.__setitem__("results", list(reversed(payload["results"]))),
        lambda payload: payload.__setitem__("schema_version", 999),
    ):
        payload = copy.deepcopy(original)
        mutation(payload)
        output.write_text(json.dumps(payload))
        calls: list[BenchmarkTarget] = []
        with pytest.raises(ValueError, match="checkpoint"):
            run_comparisons(
                [value],
                targets=targets,
                output=output,
                call=lambda target, request, observed=calls: (
                    observed.append(target) or successful_call(value)
                ),
            )
        assert calls == []


def test_summary_keeps_conditions_and_paired_deltas_separate() -> None:
    records = [
        {"pair_id": "one", "condition": "rlm", "score": 0.9, "failure": None},
        {"pair_id": "one", "condition": "direct", "score": 0.4, "failure": None},
        {"pair_id": "two", "condition": "rlm", "score": None, "failure": {"kind": "timeout"}},
        {"pair_id": "two", "condition": "direct", "score": 0.2, "failure": None},
    ]
    summary = summarize(records)
    assert summary["conditions"]["rlm"]["failures"] == 1
    assert summary["conditions"]["direct"]["mean_score"] == 0.3
    assert summary["pairs"] == {
        "total": 2,
        "valid": 1,
        "unpaired": 0,
        "failing": 1,
        "mean_rlm_minus_direct": 0.5,
    }


@pytest.mark.parametrize(
    "changed",
    [
        lambda value: replace(value, verifier_sha256="different"),
        lambda value: replace(
            value, example=replace(value.example, reference_json='["different"]')
        ),
        lambda value: replace(value, example=replace(value.example, selection_json='{"seed":99}')),
        lambda value: replace(value, example=replace(value.example, dataset_revision="different")),
    ],
)
def test_provenance_drift_rejects_resume_before_calls(tmp_path: Path, changed: Any) -> None:
    value = comparison()
    output = tmp_path / "checkpoint.json"
    targets = benchmark_targets("http://rlm", "http://direct")
    run_comparisons(
        [value], targets=targets, output=output, call=lambda target, request: successful_call(value)
    )
    calls: list[BenchmarkTarget] = []
    with pytest.raises(ValueError, match="provenance"):
        run_comparisons(
            [changed(value)],
            targets=targets,
            output=output,
            call=lambda target, request: calls.append(target) or successful_call(value),
        )
    assert calls == []


def test_typed_oolong_reference_binds_gold_raw_and_canonical_answer_type() -> None:
    class AnswerType(str, Enum):
        ENTITY = "entity"

    @dataclass(frozen=True)
    class OolongExample:
        gold_raw: dict[str, Any]
        answer_type: AnswerType

    assert typed_oolong_reference(
        OolongExample(gold_raw={"label": "ENTY"}, answer_type=AnswerType.ENTITY)
    ) == {"gold_raw": {"label": "ENTY"}, "answer_type": "entity"}
    with pytest.raises(ValueError, match="gold_raw.*answer_type"):
        typed_oolong_reference(object())


def test_verifier_module_bytes_and_identity_drift_reject_resume(tmp_path: Path) -> None:
    module_path = tmp_path / "verifier.py"
    module_path.write_text("def score_output(): return 1\ndef helper(): return 1\n")
    first = verifier_provenance("harness-rl", "1.2.3", module_path)
    module_path.write_text("def score_output(): return 1\ndef helper(): return 2\n")
    second = verifier_provenance("harness-rl", "1.2.3", module_path)
    assert first["sha256"] != second["sha256"]
    value = comparison()
    output = tmp_path / "checkpoint.json"
    targets = benchmark_targets("http://rlm", "http://direct")
    run_comparisons(
        [value], targets=targets, output=output, call=lambda target, request: successful_call(value)
    )
    calls: list[BenchmarkTarget] = []
    changed = replace(
        value, verifier_sha256=second["sha256"], verifier_source_revision=second["source_revision"]
    )
    with pytest.raises(ValueError, match="provenance"):
        run_comparisons(
            [changed],
            targets=targets,
            output=output,
            call=lambda target, request: calls.append(target) or successful_call(value),
        )
    assert calls == []


@pytest.mark.parametrize(
    "name",
    [
        "--repetitions",
        "--max-model-calls",
        "--max-total-tokens",
        "--context-length",
        "--num-examples",
    ],
)
def test_benchmark_positive_integer_flags_reject_zero_and_negative(name: str) -> None:
    parser = build_parser()
    base = [
        "--url",
        "http://rlm",
        "--direct-url",
        "http://direct",
        "--model",
        "qwen",
        "--rlm-harness-fingerprint",
        "a" * 64,
        "--timeout",
        "1",
        "--repetitions",
        "1",
        "--max-model-calls",
        "1",
        "--max-total-tokens",
        "1",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args([*base, name, "0"])
