"""Durable paired RLM/direct Oolong stress testing.

Dataset loading and verification are optional runtime dependencies; the data
types, provenance identities, and checkpoint machinery remain pure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import math
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from rlm.config import validate_controller_options
from rlm.json import StrictJSONError, strict_json_dumps, strict_json_loads, strict_json_sha256
from rlm.response import validate_terminal_response

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "harness-rl/data/benchmarks/oolong_trec_coarse-eval.parquet"
)
TAXONOMY = (
    "Choose one exact TREC coarse answer-type label: abbreviation, entity, human being, "
    "numeric value, location, description and abstract concept. Return only the label."
)
RUNNER_SCHEMA_VERSION = 1


class ConditioningMode(str, Enum):
    MINIMAL = "minimal"
    ORACLE = "oracle"


class BenchmarkCondition(str, Enum):
    RLM = "rlm"
    DIRECT = "direct"


class PromptProfile(str, Enum):
    RLM_MINIMAL_V1 = "rlm_minimal_v1"
    RLM_ORACLE_V1 = "rlm_oracle_v1"
    DIRECT_V1 = "direct_v1"


class BenchmarkFailureKind(str, Enum):
    HTTP = "http"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    DECODE = "decode"
    PROTOCOL = "protocol"
    PROVENANCE = "provenance"
    BUDGET = "budget"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def episode_id(pair_id: str, condition: BenchmarkCondition) -> str:
    """Derive one stable rollout identity without collapsing paired arms."""

    if not isinstance(pair_id, str) or len(pair_id) != 64:
        raise ValueError("pair_id must be a SHA-256 digest")
    if not isinstance(condition, BenchmarkCondition):
        raise TypeError("condition must be a BenchmarkCondition")
    return strict_json_sha256({"pair_id": pair_id, "condition": condition.value})


def typed_oolong_reference(selected_example: Any) -> dict[str, Any]:
    """Read the established Oolong row schema without ambiguous field probing."""

    try:
        gold_raw = selected_example.gold_raw
        answer_type = selected_example.answer_type
    except AttributeError as exc:
        raise ValueError("Oolong example must expose gold_raw and answer_type") from exc
    if hasattr(answer_type, "value"):
        answer_type = answer_type.value
    if not isinstance(answer_type, str) or not answer_type:
        raise ValueError("Oolong answer_type must be a non-empty string or string enum")
    try:
        gold = strict_json_loads(strict_json_dumps(gold_raw, sort_keys=True, separators=(",", ":")))
    except StrictJSONError as exc:
        raise ValueError("Oolong gold_raw must be strict JSON") from exc
    return {"gold_raw": gold, "answer_type": answer_type}


def verifier_provenance(package_name: str, version: str, module_path: Path) -> dict[str, str]:
    """Fingerprint the entire verifier module, including every parsing helper."""

    if (
        not isinstance(package_name, str)
        or not package_name
        or not isinstance(version, str)
        or not version
    ):
        raise ValueError("verifier package name and version must be identified")
    if not isinstance(module_path, Path) or not module_path.is_file():
        raise ValueError("verifier module source path must be identified and readable")
    try:
        source = module_path.read_bytes()
    except OSError as exc:
        raise ValueError("verifier module source path must be readable") from exc
    if not source:
        raise ValueError("verifier module source cannot be empty")
    digest = hashlib.sha256(source).hexdigest()
    return {
        "name": package_name,
        "version": version,
        "sha256": digest,
        "source_revision": digest,
    }


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_finite(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite value")


def _fingerprint(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("rlm_harness_fingerprint must be a lowercase 64-character SHA-256 digest")


def instructions(task: str, expected_count: int, *, mode: ConditioningMode) -> str:
    common = f"""Answer this aggregate question exactly:
{task}

The input contains {expected_count} questions. {TAXONOMY}
Return exactly {expected_count} canonical labels in the requested final answer format.
Counts must sum to {expected_count}."""
    if mode is ConditioningMode.MINIMAL:
        return common
    if mode is not ConditioningMode.ORACLE:
        raise ValueError(f"unsupported conditioning mode: {mode!r}")
    return (
        common
        + """

This is an explicitly labelled ORACLE scaffold stress condition. In a Python cell, derive prompts
from the input and call:
batch = ask_batch(prompts, options={"temperature": 0})
labels = list(batch.texts)
if batch.failed_indexes:
    retry = ask_batch(
        [prompts[index] for index in batch.failed_indexes], options={"temperature": 0}
    )
    replacements = retry.require_texts()
    for index, text in zip(batch.failed_indexes, replacements):
        labels[index] = text
assert all(isinstance(label, str) and label.strip() for label in labels)
Compute and validate the aggregate answer before submission."""
    )


def direct_instructions(task: str, expected_count: int) -> str:
    """Direct prompt intentionally contains no ABI, extraction, or retry guidance."""

    return f"""Answer this aggregate question exactly:
{task}

The input contains {expected_count} questions. {TAXONOMY}
Return exactly {expected_count} canonical labels in the requested final answer format.
Counts must sum to {expected_count}."""


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    model: str
    context: str
    task: str
    expected_count: int
    source_id: str
    context_sha256: str
    reference_json: str = "null"
    dataset_path: str = "unspecified"
    dataset_sha256: str = "unspecified"
    dataset_revision: str = "unspecified"
    selection_json: str = "{}"
    source_revision: str = "unspecified"

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, str) and item
            for item in (self.model, self.context, self.task, self.source_id)
        ):
            raise ValueError("example fields must be non-empty strings")
        _positive_int(self.expected_count, "expected_count")
        if self.context_sha256 != _sha256(self.context):
            raise ValueError("context_sha256 does not match context")
        for name in ("dataset_path", "dataset_revision", "source_revision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        for name in ("reference_json", "selection_json"):
            try:
                strict_json_loads(getattr(self, name))
            except StrictJSONError as exc:
                raise ValueError(f"{name} must be strict JSON") from exc

    @classmethod
    def from_context(
        cls,
        *,
        model: str,
        context: str,
        task: str,
        expected_count: int,
        source_id: str,
        reference: Any = None,
        dataset_path: str = "unspecified",
        dataset_sha256: str = "unspecified",
        dataset_revision: str = "unspecified",
        selection: dict[str, Any] | None = None,
        source_revision: str = "unspecified",
    ) -> BenchmarkExample:
        return cls(
            model,
            context,
            task,
            expected_count,
            source_id,
            _sha256(context),
            strict_json_dumps(reference, sort_keys=True, separators=(",", ":")),
            dataset_path,
            dataset_sha256,
            dataset_revision,
            strict_json_dumps(selection or {}, sort_keys=True, separators=(",", ":")),
            source_revision,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkBudget:
    timeout_seconds: float
    max_model_calls: int
    max_total_tokens: int | None

    def __post_init__(self) -> None:
        _positive_finite(self.timeout_seconds, "timeout_seconds")
        _positive_int(self.max_model_calls, "max_model_calls")
        if self.max_total_tokens is not None:
            _positive_int(self.max_total_tokens, "max_total_tokens")


@dataclass(frozen=True, slots=True)
class BenchmarkPrompt:
    profile: PromptProfile
    text: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile, PromptProfile) or not isinstance(self.text, str):
            raise TypeError("benchmark prompt needs a profile and text")
        object.__setattr__(self, "sha256", _sha256(self.text))


@dataclass(frozen=True, slots=True)
class BenchmarkPrompts:
    rlm: BenchmarkPrompt
    direct: BenchmarkPrompt


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    example: BenchmarkExample
    mode: ConditioningMode
    rlm_harness_fingerprint: str
    seed: int
    repetition: int
    sampling_json: str
    budget: BenchmarkBudget
    verifier_name: str = "unspecified"
    verifier_version: str = "unspecified"
    verifier_sha256: str = "unspecified"
    verifier_source_revision: str = "unspecified"
    prompts: BenchmarkPrompts = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.example, BenchmarkExample) or not isinstance(
            self.mode, ConditioningMode
        ):
            raise TypeError("comparison needs typed example and mode")
        _fingerprint(self.rlm_harness_fingerprint)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        _positive_int(self.repetition, "repetition")
        if not isinstance(self.budget, BenchmarkBudget):
            raise TypeError("budget must be a BenchmarkBudget")
        for name in (
            "verifier_name",
            "verifier_version",
            "verifier_sha256",
            "verifier_source_revision",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        try:
            sampling = strict_json_loads(self.sampling_json)
        except StrictJSONError as exc:
            raise ValueError("sampling_json must contain strict JSON") from exc
        if not isinstance(sampling, dict) or "seed" in sampling:
            raise ValueError("sampling must be an object without seed")
        try:
            validate_controller_options({**sampling, "seed": self.seed})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid sampling: {exc}") from exc
        object.__setattr__(
            self,
            "sampling_json",
            strict_json_dumps(sampling, sort_keys=True, separators=(",", ":")),
        )
        profile = (
            PromptProfile.RLM_MINIMAL_V1
            if self.mode is ConditioningMode.MINIMAL
            else PromptProfile.RLM_ORACLE_V1
        )
        object.__setattr__(
            self,
            "prompts",
            BenchmarkPrompts(
                BenchmarkPrompt(
                    profile,
                    instructions(self.example.task, self.example.expected_count, mode=self.mode),
                ),
                BenchmarkPrompt(
                    PromptProfile.DIRECT_V1,
                    direct_instructions(self.example.task, self.example.expected_count),
                ),
            ),
        )

    @classmethod
    def from_options(
        cls,
        *,
        example: BenchmarkExample,
        mode: ConditioningMode,
        rlm_harness_fingerprint: str,
        seed: int,
        repetition: int,
        sampling: dict[str, Any],
        budget: BenchmarkBudget,
        verifier_name: str = "unspecified",
        verifier_version: str = "unspecified",
        verifier_sha256: str = "unspecified",
        verifier_source_revision: str = "unspecified",
    ) -> BenchmarkComparison:
        if not isinstance(sampling, dict) or "seed" in sampling:
            raise ValueError("sampling must be an object without seed")
        try:
            owned = strict_json_loads(strict_json_dumps(copy.deepcopy(sampling), sort_keys=True))
            validate_controller_options({**owned, "seed": seed})
        except (StrictJSONError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid sampling: {exc}") from exc
        assert isinstance(owned, dict)
        return cls(
            example,
            mode,
            rlm_harness_fingerprint,
            seed,
            repetition,
            strict_json_dumps(owned, sort_keys=True, separators=(",", ":")),
            budget,
            verifier_name,
            verifier_version,
            verifier_sha256,
            verifier_source_revision,
        )

    def sampling(self) -> dict[str, Any]:
        value = strict_json_loads(self.sampling_json)
        assert isinstance(value, dict)
        return value

    def model_options(self) -> dict[str, Any]:
        return {**self.sampling(), "seed": self.seed}

    def model_options_sha256(self) -> str:
        return strict_json_sha256(self.model_options())

    def canonical_json(self) -> str:
        return strict_json_dumps(_comparison_payload(self), sort_keys=True, separators=(",", ":"))

    def pair_id(self) -> str:
        return _sha256(self.canonical_json())


def _comparison_payload(value: BenchmarkComparison) -> dict[str, Any]:
    return {
        "example": {
            "model": value.example.model,
            "context": value.example.context,
            "context_sha256": value.example.context_sha256,
            "task": value.example.task,
            "expected_count": value.example.expected_count,
            "source_id": value.example.source_id,
            "reference": strict_json_loads(value.example.reference_json),
            "dataset_path": value.example.dataset_path,
            "dataset_sha256": value.example.dataset_sha256,
            "dataset_revision": value.example.dataset_revision,
            "selection": strict_json_loads(value.example.selection_json),
            "source_revision": value.example.source_revision,
        },
        "mode": value.mode.value,
        "rlm_harness_fingerprint": value.rlm_harness_fingerprint,
        "seed": value.seed,
        "repetition": value.repetition,
        "sampling": value.sampling(),
        "budget": _budget_payload(value.budget),
        "verifier": {
            "name": value.verifier_name,
            "version": value.verifier_version,
            "sha256": value.verifier_sha256,
            "source_revision": value.verifier_source_revision,
        },
        "prompts": {
            "rlm": {
                "profile": value.prompts.rlm.profile.value,
                "text": value.prompts.rlm.text,
                "sha256": value.prompts.rlm.sha256,
            },
            "direct": {
                "profile": value.prompts.direct.profile.value,
                "text": value.prompts.direct.text,
                "sha256": value.prompts.direct.sha256,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    condition: BenchmarkCondition
    url: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.condition, BenchmarkCondition)
            or not isinstance(self.url, str)
            or not self.url
        ):
            raise ValueError("target needs a condition and URL")


def benchmark_targets(rlm_url: str, direct_url: str) -> tuple[BenchmarkTarget, BenchmarkTarget]:
    return BenchmarkTarget(BenchmarkCondition.RLM, rlm_url), BenchmarkTarget(
        BenchmarkCondition.DIRECT, direct_url
    )


def normalize_targets(
    targets: Sequence[BenchmarkTarget],
) -> tuple[BenchmarkTarget, BenchmarkTarget]:
    """Validate target topology before deriving identities or initiating calls."""

    by_condition: dict[BenchmarkCondition, BenchmarkTarget] = {}
    for target in targets:
        if not isinstance(target, BenchmarkTarget) or target.condition in by_condition:
            raise ValueError("targets must contain exactly one RLM and one direct target")
        by_condition[target.condition] = target
    if set(by_condition) != {BenchmarkCondition.RLM, BenchmarkCondition.DIRECT}:
        raise ValueError("targets must contain exactly one RLM and one direct target")
    return by_condition[BenchmarkCondition.RLM], by_condition[BenchmarkCondition.DIRECT]


@dataclass(frozen=True, slots=True)
class BenchmarkFailure:
    kind: BenchmarkFailureKind
    exception_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCall:
    status: int | None
    response: dict[str, Any] | None
    headers: dict[str, str]
    failure: BenchmarkFailure | None

    def __post_init__(self) -> None:
        if self.status is not None and (
            isinstance(self.status, bool) or not isinstance(self.status, int)
        ):
            raise TypeError("status must be an integer or None")
        if self.response is not None:
            payload = strict_json_loads(strict_json_dumps(self.response, sort_keys=True))
            if not isinstance(payload, dict):
                raise TypeError("response must be a JSON object")
            object.__setattr__(self, "response", payload)
        object.__setattr__(
            self, "headers", {str(key).lower(): str(value) for key, value in self.headers.items()}
        )


def paired_requests(comparison: BenchmarkComparison) -> tuple[dict[str, Any], dict[str, Any]]:
    base = {
        "model": comparison.example.model,
        "input": comparison.example.context,
        **comparison.model_options(),
    }
    return {**copy.deepcopy(base), "instructions": comparison.prompts.rlm.text}, {
        **copy.deepcopy(base),
        "instructions": comparison.prompts.direct.text,
    }


def output_text(response: Mapping[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    return "".join(
        part["text"]
        for item in response.get("output", [])
        if isinstance(item, Mapping)
        and item.get("type") == "message"
        and isinstance(item.get("content"), list)
        for part in item["content"]
        if isinstance(part, Mapping) and isinstance(part.get("text"), str)
    )


def complete(url: str, body: dict[str, Any], *, timeout: float) -> BenchmarkCall:
    request = urllib.request.Request(
        url,
        data=strict_json_dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return BenchmarkCall(
                response.status,
                _response_object(response.read()),
                dict(response.headers.items()),
                None,
            )
    except urllib.error.HTTPError as exc:
        try:
            return BenchmarkCall(
                exc.code,
                _response_object(exc.read()),
                dict(exc.headers.items()) if exc.headers else {},
                BenchmarkFailure(BenchmarkFailureKind.HTTP, type(exc).__name__, str(exc)),
            )
        except (StrictJSONError, UnicodeError) as decode:
            return BenchmarkCall(
                exc.code,
                None,
                dict(exc.headers.items()) if exc.headers else {},
                BenchmarkFailure(BenchmarkFailureKind.DECODE, type(decode).__name__, str(decode)),
            )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        kind = (
            BenchmarkFailureKind.TIMEOUT
            if isinstance(exc, TimeoutError)
            else BenchmarkFailureKind.TRANSPORT
        )
        return BenchmarkCall(None, None, {}, BenchmarkFailure(kind, type(exc).__name__, str(exc)))
    except (StrictJSONError, UnicodeError) as exc:
        return BenchmarkCall(
            None,
            None,
            {},
            BenchmarkFailure(BenchmarkFailureKind.DECODE, type(exc).__name__, str(exc)),
        )


def _response_object(raw: bytes) -> dict[str, Any]:
    result = strict_json_loads(raw)
    if not isinstance(result, dict):
        raise StrictJSONError("response must be a JSON object")
    return result


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _audit_rlm(
    headers: Mapping[str, str], comparison: BenchmarkComparison
) -> tuple[int | None, int | None, int | None, str | None, BenchmarkFailure | None]:
    names = (
        "x-rlm-run-id",
        "x-rlm-model-calls",
        "x-rlm-input-tokens",
        "x-rlm-output-tokens",
        "x-rlm-usage-unreported-calls",
        "x-rlm-controller-model",
        "x-rlm-controller-options-sha256",
        "x-rlm-harness-fingerprint",
    )
    missing = [name for name in names if not headers.get(name)]
    if missing:
        return (
            None,
            None,
            None,
            None,
            BenchmarkFailure(
                BenchmarkFailureKind.PROVENANCE,
                "MissingHeaders",
                "missing RLM attestation headers: " + ", ".join(missing),
            ),
        )
    try:
        calls, inputs, outputs, unreported = (int(headers[name]) for name in names[1:5])
    except (TypeError, ValueError) as exc:
        return (
            None,
            None,
            None,
            None,
            BenchmarkFailure(
                BenchmarkFailureKind.PROVENANCE, type(exc).__name__, "malformed RLM numeric headers"
            ),
        )
    if calls < 1 or inputs < 0 or outputs < 0 or unreported < 0:
        return (
            None,
            None,
            None,
            None,
            BenchmarkFailure(
                BenchmarkFailureKind.PROVENANCE, "HeaderValue", "invalid RLM numeric headers"
            ),
        )
    mismatch = (
        headers[names[5]] != comparison.example.model
        or headers[names[6]] != comparison.model_options_sha256()
        or headers[names[7]] != comparison.rlm_harness_fingerprint
        or unreported != 0
    )
    failure = (
        BenchmarkFailure(
            BenchmarkFailureKind.PROVENANCE,
            "AttestationMismatch",
            "RLM controller provenance attestation mismatched",
        )
        if mismatch
        else None
    )
    return calls, inputs, outputs, headers[names[0]], failure


def _budget_payload(value: BenchmarkBudget) -> dict[str, Any]:
    return {
        "timeout_seconds": value.timeout_seconds,
        "max_model_calls": value.max_model_calls,
        "max_total_tokens": value.max_total_tokens,
    }


def _record(
    target: BenchmarkTarget,
    comparison: BenchmarkComparison,
    call_result: BenchmarkCall,
    score: Callable[[str], Any] | None,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    prompt = (
        comparison.prompts.rlm
        if target.condition is BenchmarkCondition.RLM
        else comparison.prompts.direct
    )
    failure = call_result.failure
    answer: str | None = None
    calls: int | None = None
    inputs: int | None = None
    outputs: int | None = None
    run_id: str | None = None
    if failure is None and call_result.status != 200:
        failure = BenchmarkFailure(
            BenchmarkFailureKind.HTTP, "HTTPStatus", f"unexpected HTTP status {call_result.status}"
        )
    if failure is None:
        assert call_result.response is not None
        error = validate_terminal_response(call_result.response)
        answer = output_text(call_result.response)
        if error is not None or not answer.strip():
            failure = BenchmarkFailure(
                BenchmarkFailureKind.PROTOCOL,
                "ResponseProtocol",
                error or "response has no benchmark answer text",
            )
    if failure is None and target.condition is BenchmarkCondition.RLM:
        calls, inputs, outputs, run_id, failure = _audit_rlm(call_result.headers, comparison)
    elif failure is None:
        calls = 1
        usage = call_result.response.get("usage") if call_result.response else None
        if (
            not isinstance(usage, Mapping)
            or not _nonnegative_int(usage.get("input_tokens"))
            or not _nonnegative_int(usage.get("output_tokens"))
        ):
            failure = BenchmarkFailure(
                BenchmarkFailureKind.PROTOCOL, "ResponseUsage", "direct response has invalid usage"
            )
        else:
            inputs, outputs = int(usage["input_tokens"]), int(usage["output_tokens"])
    total = None if inputs is None or outputs is None else inputs + outputs
    if failure is None and calls is not None and calls > comparison.budget.max_model_calls:
        failure = BenchmarkFailure(
            BenchmarkFailureKind.BUDGET, "ModelCalls", "observed model calls exceed declared budget"
        )
    if (
        failure is None
        and comparison.budget.max_total_tokens is not None
        and total is not None
        and total > comparison.budget.max_total_tokens
    ):
        failure = BenchmarkFailure(
            BenchmarkFailureKind.BUDGET, "TokenUsage", "observed tokens exceed declared budget"
        )
    scored = None if failure is not None or score is None or answer is None else score(answer)
    if is_dataclass(scored) and not isinstance(scored, type):
        scored = asdict(scored)
    return _signed_record(
        {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "pair_id": comparison.pair_id(),
            "episode_id": episode_id(comparison.pair_id(), target.condition),
            "condition": target.condition.value,
            "prompt_profile": prompt.profile.value,
            "prompt_text": prompt.text,
            "prompt_sha256": prompt.sha256,
            "rlm_conditioning_mode": comparison.mode.value
            if target.condition is BenchmarkCondition.RLM
            else None,
            "model": comparison.example.model,
            "source_id": comparison.example.source_id,
            "context": comparison.example.context,
            "context_sha256": comparison.example.context_sha256,
            "task": comparison.example.task,
            "expected_count": comparison.example.expected_count,
            "reference": strict_json_loads(comparison.example.reference_json),
            "dataset_path": comparison.example.dataset_path,
            "dataset_sha256": comparison.example.dataset_sha256,
            "dataset_revision": comparison.example.dataset_revision,
            "selection": strict_json_loads(comparison.example.selection_json),
            "source_revision": comparison.example.source_revision,
            "seed": comparison.seed,
            "repetition": comparison.repetition,
            "sampling": comparison.sampling(),
            "controller_options_sha256": comparison.model_options_sha256(),
            "rlm_harness_fingerprint": comparison.rlm_harness_fingerprint,
            "verifier": {
                "name": comparison.verifier_name,
                "version": comparison.verifier_version,
                "sha256": comparison.verifier_sha256,
                "source_revision": comparison.verifier_source_revision,
            },
            "budget": _budget_payload(comparison.budget),
            "status": call_result.status,
            "response": call_result.response,
            "headers": copy.deepcopy(call_result.headers),
            "run_id": run_id,
            "answer": answer if failure is None else None,
            "score": scored,
            "failure": None if failure is None else failure.to_dict(),
            "observed_model_calls": calls,
            "observed_input_tokens": inputs,
            "observed_output_tokens": outputs,
            "observed_total_tokens": total,
            "elapsed_seconds": elapsed_seconds,
        }
    )


def run_pair(
    targets: Sequence[BenchmarkTarget],
    *,
    comparison: BenchmarkComparison,
    call: Callable[[BenchmarkTarget, dict[str, Any]], BenchmarkCall],
    score: Callable[[str], Any] | None = None,
) -> list[dict[str, Any]]:
    rlm_target, direct_target = normalize_targets(targets)
    requests = paired_requests(comparison)
    records: list[dict[str, Any]] = []
    for target, request in zip(
        (rlm_target, direct_target),
        requests,
        strict=True,
    ):
        started = time.monotonic()
        result = call(target, copy.deepcopy(request))
        records.append(_record(target, comparison, result, score, time.monotonic() - started))
    return records


@dataclass(frozen=True, slots=True)
class BenchmarkRunSpec:
    comparisons: tuple[BenchmarkComparison, ...]
    targets: tuple[BenchmarkTarget, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "comparisons": [_comparison_payload(item) for item in self.comparisons],
            "targets": [
                {"condition": item.condition.value, "url": item.url} for item in self.targets
            ],
        }

    def fingerprint(self) -> str:
        return strict_json_sha256(self.payload())


def _signed_record(record: dict[str, Any]) -> dict[str, Any]:
    """Bind every durable record field so resume rejects silent tampering."""

    payload = strict_json_loads(strict_json_dumps(record, sort_keys=True, separators=(",", ":")))
    if not isinstance(payload, dict):  # pragma: no cover - record literal
        raise TypeError("benchmark record must be a JSON object")
    payload["record_sha256"] = strict_json_sha256(payload)
    return payload


def _validate_record(record: Any, comparisons: Sequence[BenchmarkComparison]) -> None:
    if not isinstance(record, dict):
        raise ValueError("checkpoint record is malformed")
    comparison = next(
        (item for item in comparisons if item.pair_id() == record.get("pair_id")), None
    )
    if comparison is None or record.get("condition") not in {
        item.value for item in BenchmarkCondition
    }:
        raise ValueError("checkpoint record provenance does not match comparison")
    condition = BenchmarkCondition(record["condition"])
    if record.get("episode_id") != episode_id(comparison.pair_id(), condition):
        raise ValueError("checkpoint record episode identity does not match comparison")
    prompt = (
        comparison.prompts.rlm
        if record["condition"] == BenchmarkCondition.RLM.value
        else comparison.prompts.direct
    )
    if (
        record.get("prompt_profile") != prompt.profile.value
        or record.get("prompt_text") != prompt.text
        or record.get("prompt_sha256") != prompt.sha256
    ):
        raise ValueError("checkpoint record prompt provenance does not match comparison")
    expected_digest = record.get("record_sha256")
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    if not isinstance(expected_digest, str) or expected_digest != strict_json_sha256(unsigned):
        raise ValueError("checkpoint record provenance digest does not match")


def atomic_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    complete = _checkpoint_with_digest(checkpoint)
    payload = strict_json_dumps(complete, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_checkpoint(path: Path, spec: BenchmarkRunSpec) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_bytes())
    except (OSError, StrictJSONError, UnicodeError) as exc:
        raise ValueError("checkpoint is not valid strict JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("run_fingerprint") != spec.fingerprint()
        or value.get("run_spec") != spec.payload()
        or not isinstance(value.get("results"), list)
    ):
        raise ValueError("checkpoint provenance does not match requested benchmark run")
    expected = value.get("checkpoint_sha256")
    unsigned = {key: item for key, item in value.items() if key != "checkpoint_sha256"}
    if not isinstance(expected, str) or expected != strict_json_sha256(unsigned):
        raise ValueError("checkpoint digest does not match")
    return value


def _checkpoint_with_digest(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(value) for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    payload = strict_json_loads(strict_json_dumps(unsigned, sort_keys=True, separators=(",", ":")))
    if not isinstance(payload, dict):  # pragma: no cover - mapping source
        raise TypeError("checkpoint must be an object")
    payload["checkpoint_sha256"] = strict_json_sha256(payload)
    return payload


def run_comparisons(
    comparisons: Sequence[BenchmarkComparison],
    *,
    targets: Sequence[BenchmarkTarget],
    output: Path | None,
    call: Callable[[BenchmarkTarget, dict[str, Any]], BenchmarkCall],
    score: Callable[[BenchmarkComparison, str], Any] | None = None,
) -> list[dict[str, Any]]:
    if not comparisons:
        raise ValueError("benchmark selected no examples")
    normalized_targets = normalize_targets(targets)
    spec = BenchmarkRunSpec(tuple(comparisons), normalized_targets)
    checkpoint = (
        _load_checkpoint(output, spec)
        if output is not None and output.exists()
        else _checkpoint_with_digest(
            {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "run_fingerprint": spec.fingerprint(),
                "run_spec": spec.payload(),
                "results": [],
            }
        )
    )
    completed: set[tuple[str, str]] = set()
    for record in checkpoint["results"]:
        _validate_record(record, comparisons)
        key = record["pair_id"], record["condition"]
        if key in completed:
            raise ValueError("checkpoint contains duplicate pair/condition record")
        completed.add(key)
    for comparison in comparisons:
        requests = dict(
            zip(
                (BenchmarkCondition.RLM, BenchmarkCondition.DIRECT),
                paired_requests(comparison),
                strict=True,
            )
        )
        for target in normalized_targets:
            request = requests[target.condition]
            key = comparison.pair_id(), target.condition.value
            if key in completed:
                continue
            started = time.monotonic()
            result = call(target, copy.deepcopy(request))
            checkpoint["results"].append(
                _record(
                    target,
                    comparison,
                    result,
                    None
                    if score is None
                    else lambda answer, current=comparison: score(current, answer),
                    time.monotonic() - started,
                )
            )
            completed.add(key)
            if output is not None:
                atomic_checkpoint(output, checkpoint)
                checkpoint = _checkpoint_with_digest(checkpoint)
    return copy.deepcopy(checkpoint["results"])


def _score_number(value: Any) -> float:
    candidate = value.get("score") if isinstance(value, Mapping) else value
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        raise ValueError("scored benchmark record has a nonnumeric score")
    return float(candidate)


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("benchmark selected no examples")
    condition_summaries: dict[str, dict[str, Any]] = {}
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in results:
        condition = record.get("condition")
        if condition is None:
            continue
        if condition not in {item.value for item in BenchmarkCondition}:
            raise ValueError("benchmark record has an invalid condition")
        group = condition_summaries.setdefault(condition, {"total": 0, "scores": [], "failures": 0})
        group["total"] += 1
        if record.get("failure") is not None:
            group["failures"] += 1
        elif record.get("score") is not None:
            group["scores"].append(record["score"])
        pair_id = record.get("pair_id")
        if isinstance(pair_id, str):
            pairs.setdefault(pair_id, {})[condition] = record
    conditions = {
        condition: {
            "total": group["total"],
            "scored": len(group["scores"]),
            "failures": group["failures"],
            "mean_score": None
            if not group["scores"]
            else round(
                sum(_score_number(item) for item in group["scores"]) / len(group["scores"]), 12
            ),
        }
        for condition, group in condition_summaries.items()
    }
    valid_deltas: list[float] = []
    unpaired = failing = 0
    for pair in pairs.values():
        if set(pair) != {"rlm", "direct"}:
            unpaired += 1
        elif pair["rlm"].get("failure") is not None or pair["direct"].get("failure") is not None:
            failing += 1
        elif pair["rlm"].get("score") is None or pair["direct"].get("score") is None:
            unpaired += 1
        else:
            valid_deltas.append(
                _score_number(pair["rlm"]["score"]) - _score_number(pair["direct"]["score"])
            )
    scored = [
        record["score"]
        for record in results
        if record.get("failure") is None and record.get("score") is not None
    ]
    failures = [record for record in results if record.get("failure") is not None]
    return {
        "examples": len(results),
        "scored": len(scored),
        "failures": len(failures),
        "mean_score": None
        if not scored
        else sum(_score_number(value) for value in scored) / len(scored),
        "failure_rate": len(failures) / len(results),
        "conditions": conditions,
        "pairs": {
            "total": len(pairs),
            "valid": len(valid_deltas),
            "unpaired": unpaired,
            "failing": failing,
            "mean_rlm_minus_direct": None
            if not valid_deltas
            else round(sum(valid_deltas) / len(valid_deltas), 12),
        },
    }


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite value") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite value")
    return parsed


def _positive_int_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nullable_positive_int_argument(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return _positive_int_argument(value)


def _json_options(values: Sequence[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ValueError(f"sampling option must have the form KEY=VALUE: {value!r}")
        try:
            options[key] = strict_json_loads(raw)
        except StrictJSONError as exc:
            raise ValueError(f"sampling option VALUE must be valid JSON: {value!r}") from exc
    return options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired RLM/direct Oolong stress tests.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--url", required=True, help="RLM Responses endpoint")
    parser.add_argument("--direct-url", required=True, help="direct baseline Responses endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rlm-harness-fingerprint", required=True)
    parser.add_argument(
        "--conditioning",
        choices=[item.value for item in ConditioningMode],
        default=ConditioningMode.MINIMAL.value,
    )
    parser.add_argument("--timeout", type=_positive_float, required=True)
    parser.add_argument("--max-model-calls", type=_positive_int_argument, required=True)
    parser.add_argument("--max-total-tokens", type=_nullable_positive_int_argument, required=True)
    parser.add_argument("--repetitions", type=_positive_int_argument, required=True)
    parser.add_argument("--sampling-option", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument(
        "--context-length", type=_positive_int_argument, action="append", default=[]
    )
    parser.add_argument("--num-examples", type=_positive_int_argument, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    try:
        from harness_rl import oolong as oolong_dependency

        load_examples = oolong_dependency.load_examples
        score_output = oolong_dependency.score_output
        select_examples = oolong_dependency.select_examples
        package_version = importlib.metadata.version("harness-rl")
        module_file = oolong_dependency.__file__
        if not isinstance(module_file, str):
            raise ValueError("harness-rl verifier module source cannot be identified")
        verifier = verifier_provenance("harness-rl", package_version, Path(module_file))
    except (
        ImportError,
        importlib.metadata.PackageNotFoundError,
        ValueError,
    ) as exc:  # pragma: no cover
        raise SystemExit("oolong benchmark requires the harness-rl prerequisite") from exc
    args = build_parser().parse_args(argv)
    budget = BenchmarkBudget(args.timeout, args.max_model_calls, args.max_total_tokens)
    sampling = _json_options(args.sampling_option)
    dataset_path = args.dataset.resolve()
    try:
        dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SystemExit(f"oolong dataset is unavailable: {dataset_path}") from exc
    selection = {
        "context_lengths": args.context_length or [1024],
        "num_examples": args.num_examples,
        "seed": args.seed,
    }
    selected = select_examples(
        load_examples(args.dataset),
        context_lengths=args.context_length or [1024],
        num_examples=args.num_examples,
        seed=args.seed,
    )
    comparisons: list[BenchmarkComparison] = []
    examples: dict[str, Any] = {}
    for repetition in range(1, args.repetitions + 1):
        for selected_example in selected:
            comparison = BenchmarkComparison.from_options(
                example=BenchmarkExample.from_context(
                    model=args.model,
                    context=selected_example.context,
                    task=selected_example.question,
                    expected_count=selected_example.context.count(" || Instance: "),
                    source_id=selected_example.source_id,
                    reference=typed_oolong_reference(selected_example),
                    dataset_path=str(dataset_path),
                    dataset_sha256=dataset_sha256,
                    dataset_revision=str(
                        getattr(selected_example, "dataset_revision", dataset_sha256)
                    ),
                    selection=selection,
                    source_revision=str(
                        getattr(selected_example, "source_revision", "unspecified")
                    ),
                ),
                mode=ConditioningMode(args.conditioning),
                rlm_harness_fingerprint=args.rlm_harness_fingerprint,
                seed=args.seed,
                repetition=repetition,
                sampling=sampling,
                budget=budget,
                verifier_name=verifier["name"],
                verifier_version=verifier["version"],
                verifier_sha256=verifier["sha256"],
                verifier_source_revision=verifier["source_revision"],
            )
            comparisons.append(comparison)
            examples[comparison.pair_id()] = selected_example
    results = run_comparisons(
        comparisons,
        targets=benchmark_targets(args.url, args.direct_url),
        output=args.output,
        call=lambda target, request: complete(target.url, request, timeout=budget.timeout_seconds),
        score=lambda comparison, answer: score_output(examples[comparison.pair_id()], answer),
    )
    print(strict_json_dumps({"summary": summarize(results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
