from __future__ import annotations

from collections.abc import Callable

import pytest

from rlm import ControllerRunIdentity, RunAttestation, RunResult, TokenUsage
from tests.fakes import responses_text


def _result(*, duration_seconds: float = 1.2345678) -> RunResult:
    return RunResult(
        response=responses_text("done"),
        run_id="run-123",
        stop_reason="final_response",
        usage=TokenUsage(
            input_tokens=11,
            output_tokens=7,
            calls=3,
            unreported_calls=1,
        ),
        turns=2,
        duration_seconds=duration_seconds,
        controller_identity=ControllerRunIdentity("controller-model", "a" * 64),
        harness_fingerprint="b" * 64,
    )


def _headers() -> dict[str, str]:
    return {
        "X-RLM-Run-ID": "run-123",
        "X-RLM-Stop-Reason": "final_response",
        "X-RLM-Turns": "2",
        "X-RLM-Duration-Seconds": "1.234568",
        "X-RLM-Model-Calls": "3",
        "X-RLM-Input-Tokens": "11",
        "X-RLM-Output-Tokens": "7",
        "X-RLM-Usage-Unreported-Calls": "1",
        "X-RLM-Controller-Model": "controller-model",
        "X-RLM-Controller-Options-SHA256": "a" * 64,
        "X-RLM-Harness-Fingerprint": "b" * 64,
    }


def test_attestation_result_headers_round_trip_at_wire_duration_precision() -> None:
    attestation = RunAttestation.from_result(_result())

    assert attestation.to_headers() == _headers()
    decoded = RunAttestation.from_headers(attestation.to_headers())
    assert decoded == RunAttestation.from_result(_result(duration_seconds=1.234568))


def test_attestation_header_input_is_case_insensitive() -> None:
    headers = {name.swapcase(): value for name, value in _headers().items()}

    assert RunAttestation.from_headers(headers).to_headers() == _headers()


def test_attestation_defensively_owns_mutable_usage() -> None:
    usage = TokenUsage(input_tokens=1, output_tokens=2, calls=1)
    result = _result()
    object.__setattr__(result, "usage", usage)

    attestation = RunAttestation.from_result(result)
    usage.input_tokens = 99

    assert attestation.usage == TokenUsage(input_tokens=1, output_tokens=2, calls=1)
    with pytest.raises(AttributeError):
        attestation.run_id = "changed"  # type: ignore[misc]


def test_attestation_usage_access_cannot_mutate_the_snapshot() -> None:
    attestation = RunAttestation.from_result(_result())

    retrieved = attestation.usage
    retrieved.input_tokens = 999

    assert attestation.usage.input_tokens == 11
    assert attestation.to_headers()["X-RLM-Input-Tokens"] == "11"


def test_run_result_access_cannot_mutate_response_or_usage_snapshot() -> None:
    result = _result()

    response = result.response
    usage = result.usage
    response["output"][0]["content"][0]["text"] = "mutated"
    usage.calls = 999

    assert result.response["output"][0]["content"][0]["text"] == "done"
    assert result.usage.calls == 3
    assert result.to_dict()["usage"]["calls"] == 3


@pytest.mark.parametrize("missing", tuple(_headers()))
def test_attestation_rejects_each_missing_header(missing: str) -> None:
    headers = _headers()
    del headers[missing]

    with pytest.raises(ValueError, match="missing"):
        RunAttestation.from_headers(headers)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("X-RLM-Run-ID", ""),
        ("X-RLM-Stop-Reason", ""),
        ("X-RLM-Controller-Model", ""),
        ("X-RLM-Turns", "-1"),
        ("X-RLM-Turns", "+1"),
        ("X-RLM-Turns", "1.0"),
        ("X-RLM-Input-Tokens", " 1"),
        ("X-RLM-Output-Tokens", "1 "),
        ("X-RLM-Usage-Unreported-Calls", ""),
        ("X-RLM-Model-Calls", "0"),
        ("X-RLM-Duration-Seconds", "-0.1"),
        ("X-RLM-Duration-Seconds", "nan"),
        ("X-RLM-Duration-Seconds", "inf"),
        ("X-RLM-Controller-Options-SHA256", "A" * 64),
        ("X-RLM-Controller-Options-SHA256", "a" * 63),
        ("X-RLM-Harness-Fingerprint", "z" * 64),
    ],
)
def test_attestation_rejects_malformed_header(name: str, value: str) -> None:
    headers = _headers()
    headers[name] = value

    with pytest.raises(ValueError):
        RunAttestation.from_headers(headers)


def test_attestation_rejects_case_colliding_headers() -> None:
    headers = _headers()
    headers["x-rlm-run-id"] = "other-run"

    with pytest.raises(ValueError, match="duplicate"):
        RunAttestation.from_headers(headers)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: object.__setattr__(result, "turns", -1),
        lambda result: object.__setattr__(result, "duration_seconds", float("nan")),
        lambda result: object.__setattr__(result, "usage", TokenUsage(calls=0)),
    ],
)
def test_attestation_rejects_invalid_run_result(
    mutate: Callable[[RunResult], None],
) -> None:
    result = _result()
    mutate(result)

    with pytest.raises((TypeError, ValueError)):
        RunAttestation.from_result(result)
