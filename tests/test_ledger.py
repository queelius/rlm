from __future__ import annotations

import math

import pytest

from rlm.config import RLMConfig, RunLimits
from rlm.errors import BackendProtocolError, LimitExceededError
from rlm.ledger import Ledger
from tests.fakes import responses_text


def _ledger(*, max_total_tokens: int | None) -> Ledger:
    return Ledger(RLMConfig(limits=RunLimits(max_total_tokens=max_total_tokens)))


def test_aggregate_token_accounting_records_valid_usage() -> None:
    ledger = _ledger(max_total_tokens=10)
    ledger.reserve_model_call()

    usage = ledger.record_response(responses_text("ok", input_tokens=3, output_tokens=4))

    assert usage.to_dict() == {
        "input_tokens": 3,
        "output_tokens": 4,
        "calls": 1,
        "unreported_calls": 0,
    }
    assert ledger.snapshot().to_dict() == {
        "input_tokens": 3,
        "output_tokens": 4,
        "calls": 1,
        "unreported_calls": 0,
    }


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": True, "output_tokens": 1}])
def test_aggregate_budget_rejects_missing_or_malformed_usage(usage: object) -> None:
    ledger = _ledger(max_total_tokens=10)
    response = responses_text("ok")
    if usage is None:
        response.pop("usage")
    else:
        response["usage"] = usage

    with pytest.raises(BackendProtocolError) as caught:
        ledger.record_response(response)

    assert caught.value.code == "backend_protocol_error"


def test_aggregate_budget_rejects_cumulative_token_exhaustion() -> None:
    ledger = _ledger(max_total_tokens=10)
    ledger.record_response(responses_text("first", input_tokens=3, output_tokens=4))

    with pytest.raises(LimitExceededError) as caught:
        ledger.record_response(responses_text("second", input_tokens=2, output_tokens=2))

    assert caught.value.code == "token_limit"


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_turns": True},
        {"max_model_calls": True},
        {"max_subcalls": True},
        {"max_parallel_model_calls": True},
        {"max_depth": True},
        {"max_total_tokens": True},
        {"max_execution_output_chars": True},
        {"max_observation_chars": True},
        {"deadline_seconds": True},
        {"execution_timeout_seconds": True},
        {"deadline_seconds": math.nan},
        {"deadline_seconds": math.inf},
        {"execution_timeout_seconds": -math.inf},
    ],
)
def test_run_limits_reject_bool_and_nonfinite_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RunLimits(**overrides)


def test_run_limits_enforce_the_supported_observation_floor() -> None:
    with pytest.raises(ValueError, match="max_observation_chars.*512"):
        RunLimits(max_observation_chars=511)

    assert RunLimits(max_observation_chars=512).max_observation_chars == 512
