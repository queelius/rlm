from rlm.response import validate_response_envelope, validate_terminal_response
from tests.fakes import responses_text


def test_terminal_response_requires_complete_responses_envelope() -> None:
    assert validate_terminal_response({"output": [{}]}) is not None
    assert validate_terminal_response({"output": [{"type": "bogus"}]}) is not None


def test_valid_unknown_output_item_is_preserved() -> None:
    response = responses_text("done")
    response["output"] = [{"id": "future_1", "type": "future_terminal", "payload": {"value": 1}}]
    assert validate_terminal_response(response) is None


def test_reasoning_only_response_is_not_terminal() -> None:
    response = responses_text("done")
    response["output"] = [{"id": "rs_1", "type": "reasoning", "summary": []}]
    assert validate_response_envelope(response) is None
    assert validate_terminal_response(response) == "Responses response contains no terminal output"


def test_empty_text_is_structurally_valid_but_strict_helpers_may_reject_it() -> None:
    response = responses_text("")
    assert validate_response_envelope(response) is None
    assert validate_terminal_response(response) is None


def test_non_string_status_is_a_validation_error_not_a_validator_exception() -> None:
    response = responses_text("done")
    response["status"] = []
    assert validate_response_envelope(response) is not None
