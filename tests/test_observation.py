from __future__ import annotations

from rlm.json import strict_json_dumps
from rlm.observation import controller_observation
from rlm.recovery import ControllerFault, FaultKind
from rlm.specs import ObservationSpec
from rlm.types import ExecutionException, ExecutionResult


def test_controller_observation_preserves_fault_identity_when_text_is_truncated() -> None:
    """Removing structured fault fields or cap enforcement must break this test."""

    execution = ExecutionResult(
        stdout="x" * 10_000,
        exception=ExecutionException(type="NameError", message="missing value"),
    )

    observation = controller_observation(
        spec=ObservationSpec(),
        fault=ControllerFault(FaultKind.EXECUTION, "missing value"),
        execution=execution,
        max_chars=384,
    )

    encoded = strict_json_dumps(observation, sort_keys=True, separators=(",", ":"))
    assert len(encoded) <= 384
    assert observation["type"] == "rlm.controller_observation"
    assert observation["schema_version"] == "2"
    assert observation["request_binding"] == "request"
    assert observation["submission"] == {"status": "absent", "required": True}
    assert observation["execution"]["status"] == "error"
    assert observation["execution"]["fault"]["kind"] == "execution"
    assert observation["execution"]["exception"]["type"] == "NameError"
    assert observation["execution"]["truncation"]["omitted_chars"] > 0


def test_controller_observation_retains_native_outcome_values_when_they_fit() -> None:
    observation = controller_observation(
        spec=ObservationSpec(),
        fault=None,
        execution=ExecutionResult(namespace={"total": "int"}, duration_seconds=0.25),
        max_chars=1_000,
    )

    assert observation["execution"]["namespace"] == {"total": "int"}
    assert observation["execution"]["duration_seconds"] == 0.25
