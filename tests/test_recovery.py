from __future__ import annotations

import math

import pytest

from rlm.recovery import ControllerFault, FaultKind, RecoveryPolicy, Repair
from rlm.specs import RecoverySpec


def test_default_policy_repairs_controller_fault() -> None:
    fault = ControllerFault(FaultKind.EXECUTION, "NameError: missing")
    decision = RecoveryPolicy(RecoverySpec()).decide(fault, turn=2)

    assert isinstance(decision, Repair)
    assert decision.fault is fault


def test_controller_fault_details_are_copied_as_strict_json() -> None:
    details = {"nested": [1, {"value": True}]}
    fault = ControllerFault(FaultKind.EXECUTION, "missing", details)

    value = fault.to_dict()
    details["nested"][1]["value"] = False

    assert value == {
        "kind": "execution",
        "message": "missing",
        "details": {"nested": [1, {"value": True}]},
    }


@pytest.mark.parametrize(
    "details",
    [
        {"tuple": (1, 2)},
        {1: "non-string key"},
        {"cycle": None},
        {"surrogate": "\ud800"},
        {"not_a_number": math.nan},
        {"infinite": math.inf},
    ],
)
def test_controller_fault_details_reject_non_strict_json(details: dict[object, object]) -> None:
    if "cycle" in details:
        details["cycle"] = details

    with pytest.raises(ValueError, match="strict JSON"):
        ControllerFault(FaultKind.EXECUTION, "missing", details)
