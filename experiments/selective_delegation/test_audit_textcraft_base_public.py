import pytest
from audit_textcraft_base_public import success_bounds


def test_success_bounds_uses_observed_unknown_count_not_a_hardcoded_three() -> None:
    rows = [
        {"status": "known", "base": 0.0, "public": 1.0},
        {"status": "known", "base": 1.0, "public": 1.0},
        {"status": "base_missing_unknown", "base": None, "public": 1.0},
    ]

    assert success_bounds(rows) == {
        "planned": 3,
        "known": 2,
        "unknown": 1,
        "public": [1.0, 1.0],
        "base": [1 / 3, 2 / 3],
        "public_minus_base": [1 / 3, 2 / 3],
    }


def test_success_bounds_rejects_unknown_public_result() -> None:
    with pytest.raises(ValueError, match="public outcome unavailable"):
        success_bounds([{"status": "base_missing_unknown", "base": None, "public": None}])
