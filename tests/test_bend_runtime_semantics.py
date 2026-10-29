"""Source-only observation algebra: no geometry, fixtures or model IO."""
from itertools import product

import pytest

from domains.bend.services.runtime_semantics import (
    is_explicit_observability_evidence,
    legacy_observability_detail_state,
    normalize_observability_state,
)


@pytest.mark.parametrize("state", [None, "", "UNKNOWN", "UNOBSERVED", "PARTIALLY_OBSERVED", "PARTIALLY_FORMED", "bad-state"])
def test_unknown_observation_is_invariant_to_verdict_and_completion_claim(state):
    for status, physical in product(
        [None, "PASS", "FAIL", "WARNING", "NOT_DETECTED"],
        [None, "FORMED", "UNFORMED", "NOT_FORMED", "PARTIALLY_FORMED", "UNKNOWN"],
    ):
        assert normalize_observability_state(state, status=status, physical_completion_state=physical) == "UNKNOWN"
        assert not is_explicit_observability_evidence(state, status=status, physical_completion_state=physical)


@pytest.mark.parametrize("state,expected", [
    ("FORMED", "FORMED"), ("OBSERVED_FORMED", "FORMED"),
    ("UNFORMED", "UNFORMED"), ("OBSERVED_NOT_FORMED", "UNFORMED"),
    (" observed_formed ", "FORMED"),
])
def test_explicit_observation_aliases_are_stable_and_idempotent(state, expected):
    for status in [None, "PASS", "FAIL", "WARNING", "NOT_DETECTED"]:
        result = normalize_observability_state(state, status=status)
        assert result == expected
        assert normalize_observability_state(result) == result
        assert is_explicit_observability_evidence(state, status=status)


def test_one_surface_measurement_pass_is_not_two_surface_observation():
    state = normalize_observability_state("PARTIALLY_OBSERVED", status="PASS", physical_completion_state="FORMED")
    assert state == "UNKNOWN"
    assert legacy_observability_detail_state("PARTIALLY_OBSERVED", status="PASS") == "PARTIALLY_OBSERVED"


def test_explicit_detail_is_preserved_and_missing_detail_cannot_use_verdict():
    assert legacy_observability_detail_state("UNOBSERVED", physical_completion_state="FORMED", status="PASS") == "UNOBSERVED"
    assert legacy_observability_detail_state(None, physical_completion_state="FORMED", status="PASS") == "UNOBSERVED"
    assert legacy_observability_detail_state("FORMED") == "OBSERVED_FORMED"
