import pytest

from scripts.augment_f1_candidate_pool_with_scan_mesh import (
    FEATURE_SCHEMA_VERSION,
    REQUIRED_LINE_OBJECT_FEATURES,
    _namespace_existing_candidate,
    validate_line_object_feature_schema,
)


def test_line_object_feature_schema_requires_uniform_enrichment():
    complete = {name: 0.0 for name in REQUIRED_LINE_OBJECT_FEATURES}
    validate_line_object_feature_schema([complete])

    incomplete = dict(complete)
    incomplete.pop("two_sided_transition_score")
    with pytest.raises(ValueError, match=FEATURE_SCHEMA_VERSION):
        validate_line_object_feature_schema([incomplete])


def test_existing_candidate_identity_is_namespaced_idempotently():
    candidate = {
        "candidate_source": "cylindrical_wide",
        "candidate_id": "CB2_301",
    }

    first = _namespace_existing_candidate(candidate)
    second = _namespace_existing_candidate(first)

    assert first["candidate_id"] == "cylindrical_wide::CB2_301"
    assert second == first
