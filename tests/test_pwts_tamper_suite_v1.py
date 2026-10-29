from __future__ import annotations

from research.pwts_v1.independent_validator import ValidationPolicy
from research.pwts_v1.tamper_suite import (
    MANDATORY_TAMPER_REGRESSIONS,
    require_all_tampers_rejected,
    run_tamper_suite,
)
from tests.test_pwts_independent_validator_v1 import (
    CASES,
    CONFIG_SHA,
    SCHEMA_SHA,
    SOURCE_MANIFEST_SHA,
    make_real_like_compact_evidence,
)


def _policy():
    return ValidationPolicy(
        expected_configuration_sha256=CONFIG_SHA,
        expected_schema_sha256=SCHEMA_SHA,
        expected_source_manifest_sha256=SOURCE_MANIFEST_SHA,
        expected_case_tokens=CASES,
    )


def test_exactly_fourteen_mandatory_regressions_are_unique():
    assert len(MANDATORY_TAMPER_REGRESSIONS) == 14
    ids = [regression.regression_id for regression in MANDATORY_TAMPER_REGRESSIONS]
    assert len(ids) == len(set(ids))
    assert ids == [f"T{index:02d}_{suffix}" for index, suffix in enumerate(
        (
            "RAW_BYTE_CHANGED",
            "CASE_SOURCE_BINDING_EXCHANGE",
            "SAMPLED_POINT_ID_SUBSTITUTION",
            "UNAUTHORIZED_CONFIGURATION",
            "SEED_ROW_REMOVAL",
            "TRACE_NODE_ID_MUTATION",
            "MEMBERSHIP_SIDE_EXCHANGE",
            "FORGED_CERTIFICATE_VALIDITY",
            "DELETION_EVIDENCE_FORGERY",
            "ROW_ORDER_DUPLICATE_DELETE",
            "SOURCE_POINT_SERIALIZATION",
            "RESOURCE_ACCOUNTING_FORGERY",
            "FORGED_EQUIVALENCE",
            "SEMANTIC_IMPORT_BOUNDARY",
        ),
        start=1,
    )]


def test_all_mandatory_tampers_fail_closed_on_copied_real_like_tables():
    baseline = make_real_like_compact_evidence()
    results = run_tamper_suite(baseline, policy=_policy())
    require_all_tampers_rejected(results)
    assert all(result.observed_errors for result in results)


def test_each_tamper_reports_its_expected_rejection_class():
    baseline = make_real_like_compact_evidence()
    results = run_tamper_suite(baseline, policy=_policy())
    misses = {
        result.regression_id: (result.expected_rejection, result.observed_errors)
        for result in results
        if not any(
            result.expected_rejection.lower() in error.lower()
            for error in result.observed_errors
        )
    }
    assert misses == {}
