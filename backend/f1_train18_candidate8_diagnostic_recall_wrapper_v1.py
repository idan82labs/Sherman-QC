"""Filesystem-blind, value-free diagnostic wrapper for the frozen recall hypothesis.

Candidate8 changes no scientific acceptance, eligibility, sanitization, ranking,
or ownership behavior.  A separate lineage gate validates exact dependencies;
then five fixed diagnostic phases invoke the frozen Candidate6 hypothesis and
Candidate7 wrapper.  Only a bounded phase/check code is returned.  Exception
text, exception classes, counts, identifiers, candidate values, geometry,
predictions, labels, and metrics are never returned.
"""

from __future__ import annotations

import hashlib
import json
import re
from types import ModuleType
from typing import Any, Mapping


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 8
ENGINEERING_ATTEMPT_ID = "CANDIDATE8_ENGINEERING_ATTEMPT_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256 = (
    "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252"
)
SCIENTIFIC_HYPOTHESIS_SOURCE_SIZE_BYTES = 53837
SCIENTIFIC_RULE_SPACE_SHA256 = (
    "0bb1f5fda0b06f1c7b0d7eb4348dcaa32e68d9abf386f55f27e06fcd36a6fcc8"
)
SCIENTIFIC_OWNERSHIP_CONTRACT_SHA256 = (
    "897db33ea491601af4f685dac85272a3a9cbc8ec756dd82a81949006a293dcbc"
)
CANDIDATE7_RUNTIME_SOURCE_SHA256 = (
    "a6bd7a23ead5bb07040a7bb1d02a8aab3588bf544b1ec1c9feaf534c9498f04d"
)
CANDIDATE7_RUNTIME_SOURCE_SIZE_BYTES = 28098
CANDIDATE7_PREFLIGHT_RESULT_KIND = (
    "sherman_qc_f1_aug30_train18_candidate7_semantic_pool_compatibility_result_v1"
)
CANDIDATE7_PREFLIGHT_RESULT_KEYS = frozenset(
    {
        "all_fixed_rule_forms_exercised",
        "attempt_ordinal",
        "candidate_metric_computed",
        "candidate_rule_selected",
        "candidate_source_rows_sanitized",
        "deterministic_geometry_source_tie_break_exercised",
        "engineering_attempt_id",
        "fixed_allowed_source_kind_eligibility_exercised",
        "generic_pool_accounting_ignored",
        "holdout_score_opened",
        "input_pool_sha256",
        "input_v51_semantic_sha256",
        "kind",
        "labels_used",
        "missing_evidence_abstention_exercised",
        "model_fit_performed",
        "ownership_geometry_exercised",
        "raw_candidates_ignored_for_selection",
        "recursive_forbidden_key_validation_exercised",
        "runtime_threshold_match_implies_conservative_conflict_for_bound_pairs",
        "schema_version",
        "scientific_hypothesis_exactly_preserved",
        "scientific_hypothesis_id",
        "scientific_hypothesis_source_sha256",
        "source_only",
        "status",
        "v51_prediction_values_and_order_preserved",
    }
)

CANDIDATE7_DISPOSITION_PATH = (
    "research/f1_aug30_train18_candidate7_terminal_disposition_v1.json"
)
CANDIDATE7_DISPOSITION_SHA256 = (
    "1a64d3148d018b224a3a44d9d36d2620f1d87f18063e32df14050210aa03f56b"
)
CANDIDATE7_DISPOSITION_SIZE_BYTES = 3666
CANDIDATE7_DISPOSITION_KIND = (
    "sherman_qc_f1_aug30_train18_candidate7_terminal_disposition_v1"
)

RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate8_diagnostic_result_v1"
PHASE_CONTRACT = (
    "D01_SEMANTIC_VALIDATION",
    "D02_POOL_PARSE_SANITIZE",
    "D03_OWNERSHIP_EXERCISE",
    "D04_FIXED_RULE_EXERCISE",
    "D05_WRAPPER_RESULT_VALIDATION",
)
FAILURE_CODES = {
    "LINEAGE": "C8_LINEAGE_VALIDATION_FAILED",
    PHASE_CONTRACT[0]: "C8_SEMANTIC_VALIDATION_FAILED",
    PHASE_CONTRACT[1]: "C8_POOL_PARSE_SANITIZE_FAILED",
    PHASE_CONTRACT[2]: "C8_OWNERSHIP_EXERCISE_FAILED",
    PHASE_CONTRACT[3]: "C8_FIXED_RULE_EXERCISE_FAILED",
    PHASE_CONTRACT[4]: "C8_WRAPPER_RESULT_VALIDATION_FAILED",
}
PASS_CODE = "C8_ALL_DIAGNOSTIC_PHASES_PASSED"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

RESULT_KEYS = frozenset(
    {
        "attempt_ordinal",
        "candidate_metric_computed",
        "candidate_rule_selected",
        "check_code",
        "diagnostic_complete",
        "engineering_attempt_id",
        "exception_class_exposed",
        "exception_text_exposed",
        "failure_phase",
        "holdout_score_opened",
        "identifier_or_geometry_exposed",
        "kind",
        "labels_used",
        "lineage_gate_passed",
        "model_fit_performed",
        "numeric_or_categorical_source_value_exposed",
        "phase_contract_sha256",
        "schema_version",
        "scientific_hypothesis_exactly_preserved",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "scientific_hypothesis_source_sha256",
        "source_only",
        "status",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


PHASE_CONTRACT_SHA256 = sha256_bytes(canonical_bytes(list(PHASE_CONTRACT)))


def _require_sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if value == "0" * 64:
        raise ValueError(f"{name} must not be unresolved")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _parse_object(data: bytes, *, name: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError(f"{name} must be immutable bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict) or canonical_bytes(value) != data:
        raise ValueError(f"{name} must be one LF-canonical JSON object")
    return value


def verify_scientific_hypothesis_runtime(
    hypothesis_runtime: ModuleType, *, source_sha256: str
) -> None:
    if _require_sha(source_sha256, name="hypothesis source SHA") != (
        SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
    ):
        raise ValueError("Candidate8 hypothesis source binding mismatch")
    try:
        rules = list(hypothesis_runtime.fixed_rule_space())
        ownership = hypothesis_runtime._ownership_contract()
    except AttributeError as error:
        raise ValueError("Candidate8 hypothesis API mismatch") from error
    checks = (
        hypothesis_runtime.ATTEMPT_ORDINAL == 6,
        sha256_bytes(canonical_bytes(rules)) == SCIENTIFIC_RULE_SPACE_SHA256,
        sha256_bytes(canonical_bytes(ownership))
        == SCIENTIFIC_OWNERSHIP_CONTRACT_SHA256,
        hypothesis_runtime.MAX_ADDITIONS_PER_CASE == 1,
        hypothesis_runtime.MAX_ADDED_FALSE_POSITIVES_TRAIN18 == 2,
        tuple(hypothesis_runtime.FIXED_ELIGIBILITY_INPUTS)
        == ("finite_line_geometry", "source_kind"),
        tuple(hypothesis_runtime.DETERMINISTIC_TIE_BREAK_INPUTS)
        == ("canonical_line_endpoints", "source_kind"),
        hypothesis_runtime.DETERMINISTIC_TIE_BREAK_DIRECTION
        == "CANONICAL_JSON_BYTES_ASCENDING",
    )
    if not all(checks):
        raise ValueError("Candidate8 hypothesis contract mismatch")


def verify_candidate7_wrapper_runtime(
    candidate7_runtime: ModuleType,
    *,
    source_sha256: str,
    hypothesis_runtime: ModuleType,
) -> None:
    if _require_sha(source_sha256, name="Candidate7 runtime source SHA") != (
        CANDIDATE7_RUNTIME_SOURCE_SHA256
    ):
        raise ValueError("Candidate8 Candidate7 wrapper source binding mismatch")
    if (
        candidate7_runtime.ATTEMPT_ORDINAL != 7
        or candidate7_runtime.ENGINEERING_ATTEMPT_ID
        != "CANDIDATE7_ENGINEERING_ATTEMPT_V1"
        or candidate7_runtime.SCIENTIFIC_HYPOTHESIS_ID != SCIENTIFIC_HYPOTHESIS_ID
        or candidate7_runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
        != SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
        or candidate7_runtime.PREFLIGHT_RESULT_KIND
        != CANDIDATE7_PREFLIGHT_RESULT_KIND
    ):
        raise ValueError("Candidate8 Candidate7 wrapper contract mismatch")
    candidate7_runtime.verify_scientific_hypothesis_runtime(
        hypothesis_runtime,
        source_sha256=SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
    )


def _validate_candidate7_disposition(data: bytes, *, expected_sha256: str) -> None:
    if expected_sha256 != CANDIDATE7_DISPOSITION_SHA256:
        raise ValueError("Candidate7 disposition external binding mismatch")
    if len(data) != CANDIDATE7_DISPOSITION_SIZE_BYTES or sha256_bytes(data) != (
        expected_sha256
    ):
        raise ValueError("Candidate7 disposition exact byte mismatch")
    value = _parse_object(data, name="Candidate7 disposition")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != CANDIDATE7_DISPOSITION_KIND
        or value.get("attempt_ordinal") != 7
        or value.get("status")
        != "CONFIRMED_FAILED_ENGINEERING_SOURCE_COMPATIBILITY"
        or value.get("candidate7_continuation_authorized") is not False
        or value.get("candidate7_preflight_retry_authorized") is not False
        or value.get("candidate8_engineering_successor_authorized") is not True
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("scientific_hypothesis_performance_tested") is not False
        or value.get("label_attestation_or_projection_opened") is not False
        or value.get("holdout_path_hash_status_or_score_opened") is not False
        or value.get("candidate7_metric_computed") is not False
        or value.get("candidate7_fit_started") is not False
    ):
        raise ValueError("Candidate7 disposition status contract mismatch")


def _result(*, failure_phase: str | None) -> dict[str, Any]:
    code = PASS_CODE if failure_phase is None else FAILURE_CODES[failure_phase]
    lineage_gate_passed = failure_phase != "LINEAGE"
    value = {
        "attempt_ordinal": 8,
        "candidate_metric_computed": False,
        "candidate_rule_selected": False,
        "check_code": code,
        "diagnostic_complete": failure_phase is None,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "exception_class_exposed": False,
        "exception_text_exposed": False,
        "failure_phase": failure_phase,
        "holdout_score_opened": False,
        "identifier_or_geometry_exposed": False,
        "kind": RESULT_KIND,
        "labels_used": False,
        "lineage_gate_passed": lineage_gate_passed,
        "model_fit_performed": False,
        "numeric_or_categorical_source_value_exposed": False,
        "phase_contract_sha256": PHASE_CONTRACT_SHA256,
        "schema_version": 1,
        "scientific_hypothesis_exactly_preserved": lineage_gate_passed,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "scientific_hypothesis_source_sha256": SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
        "source_only": True,
        "status": "SOURCE_COMPATIBLE" if failure_phase is None else "SOURCE_INCOMPATIBLE",
    }
    _validate_result(value)
    return value


def _validate_result(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != RESULT_KEYS:
        raise ValueError("Candidate8 diagnostic result schema mismatch")
    failure_phase = value["failure_phase"]
    if failure_phase is not None and failure_phase not in {"LINEAGE", *PHASE_CONTRACT}:
        raise ValueError("Candidate8 diagnostic phase is outside the fixed contract")
    expected_code = PASS_CODE if failure_phase is None else FAILURE_CODES[failure_phase]
    if (
        value["schema_version"] != 1
        or value["kind"] != RESULT_KIND
        or value["attempt_ordinal"] != 8
        or value["engineering_attempt_id"] != ENGINEERING_ATTEMPT_ID
        or value["scientific_hypothesis_id"] != SCIENTIFIC_HYPOTHESIS_ID
        or value["scientific_hypothesis_source_sha256"]
        != SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
        or value["phase_contract_sha256"] != PHASE_CONTRACT_SHA256
        or value["check_code"] != expected_code
        or value["diagnostic_complete"] is not (failure_phase is None)
        or value["status"]
        != ("SOURCE_COMPATIBLE" if failure_phase is None else "SOURCE_INCOMPATIBLE")
        or value["lineage_gate_passed"] is not (failure_phase != "LINEAGE")
        or value["scientific_hypothesis_exactly_preserved"]
        is not (failure_phase != "LINEAGE")
        or value["source_only"] is not True
        or any(
            value[key] is not False
            for key in (
                "candidate_metric_computed",
                "candidate_rule_selected",
                "exception_class_exposed",
                "exception_text_exposed",
                "holdout_score_opened",
                "identifier_or_geometry_exposed",
                "labels_used",
                "model_fit_performed",
                "numeric_or_categorical_source_value_exposed",
                "scientific_hypothesis_performance_tested",
            )
        )
    ):
        raise ValueError("Candidate8 diagnostic result contract mismatch")


def diagnose_source_bytes(
    v51_semantic_bytes: bytes,
    pool_bytes: bytes,
    candidate7_disposition_bytes: bytes,
    *,
    v51_semantic_sha256: str,
    pool_sha256: str,
    candidate7_disposition_sha256: str,
    hypothesis_runtime: ModuleType,
    hypothesis_source_sha256: str,
    candidate7_runtime: ModuleType,
    candidate7_runtime_source_sha256: str,
) -> dict[str, Any]:
    """Return only the first fixed phase/check code that fails."""

    try:
        verify_scientific_hypothesis_runtime(
            hypothesis_runtime, source_sha256=hypothesis_source_sha256
        )
        verify_candidate7_wrapper_runtime(
            candidate7_runtime,
            source_sha256=candidate7_runtime_source_sha256,
            hypothesis_runtime=hypothesis_runtime,
        )
        _validate_candidate7_disposition(
            candidate7_disposition_bytes,
            expected_sha256=candidate7_disposition_sha256,
        )
    except BaseException:
        return _result(failure_phase="LINEAGE")

    try:
        _semantic, predictions = hypothesis_runtime._validate_semantic_bytes(
            v51_semantic_bytes, expected_sha256=v51_semantic_sha256
        )
    except BaseException:
        return _result(failure_phase=PHASE_CONTRACT[0])

    try:
        _pool, candidates = hypothesis_runtime._validate_pool_bytes(
            pool_bytes, expected_sha256=pool_sha256
        )
    except BaseException:
        return _result(failure_phase=PHASE_CONTRACT[1])

    try:
        incumbents = [
            hypothesis_runtime._endpoints_from_row(
                prediction,
                name="Candidate8 incumbent",
                require_prediction_lines=True,
            )
            for prediction in predictions
        ]
        for incumbent in incumbents:
            if incumbent is None:
                raise RuntimeError("unreachable incumbent representation")
            for candidate in candidates:
                projection = candidate["identity_projection"]
                candidate_line = (
                    tuple(projection["line_start"]),
                    tuple(projection["line_end"]),
                )
                metrics = hypothesis_runtime.line_pair_metrics(
                    incumbent, candidate_line
                )
                runtime_threshold_match = bool(
                    metrics["axis_angle_deg"]
                    <= hypothesis_runtime.OWNERSHIP_ANGLE_DEG
                    and metrics["transverse_offset_mm"]
                    <= hypothesis_runtime.OWNERSHIP_TRANSVERSE_OFFSET_MM
                    and metrics["incumbent_span_overlap"]
                    >= hypothesis_runtime.OWNERSHIP_INCUMBENT_SPAN_OVERLAP
                    and metrics["candidate_span_overlap"]
                    >= hypothesis_runtime.OWNERSHIP_CANDIDATE_SPAN_OVERLAP
                )
                if runtime_threshold_match and not hypothesis_runtime._ownership_conflict(
                    candidate_line, [incumbent]
                ):
                    raise RuntimeError("ownership implication failed")
        identical = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
        separated = ((0.0, 5.0, 0.0), (10.0, 5.0, 0.0))
        if not hypothesis_runtime._ownership_conflict(identical, [identical]):
            raise RuntimeError("ownership self-check failed")
        if hypothesis_runtime._ownership_conflict(separated, [identical]):
            raise RuntimeError("ownership separation self-check failed")
    except BaseException:
        return _result(failure_phase=PHASE_CONTRACT[2])

    try:
        original_prefix = hypothesis_runtime.canonical_bytes(predictions)
        for rule in hypothesis_runtime.fixed_rule_space():
            hypothesis_runtime.validate_rule(rule)
            if rule["operator"] == "ge":
                sentinel = {
                    feature: None
                    for feature in hypothesis_runtime.ALLOWED_RULE_FEATURES
                }
                sentinel[rule["feature"]] = rule["threshold"]
                if not hypothesis_runtime._rule_matches(rule, sentinel):
                    raise RuntimeError("numeric rule direction failed")
                sentinel[rule["feature"]] = None
                if hypothesis_runtime._rule_matches(rule, sentinel):
                    raise RuntimeError("numeric missingness failed")
            else:
                sentinel = {
                    feature: None
                    for feature in hypothesis_runtime.ALLOWED_RULE_FEATURES
                }
                sentinel["source_kind"] = rule["value"]
                if not hypothesis_runtime._rule_matches(rule, sentinel):
                    raise RuntimeError("categorical rule direction failed")
            selected = hypothesis_runtime._select_addition(
                predictions, candidates, rule
            )
            output = list(predictions)
            if selected is not None:
                output.append(hypothesis_runtime._addition_prediction(selected))
            if hypothesis_runtime.canonical_bytes(
                output[: len(predictions)]
            ) != original_prefix:
                raise RuntimeError("V51 prefix preservation failed")
    except BaseException:
        return _result(failure_phase=PHASE_CONTRACT[3])

    try:
        upstream = candidate7_runtime.preflight_candidate7_source_bytes(
            v51_semantic_bytes,
            pool_bytes,
            v51_semantic_sha256=v51_semantic_sha256,
            pool_sha256=pool_sha256,
            hypothesis_runtime=hypothesis_runtime,
            hypothesis_source_sha256=hypothesis_source_sha256,
        )
        required_true = {
            "all_fixed_rule_forms_exercised",
            "candidate_source_rows_sanitized",
            "deterministic_geometry_source_tie_break_exercised",
            "fixed_allowed_source_kind_eligibility_exercised",
            "generic_pool_accounting_ignored",
            "missing_evidence_abstention_exercised",
            "ownership_geometry_exercised",
            "raw_candidates_ignored_for_selection",
            "recursive_forbidden_key_validation_exercised",
            "runtime_threshold_match_implies_conservative_conflict_for_bound_pairs",
            "scientific_hypothesis_exactly_preserved",
            "source_only",
            "v51_prediction_values_and_order_preserved",
        }
        if (
            not isinstance(upstream, Mapping)
            or set(upstream) != CANDIDATE7_PREFLIGHT_RESULT_KEYS
            or upstream.get("schema_version") != 1
            or upstream.get("kind") != CANDIDATE7_PREFLIGHT_RESULT_KIND
            or upstream.get("attempt_ordinal") != 7
            or upstream.get("engineering_attempt_id")
            != "CANDIDATE7_ENGINEERING_ATTEMPT_V1"
            or upstream.get("scientific_hypothesis_id")
            != SCIENTIFIC_HYPOTHESIS_ID
            or upstream.get("status") != "SOURCE_COMPATIBLE"
            or upstream.get("input_v51_semantic_sha256") != v51_semantic_sha256
            or upstream.get("input_pool_sha256") != pool_sha256
            or upstream.get("scientific_hypothesis_source_sha256")
            != SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
            or upstream.get("candidate_rule_selected") is not False
            or upstream.get("labels_used") is not False
            or upstream.get("candidate_metric_computed") is not False
            or upstream.get("model_fit_performed") is not False
            or upstream.get("holdout_score_opened") is not False
            or any(upstream.get(key) is not True for key in required_true)
        ):
            raise RuntimeError("upstream aggregate contract mismatch")
        result = _result(failure_phase=None)
        _validate_result(result)
        return result
    except BaseException:
        return _result(failure_phase=PHASE_CONTRACT[4])
