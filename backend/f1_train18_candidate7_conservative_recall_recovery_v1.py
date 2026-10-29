"""Filesystem-blind Candidate7 wrapper for the unconsumed recall hypothesis.

Candidate7 is a new engineering attempt, not a continuation of Candidate6.
Its scientific hypothesis is implemented by the exact frozen Candidate6 runtime
bytes.  Callers must independently verify and compile those bytes, then pass the
resulting module together with its exact SHA-256.  This wrapper changes only
attempt/model/result lineage.  It does not change the 32-rule grid, candidate
sanitization, source eligibility, deterministic tie-break, or 4 mm ownership
semantics.

The module is filesystem blind.  It cannot discover or open source artifacts,
labels, evaluator outputs, expected counts, CAD, filenames, or holdout data.
"""

from __future__ import annotations

import hashlib
import json
import re
from types import ModuleType
from typing import Any, Mapping


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 7
ENGINEERING_ATTEMPT_ID = "CANDIDATE7_ENGINEERING_ATTEMPT_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
SCIENTIFIC_HYPOTHESIS_SOURCE_PATH = (
    "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py"
)
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

CANDIDATE6_DISPOSITION_PATH = (
    "research/f1_aug30_train18_candidate6_terminal_disposition_v1.json"
)
CANDIDATE6_DISPOSITION_SHA256 = (
    "5849807fa82ab0755ab20b116774dccca68890ffa0cebb77e20a36060af88faf"
)
CANDIDATE6_DISPOSITION_SIZE_BYTES = 2830
CANDIDATE6_DISPOSITION_KIND = (
    "sherman_qc_f1_aug30_train18_candidate6_terminal_disposition_v1"
)

PROFILE = "F1_AUG30_TRAIN18_CANDIDATE7_CONSERVATIVE_RECALL_RECOVERY_V1"
MODEL_KIND = "sherman_qc_f1_aug30_train18_candidate7_recall_model_v1"
PASS_KIND = "sherman_qc_f1_aug30_train18_candidate7_train_gate_pass_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate7_recall_result_v1"
PREFLIGHT_RESULT_KIND = (
    "sherman_qc_f1_aug30_train18_candidate7_semantic_pool_compatibility_result_v1"
)
RESERVED_BEND_ID_PREFIX = "C7R-"
RESERVED_SOURCE_HYPOTHESIS_ID_PREFIX = "C7G-"

KNOWN_BINDINGS = {
    "candidate1_terminal_sha256": "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8",
    "candidate2_terminal_sha256": "ecdbe871364044214fc46e5b5e59095cc9b38cefe9678292f1a8462dffdf3515",
    "candidate3_terminal_sha256": "00449059f9e4cd989f3fbcb67e9fc8e7a499ad1dbca0a947deccd75d21019116",
    "candidate4_terminal_sha256": "10d6bd14ad10dfd1ace464e551cb79f65abeb3c12bc0094fb2695f61a1aa2a50",
    "candidate5_terminal_sha256": "6b7d448b1c0b16c12cc32debb61c5255dc3ddb18956f4807b0c0090fbddc9049",
    "candidate6_terminal_disposition_sha256": CANDIDATE6_DISPOSITION_SHA256,
    "custody_cache_attestation_sha256": "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71",
    "custody_label_attestation_sha256": "fe9313bbd09de945fc4da0c238a6b1f2769979e78cccc2abff29ca434dbf84ad",
    "custody_staged_validator_pass_sha256": "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433",
    "original_preregistration_sha256": "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a",
    "scientific_hypothesis_source_sha256": SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
    "sealed_split_sha256": "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185",
    "train_allowlist_sha256": "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0",
}
BINDING_KEYS = frozenset(
    {
        *KNOWN_BINDINGS,
        "candidate7_optimizer_source_sha256",
        "candidate7_optimizer_tests_sha256",
        "candidate7_preflight_lock_sha256",
        "candidate7_preflight_read_ledger_sha256",
        "candidate7_preflight_report_sha256",
        "candidate7_preflight_source_sha256",
        "candidate7_preflight_tests_sha256",
        "candidate7_preregistration_sha256",
        "candidate7_runtime_source_sha256",
        "candidate7_runtime_tests_sha256",
        "evaluator_source_sha256",
        "fit_audit_core_sha256",
        "train_cache_manifest_sha256",
        "train_label_projection_sha256",
        "train_read_ledger_sha256",
        "train_source_cache_binding_sha256",
    }
)

FALSE_INFERENCE_DECLARATIONS = (
    "audit_metadata_feature_used",
    "bend_id_feature_used",
    "cad_feature_used",
    "candidate_count_feature_used",
    "case_identity_feature_used",
    "drawing_feature_used",
    "expected_count_feature_used",
    "filename_feature_used",
    "global_count_feature_used",
    "gold_or_truth_feature_used",
    "input_list_order_feature_used",
    "label_feature_used",
    "local_multiplicity_feature_used",
    "observed_source_or_prediction_multiplicity_used",
    "part_identity_feature_used",
    "runtime_identity_feature_used",
    "semantic_internal_scan_identity_feature_used",
    "sender_count_feature_used",
    "source_hypothesis_id_feature_used",
)
FEATURE_DECLARATION_KEYS = {
    "endpoint_support_fraction": "endpoint_support_fraction_feature_used",
    "finite_support_fraction": "finite_support_fraction_feature_used",
    "line_length_mm": "line_length_feature_used",
    "normal_second_eigenvalue": "normal_second_eigenvalue_feature_used",
    "score": "score_feature_used",
    "source_kind": "source_kind_feature_used",
    "two_sided_transition_score": "two_sided_transition_score_feature_used",
}
GATE_CHECK_KEYS = frozenset(
    {
        "added_false_positives_within_preregistered_bound",
        "cache_pool_preflight_passed",
        "deployed_model_exact_bytes_match_final_rule_train_replay",
        "every_reviewer_support_tier_tp_nondecrease",
        "exact_case_count_nondecrease",
        "fixed_preregistered_search_space",
        "holdout_access_zero",
        "micro_f1_increase",
        "nested_complete_case_group_validation",
        "outer_all_18_cases_scored_exactly_once",
        "outer_train_validation_fold_disjoint",
        "per_case_exact_status_nondecrease",
        "primary_replay_exact",
        "recall_strictly_increased",
        "replay_execution_succeeded_before_publication",
        "scientific_hypothesis_exactly_preserved",
        "train18_allowlist_exact",
        "train_read_ledger_exact",
        "true_positive_nondecrease_every_case",
        "v51_prediction_values_and_order_preserved",
    }
)
MODEL_KEYS = frozenset(
    {
        "additive_only",
        "attempt_ordinal",
        "bindings",
        "deterministic_tie_break_direction",
        "deterministic_tie_break_inputs",
        "engineering_attempt_id",
        "external_launcher_authentication_required",
        "fixed_eligibility_inputs",
        "freeze_status",
        "holdout_score_opened",
        "inference_features",
        "kind",
        "line_objects_only",
        "max_added_false_positives_train18",
        "max_additions_per_case",
        "missing_evidence_abstains",
        "ownership_contract",
        "post_holdout_model_or_threshold_mutation",
        "pre_holdout_execution_anchor_created",
        "preserve_v51_prediction_values_and_order",
        "profile",
        "raw_candidates_ignored_for_selection",
        "rule",
        "schema_version",
        "scientific_hypothesis_id",
        "scientific_hypothesis_source_sha256",
        "source_only",
        "train_gate_pass_binding",
        "training_gate_passed",
        *FALSE_INFERENCE_DECLARATIONS,
        *FEATURE_DECLARATION_KEYS.values(),
    }
)
PASS_KEYS = frozenset(
    {
        "attempt_ordinal",
        "bindings",
        "checks",
        "engineering_attempt_id",
        "gate_result_value_free",
        "holdout_execution_authorized",
        "holdout_score_opened",
        "kind",
        "label_derived_metric_values_present",
        "pre_holdout_execution_anchor_created",
        "schema_version",
        "scientific_hypothesis_id",
        "selected_rule",
        "status",
        "training_gate_passed",
    }
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _parse_object(data: bytes, *, name: str, canonical_required: bool) -> dict[str, Any]:
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
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    if canonical_required and canonical_bytes(value) != data:
        raise ValueError(f"{name} must use LF-canonical JSON")
    return value


def verify_scientific_hypothesis_runtime(
    hypothesis_runtime: ModuleType, *, source_sha256: str
) -> None:
    if _require_sha(source_sha256, name="scientific hypothesis source SHA") != (
        SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
    ):
        raise ValueError("Candidate7 scientific hypothesis source binding mismatch")
    try:
        rules = list(hypothesis_runtime.fixed_rule_space())
        ownership = hypothesis_runtime._ownership_contract()
    except AttributeError as error:
        raise ValueError("Candidate7 scientific hypothesis API mismatch") from error
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
        raise ValueError("Candidate7 scientific hypothesis contract mismatch")


def fixed_rule_space(
    hypothesis_runtime: ModuleType, *, source_sha256: str
) -> tuple[dict[str, Any], ...]:
    verify_scientific_hypothesis_runtime(
        hypothesis_runtime, source_sha256=source_sha256
    )
    return tuple(dict(rule) for rule in hypothesis_runtime.fixed_rule_space())


def _validate_candidate6_disposition(data: bytes, *, expected_sha256: str) -> None:
    if expected_sha256 != CANDIDATE6_DISPOSITION_SHA256:
        raise ValueError("Candidate6 disposition external binding mismatch")
    if len(data) != CANDIDATE6_DISPOSITION_SIZE_BYTES or sha256_bytes(data) != expected_sha256:
        raise ValueError("Candidate6 disposition exact byte binding mismatch")
    value = _parse_object(
        data, name="Candidate6 disposition", canonical_required=True
    )
    if (
        value.get("schema_version") != 1
        or value.get("kind") != CANDIDATE6_DISPOSITION_KIND
        or value.get("attempt_ordinal") != 6
        or value.get("status")
        != "CONFIRMED_FAILED_ENGINEERING_SOURCE_COMPATIBILITY"
        or value.get("candidate6_continuation_authorized") is not False
        or value.get("candidate6_preflight_retry_authorized") is not False
        or value.get("candidate7_engineering_successor_authorized") is not True
        or value.get("scientific_hypothesis_consumed") is not False
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("cache_manifest_opened") is not False
        or value.get("semantic_artifact_opened") is not False
        or value.get("pool_artifact_opened") is not False
        or value.get("label_attestation_or_projection_opened") is not False
        or value.get("holdout_path_hash_status_or_score_opened") is not False
    ):
        raise ValueError("Candidate6 disposition status contract mismatch")


def _validate_bindings(
    value: Any, *, verified_runtime_source_sha256: str
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != BINDING_KEYS:
        raise ValueError("Candidate7 binding schema mismatch")
    bindings = {
        key: _require_sha(value[key], name=f"bindings.{key}")
        for key in sorted(BINDING_KEYS)
    }
    for key, expected in KNOWN_BINDINGS.items():
        if bindings[key] != expected:
            raise ValueError(f"Candidate7 immutable binding mismatch: {key}")
    if bindings["candidate7_runtime_source_sha256"] != _require_sha(
        verified_runtime_source_sha256, name="verified Candidate7 runtime SHA"
    ):
        raise ValueError("Candidate7 runtime source binding mismatch")
    return bindings


def _validate_pass(
    data: bytes,
    *,
    expected_sha256: str,
    expected_rule: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
    verified_runtime_source_sha256: str,
) -> None:
    if sha256_bytes(data) != _require_sha(expected_sha256, name="Candidate7 PASS SHA"):
        raise ValueError("Candidate7 PASS byte binding mismatch")
    value = _parse_object(data, name="Candidate7 PASS", canonical_required=True)
    if set(value) != PASS_KEYS:
        raise ValueError("Candidate7 PASS schema mismatch")
    if (
        value["schema_version"] != 1
        or value["kind"] != PASS_KIND
        or value["status"] != "PASS"
        or value["attempt_ordinal"] != 7
        or value["engineering_attempt_id"] != ENGINEERING_ATTEMPT_ID
        or value["scientific_hypothesis_id"] != SCIENTIFIC_HYPOTHESIS_ID
        or value["training_gate_passed"] is not True
        or value["holdout_execution_authorized"] is not False
        or value["holdout_score_opened"] is not False
        or value["pre_holdout_execution_anchor_created"] is not False
        or value["gate_result_value_free"] is not True
        or value["label_derived_metric_values_present"] is not False
        or canonical_bytes(value["selected_rule"])
        != canonical_bytes(dict(expected_rule))
    ):
        raise ValueError("Candidate7 PASS status/rule mismatch")
    bindings = _validate_bindings(
        value["bindings"],
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    if bindings != dict(expected_bindings):
        raise ValueError("Candidate7 model/PASS binding mismatch")
    checks = value["checks"]
    if not isinstance(checks, dict) or set(checks) != GATE_CHECK_KEYS:
        raise ValueError("Candidate7 PASS gate schema mismatch")
    if any(checks[key] is not True for key in GATE_CHECK_KEYS):
        raise ValueError("Candidate7 PASS contains a failed gate")


def _validate_model(
    model: Mapping[str, Any],
    *,
    pass_bytes: bytes,
    pass_sha256: str,
    verified_runtime_source_sha256: str,
    hypothesis_runtime: ModuleType,
    hypothesis_source_sha256: str,
) -> Mapping[str, Any]:
    verify_scientific_hypothesis_runtime(
        hypothesis_runtime, source_sha256=hypothesis_source_sha256
    )
    if not isinstance(model, Mapping) or set(model) != MODEL_KEYS:
        raise ValueError("Candidate7 model schema mismatch")
    if (
        model["schema_version"] != 1
        or model["kind"] != MODEL_KIND
        or model["profile"] != PROFILE
        or model["attempt_ordinal"] != 7
        or model["engineering_attempt_id"] != ENGINEERING_ATTEMPT_ID
        or model["scientific_hypothesis_id"] != SCIENTIFIC_HYPOTHESIS_ID
        or model["scientific_hypothesis_source_sha256"]
        != SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
        or model["training_gate_passed"] is not True
        or model["freeze_status"] != "FROZEN_AFTER_TRAIN_GATE_PASS"
        or model["holdout_score_opened"] is not False
        or model["pre_holdout_execution_anchor_created"] is not False
        or model["post_holdout_model_or_threshold_mutation"]
        != "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
        or model["source_only"] is not True
        or model["additive_only"] is not True
        or model["line_objects_only"] is not True
        or model["raw_candidates_ignored_for_selection"] is not True
        or model["preserve_v51_prediction_values_and_order"] is not True
        or model["missing_evidence_abstains"] is not True
        or model["external_launcher_authentication_required"] is not True
        or model["max_additions_per_case"] != 1
        or model["max_added_false_positives_train18"] != 2
        or model["ownership_contract"] != hypothesis_runtime._ownership_contract()
        or model["fixed_eligibility_inputs"]
        != list(hypothesis_runtime.FIXED_ELIGIBILITY_INPUTS)
        or model["deterministic_tie_break_inputs"]
        != list(hypothesis_runtime.DETERMINISTIC_TIE_BREAK_INPUTS)
        or model["deterministic_tie_break_direction"]
        != hypothesis_runtime.DETERMINISTIC_TIE_BREAK_DIRECTION
        or any(model[field] is not False for field in FALSE_INFERENCE_DECLARATIONS)
    ):
        raise ValueError("Candidate7 model status/source contract mismatch")
    rule = model["rule"]
    hypothesis_runtime.validate_rule(rule)
    expected_features = [rule["feature"]]
    if rule["feature"] != "source_kind":
        expected_features.append("source_kind")
    if model["inference_features"] != expected_features:
        raise ValueError("Candidate7 model inference feature disclosure mismatch")
    for feature, declaration in FEATURE_DECLARATION_KEYS.items():
        expected = rule["feature"] == feature or feature == "source_kind"
        if model[declaration] is not expected:
            raise ValueError("Candidate7 feature declaration mismatch")
    bindings = _validate_bindings(
        model["bindings"],
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    pass_binding = model["train_gate_pass_binding"]
    if (
        not isinstance(pass_binding, dict)
        or set(pass_binding) != {"kind", "sha256"}
        or pass_binding["kind"] != PASS_KIND
        or pass_binding["sha256"] != pass_sha256
    ):
        raise ValueError("Candidate7 PASS binding mismatch")
    _validate_pass(
        pass_bytes,
        expected_sha256=pass_sha256,
        expected_rule=rule,
        expected_bindings=bindings,
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    return rule


def _assert_no_reserved_c7_ids(predictions: list[dict[str, Any]]) -> None:
    for prediction in predictions:
        if str(prediction["bend_id"]).startswith(RESERVED_BEND_ID_PREFIX):
            raise ValueError("V51 prediction uses reserved Candidate7 bend-id namespace")
        if str(prediction["source_hypothesis_id"]).startswith(
            RESERVED_SOURCE_HYPOTHESIS_ID_PREFIX
        ):
            raise ValueError("V51 prediction uses reserved Candidate7 source-id namespace")


def _addition_prediction(candidate: Mapping[str, Any]) -> dict[str, Any]:
    projection = candidate["identity_projection"]
    digest = str(candidate["digest"])
    return {
        "bend_id": f"{RESERVED_BEND_ID_PREFIX}{digest[:20]}",
        "line_end": list(projection["line_end"]),
        "line_start": list(projection["line_start"]),
        "selection_route": "candidate7_conservative_recall_recovery_v1",
        "selector_score": None,
        "source_hypothesis_id": (
            f"{RESERVED_SOURCE_HYPOTHESIS_ID_PREFIX}{digest[:20]}"
        ),
        "source_kind": str(projection["source_kind"]),
    }


def preflight_candidate7_source_bytes(
    v51_semantic_bytes: bytes,
    pool_bytes: bytes,
    *,
    v51_semantic_sha256: str,
    pool_sha256: str,
    hypothesis_runtime: ModuleType,
    hypothesis_source_sha256: str,
) -> dict[str, Any]:
    verify_scientific_hypothesis_runtime(
        hypothesis_runtime, source_sha256=hypothesis_source_sha256
    )
    upstream = hypothesis_runtime.preflight_candidate6_source_bytes(
        v51_semantic_bytes,
        pool_bytes,
        v51_semantic_sha256=v51_semantic_sha256,
        pool_sha256=pool_sha256,
    )
    required_upstream_true = {
        "all_fixed_rule_forms_exercised",
        "candidate_source_rows_sanitized",
        "deterministic_geometry_source_tie_break_exercised",
        "fixed_allowed_source_kind_eligibility_exercised",
        "generic_pool_accounting_ignored",
        "missing_evidence_abstention_exercised",
        "ownership_geometry_exercised",
        "raw_candidates_ignored",
        "recursive_forbidden_key_validation_exercised",
        "runtime_threshold_match_implies_conservative_conflict_for_bound_pairs",
        "source_only",
        "v51_prediction_values_and_order_preserved",
    }
    if (
        upstream.get("attempt_ordinal") != 6
        or upstream.get("status") != "SOURCE_COMPATIBLE"
        or upstream.get("labels_used") is not False
        or upstream.get("model_fit_performed") is not False
        or upstream.get("holdout_score_opened") is not False
        or any(upstream.get(key) is not True for key in required_upstream_true)
    ):
        raise ValueError("Candidate7 frozen hypothesis preflight contract mismatch")
    _semantic, predictions = hypothesis_runtime._validate_semantic_bytes(
        v51_semantic_bytes, expected_sha256=v51_semantic_sha256
    )
    _assert_no_reserved_c7_ids(predictions)
    return {
        "all_fixed_rule_forms_exercised": True,
        "attempt_ordinal": 7,
        "candidate_metric_computed": False,
        "candidate_rule_selected": False,
        "candidate_source_rows_sanitized": True,
        "deterministic_geometry_source_tie_break_exercised": True,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "fixed_allowed_source_kind_eligibility_exercised": True,
        "generic_pool_accounting_ignored": True,
        "holdout_score_opened": False,
        "input_pool_sha256": pool_sha256,
        "input_v51_semantic_sha256": v51_semantic_sha256,
        "kind": PREFLIGHT_RESULT_KIND,
        "labels_used": False,
        "missing_evidence_abstention_exercised": True,
        "model_fit_performed": False,
        "ownership_geometry_exercised": True,
        "raw_candidates_ignored_for_selection": True,
        "recursive_forbidden_key_validation_exercised": True,
        "runtime_threshold_match_implies_conservative_conflict_for_bound_pairs": True,
        "schema_version": 1,
        "scientific_hypothesis_exactly_preserved": True,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_source_sha256": SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
        "source_only": True,
        "status": "SOURCE_COMPATIBLE",
        "v51_prediction_values_and_order_preserved": True,
    }


def apply_candidate7(
    v51_semantic_bytes: bytes,
    pool_bytes: bytes,
    model: Mapping[str, Any],
    train_gate_pass_bytes: bytes,
    candidate6_disposition_bytes: bytes,
    *,
    v51_semantic_sha256: str,
    pool_sha256: str,
    train_gate_pass_sha256: str,
    candidate6_disposition_sha256: str,
    verified_runtime_source_sha256: str,
    hypothesis_runtime: ModuleType,
    hypothesis_source_sha256: str,
) -> dict[str, Any]:
    """Apply the frozen hypothesis after external artifact authentication."""

    _validate_candidate6_disposition(
        candidate6_disposition_bytes,
        expected_sha256=candidate6_disposition_sha256,
    )
    rule = _validate_model(
        model,
        pass_bytes=train_gate_pass_bytes,
        pass_sha256=train_gate_pass_sha256,
        verified_runtime_source_sha256=verified_runtime_source_sha256,
        hypothesis_runtime=hypothesis_runtime,
        hypothesis_source_sha256=hypothesis_source_sha256,
    )
    if model["bindings"]["candidate6_terminal_disposition_sha256"] != (
        candidate6_disposition_sha256
    ):
        raise ValueError("Candidate7 model does not bind Candidate6 disposition")
    _semantic, predictions = hypothesis_runtime._validate_semantic_bytes(
        v51_semantic_bytes, expected_sha256=v51_semantic_sha256
    )
    _assert_no_reserved_c7_ids(predictions)
    _pool, candidates = hypothesis_runtime._validate_pool_bytes(
        pool_bytes, expected_sha256=pool_sha256
    )
    original_prefix = canonical_bytes(predictions)
    selected = hypothesis_runtime._select_addition(predictions, candidates, rule)
    output = list(predictions)
    if selected is not None:
        output.append(_addition_prediction(selected))
    if canonical_bytes(output[: len(predictions)]) != original_prefix:
        raise RuntimeError("Candidate7 changed V51 prediction values/order")
    output_ids = [str(prediction["bend_id"]) for prediction in output]
    if len(output_ids) != len(set(output_ids)):
        raise RuntimeError("Candidate7 produced duplicate prediction bend_id")
    result = {
        "addition_made": selected is not None,
        "additive_only": True,
        "attempt_ordinal": 7,
        "candidate6_terminal_disposition_sha256": candidate6_disposition_sha256,
        "candidate7_model_sha256": sha256_bytes(canonical_bytes(dict(model))),
        "deterministic_tie_break_direction": (
            hypothesis_runtime.DETERMINISTIC_TIE_BREAK_DIRECTION
        ),
        "deterministic_tie_break_inputs": list(
            hypothesis_runtime.DETERMINISTIC_TIE_BREAK_INPUTS
        ),
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "external_launcher_authentication_required": True,
        "fixed_eligibility_inputs": list(hypothesis_runtime.FIXED_ELIGIBILITY_INPUTS),
        "holdout_score_opened": False,
        "input_pool_sha256": pool_sha256,
        "input_v51_semantic_sha256": v51_semantic_sha256,
        "kind": RESULT_KIND,
        "labels_used": False,
        "line_objects_only": True,
        "max_additions_per_case": 1,
        "missing_evidence_abstains": True,
        "ownership_contract": hypothesis_runtime._ownership_contract(),
        "predictions": output,
        "profile": PROFILE,
        "promotion_eligible": False,
        "raw_candidates_ignored_for_selection": True,
        "research_only": True,
        "rule": dict(rule),
        "scan_identity_authority": "ATTESTED_CACHE_MANIFEST_ONLY",
        "schema_version": 1,
        "scientific_hypothesis_exactly_preserved": True,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_source_sha256": hypothesis_source_sha256,
        "selected_candidate_geometry_digest": (
            None if selected is None else selected["digest"]
        ),
        "source_only": True,
        "train_gate_pass_sha256": train_gate_pass_sha256,
        "training_gate_passed": True,
        "v51_prediction_values_and_order_preserved": True,
        **{field: False for field in FALSE_INFERENCE_DECLARATIONS},
        **{
            declaration: rule["feature"] == feature or feature == "source_kind"
            for feature, declaration in FEATURE_DECLARATION_KEYS.items()
        },
    }
    result["candidate7_result_payload_sha256"] = sha256_bytes(
        canonical_bytes(result)
    )
    return result
