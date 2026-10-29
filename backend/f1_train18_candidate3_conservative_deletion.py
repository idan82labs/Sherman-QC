"""Fail-closed runtime for TRAIN18 Candidate3 conservative deletion.

Candidate3 can only delete frozen V51 predictions. Runtime execution requires a
distinct, externally byte-hash-bound TRAIN gate PASS artifact. The runtime uses
no labels, counts, case identity, filenames, CAD, drawings, or expected counts.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


PROFILE = "F1_AUG30_TRAIN18_CANDIDATE3_CONSERVATIVE_DELETION_V1"
MODEL_KIND = "sherman_qc_f1_train18_candidate3_conservative_deletion_model_v1"
PASS_KIND = "sherman_qc_f1_train18_candidate3_train_gate_pass_v1"
RESULT_KIND = "sherman_qc_f1_train18_candidate3_conservative_deletion_result_v1"
PREFLIGHT_RESULT_KIND = "sherman_qc_f1_train18_candidate3_source_compatibility_preflight_v1"
ATTEMPT_ORDINAL = 3

ALLOWED_NUMERIC_FEATURES = (
    "line_length_mm",
    "selector_score",
    "normal_second_eigenvalue",
    "p90_line_distance_mm",
    "median_line_distance_mm",
    "two_sided_transition_score",
    "two_sided_normal_angle_deg",
    "two_sided_support_balance",
    "finite_support_fraction",
    "endpoint_support_fraction",
    "transverse_angular_bin_fraction",
    "surface_gap_mm",
)
ALLOWED_FEATURES = frozenset(ALLOWED_NUMERIC_FEATURES) | {"source_kind"}
NUMERIC_OPERATORS = frozenset({"le", "ge"})
PREDICTION_KEYS = frozenset(
    {
        "bend_id",
        "line_start",
        "line_end",
        "selection_route",
        "selector_score",
        "source_hypothesis_id",
        "source_kind",
    }
)
BINDING_KEYS = frozenset(
    {
        "candidate1_terminal_sha256",
        "candidate2_terminal_sha256",
        "split_sha256",
        "train_allowlist_sha256",
        "original_prereg_sha256",
        "candidate3_prereg_sha256",
        "custody_label_attestation_sha256",
        "custody_cache_attestation_sha256",
        "runtime_source_sha256",
        "optimizer_source_sha256",
        "tests_source_sha256",
    }
)
CHECK_KEYS = frozenset(
    {
        "nested_oof_f1_increase",
        "nested_oof_true_positive_nondecrease",
        "nested_oof_false_positive_decrease",
        "nested_oof_exact_case_nondecrease",
        "nested_oof_support_tier_tp_nondecrease",
        "primary_replay_exact",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def runtime_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _strict_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_number(row: dict[str, Any], key: str) -> float | None:
    if key not in row or row[key] is None:
        return None
    return _strict_number(row[key], key)


def _validate_prediction(prediction: dict[str, Any]) -> None:
    if set(prediction) != PREDICTION_KEYS:
        raise ValueError("V51 prediction schema mismatch")
    for key in ("bend_id", "selection_route", "source_hypothesis_id", "source_kind"):
        if not isinstance(prediction[key], str) or not prediction[key]:
            raise ValueError(f"V51 prediction {key} must be a nonempty string")
    for endpoint_key in ("line_start", "line_end"):
        endpoint = prediction[endpoint_key]
        if not isinstance(endpoint, list) or len(endpoint) != 3:
            raise ValueError(f"V51 prediction {endpoint_key} must have three coordinates")
        for index, coordinate in enumerate(endpoint):
            _strict_number(coordinate, f"{endpoint_key}[{index}]")
    _optional_number(prediction, "selector_score")
    _line_length(prediction)


def _line_length(prediction: dict[str, Any]) -> float:
    start = [float(value) for value in prediction["line_start"]]
    end = [float(value) for value in prediction["line_end"]]
    result = math.dist(start, end)
    if not math.isfinite(result) or result <= 1e-9:
        raise ValueError("V51 prediction line must have finite nonzero length")
    return result


def extract_prediction_features(
    prediction: dict[str, Any], candidate: dict[str, Any] | None
) -> dict[str, float | str | None]:
    """Return the complete allowlisted inference feature surface."""

    _validate_prediction(prediction)
    row = candidate or {}
    return {
        "line_length_mm": _line_length(prediction),
        "selector_score": _optional_number(prediction, "selector_score"),
        "normal_second_eigenvalue": _optional_number(row, "normal_second_eigenvalue"),
        "p90_line_distance_mm": _optional_number(row, "p90_line_distance_mm"),
        "median_line_distance_mm": _optional_number(row, "median_line_distance_mm"),
        "two_sided_transition_score": _optional_number(row, "two_sided_transition_score"),
        "two_sided_normal_angle_deg": _optional_number(row, "two_sided_normal_angle_deg"),
        "two_sided_support_balance": _optional_number(row, "two_sided_support_balance"),
        "finite_support_fraction": _optional_number(row, "finite_support_fraction"),
        "endpoint_support_fraction": _optional_number(row, "endpoint_support_fraction"),
        "transverse_angular_bin_fraction": _optional_number(
            row, "transverse_angular_bin_fraction"
        ),
        "surface_gap_mm": _optional_number(row, "min_surface_gap"),
        "source_kind": prediction["source_kind"],
    }


def _candidate_map(pool: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = pool.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("candidate pool must contain a candidates list")
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, dict):
            raise ValueError("candidate pool rows must be objects")
        candidate_id = value.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate pool row is missing candidate_id")
        if candidate_id in result and _canonical(result[candidate_id]) != _canonical(value):
            raise ValueError(f"conflicting candidate rows for {candidate_id}")
        result[candidate_id] = deepcopy(value)
    return result


def validate_rule(rule: dict[str, Any]) -> None:
    if not isinstance(rule, dict):
        raise ValueError("deletion_rule must be an object")
    feature = rule.get("feature")
    operator = rule.get("operator")
    if not isinstance(feature, str) or feature not in ALLOWED_FEATURES:
        raise ValueError(f"deletion rule uses prohibited or unknown feature: {feature}")
    if feature == "source_kind":
        if set(rule) != {"feature", "operator", "value"} or operator != "eq":
            raise ValueError("source_kind rule must be exact equality")
        if not isinstance(rule.get("value"), str) or not rule["value"]:
            raise ValueError("source_kind rule requires a nonempty string value")
        return
    if set(rule) != {"feature", "operator", "threshold"} or operator not in NUMERIC_OPERATORS:
        raise ValueError("numeric deletion rule requires le/ge and one threshold")
    _strict_number(rule.get("threshold"), "deletion_rule.threshold")


def _validate_bindings(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != BINDING_KEYS:
        raise ValueError("Candidate3 binding schema mismatch")
    result = {key: _require_hash(value[key], f"bindings.{key}") for key in sorted(BINDING_KEYS)}
    if result["runtime_source_sha256"] != runtime_source_sha256():
        raise ValueError("Candidate3 runtime source hash does not match the frozen model binding")
    return result


def _validate_support_tiers(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"1", "2", "3"}:
        raise ValueError(f"{field} support-tier schema mismatch")
    return {key: _strict_int(value[key], f"{field}.{key}") for key in ("1", "2", "3")}


def _validate_metric_block(value: Any, field: str) -> dict[str, Any]:
    keys = {
        "true_positive",
        "false_positive",
        "false_negative",
        "f1",
        "exact_case_count",
        "support_tier_tp",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} metric schema mismatch")
    return {
        "true_positive": _strict_int(value["true_positive"], f"{field}.true_positive"),
        "false_positive": _strict_int(value["false_positive"], f"{field}.false_positive"),
        "false_negative": _strict_int(value["false_negative"], f"{field}.false_negative"),
        "f1": _strict_number(value["f1"], f"{field}.f1"),
        "exact_case_count": _strict_int(value["exact_case_count"], f"{field}.exact_case_count"),
        "support_tier_tp": _validate_support_tiers(
            value["support_tier_tp"], f"{field}.support_tier_tp"
        ),
    }


def _validate_pass_artifact(
    artifact: dict[str, Any], model_rule: dict[str, Any], bindings: dict[str, str]
) -> None:
    required = {
        "schema_version",
        "kind",
        "status",
        "attempt_ordinal",
        "holdout_score_opened",
        "selected_deletion_rule",
        "bindings",
        "checks",
        "nested_oof_metrics",
        "fit_audit_binding",
        "train_read_ledger_binding",
    }
    if not isinstance(artifact, dict) or set(artifact) != required:
        raise ValueError("Candidate3 TRAIN PASS artifact schema mismatch")
    if artifact["schema_version"] != 1 or artifact["kind"] != PASS_KIND:
        raise ValueError("Candidate3 TRAIN PASS kind/version mismatch")
    if artifact["status"] != "PASS" or artifact["attempt_ordinal"] != ATTEMPT_ORDINAL:
        raise ValueError("Candidate3 TRAIN PASS artifact is not executable")
    if artifact["holdout_score_opened"] is not False:
        raise ValueError("Candidate3 TRAIN PASS artifact observed holdout state")
    for binding_key, expected_kind in (
        ("fit_audit_binding", "sherman_qc_f1_aug30_train18_candidate3_fit_audit_v1"),
        ("train_read_ledger_binding", "sherman_qc_f1_aug30_train18_candidate3_read_ledger_v1"),
    ):
        binding = artifact[binding_key]
        if not isinstance(binding, dict) or set(binding) != {"kind", "sha256"}:
            raise ValueError(f"Candidate3 {binding_key} schema mismatch")
        if binding["kind"] != expected_kind:
            raise ValueError(f"Candidate3 {binding_key} kind mismatch")
        _require_hash(binding["sha256"], f"{binding_key}.sha256")
    if _canonical(artifact["selected_deletion_rule"]) != _canonical(model_rule):
        raise ValueError("Candidate3 model rule differs from TRAIN PASS rule")
    if _validate_bindings(artifact["bindings"]) != bindings:
        raise ValueError("Candidate3 model and TRAIN PASS bindings differ")
    checks = artifact["checks"]
    if not isinstance(checks, dict) or set(checks) != CHECK_KEYS:
        raise ValueError("Candidate3 TRAIN PASS check schema mismatch")
    if any(checks[key] is not True for key in CHECK_KEYS):
        raise ValueError("Candidate3 TRAIN PASS contains a failed or non-boolean check")

    metrics = artifact["nested_oof_metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {"reference", "candidate", "delta"}:
        raise ValueError("Candidate3 nested OOF metric container mismatch")
    reference = _validate_metric_block(metrics["reference"], "reference")
    candidate = _validate_metric_block(metrics["candidate"], "candidate")
    delta = _validate_metric_block(metrics["delta"], "delta")
    for key in ("true_positive", "false_positive", "false_negative", "exact_case_count"):
        if candidate[key] - reference[key] != delta[key]:
            raise ValueError(f"Candidate3 nested OOF {key} delta is inconsistent")
    if abs((candidate["f1"] - reference["f1"]) - delta["f1"]) > 1e-6:
        raise ValueError("Candidate3 nested OOF F1 delta is inconsistent")
    for tier in ("1", "2", "3"):
        if candidate["support_tier_tp"][tier] - reference["support_tier_tp"][tier] != delta[
            "support_tier_tp"
        ][tier]:
            raise ValueError("Candidate3 nested OOF support-tier delta is inconsistent")
    if (
        delta["true_positive"] != 0
        or delta["false_positive"] >= 0
        or delta["false_negative"] != 0
        or delta["f1"] <= 0.0
        or delta["exact_case_count"] < 0
        or min(delta["support_tier_tp"].values()) < 0
    ):
        raise ValueError("Candidate3 nested OOF metrics do not satisfy conservative deletion gates")


def _validate_model(
    model: dict[str, Any], pass_artifact: dict[str, Any], pass_artifact_sha256: str
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "profile",
        "attempt_ordinal",
        "holdout_score_opened",
        "deletion_rule",
        "source_only",
        "uses_prediction_or_candidate_multiplicity",
        "forbidden_inference_inputs",
        "bindings",
        "train_gate_pass_binding",
    }
    if not isinstance(model, dict) or set(model) != required:
        raise ValueError("Candidate3 model schema mismatch")
    if model["schema_version"] != 1 or model["kind"] != MODEL_KIND or model["profile"] != PROFILE:
        raise ValueError("Candidate3 model kind/profile/version mismatch")
    if model["attempt_ordinal"] != ATTEMPT_ORDINAL or model["holdout_score_opened"] is not False:
        raise ValueError("Candidate3 model attempt or holdout state mismatch")
    if model["source_only"] is not True:
        raise ValueError("Candidate3 model must declare source_only=true")
    if model["uses_prediction_or_candidate_multiplicity"] is not False:
        raise ValueError("Candidate3 forbids prediction/candidate multiplicity")
    if model["forbidden_inference_inputs"] != []:
        raise ValueError("Candidate3 model observed a forbidden inference input")
    validate_rule(model["deletion_rule"])
    bindings = _validate_bindings(model["bindings"])
    pass_binding = model["train_gate_pass_binding"]
    if not isinstance(pass_binding, dict) or set(pass_binding) != {"kind", "sha256"}:
        raise ValueError("Candidate3 TRAIN PASS binding schema mismatch")
    if pass_binding["kind"] != PASS_KIND:
        raise ValueError("Candidate3 TRAIN PASS binding kind mismatch")
    bound_sha = _require_hash(pass_binding["sha256"], "train_gate_pass_binding.sha256")
    external_sha = _require_hash(pass_artifact_sha256, "train_gate_pass_artifact_sha256")
    if bound_sha != external_sha or canonical_sha256(pass_artifact) != external_sha:
        raise ValueError("Candidate3 TRAIN PASS artifact hash mismatch")
    _validate_pass_artifact(pass_artifact, model["deletion_rule"], bindings)
    return model["deletion_rule"]


def _matches(rule: dict[str, Any], features: dict[str, float | str | None]) -> bool:
    feature = rule["feature"]
    value = features[feature]
    if feature == "source_kind":
        return value == rule["value"]
    if value is None:
        return False
    number = float(value)
    threshold = float(rule["threshold"])
    return number <= threshold if rule["operator"] == "le" else number >= threshold


def _validated_feature_rows(
    v51_result: dict[str, Any], pool: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, float | str | None]]]:
    for flag in ("labels_used", "cad_used", "part_identity_used", "expected_count_used"):
        if v51_result.get(flag) is not False:
            raise ValueError(f"Candidate3 source-only runtime requires {flag}=false")
    candidate_by_id = _candidate_map(pool)
    predictions = v51_result.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("V51 result must contain a predictions list")
    prediction_ids: set[str] = set()
    rows = []
    for value in predictions:
        if not isinstance(value, dict):
            raise ValueError("V51 predictions must be objects")
        prediction = deepcopy(value)
        _validate_prediction(prediction)
        prediction_id = prediction["bend_id"]
        if prediction_id in prediction_ids:
            raise ValueError(f"duplicate V51 prediction bend_id: {prediction_id}")
        prediction_ids.add(prediction_id)
        hypothesis_id = prediction["source_hypothesis_id"]
        rows.append(
            (
                prediction,
                extract_prediction_features(prediction, candidate_by_id.get(hypothesis_id)),
            )
        )
    return rows


def preflight_candidate3_source_compatibility(
    v51_result: dict[str, Any],
    pool: dict[str, Any],
    *,
    v51_artifact_sha256: str,
    pool_artifact_sha256: str,
) -> dict[str, Any]:
    """Exercise the exact runtime feature path without labels or a deletion rule."""

    input_v51_sha256 = _require_hash(v51_artifact_sha256, "v51_artifact_sha256")
    input_pool_sha256 = _require_hash(pool_artifact_sha256, "pool_artifact_sha256")
    rows = _validated_feature_rows(v51_result, pool)
    for _prediction, features in rows:
        for feature in ALLOWED_NUMERIC_FEATURES:
            value = features[feature]
            threshold = 0.0 if value is None else float(value)
            for operator in ("le", "ge"):
                rule = {"feature": feature, "operator": operator, "threshold": threshold}
                validate_rule(rule)
                matched = _matches(rule, features)
                if matched is not (value is not None):
                    raise ValueError(f"Candidate3 {feature}/{operator} rule path is inconsistent")
        own_source_rule = {
            "feature": "source_kind",
            "operator": "eq",
            "value": str(features["source_kind"]),
        }
        absent_source_rule = {
            "feature": "source_kind",
            "operator": "eq",
            "value": "__candidate3_source_compatibility_absent__",
        }
        validate_rule(own_source_rule)
        validate_rule(absent_source_rule)
        if not _matches(own_source_rule, features) or _matches(absent_source_rule, features):
            raise ValueError("Candidate3 categorical rule path is inconsistent")
    result = {
        "schema_version": 1,
        "kind": PREFLIGHT_RESULT_KIND,
        "profile": PROFILE,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "status": "SOURCE_COMPATIBLE",
        "input_v51_artifact_sha256": input_v51_sha256,
        "input_pool_artifact_sha256": input_pool_sha256,
        "feature_extraction_complete": True,
        "all_numeric_le_ge_and_categorical_paths_exercised": True,
        "semantic_internal_scan_identity_trusted": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "expected_count_used": False,
        "case_identity_used": False,
        "filename_used": False,
        "sender_count_used": False,
        "drawing_used": False,
        "source_only": True,
        "uses_prediction_or_candidate_multiplicity": False,
        "holdout_score_opened": False,
        "research_only": True,
        "promotion_eligible": False,
    }
    result["preflight_result_sha256"] = canonical_sha256(result)
    return result


def apply_candidate3(
    v51_result: dict[str, Any],
    pool: dict[str, Any],
    model: dict[str, Any],
    train_gate_pass_artifact: dict[str, Any],
    *,
    train_gate_pass_artifact_sha256: str,
    v51_artifact_sha256: str,
    pool_artifact_sha256: str,
) -> dict[str, Any]:
    """Execute only a fully bound Candidate3 model with a distinct TRAIN PASS."""

    input_v51_sha256 = _require_hash(v51_artifact_sha256, "v51_artifact_sha256")
    input_pool_sha256 = _require_hash(pool_artifact_sha256, "pool_artifact_sha256")
    rule = _validate_model(model, train_gate_pass_artifact, train_gate_pass_artifact_sha256)
    feature_rows = _validated_feature_rows(v51_result, pool)
    retained: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for prediction, features in feature_rows:
        prediction_id = prediction["bend_id"]
        hypothesis_id = prediction["source_hypothesis_id"]
        if not _matches(rule, features):
            retained.append(prediction)
            continue
        feature = rule["feature"]
        audit.append(
            {
                "prediction_id": prediction_id,
                "source_hypothesis_id": hypothesis_id,
                "feature": feature,
                "feature_value": features[feature],
                "reason": "frozen_train_nested_rule",
            }
        )
    model_sha256 = canonical_sha256(model)
    result = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "profile": PROFILE,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "input_v51_artifact_sha256": input_v51_sha256,
        "input_pool_artifact_sha256": input_pool_sha256,
        "candidate3_model_sha256": model_sha256,
        "train_gate_pass_artifact_sha256": train_gate_pass_artifact_sha256,
        "candidate_count": len(retained),
        "predictions": retained,
        "deletion_audit": audit,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "expected_count_used": False,
        "case_identity_used": False,
        "filename_used": False,
        "sender_count_used": False,
        "drawing_used": False,
        "source_only": True,
        "uses_prediction_or_candidate_multiplicity": False,
        "holdout_score_opened": False,
        "research_only": True,
        "promotion_eligible": False,
    }
    result["candidate3_result_sha256"] = canonical_sha256(
        {
            "profile": PROFILE,
            "input_v51_artifact_sha256": input_v51_sha256,
            "input_pool_artifact_sha256": input_pool_sha256,
            "model_sha256": model_sha256,
            "train_gate_pass_artifact_sha256": train_gate_pass_artifact_sha256,
            "predictions": retained,
            "deletion_audit": audit,
        }
    )
    return result
