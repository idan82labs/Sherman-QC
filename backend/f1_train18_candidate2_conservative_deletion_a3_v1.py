"""Frozen source-only runtime for TRAIN18 candidate2 engineering amendment A3.

The successor can only delete V51 incumbents.  It never creates, refits, moves,
or truncates a line.  Two independently selectable mechanisms are supported:

* a fixed sidecar-burst guard; and
* a portable incumbent TP-probability veto.

The module is filesystem- and truth-blind.  A model is executable only when it
contains an explicit passing nested-TRAIN gate binding.  Failed TRAIN attempts
therefore cannot be turned into a holdout-executable model.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_train18_frozen_source_only_candidate_v1 import (
    _validate_classifier,
    build_observations,
    canonical_bytes,
    classifier_probabilities,
    endpoints_from_record,
    feature_matrix,
    feature_names,
    validate_source_input,
)


SCHEMA_VERSION = 1
MODEL_KIND = "sherman_qc_f1_aug30_train18_candidate2_a3_conservative_deletion_model_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate2_a3_conservative_deletion_result_v1"
PREREG_KIND = "sherman_qc_f1_aug30_train18_candidate2_a3_preregistration_v1"

POLICY_MODES = ("EXPLOSION_ONLY", "VETO_ONLY", "EXPLOSION_THEN_VETO")
EXPLOSION_RULES = (
    "DROP_ALL_SIDECARS_IF_AT_LEAST_2",
    "DROP_ALL_SIDECARS_IF_AT_LEAST_3",
)
SIDECAR_SOURCE_KIND = "v41_fragment_tolerant_parent_bridge_sidecar"
SIDECAR_ROUTE_TOKEN = "fragment_tolerant_parent_bridge"

HASH_BINDINGS = (
    "candidate2_a2_prefit_disposition_sha256",
    "candidate2_a2_runtime_code_sha256",
    "candidate2_a2_optimizer_code_sha256",
    "candidate2_a2_tests_code_sha256",
    "candidate2_a2_p0_preflight_code_sha256",
    "candidate2_a2_p0_failure_sha256",
    "candidate2_a2_p1_preflight_code_sha256",
    "candidate2_a2_p1_tests_code_sha256",
    "candidate2_a2_p1_failure_sha256",
    "candidate2_a1_abort_sha256",
    "candidate2_a1_preregistration_sha256",
    "candidate2_a1_runtime_code_sha256",
    "candidate2_a1_optimizer_code_sha256",
    "candidate2_a1_tests_code_sha256",
    "candidate2_a0_abort_sha256",
    "candidate2_a0_preregistration_sha256",
    "candidate1_terminal_sha256",
    "candidate2_preregistration_sha256",
    "candidate2_a3_preregistration_sha256",
    "split_sha256",
    "train_allowlist_sha256",
    "original_preregistration_sha256",
    "label_attestation_sha256",
    "cache_attestation_sha256",
    "custody_pass_sha256",
    "base_runtime_code_sha256",
    "base_optimizer_code_sha256",
    "runtime_code_sha256",
    "optimizer_code_sha256",
    "tests_code_sha256",
    "fit_audit_core_sha256",
    "train_gate_result_sha256",
    "label_projection_sha256",
    "train_cache_manifest_sha256",
    "train_source_cache_binding_sha256",
    "schema_preflight_pass_sha256",
    "schema_preflight_code_sha256",
)


def _lower_sha(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite in [0,1]")
    return number


def _is_sidecar(row: Mapping[str, Any]) -> bool:
    kind = row["source_kind"]
    route = row["selection_route"]
    assert isinstance(kind, str) and isinstance(route, str)
    return kind == SIDECAR_SOURCE_KIND or SIDECAR_ROUTE_TOKEN in route


def validate_incumbent_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the incumbent stream before any deletion decision.

    The burst policy may be the only active arm, so it cannot rely on the
    classifier feature builder to validate prediction structure for it.
    """

    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise ValueError("V51 predictions must be a sequence of objects")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, value in enumerate(predictions):
        if not isinstance(value, Mapping):
            raise ValueError(f"V51 prediction {index} must be an object")
        row = deepcopy(dict(value))
        bend_id = row.get("bend_id")
        if not isinstance(bend_id, str) or not bend_id:
            raise ValueError(f"V51 prediction {index} has no nonempty bend_id")
        if bend_id in identities:
            raise ValueError(f"V51 prediction bend_id is duplicated: {bend_id}")
        identities.add(bend_id)
        endpoints_from_record(row)
        for field in ("source_kind", "selection_route"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"V51 prediction {bend_id} has invalid {field}")
        score_available = "selector_score" in row and row.get("selector_score") is not None
        if score_available:
            score = row.get("selector_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"prediction {bend_id} selector_score must be numeric")
            if not math.isfinite(float(score)):
                raise ValueError(f"prediction {bend_id} selector_score must be finite")
        rows.append(row)
    return rows


def incumbent_probability_map(
    predictions: Sequence[Mapping[str, Any]], probabilities: Sequence[float]
) -> dict[str, float]:
    """Use the one frozen alignment and rounding rule in fit and inference."""

    rows = validate_incumbent_predictions(predictions)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("candidate2 probability/incumbent alignment is invalid")
    return {
        str(row["bend_id"]): round(float(probability), 12)
        for row, probability in zip(rows, values)
    }


def apply_explosion_guard(
    predictions: Sequence[Mapping[str, Any]], rule: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply a fixed, source-route-only burst rule while preserving all others."""

    if rule not in EXPLOSION_RULES:
        raise ValueError(f"unsupported explosion rule: {rule}")
    rows = validate_incumbent_predictions(predictions)
    sidecar_indices = [index for index, row in enumerate(rows) if _is_sidecar(row)]
    trigger = 2 if rule == "DROP_ALL_SIDECARS_IF_AT_LEAST_2" else 3
    if len(sidecar_indices) < trigger:
        return rows, []
    removed = set(sidecar_indices)
    removed_ids = [str(rows[index].get("bend_id") or "") for index in sorted(removed)]
    return [row for index, row in enumerate(rows) if index not in removed], removed_ids


def validate_model(model: Mapping[str, Any]) -> dict[str, Any]:
    if model.get("schema_version") != SCHEMA_VERSION or model.get("kind") != MODEL_KIND:
        raise ValueError("candidate2 model schema/kind mismatch")
    if model.get("attempt_ordinal") != 2:
        raise ValueError("candidate2 model attempt ordinal mismatch")
    if model.get("engineering_amendment") != 3:
        raise ValueError("candidate2 engineering amendment mismatch")
    if model.get("training_gate_passed") is not True:
        raise ValueError("candidate2 is not executable without a passing TRAIN gate")
    if model.get("holdout_score_opened") is not False:
        raise ValueError("candidate2 model was not frozen before holdout")
    if model.get("freeze_status") != "FROZEN_AFTER_TRAIN_GATE_PASS":
        raise ValueError("candidate2 freeze status mismatch")
    for field in (
        "source_only",
        "v51_fallback_default",
        "deletion_only",
    ):
        if model.get(field) is not True:
            raise ValueError(f"candidate2 model must declare {field}=true")
    for field in (
        "case_id_feature_used",
        "runtime_id_feature_used",
        "filename_feature_used",
        "cad_feature_used",
    ):
        if model.get(field) is not False:
            raise ValueError(f"candidate2 model must declare {field}=false")
    for field in HASH_BINDINGS:
        _lower_sha(model.get(field), name=field)
    if model.get("train_gate_result_kind") != (
        "sherman_qc_f1_aug30_train18_candidate2_a3_train_gate_pass_v1"
    ):
        raise ValueError("candidate2 TRAIN gate PASS kind mismatch")
    if model.get("train_gate_status") != "PASS":
        raise ValueError("candidate2 TRAIN gate PASS status mismatch")
    profile = model.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("candidate2 profile is missing")
    mode = str(profile.get("policy_mode") or "")
    if mode not in POLICY_MODES:
        raise ValueError("candidate2 policy mode is invalid")
    expected_profile_fields = {
        "EXPLOSION_ONLY": {"policy_mode", "explosion_rule"},
        "VETO_ONLY": {
            "policy_mode",
            "learner",
            "representation",
            "veto_keep_threshold",
        },
        "EXPLOSION_THEN_VETO": {
            "policy_mode",
            "explosion_rule",
            "learner",
            "representation",
            "veto_keep_threshold",
        },
    }
    if set(profile) != expected_profile_fields[mode]:
        raise ValueError("candidate2 profile fields do not match the locked policy mode")
    explosion_rule = profile.get("explosion_rule")
    if mode in ("EXPLOSION_ONLY", "EXPLOSION_THEN_VETO"):
        if explosion_rule not in EXPLOSION_RULES:
            raise ValueError("candidate2 explosion rule is invalid")
    elif explosion_rule is not None:
        raise ValueError("VETO_ONLY must not carry an explosion rule")
    representation = profile.get("representation")
    classifier = model.get("classifier")
    if mode in ("VETO_ONLY", "EXPLOSION_THEN_VETO"):
        if not isinstance(profile.get("learner"), str) or not profile["learner"]:
            raise ValueError("candidate2 veto learner name is invalid")
        expected = feature_names(str(representation))
        if tuple(model.get("feature_names") or ()) != expected:
            raise ValueError("candidate2 feature contract mismatch")
        _probability(profile.get("veto_keep_threshold"), name="veto keep threshold")
        if not isinstance(classifier, dict):
            raise ValueError("candidate2 veto classifier is missing")
        _validate_classifier(classifier, len(expected))
    else:
        if representation is not None or classifier is not None or model.get("feature_names") not in (None, []):
            raise ValueError("EXPLOSION_ONLY must not carry veto model state")
    uses_multiplicity = mode in ("EXPLOSION_ONLY", "EXPLOSION_THEN_VETO")
    if model.get("observed_source_multiplicity_used") is not uses_multiplicity:
        raise ValueError("observed source multiplicity declaration mismatch")
    if model.get("classifier_count_feature_used") is not False:
        raise ValueError("classifier count features are forbidden")
    for field in (
        "expected_count_used",
        "sender_count_used",
        "gold_count_used",
        "label_count_used_at_inference",
    ):
        if model.get(field) is not False:
            raise ValueError(f"candidate2 model must declare {field}=false")
    return dict(profile)


def compose_candidate2(
    v51_result: Mapping[str, Any],
    pool: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the passing deletion-only candidate2 model to one source case."""

    validate_source_input(v51_result)
    profile = validate_model(model)
    predictions = v51_result.get("predictions")
    rows = validate_incumbent_predictions(predictions)
    removed_by_explosion: list[str] = []
    if profile["policy_mode"] in ("EXPLOSION_ONLY", "EXPLOSION_THEN_VETO"):
        rows, removed_by_explosion = apply_explosion_guard(rows, str(profile["explosion_rule"]))

    removed_by_veto: list[str] = []
    if profile["policy_mode"] in ("VETO_ONLY", "EXPLOSION_THEN_VETO"):
        observations, raw = build_observations(
            v51_result,
            pool,
            universe="incumbent_only",
        )
        matrix = feature_matrix(
            observations,
            raw,
            representation=str(profile["representation"]),
        )
        probabilities = classifier_probabilities(matrix, model["classifier"])
        probability_by_id = incumbent_probability_map(predictions, probabilities)
        threshold = _probability(
            profile["veto_keep_threshold"], name="veto keep threshold"
        )
        retained: list[dict[str, Any]] = []
        for row in rows:
            bend_id = str(row.get("bend_id") or "")
            probability = probability_by_id.get(bend_id)
            if probability is None:
                raise RuntimeError("candidate2 veto lost incumbent identity alignment")
            if probability >= threshold:
                retained.append(row)
            else:
                removed_by_veto.append(bend_id)
        rows = retained

    original_by_id = {
        str(row.get("bend_id") or ""): json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in predictions
    }
    for row in rows:
        bend_id = str(row.get("bend_id") or "")
        if original_by_id.get(bend_id) != json.dumps(row, sort_keys=True, separators=(",", ":")):
            raise RuntimeError("candidate2 modified a retained V51 line")
    model_sha = hashlib.sha256(canonical_bytes(model)).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "engineering_amendment": 3,
        "profile": str(model.get("profile_name") or "TRAIN18_CANDIDATE2"),
        "predictions": rows,
        "candidate_count": len(rows),
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "cad_used": False,
        "drawing_used": False,
        "part_identity_used": False,
        "case_id_used": False,
        "runtime_id_used": False,
        "filename_used": False,
        "observed_source_multiplicity_used": bool(
            profile["policy_mode"] in ("EXPLOSION_ONLY", "EXPLOSION_THEN_VETO")
        ),
        "classifier_count_feature_used": False,
        "expected_count_used": False,
        "sender_count_used": False,
        "gold_count_used": False,
        "label_count_used_at_inference": False,
        "deletion_only": True,
        "v51_fallback_default": True,
        "training_gate_passed": True,
        "holdout_score_opened": False,
        "frozen_model_sha256": model_sha,
        "deletion_audit": {
            "input_prediction_count": len(predictions),
            "removed_by_explosion_count": len(removed_by_explosion),
            "removed_by_veto_count": len(removed_by_veto),
            "output_prediction_count": len(rows),
            "retained_lines_byte_identical_to_v51": True,
        },
    }
