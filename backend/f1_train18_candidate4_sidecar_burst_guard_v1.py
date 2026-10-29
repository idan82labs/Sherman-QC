"""Engineering-only runtime for Candidate4's minimal V51 sidecar-burst guard.

The runtime is filesystem-blind for inference inputs: callers provide the exact
V51 semantic bytes and their external SHA-256 commitment.  Candidate4 can only
delete V41 fragment-tolerant-parent-bridge sidecars when their observed
multiplicity reaches one of two frozen thresholds.  Every retained prediction
is returned without field mutation and in its original relative order.

No model is executable without a distinct, canonical TRAIN-gate PASS artifact
and an immutable Candidate3 terminal-negative artifact.  This module contains
no label, cache, fitting, scoring, preregistration, or holdout I/O.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 4
PROFILE = "F1_AUG30_TRAIN18_CANDIDATE4_SIDECAR_BURST_GUARD_V1"
MODEL_KIND = "sherman_qc_f1_aug30_train18_candidate4_sidecar_burst_guard_model_v1"
PASS_KIND = "sherman_qc_f1_aug30_train18_candidate4_train_gate_pass_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate4_sidecar_burst_guard_result_v1"
PREFLIGHT_RESULT_KIND = (
    "sherman_qc_f1_aug30_train18_candidate4_semantic_compatibility_result_v1"
)
CANDIDATE3_TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate_terminal_v1"
CANDIDATE3_TERMINAL_STATUS = "REJECTED_PREFIT_CUSTODY_BOUNDARY_VIOLATION"
CANDIDATE3_TERMINAL_SHA256 = (
    "00449059f9e4cd989f3fbcb67e9fc8e7a499ad1dbca0a947deccd75d21019116"
)
CANDIDATE3_TERMINAL_SIZE_BYTES = 5518
CANDIDATE3_INCIDENT_PATH = (
    "research/f1_aug30_train18_candidate3_prefit_custody_incident_v1.json"
)
CANDIDATE3_INCIDENT_SHA256 = (
    "05c044224918a9400f7c7da23f508ec3ca69c44c1b9f5c3700dcee1c8f4021f1"
)
CANDIDATE3_INCIDENT_SIZE_BYTES = 2277

EXPLOSION_RULES = (
    "DROP_ALL_SIDECARS_IF_AT_LEAST_2",
    "DROP_ALL_SIDECARS_IF_AT_LEAST_3",
)
SIDECAR_SOURCE_KIND = "v41_fragment_tolerant_parent_bridge_sidecar"

KNOWN_BINDINGS = {
    "candidate1_terminal_sha256": (
        "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8"
    ),
    "candidate2_terminal_sha256": (
        "ecdbe871364044214fc46e5b5e59095cc9b38cefe9678292f1a8462dffdf3515"
    ),
    "candidate3_terminal_sha256": CANDIDATE3_TERMINAL_SHA256,
    "candidate3_incident_sha256": CANDIDATE3_INCIDENT_SHA256,
    "split_sha256": (
        "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185"
    ),
    "train_allowlist_sha256": (
        "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0"
    ),
    "original_preregistration_sha256": (
        "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a"
    ),
    "custody_label_attestation_sha256": (
        "fe9313bbd09de945fc4da0c238a6b1f2769979e78cccc2abff29ca434dbf84ad"
    ),
    "custody_cache_attestation_sha256": (
        "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71"
    ),
    "custody_staged_validator_pass_sha256": (
        "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433"
    ),
}

BINDING_KEYS = frozenset(
    {
        *KNOWN_BINDINGS,
        "candidate4_preregistration_sha256",
        "candidate4_runtime_source_sha256",
        "candidate4_optimizer_source_sha256",
        "candidate4_tests_source_sha256",
        "candidate4_cache_preflight_source_sha256",
        "candidate4_cache_preflight_tests_sha256",
        "candidate4_cache_preflight_report_sha256",
        "candidate4_cache_preflight_lock_sha256",
        "candidate4_cache_preflight_read_ledger_sha256",
        "fit_audit_core_sha256",
        "train_read_ledger_sha256",
        "train_label_projection_sha256",
        "train_cache_manifest_sha256",
        "train_source_cache_binding_sha256",
    }
)

GATE_CHECK_KEYS = frozenset(
    {
        "cache_preflight_passed",
        "deployed_model_exact_bytes_match_evaluated_predictions",
        "every_label_support_tier_tp_nondecrease",
        "exact_case_count_nondecrease",
        "false_positive_increase_bounded",
        "fixed_preregistered_search_space",
        "fixed_holdout_stage_order_similar4_adjacent3_distant3",
        "holdout_access_zero",
        "nested_complete_case_group_validation",
        "outer_all_18_cases_scored_exactly_once",
        "outer_train_validation_fold_disjoint",
        "overall_micro_f1_delta_at_least_0_05",
        "primary_replay_bytes_exact",
        "replay_execution_succeeded",
        "train18_allowlist_exact",
        "train_read_ledger_exact",
        "true_positive_nondecrease",
    }
)

MODEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile",
        "attempt_ordinal",
        "training_gate_passed",
        "freeze_status",
        "holdout_score_opened",
        "pre_holdout_execution_anchor_created",
        "post_holdout_model_or_threshold_mutation",
        "source_only",
        "deletion_only",
        "v51_fallback_default",
        "rule",
        "source_kind_used",
        "selection_route_used",
        "selector_score_used",
        "geometry_used_for_deletion",
        "observed_source_multiplicity_used",
        "case_id_feature_used",
        "runtime_id_feature_used",
        "filename_feature_used",
        "cad_feature_used",
        "expected_count_used",
        "sender_count_used",
        "gold_count_used",
        "label_count_used_at_inference",
        "bindings",
        "train_gate_pass_binding",
    }
)

PASS_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "attempt_ordinal",
        "training_gate_passed",
        "holdout_score_opened",
        "holdout_execution_authorized",
        "pre_holdout_execution_anchor_created",
        "selected_rule",
        "bindings",
        "checks",
        "gate_result_value_free",
        "label_derived_metric_values_present",
    }
)

FORBIDDEN_FALSE_FLAGS = (
    "labels_used",
    "expected_count_used",
    "cad_used",
    "drawing_used",
    "part_identity_used",
    "case_id_used",
    "runtime_id_used",
    "filename_used",
    "sender_count_used",
    "gold_count_used",
)

FORBIDDEN_PAYLOAD_FIELDS = frozenset(
    {
        "cad_bends",
        "drawing_bends",
        "expected_bend_count",
        "expected_count",
        "filename",
        "gold_bends",
        "ground_truth",
        "labels",
        "manual_bends",
        "outcomes",
        "part_id",
        "sender_count",
    }
)

FORBIDDEN_PREDICTION_FIELDS = FORBIDDEN_PAYLOAD_FIELDS | {
    "case_id",
    "file_name",
    "file_path",
    "is_true_positive",
    "label",
    "outcome",
    "runtime_case_id",
    "scan_id",
}

ROOT_OBSERVED_COUNT_FIELDS = frozenset(
    {
        "candidate_count",
        "line_object_count",
    }
)

ROOT_UNUSED_IDENTITY_FIELDS = frozenset({"case_id", "runtime_case_id", "scan_id"})

FORBIDDEN_NESTED_KEY_TOKENS = frozenset(
    {
        "cad",
        "count",
        "drawing",
        "filename",
        "gold",
        "label",
        "labels",
        "outcome",
        "outcomes",
        "truth",
    }
)

FORBIDDEN_COMPACT_KEYS = frozenset(
    re.sub(r"[^a-z0-9]+", "", key.lower())
    for key in FORBIDDEN_PAYLOAD_FIELDS | FORBIDDEN_PREDICTION_FIELDS
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def runtime_source_sha256() -> str:
    return sha256_bytes(Path(__file__).read_bytes())


def _require_sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if value == "0" * 64:
        raise ValueError(f"{name} must not be an unresolved zero commitment")
    return value


def _strict_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric and non-boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _key_tokens(value: str) -> set[str]:
    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    tokens = {
        token for token in re.split(r"[^a-z0-9]+", separated.lower()) if token
    }
    tokens.update(
        token[:-1] for token in tuple(tokens) if len(token) > 3 and token.endswith("s")
    )
    return tokens


def _validate_no_forbidden_nested_fields(
    value: Any, *, path: str, depth: int = 0
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string JSON key")
            if depth == 0 and key in FORBIDDEN_FALSE_FLAGS:
                continue
            if depth == 0 and key in ROOT_OBSERVED_COUNT_FIELDS:
                _validate_no_forbidden_nested_fields(
                    child, path=f"{path}.{key}", depth=depth + 1
                )
                continue
            if depth == 0 and key in ROOT_UNUSED_IDENTITY_FIELDS:
                _validate_no_forbidden_nested_fields(
                    child, path=f"{path}.{key}", depth=depth + 1
                )
                continue
            lowered = key.lower()
            compact = re.sub(r"[^a-z0-9]+", "", lowered)
            tokens = _key_tokens(key)
            if (
                lowered in FORBIDDEN_PAYLOAD_FIELDS
                or lowered in FORBIDDEN_PREDICTION_FIELDS
                or compact in FORBIDDEN_COMPACT_KEYS
                or tokens.intersection(FORBIDDEN_NESTED_KEY_TOKENS)
                or lowered in {"part_id", "sender_count", "expected_count"}
            ):
                raise ValueError(f"{path} exposes forbidden nested field: {key}")
            _validate_no_forbidden_nested_fields(
                child, path=f"{path}.{key}", depth=depth + 1
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_forbidden_nested_fields(
                child, path=f"{path}[{index}]", depth=depth + 1
            )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _parse_json_bytes(
    data: bytes, *, name: str, canonical_required: bool
) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError(f"{name} must be immutable bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8 JSON") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    if canonical_required and canonical_bytes(value) != data:
        raise ValueError(f"{name} must use exact canonical JSON bytes")
    return value


def _point(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-coordinate JSON array")
    return tuple(
        _strict_number(coordinate, name=f"{name}[{index}]")
        for index, coordinate in enumerate(value)
    )


def _validate_endpoint_pair(start: Any, end: Any, *, name: str) -> None:
    first = _point(start, name=f"{name}[0]")
    second = _point(end, name=f"{name}[1]")
    length = math.dist(first, second)
    if not math.isfinite(length) or length <= 1e-9:
        raise ValueError(f"{name} must define a nondegenerate line")


def _validate_flexible_geometry(row: Mapping[str, Any], *, bend_id: str) -> None:
    forms = 0

    span = row.get("span_endpoints")
    if span is not None:
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f"prediction {bend_id} span_endpoints must be 2x3")
        _validate_endpoint_pair(span[0], span[1], name="span_endpoints")
        forms += 1

    line_values = (row.get("line_start"), row.get("line_end"))
    if any(value is not None for value in line_values):
        if any(value is None for value in line_values):
            raise ValueError(f"prediction {bend_id} line_start/line_end are incomplete")
        _validate_endpoint_pair(line_values[0], line_values[1], name="line")
        forms += 1

    axial_values = (
        row.get("anchor"),
        row.get("axis_direction"),
        row.get("span_mm"),
    )
    if any(value is not None for value in axial_values):
        if any(value is None for value in axial_values):
            raise ValueError(f"prediction {bend_id} anchor/axis/span are incomplete")
        _point(axial_values[0], name="anchor")
        axis = _point(axial_values[1], name="axis_direction")
        span_mm = _strict_number(axial_values[2], name="span_mm")
        axis_norm = math.hypot(*axis)
        if not math.isfinite(axis_norm) or axis_norm <= 1e-12 or span_mm <= 1e-9:
            raise ValueError(f"prediction {bend_id} anchor/axis/span line is degenerate")
        forms += 1

    if forms == 0:
        raise ValueError(f"prediction {bend_id} has no supported line geometry")


def _validate_prediction_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("V51 semantic result must contain a predictions list")
    rows: list[dict[str, Any]] = []
    bend_ids: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"V51 prediction {index} must be a JSON object")
        forbidden = sorted(set(raw).intersection(FORBIDDEN_PREDICTION_FIELDS))
        if forbidden:
            raise ValueError(
                f"V51 prediction {index} exposes forbidden fields: {forbidden}"
            )
        bend_id = raw.get("bend_id")
        if not isinstance(bend_id, str) or not bend_id:
            raise ValueError(f"V51 prediction {index} must have a nonempty bend_id")
        if bend_id in bend_ids:
            raise ValueError(f"duplicate V51 prediction bend_id: {bend_id}")
        bend_ids.add(bend_id)
        for field in ("source_kind", "selection_route"):
            if not isinstance(raw.get(field), str) or not raw[field]:
                raise ValueError(f"prediction {bend_id} {field} must be a nonempty string")
        if "source_hypothesis_id" in raw and (
            not isinstance(raw["source_hypothesis_id"], str)
            or not raw["source_hypothesis_id"]
        ):
            raise ValueError(
                f"prediction {bend_id} source_hypothesis_id must be a nonempty string"
            )
        if "selector_score" in raw and raw["selector_score"] is not None:
            _strict_number(raw["selector_score"], name=f"prediction {bend_id} selector_score")
        _validate_flexible_geometry(raw, bend_id=bend_id)
        rows.append(raw)
    return rows


def _validate_semantic_bytes(
    data: bytes, *, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = _require_sha(expected_sha256, name="V51 semantic SHA")
    if sha256_bytes(data) != expected:
        raise ValueError("V51 semantic bytes differ from the external SHA binding")
    semantic = _parse_json_bytes(
        data, name="V51 semantic artifact", canonical_required=False
    )
    if semantic.get("source_only") is not True or semantic.get("research_only") is not True:
        raise ValueError("V51 semantic artifact must be source-only research output")
    forbidden = sorted(set(semantic).intersection(FORBIDDEN_PAYLOAD_FIELDS))
    if forbidden:
        raise ValueError(f"V51 semantic artifact exposes forbidden fields: {forbidden}")
    for field in FORBIDDEN_FALSE_FLAGS:
        if semantic.get(field) not in (None, False):
            raise ValueError(f"V51 semantic artifact reports forbidden input: {field}")
    _validate_no_forbidden_nested_fields(semantic, path="V51 semantic artifact")
    return semantic, _validate_prediction_rows(semantic.get("predictions"))


def _validate_bindings(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != BINDING_KEYS:
        raise ValueError("Candidate4 binding schema mismatch")
    result = {
        key: _require_sha(value[key], name=f"bindings.{key}")
        for key in sorted(BINDING_KEYS)
    }
    for key, expected in KNOWN_BINDINGS.items():
        if result[key] != expected:
            raise ValueError(f"Candidate4 immutable binding mismatch: {key}")
    if result["candidate4_runtime_source_sha256"] != runtime_source_sha256():
        raise ValueError("Candidate4 runtime source hash binding mismatch")
    terminal_hashes = {
        result["candidate1_terminal_sha256"],
        result["candidate2_terminal_sha256"],
        result["candidate3_terminal_sha256"],
    }
    if len(terminal_hashes) != 3:
        raise ValueError("Candidate4 terminal bindings must be distinct")
    if result["fit_audit_core_sha256"] == result["train_read_ledger_sha256"]:
        raise ValueError("Candidate4 fit-audit and read-ledger bindings must be distinct")
    return result


def _validate_candidate3_terminal(
    data: bytes, *, expected_sha256: str
) -> dict[str, Any]:
    expected = _require_sha(expected_sha256, name="Candidate3 terminal SHA")
    if expected != CANDIDATE3_TERMINAL_SHA256:
        raise ValueError("Candidate3 terminal SHA differs from the exact audited seal")
    if len(data) != CANDIDATE3_TERMINAL_SIZE_BYTES:
        raise ValueError("Candidate3 terminal size differs from the exact audited seal")
    if sha256_bytes(data) != expected:
        raise ValueError("Candidate3 terminal bytes differ from the external SHA binding")
    terminal = _parse_json_bytes(
        data, name="Candidate3 terminal artifact", canonical_required=True
    )
    incident_binding = (terminal.get("artifact_bindings") or {}).get(
        "candidate3_custody_incident"
    )
    if incident_binding != {
        "path": CANDIDATE3_INCIDENT_PATH,
        "sha256": CANDIDATE3_INCIDENT_SHA256,
        "size_bytes": CANDIDATE3_INCIDENT_SIZE_BYTES,
    }:
        raise ValueError("Candidate3 terminal does not bind the exact custody incident")
    if (
        terminal.get("schema_version") != 1
        or terminal.get("kind") != CANDIDATE3_TERMINAL_KIND
        or terminal.get("candidate_id") != 3
        or terminal.get("attempt_ordinal") != 3
        or terminal.get("status") != CANDIDATE3_TERMINAL_STATUS
        or terminal.get("candidate3_continuation_authorized") is not False
        or terminal.get("candidate3_preregistration_authorized") is not False
        or terminal.get("candidate3_preregistration_created") is not False
        or terminal.get("candidate3_fit_authorized") is not False
        or terminal.get("candidate3_fit_started") is not False
        or terminal.get("candidate3_metric_computed") is not False
        or terminal.get("candidate3_model_created") is not False
        or terminal.get("candidate3_rule_selected") is not False
        or terminal.get("candidate3_optimizer_draft_executed_on_train_labels") is not False
        or terminal.get("nested_training_retain_gate_evaluated") is not False
        or terminal.get("primary_replay_candidate_evaluated") is not False
        or terminal.get("holdout_execution_authorized") is not False
        or terminal.get("holdout_access_count") != 0
        or terminal.get("holdout_score_opened") is not False
        or terminal.get("pre_holdout_execution_anchor_created") is not False
        or terminal.get("local_pre_holdout_seal_disposition") != "NOT_CREATED"
        or terminal.get("promotion_eligible") is not False
        or terminal.get("successor_candidate_authorized_by_this_terminal") is not False
        or terminal.get("post_terminal_candidate3_model_or_threshold_mutation")
        != "FORBIDDEN"
    ):
        raise ValueError(
            "Candidate3 terminal artifact is not the exact noncontinuing custody rejection"
        )
    return terminal


def _validate_pass_artifact(
    data: bytes,
    *,
    expected_sha256: str,
    expected_rule: str,
    expected_bindings: Mapping[str, str],
) -> dict[str, Any]:
    expected = _require_sha(expected_sha256, name="Candidate4 TRAIN PASS SHA")
    if sha256_bytes(data) != expected:
        raise ValueError("Candidate4 TRAIN PASS bytes differ from the external SHA binding")
    artifact = _parse_json_bytes(
        data, name="Candidate4 TRAIN PASS artifact", canonical_required=True
    )
    if not isinstance(artifact, dict) or set(artifact) != PASS_KEYS:
        raise ValueError("Candidate4 TRAIN PASS artifact schema mismatch")
    if (
        artifact["schema_version"] != 1
        or artifact["kind"] != PASS_KIND
        or artifact["status"] != "PASS"
        or artifact["attempt_ordinal"] != ATTEMPT_ORDINAL
        or artifact["training_gate_passed"] is not True
        or artifact["holdout_score_opened"] is not False
        or artifact["holdout_execution_authorized"] is not False
        or artifact["pre_holdout_execution_anchor_created"] is not False
        or artifact["gate_result_value_free"] is not True
        or artifact["label_derived_metric_values_present"] is not False
    ):
        raise ValueError("Candidate4 TRAIN PASS status contract mismatch")
    if artifact["selected_rule"] != expected_rule:
        raise ValueError("Candidate4 model rule differs from TRAIN PASS rule")
    bindings = _validate_bindings(artifact["bindings"])
    if bindings != dict(expected_bindings):
        raise ValueError("Candidate4 model and TRAIN PASS bindings differ")
    checks = artifact["checks"]
    if not isinstance(checks, dict) or set(checks) != GATE_CHECK_KEYS:
        raise ValueError("Candidate4 TRAIN PASS gate-check schema mismatch")
    if any(checks[key] is not True for key in GATE_CHECK_KEYS):
        raise ValueError("Candidate4 TRAIN PASS contains a failed gate")
    return artifact


def _validate_model(
    model: Mapping[str, Any],
    *,
    train_gate_pass_bytes: bytes,
    train_gate_pass_sha256: str,
    candidate3_terminal_sha256: str,
) -> str:
    if not isinstance(model, Mapping) or set(model) != MODEL_KEYS:
        raise ValueError("Candidate4 model schema mismatch")
    if (
        model["schema_version"] != SCHEMA_VERSION
        or model["kind"] != MODEL_KIND
        or model["profile"] != PROFILE
        or model["attempt_ordinal"] != ATTEMPT_ORDINAL
        or model["training_gate_passed"] is not True
        or model["freeze_status"] != "FROZEN_AFTER_TRAIN_GATE_PASS"
        or model["holdout_score_opened"] is not False
        or model["pre_holdout_execution_anchor_created"] is not False
        or model["post_holdout_model_or_threshold_mutation"]
        != "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
        or model["source_only"] is not True
        or model["deletion_only"] is not True
        or model["v51_fallback_default"] is not True
    ):
        raise ValueError("Candidate4 model status contract mismatch")
    declarations = {
        "source_kind_used": True,
        "selection_route_used": False,
        "selector_score_used": False,
        "geometry_used_for_deletion": False,
        "observed_source_multiplicity_used": True,
        "case_id_feature_used": False,
        "runtime_id_feature_used": False,
        "filename_feature_used": False,
        "cad_feature_used": False,
        "expected_count_used": False,
        "sender_count_used": False,
        "gold_count_used": False,
        "label_count_used_at_inference": False,
    }
    for field, expected in declarations.items():
        if model[field] is not expected:
            raise ValueError(f"Candidate4 model declaration mismatch: {field}")
    rule = model["rule"]
    if rule not in EXPLOSION_RULES:
        raise ValueError("Candidate4 model contains an unsupported burst rule")
    bindings = _validate_bindings(model["bindings"])
    terminal_sha = _require_sha(
        candidate3_terminal_sha256, name="Candidate3 terminal SHA"
    )
    if bindings["candidate3_terminal_sha256"] != terminal_sha:
        raise ValueError("Candidate4 model does not bind the supplied Candidate3 terminal")
    pass_binding = model["train_gate_pass_binding"]
    if not isinstance(pass_binding, dict) or set(pass_binding) != {"kind", "sha256"}:
        raise ValueError("Candidate4 TRAIN PASS binding schema mismatch")
    if pass_binding["kind"] != PASS_KIND:
        raise ValueError("Candidate4 TRAIN PASS binding kind mismatch")
    bound_pass_sha = _require_sha(
        pass_binding["sha256"], name="train_gate_pass_binding.sha256"
    )
    external_pass_sha = _require_sha(
        train_gate_pass_sha256, name="Candidate4 TRAIN PASS SHA"
    )
    if bound_pass_sha != external_pass_sha:
        raise ValueError("Candidate4 model does not bind the supplied TRAIN PASS")
    if external_pass_sha in set(bindings.values()):
        raise ValueError("Candidate4 TRAIN PASS must be distinct from every bound artifact")
    _validate_pass_artifact(
        train_gate_pass_bytes,
        expected_sha256=external_pass_sha,
        expected_rule=rule,
        expected_bindings=bindings,
    )
    return rule


def _is_sidecar(row: Mapping[str, Any]) -> bool:
    return row["source_kind"] == SIDECAR_SOURCE_KIND


def preflight_candidate4_semantic_bytes(
    v51_semantic_bytes: bytes, *, v51_semantic_sha256: str
) -> dict[str, Any]:
    """Exercise every frozen deletion path without labels, fitting, or a model."""

    _semantic, predictions = _validate_semantic_bytes(
        v51_semantic_bytes, expected_sha256=v51_semantic_sha256
    )
    original_bytes = {
        row["bend_id"]: canonical_bytes(row) for row in predictions
    }
    sidecar_count = sum(_is_sidecar(row) for row in predictions)
    for rule in EXPLOSION_RULES:
        threshold = 2 if rule == EXPLOSION_RULES[0] else 3
        triggered = sidecar_count >= threshold
        retained = [
            row for row in predictions if not (triggered and _is_sidecar(row))
        ]
        if any(
            canonical_bytes(row) != original_bytes[row["bend_id"]]
            for row in retained
        ):
            raise RuntimeError("Candidate4 preflight modified a retained prediction")
        if triggered and any(_is_sidecar(row) for row in retained):
            raise RuntimeError("Candidate4 preflight retained a triggered sidecar")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PREFLIGHT_RESULT_KIND,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "status": "SOURCE_COMPATIBLE",
        "input_v51_semantic_sha256": v51_semantic_sha256,
        "all_frozen_rules_exercised": True,
        "all_retained_predictions_canonical_byte_identical": True,
        "flexible_geometry_schema_validated": True,
        "recursive_forbidden_field_validation_exercised": True,
        "manifest_scan_identity_required_externally": True,
        "semantic_internal_scan_identity_used": False,
        "source_only": True,
        "labels_used": False,
        "candidate_metric_computed": False,
        "model_fit_performed": False,
        "deletion_rule_selected": False,
        "holdout_score_opened": False,
    }


def apply_candidate4(
    v51_semantic_bytes: bytes,
    model: Mapping[str, Any],
    train_gate_pass_bytes: bytes,
    candidate3_terminal_bytes: bytes,
    *,
    v51_semantic_sha256: str,
    train_gate_pass_sha256: str,
    candidate3_terminal_sha256: str,
) -> dict[str, Any]:
    """Apply one frozen sidecar-burst rule to externally hash-bound V51 bytes."""

    _validate_candidate3_terminal(
        candidate3_terminal_bytes, expected_sha256=candidate3_terminal_sha256
    )
    rule = _validate_model(
        model,
        train_gate_pass_bytes=train_gate_pass_bytes,
        train_gate_pass_sha256=train_gate_pass_sha256,
        candidate3_terminal_sha256=candidate3_terminal_sha256,
    )
    _semantic, predictions = _validate_semantic_bytes(
        v51_semantic_bytes, expected_sha256=v51_semantic_sha256
    )

    original_bytes = {
        row["bend_id"]: canonical_bytes(row) for row in predictions
    }
    sidecar_count = sum(_is_sidecar(row) for row in predictions)
    threshold = 2 if rule == EXPLOSION_RULES[0] else 3
    triggered = sidecar_count >= threshold
    retained = [
        row for row in predictions if not (triggered and _is_sidecar(row))
    ]
    if any(
        canonical_bytes(row) != original_bytes[row["bend_id"]] for row in retained
    ):
        raise RuntimeError("Candidate4 modified a retained V51 prediction")

    model_sha = sha256_bytes(canonical_bytes(dict(model)))
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "profile": PROFILE,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "input_v51_semantic_sha256": v51_semantic_sha256,
        "candidate4_model_sha256": model_sha,
        "train_gate_pass_sha256": train_gate_pass_sha256,
        "candidate3_terminal_sha256": candidate3_terminal_sha256,
        "rule": rule,
        "predictions": retained,
        "candidate_count": len(retained),
        "deletion_audit": {
            "input_prediction_count": len(predictions),
            "observed_sidecar_multiplicity": sidecar_count,
            "trigger_threshold": threshold,
            "triggered": triggered,
            "deleted_sidecar_count": len(predictions) - len(retained),
            "output_prediction_count": len(retained),
            "retained_predictions_canonical_byte_identical": True,
        },
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "cad_used": False,
        "drawing_used": False,
        "part_identity_used": False,
        "case_id_used": False,
        "runtime_id_used": False,
        "filename_used": False,
        "source_kind_used": True,
        "selection_route_used": False,
        "selector_score_used": False,
        "geometry_used_for_deletion": False,
        "observed_source_multiplicity_used": True,
        "expected_count_used": False,
        "sender_count_used": False,
        "gold_count_used": False,
        "label_count_used_at_inference": False,
        "deletion_only": True,
        "v51_fallback_default": True,
        "training_gate_passed": True,
        "holdout_score_opened": False,
        "post_holdout_model_or_threshold_mutation": (
            "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
        ),
        "promotion_eligible": False,
    }
    result["candidate4_result_sha256"] = sha256_bytes(canonical_bytes(result))
    return result
