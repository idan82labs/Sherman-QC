"""Filesystem-blind runtime for Candidate5 conservative single-rule deletion.

Candidate5 is a new attempt, not a continuation of Candidate3 or Candidate4.
It preserves Candidate3's scientifically unconsumed design: select at most one
source-only deletion rule, default to the frozen V51 prediction set whenever
evidence is missing, and execute only after a distinct TRAIN gate PASS.

The runtime accepts externally SHA-256-bound V51 semantic bytes.  It never
opens cache, pool, label, preregistration, or holdout files.  Generic source
audit/count metadata may be present but is ignored.  Forbidden semantic keys
are rejected recursively, and retained predictions are emitted through an
explicit sanitized projection so ignored metadata cannot leak.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 5
PROFILE = "F1_AUG30_TRAIN18_CANDIDATE5_CONSERVATIVE_DELETION_V1"
MODEL_KIND = "sherman_qc_f1_aug30_train18_candidate5_conservative_deletion_model_v1"
PASS_KIND = "sherman_qc_f1_aug30_train18_candidate5_train_gate_pass_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate5_conservative_deletion_result_v1"
PREFLIGHT_RESULT_KIND = (
    "sherman_qc_f1_aug30_train18_candidate5_semantic_compatibility_result_v1"
)
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate_terminal_v1"

CANDIDATE3_TERMINAL_SHA256 = (
    "00449059f9e4cd989f3fbcb67e9fc8e7a499ad1dbca0a947deccd75d21019116"
)
CANDIDATE3_TERMINAL_SIZE_BYTES = 5518
CANDIDATE3_TERMINAL_STATUS = "REJECTED_PREFIT_CUSTODY_BOUNDARY_VIOLATION"
CANDIDATE4_TERMINAL_SHA256 = (
    "10d6bd14ad10dfd1ace464e551cb79f65abeb3c12bc0094fb2695f61a1aa2a50"
)
CANDIDATE4_TERMINAL_SIZE_BYTES = 5384
CANDIDATE4_TERMINAL_STATUS = "REJECTED_PREFIT_SOURCE_SCHEMA_INCOMPATIBILITY"

KNOWN_BINDINGS = {
    "candidate1_terminal_sha256": (
        "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8"
    ),
    "candidate2_terminal_sha256": (
        "ecdbe871364044214fc46e5b5e59095cc9b38cefe9678292f1a8462dffdf3515"
    ),
    "candidate3_terminal_sha256": CANDIDATE3_TERMINAL_SHA256,
    "candidate4_terminal_sha256": CANDIDATE4_TERMINAL_SHA256,
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
        "candidate5_preregistration_sha256",
        "candidate5_runtime_source_sha256",
        "candidate5_optimizer_source_sha256",
        "candidate5_runtime_tests_sha256",
        "candidate5_preflight_source_sha256",
        "candidate5_preflight_tests_sha256",
        "candidate5_preflight_report_sha256",
        "candidate5_preflight_lock_sha256",
        "candidate5_preflight_read_ledger_sha256",
        "fit_audit_core_sha256",
        "train_read_ledger_sha256",
        "train_label_projection_sha256",
        "train_cache_manifest_sha256",
        "train_source_cache_binding_sha256",
    }
)

NUMERIC_FEATURES = frozenset({"line_length_mm", "selector_score"})
CATEGORICAL_FEATURES = frozenset({"source_kind", "selection_route"})
ALLOWED_RULE_FEATURES = NUMERIC_FEATURES | CATEGORICAL_FEATURES
NUMERIC_OPERATORS = frozenset({"le", "ge"})

REQUIRED_SOURCE_FLAGS = {
    "source_only": True,
    "research_only": True,
    "labels_used": False,
    "cad_used": False,
    "part_identity_used": False,
    "expected_count_used": False,
}
OPTIONAL_FALSE_SOURCE_FLAGS = frozenset(
    {
        "case_id_used",
        "drawing_used",
        "filename_used",
        "gold_count_used",
        "label_count_used_at_inference",
        "runtime_id_used",
        "sender_count_used",
    }
)
TOP_LEVEL_GUARD_FLAGS = frozenset(REQUIRED_SOURCE_FLAGS) | OPTIONAL_FALSE_SOURCE_FLAGS

FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "cad",
        "drawing",
        "expected",
        "filename",
        "gold",
        "label",
        "labels",
        "outcome",
        "outcomes",
        "part",
        "sender",
        "target",
        "truth",
    }
)
FORBIDDEN_COMPACT_KEYS = frozenset(
    {
        "cadbends",
        "drawingbends",
        "expectedbendcount",
        "expectedcount",
        "filename",
        "goldbends",
        "groundtruth",
        "labels",
        "manualbends",
        "outcome",
        "outcomes",
        "partid",
        "sendercount",
        "targettruth",
        "falsenegative",
        "falsepositive",
        "isfalsenegative",
        "isfalsepositive",
        "istruenegative",
        "istruepositive",
        "truenegative",
        "truepositive",
    }
)

SANITIZED_PREDICTION_KEYS = frozenset(
    {
        "bend_id",
        "line_end",
        "line_start",
        "selection_route",
        "selector_score",
        "source_hypothesis_id",
        "source_kind",
    }
)
REQUIRED_PREDICTION_STRING_FIELDS = (
    "bend_id",
    "selection_route",
    "source_hypothesis_id",
    "source_kind",
)

MODEL_KEYS = frozenset(
    {
        "attempt_ordinal",
        "audit_metadata_feature_used",
        "bindings",
        "bend_id_feature_used",
        "cad_feature_used",
        "case_identity_feature_used",
        "candidate_count_feature_used",
        "deletion_only",
        "drawing_feature_used",
        "expected_count_feature_used",
        "filename_feature_used",
        "freeze_status",
        "gold_or_truth_feature_used",
        "holdout_score_opened",
        "inference_features",
        "kind",
        "label_feature_used",
        "line_length_feature_used",
        "observed_source_or_prediction_multiplicity_used",
        "part_identity_feature_used",
        "post_holdout_model_or_threshold_mutation",
        "pre_holdout_execution_anchor_created",
        "profile",
        "rule",
        "runtime_identity_feature_used",
        "schema_version",
        "selection_route_feature_used",
        "selector_score_feature_used",
        "semantic_internal_scan_identity_feature_used",
        "sender_count_feature_used",
        "source_only",
        "source_hypothesis_id_feature_used",
        "source_kind_feature_used",
        "train_gate_pass_binding",
        "training_gate_passed",
        "v51_fallback_default",
    }
)
PASS_KEYS = frozenset(
    {
        "attempt_ordinal",
        "bindings",
        "checks",
        "gate_result_value_free",
        "holdout_execution_authorized",
        "holdout_score_opened",
        "kind",
        "label_derived_metric_values_present",
        "pre_holdout_execution_anchor_created",
        "schema_version",
        "selected_rule",
        "status",
        "training_gate_passed",
    }
)
GATE_CHECK_KEYS = frozenset(
    {
        "cache_preflight_passed",
        "deployed_model_exact_bytes_match_evaluated_predictions",
        "every_label_support_tier_tp_nondecrease",
        "exact_case_count_nondecrease",
        "false_positive_decrease",
        "fixed_holdout_stage_order_similar4_adjacent3_distant3",
        "fixed_preregistered_search_space",
        "holdout_access_zero",
        "nested_complete_case_group_validation",
        "outer_all_18_cases_scored_exactly_once",
        "outer_train_validation_fold_disjoint",
        "overall_micro_f1_increase",
        "primary_replay_bytes_exact",
        "replay_execution_succeeded_before_publication",
        "train18_allowlist_exact",
        "train_read_ledger_exact",
        "true_positive_nondecrease",
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
        raise ValueError(f"{name} must not be an unresolved zero commitment")
    return value


def _strict_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric and non-boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


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
        raise ValueError(f"{name} must use exact LF-canonical JSON bytes")
    return value


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


def _forbidden_key(key: str) -> bool:
    lowered = key.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    tokens = _key_tokens(key)
    truth_class_alias = bool(
        ({"true", "positive"}.issubset(tokens))
        or ({"false", "positive"}.issubset(tokens))
        or ({"true", "negative"}.issubset(tokens))
        or ({"false", "negative"}.issubset(tokens))
        or tokens.intersection({"tp", "fp", "tn", "fn"})
    )
    return bool(
        tokens.intersection(FORBIDDEN_KEY_TOKENS)
        or compact in FORBIDDEN_COMPACT_KEYS
        or {"file", "name"}.issubset(tokens)
        or truth_class_alias
    )


def _validate_no_forbidden_keys(value: Any, *, path: str, depth: int = 0) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string JSON key")
            if depth == 0 and key in TOP_LEVEL_GUARD_FLAGS:
                _validate_no_forbidden_keys(
                    child, path=f"{path}.{key}", depth=depth + 1
                )
                continue
            if _forbidden_key(key):
                raise ValueError(f"{path} exposes forbidden key: {key}")
            _validate_no_forbidden_keys(
                child, path=f"{path}.{key}", depth=depth + 1
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_forbidden_keys(
                child, path=f"{path}[{index}]", depth=depth + 1
            )


def _validate_top_level_source_flags(semantic: Mapping[str, Any]) -> None:
    for field, expected in REQUIRED_SOURCE_FLAGS.items():
        if field not in semantic or semantic[field] is not expected:
            raise ValueError(f"V51 semantic artifact requires {field}={expected!r}")
    for field in OPTIONAL_FALSE_SOURCE_FLAGS:
        if field in semantic and semantic[field] is not False:
            raise ValueError(f"V51 semantic artifact requires optional {field}=false")


def _point(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-coordinate JSON array")
    return tuple(
        _strict_number(coordinate, name=f"{name}[{index}]")
        for index, coordinate in enumerate(value)
    )


def _endpoint_pair(start: Any, end: Any, *, name: str) -> float:
    first = _point(start, name=f"{name}[0]")
    second = _point(end, name=f"{name}[1]")
    length = math.dist(first, second)
    if not math.isfinite(length) or length <= 1e-9:
        raise ValueError(f"{name} must define a nondegenerate line")
    return length


def _validate_geometry(row: Mapping[str, Any], *, bend_id: str) -> float:
    if "line_start" not in row or "line_end" not in row:
        raise ValueError(f"prediction {bend_id} requires line_start and line_end")
    return _endpoint_pair(row["line_start"], row["line_end"], name="line")


def _validate_prediction_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("V51 semantic result must contain a predictions list")
    rows: list[dict[str, Any]] = []
    bend_ids: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"V51 prediction {index} must be a JSON object")
        for field in REQUIRED_PREDICTION_STRING_FIELDS:
            field_value = raw.get(field)
            if (
                not isinstance(field_value, str)
                or not field_value
                or field_value != field_value.strip()
                or len(field_value) > 256
                or any(ord(character) < 32 for character in field_value)
            ):
                raise ValueError(f"V51 prediction {index} {field} must be nonempty")
        bend_id = raw["bend_id"]
        if bend_id in bend_ids:
            raise ValueError(f"duplicate V51 prediction bend_id: {bend_id}")
        bend_ids.add(bend_id)
        if "selector_score" in raw and raw["selector_score"] is not None:
            _strict_number(
                raw["selector_score"], name=f"prediction {bend_id} selector_score"
            )
        _validate_geometry(raw, bend_id=bend_id)
        rows.append(dict(raw))
    return rows


def _sanitize_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: row[key]
        for key in sorted(SANITIZED_PREDICTION_KEYS)
        if key in row
    }
    for field in REQUIRED_PREDICTION_STRING_FIELDS:
        if field not in sanitized:
            raise AssertionError(f"sanitized prediction lost required field: {field}")
    _validate_geometry(sanitized, bend_id=str(sanitized["bend_id"]))
    if any(key not in SANITIZED_PREDICTION_KEYS for key in sanitized):
        raise AssertionError("sanitized prediction contains a non-allowlisted key")
    return sanitized


def _validate_semantic_bytes(
    data: bytes, *, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = _require_sha(expected_sha256, name="V51 semantic SHA")
    if sha256_bytes(data) != expected:
        raise ValueError("V51 semantic bytes differ from the external SHA binding")
    semantic = _parse_json_bytes(
        data, name="V51 semantic artifact", canonical_required=False
    )
    _validate_top_level_source_flags(semantic)
    _validate_no_forbidden_keys(semantic, path="V51 semantic artifact")
    rows = _validate_prediction_rows(semantic.get("predictions"))
    return semantic, [_sanitize_prediction(row) for row in rows]


def extract_prediction_features(row: Mapping[str, Any]) -> dict[str, float | str | None]:
    bend_id = str(row.get("bend_id") or "")
    if not bend_id:
        raise ValueError("prediction bend_id is required for feature extraction")
    line_length = _validate_geometry(row, bend_id=bend_id)
    selector_score = (
        None
        if row.get("selector_score") is None
        else _strict_number(row["selector_score"], name="selector_score")
    )
    return {
        "line_length_mm": line_length,
        "selector_score": selector_score,
        "selection_route": str(row["selection_route"]),
        "source_kind": str(row["source_kind"]),
    }


def validate_rule(rule: Mapping[str, Any]) -> None:
    if not isinstance(rule, Mapping):
        raise ValueError("Candidate5 rule must be an object")
    feature = rule.get("feature")
    operator = rule.get("operator")
    if feature not in ALLOWED_RULE_FEATURES:
        raise ValueError(f"Candidate5 rule uses an unsupported feature: {feature}")
    if feature in CATEGORICAL_FEATURES:
        if set(rule) != {"feature", "operator", "value"} or operator != "eq":
            raise ValueError("categorical Candidate5 rule requires exact equality")
        if not isinstance(rule.get("value"), str) or not rule["value"]:
            raise ValueError("categorical Candidate5 rule value must be nonempty")
        return
    if set(rule) != {"feature", "operator", "threshold"}:
        raise ValueError("numeric Candidate5 rule has an invalid schema")
    if operator not in NUMERIC_OPERATORS:
        raise ValueError("numeric Candidate5 rule requires le/ge")
    _strict_number(rule.get("threshold"), name="Candidate5 rule threshold")


def _matches(
    rule: Mapping[str, Any], features: Mapping[str, float | str | None]
) -> bool:
    feature = str(rule["feature"])
    value = features[feature]
    if feature in CATEGORICAL_FEATURES:
        return value == rule["value"]
    if value is None:
        return False
    number = float(value)
    threshold = float(rule["threshold"])
    return number <= threshold if rule["operator"] == "le" else number >= threshold


def _validate_terminal(
    data: bytes,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    candidate_id: int,
    attempt_ordinal: int,
    expected_status: str,
) -> dict[str, Any]:
    expected = _require_sha(expected_sha256, name=f"Candidate{candidate_id} terminal SHA")
    if len(data) != expected_size_bytes or sha256_bytes(data) != expected:
        raise ValueError(f"Candidate{candidate_id} terminal byte binding mismatch")
    terminal = _parse_json_bytes(
        data, name=f"Candidate{candidate_id} terminal", canonical_required=True
    )
    continuation_key = f"candidate{candidate_id}_continuation_authorized"
    mutation_key = f"post_terminal_candidate{candidate_id}_model_or_threshold_mutation"
    if (
        terminal.get("schema_version") != 1
        or terminal.get("kind") != TERMINAL_KIND
        or terminal.get("candidate_id") != candidate_id
        or terminal.get("attempt_ordinal") != attempt_ordinal
        or terminal.get("status") != expected_status
        or terminal.get(continuation_key) is not False
        or terminal.get("holdout_execution_authorized") is not False
        or terminal.get("holdout_access_count") != 0
        or terminal.get("holdout_score_opened") is not False
        or terminal.get("promotion_eligible") is not False
        or terminal.get(mutation_key) != "FORBIDDEN"
    ):
        raise ValueError(f"Candidate{candidate_id} terminal status contract mismatch")
    return terminal


def validate_candidate3_terminal_bytes(data: bytes, *, expected_sha256: str) -> dict[str, Any]:
    if expected_sha256 != CANDIDATE3_TERMINAL_SHA256:
        raise ValueError("Candidate3 terminal differs from the audited seal")
    return _validate_terminal(
        data,
        expected_sha256=expected_sha256,
        expected_size_bytes=CANDIDATE3_TERMINAL_SIZE_BYTES,
        candidate_id=3,
        attempt_ordinal=3,
        expected_status=CANDIDATE3_TERMINAL_STATUS,
    )


def validate_candidate4_terminal_bytes(data: bytes, *, expected_sha256: str) -> dict[str, Any]:
    if expected_sha256 != CANDIDATE4_TERMINAL_SHA256:
        raise ValueError("Candidate4 terminal differs from the audited seal")
    terminal = _validate_terminal(
        data,
        expected_sha256=expected_sha256,
        expected_size_bytes=CANDIDATE4_TERMINAL_SIZE_BYTES,
        candidate_id=4,
        attempt_ordinal=4,
        expected_status=CANDIDATE4_TERMINAL_STATUS,
    )
    parent = (terminal.get("artifact_bindings") or {}).get("candidate3_terminal")
    if parent != {
        "path": "research/f1_aug30_train18_candidate3_terminal_negative_v1.json",
        "sha256": CANDIDATE3_TERMINAL_SHA256,
        "size_bytes": CANDIDATE3_TERMINAL_SIZE_BYTES,
    }:
        raise ValueError("Candidate4 terminal does not bind exact Candidate3 terminal")
    return terminal


def _validate_bindings(
    value: Any, *, verified_runtime_source_sha256: str
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != BINDING_KEYS:
        raise ValueError("Candidate5 binding schema mismatch")
    result = {
        key: _require_sha(value[key], name=f"bindings.{key}")
        for key in sorted(BINDING_KEYS)
    }
    for key, expected in KNOWN_BINDINGS.items():
        if result[key] != expected:
            raise ValueError(f"Candidate5 immutable binding mismatch: {key}")
    verified_runtime = _require_sha(
        verified_runtime_source_sha256, name="verified Candidate5 runtime source SHA"
    )
    if result["candidate5_runtime_source_sha256"] != verified_runtime:
        raise ValueError("Candidate5 runtime source binding mismatch")
    if result["fit_audit_core_sha256"] == result["train_read_ledger_sha256"]:
        raise ValueError("Candidate5 fit audit and read ledger must be distinct")
    return result


def _validate_pass_artifact(
    data: bytes,
    *,
    expected_sha256: str,
    expected_rule: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
    verified_runtime_source_sha256: str,
) -> dict[str, Any]:
    expected = _require_sha(expected_sha256, name="Candidate5 TRAIN PASS SHA")
    if sha256_bytes(data) != expected:
        raise ValueError("Candidate5 TRAIN PASS byte binding mismatch")
    artifact = _parse_json_bytes(
        data, name="Candidate5 TRAIN PASS", canonical_required=True
    )
    if set(artifact) != PASS_KEYS:
        raise ValueError("Candidate5 TRAIN PASS schema mismatch")
    if (
        artifact["schema_version"] != SCHEMA_VERSION
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
        raise ValueError("Candidate5 TRAIN PASS status contract mismatch")
    if canonical_bytes(artifact["selected_rule"]) != canonical_bytes(dict(expected_rule)):
        raise ValueError("Candidate5 model rule differs from TRAIN PASS rule")
    bindings = _validate_bindings(
        artifact["bindings"],
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    if bindings != dict(expected_bindings):
        raise ValueError("Candidate5 model and TRAIN PASS bindings differ")
    checks = artifact["checks"]
    if not isinstance(checks, dict) or set(checks) != GATE_CHECK_KEYS:
        raise ValueError("Candidate5 TRAIN PASS check schema mismatch")
    if any(checks[key] is not True for key in GATE_CHECK_KEYS):
        raise ValueError("Candidate5 TRAIN PASS contains a failed gate")
    return artifact


def _validate_model(
    model: Mapping[str, Any],
    *,
    train_gate_pass_bytes: bytes,
    train_gate_pass_sha256: str,
    verified_runtime_source_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(model, Mapping) or set(model) != MODEL_KEYS:
        raise ValueError("Candidate5 model schema mismatch")
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
        raise ValueError("Candidate5 model status contract mismatch")
    false_inference_declarations = (
        "audit_metadata_feature_used",
        "bend_id_feature_used",
        "cad_feature_used",
        "candidate_count_feature_used",
        "case_identity_feature_used",
        "drawing_feature_used",
        "expected_count_feature_used",
        "filename_feature_used",
        "gold_or_truth_feature_used",
        "label_feature_used",
        "observed_source_or_prediction_multiplicity_used",
        "part_identity_feature_used",
        "runtime_identity_feature_used",
        "semantic_internal_scan_identity_feature_used",
        "sender_count_feature_used",
        "source_hypothesis_id_feature_used",
    )
    if any(model[field] is not False for field in false_inference_declarations):
        raise ValueError("Candidate5 model contains a forbidden inference declaration")
    rule = model["rule"]
    validate_rule(rule)
    if model["inference_features"] != [rule["feature"]]:
        raise ValueError("Candidate5 model must use exactly its single rule feature")
    allowed_feature_declarations = {
        "line_length_feature_used": rule["feature"] == "line_length_mm",
        "selection_route_feature_used": rule["feature"] == "selection_route",
        "selector_score_feature_used": rule["feature"] == "selector_score",
        "source_kind_feature_used": rule["feature"] == "source_kind",
    }
    if any(
        model[field] is not expected
        for field, expected in allowed_feature_declarations.items()
    ):
        raise ValueError("Candidate5 allowed feature-use declaration mismatch")
    bindings = _validate_bindings(
        model["bindings"],
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    pass_binding = model["train_gate_pass_binding"]
    if not isinstance(pass_binding, dict) or set(pass_binding) != {"kind", "sha256"}:
        raise ValueError("Candidate5 TRAIN PASS binding schema mismatch")
    if pass_binding["kind"] != PASS_KIND:
        raise ValueError("Candidate5 TRAIN PASS kind mismatch")
    pass_sha = _require_sha(
        train_gate_pass_sha256, name="Candidate5 supplied TRAIN PASS SHA"
    )
    if pass_binding["sha256"] != pass_sha or pass_sha in set(bindings.values()):
        raise ValueError("Candidate5 TRAIN PASS is not a distinct exact binding")
    _validate_pass_artifact(
        train_gate_pass_bytes,
        expected_sha256=pass_sha,
        expected_rule=rule,
        expected_bindings=bindings,
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    return rule


def preflight_candidate5_semantic_bytes(
    v51_semantic_bytes: bytes, *, v51_semantic_sha256: str
) -> dict[str, Any]:
    """Exercise the full semantic-only feature/rule path without labels or fit."""

    _semantic, predictions = _validate_semantic_bytes(
        v51_semantic_bytes, expected_sha256=v51_semantic_sha256
    )
    for row in predictions:
        features = extract_prediction_features(row)
        for feature in sorted(NUMERIC_FEATURES):
            value = features[feature]
            threshold = 0.0 if value is None else float(value)
            for operator in sorted(NUMERIC_OPERATORS):
                rule = {
                    "feature": feature,
                    "operator": operator,
                    "threshold": threshold,
                }
                validate_rule(rule)
                matched = _matches(rule, features)
                if matched is not (value is not None):
                    raise RuntimeError(
                        f"Candidate5 numeric rule path inconsistent: {feature}/{operator}"
                    )
        for feature in sorted(CATEGORICAL_FEATURES):
            own_rule = {
                "feature": feature,
                "operator": "eq",
                "value": str(features[feature]),
            }
            absent_rule = {
                "feature": feature,
                "operator": "eq",
                "value": "__candidate5_absent_source_value__",
            }
            validate_rule(own_rule)
            validate_rule(absent_rule)
            if not _matches(own_rule, features) or _matches(absent_rule, features):
                raise RuntimeError(
                    f"Candidate5 categorical rule path inconsistent: {feature}"
                )
        sanitized = _sanitize_prediction(row)
        if set(sanitized) - SANITIZED_PREDICTION_KEYS:
            raise RuntimeError("Candidate5 sanitized prediction projection failed")
    return {
        "all_frozen_rule_forms_exercised": True,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "candidate_metric_computed": False,
        "deletion_rule_selected": False,
        "feature_extraction_complete": True,
        "generic_source_count_and_audit_metadata_ignored": True,
        "holdout_score_opened": False,
        "input_v51_semantic_sha256": v51_semantic_sha256,
        "kind": PREFLIGHT_RESULT_KIND,
        "labels_used": False,
        "manifest_scan_identity_required_externally": True,
        "model_fit_performed": False,
        "prediction_output_projection_sanitized": True,
        "recursive_forbidden_key_validation_exercised": True,
        "schema_version": SCHEMA_VERSION,
        "semantic_internal_scan_identity_used": False,
        "source_only": True,
        "status": "SOURCE_COMPATIBLE",
    }


def apply_candidate5(
    v51_semantic_bytes: bytes,
    model: Mapping[str, Any],
    train_gate_pass_bytes: bytes,
    candidate3_terminal_bytes: bytes,
    candidate4_terminal_bytes: bytes,
    *,
    v51_semantic_sha256: str,
    train_gate_pass_sha256: str,
    candidate3_terminal_sha256: str,
    candidate4_terminal_sha256: str,
    verified_runtime_source_sha256: str,
) -> dict[str, Any]:
    """Apply one frozen rule only after all external seals validate."""

    validate_candidate3_terminal_bytes(
        candidate3_terminal_bytes, expected_sha256=candidate3_terminal_sha256
    )
    validate_candidate4_terminal_bytes(
        candidate4_terminal_bytes, expected_sha256=candidate4_terminal_sha256
    )
    rule = _validate_model(
        model,
        train_gate_pass_bytes=train_gate_pass_bytes,
        train_gate_pass_sha256=train_gate_pass_sha256,
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    bindings = model["bindings"]
    if (
        bindings["candidate3_terminal_sha256"] != candidate3_terminal_sha256
        or bindings["candidate4_terminal_sha256"] != candidate4_terminal_sha256
    ):
        raise ValueError("Candidate5 model does not bind supplied predecessor terminals")

    _semantic, predictions = _validate_semantic_bytes(
        v51_semantic_bytes, expected_sha256=v51_semantic_sha256
    )
    retained: list[dict[str, Any]] = []
    deletion_audit: list[dict[str, Any]] = []
    for row in predictions:
        features = extract_prediction_features(row)
        if not _matches(rule, features):
            retained.append(_sanitize_prediction(row))
            continue
        feature = str(rule["feature"])
        deletion_audit.append(
            {
                "bend_id": row["bend_id"],
                "feature": feature,
                "feature_value": features[feature],
                "reason": "frozen_train_nested_single_rule",
            }
        )

    result = {
        "accounting": {
            "deleted_prediction_count": len(predictions) - len(retained),
            "input_prediction_count": len(predictions),
            "output_prediction_count": len(retained),
        },
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "audit_metadata_feature_used": False,
        "bend_id_feature_used": False,
        "cad_feature_used": False,
        "cad_used": False,
        "candidate3_terminal_sha256": candidate3_terminal_sha256,
        "candidate4_terminal_sha256": candidate4_terminal_sha256,
        "candidate5_model_sha256": sha256_bytes(canonical_bytes(dict(model))),
        "candidate_count_feature_used": False,
        "case_id_used": False,
        "case_identity_feature_used": False,
        "case_identity_used": False,
        "deletion_audit": deletion_audit,
        "deletion_only": True,
        "drawing_used": False,
        "drawing_feature_used": False,
        "expected_count_feature_used": False,
        "expected_count_used": False,
        "filename_feature_used": False,
        "filename_used": False,
        "gold_or_truth_feature_used": False,
        "gold_count_used": False,
        "holdout_score_opened": False,
        "inference_features": list(model["inference_features"]),
        "input_v51_semantic_sha256": v51_semantic_sha256,
        "kind": RESULT_KIND,
        "label_feature_used": False,
        "label_count_used_at_inference": False,
        "labels_used": False,
        "line_length_feature_used": rule["feature"] == "line_length_mm",
        "observed_source_or_prediction_multiplicity_used": False,
        "part_identity_feature_used": False,
        "part_identity_used": False,
        "post_holdout_model_or_threshold_mutation": (
            "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
        ),
        "predictions": retained,
        "profile": PROFILE,
        "promotion_eligible": False,
        "research_only": True,
        "rule": dict(rule),
        "runtime_id_used": False,
        "runtime_identity_feature_used": False,
        "scan_identity_authority": "ATTESTED_CACHE_MANIFEST_ONLY",
        "schema_version": SCHEMA_VERSION,
        "selection_route_feature_used": rule["feature"] == "selection_route",
        "selector_score_feature_used": rule["feature"] == "selector_score",
        "semantic_internal_scan_identity_used": False,
        "semantic_internal_scan_identity_feature_used": False,
        "sender_count_used": False,
        "sender_count_feature_used": False,
        "source_only": True,
        "source_hypothesis_id_feature_used": False,
        "source_kind_feature_used": rule["feature"] == "source_kind",
        "train_gate_pass_sha256": train_gate_pass_sha256,
        "training_gate_passed": True,
        "v51_fallback_default": True,
    }
    result["candidate5_result_sha256"] = sha256_bytes(canonical_bytes(result))
    return result
