"""Filesystem-blind Candidate6 conservative recall recovery runtime.

Candidate6 is a new additive attempt after Candidate5 terminalized.  It keeps
every V51 prediction in its original order and may append at most one line
derived from the scan-only candidate pool's ``line_objects`` array.  Raw pool
candidates, identifiers, list order, counts, and audit metadata are ignored.

The model family is a fixed 32-rule grid over one selected source-derived
feature plus a fixed allowed-source-kind eligibility predicate.  Missing
evidence abstains.  Before addition, evaluator-aligned finite-line
geometry enforces a 4.0 mm ownership/NMS boundary with symmetric transverse
offset and bidirectional span-overlap checks.  This module has no filesystem,
label, evaluator, case, filename, expected-count, CAD, drawing, or holdout
capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 6
PROFILE = "F1_AUG30_TRAIN18_CANDIDATE6_CONSERVATIVE_RECALL_RECOVERY_V1"
MODEL_KIND = "sherman_qc_f1_aug30_train18_candidate6_recall_model_v1"
PASS_KIND = "sherman_qc_f1_aug30_train18_candidate6_train_gate_pass_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate6_recall_result_v1"
PREFLIGHT_RESULT_KIND = (
    "sherman_qc_f1_aug30_train18_candidate6_semantic_pool_compatibility_result_v1"
)
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate_terminal_v1"
DISPOSITION_KIND = "sherman_qc_f1_aug30_train18_candidate5_terminal_disposition_v1"

CANDIDATE5_TERMINAL_PATH = (
    "output/research/f1-aug30-train18-candidate5-conservative-deletion-v1/"
    "terminal_negative.json"
)
CANDIDATE5_TERMINAL_SHA256 = (
    "6b7d448b1c0b16c12cc32debb61c5255dc3ddb18956f4807b0c0090fbddc9049"
)
CANDIDATE5_TERMINAL_SIZE_BYTES = 2896
CANDIDATE5_TERMINAL_STATUS = "REJECTED_TRAIN_GATE"
CANDIDATE5_DISPOSITION_SHA256 = (
    "a215a4a514e631086ced6bcb2deb0e1b01f218defdb6d70432ac1a57435aebe4"
)
CANDIDATE5_DISPOSITION_SIZE_BYTES = 3335

PREDECESSOR_LINEAGE = {
    "candidate1_terminal": {
        "path": "research/f1_aug30_train18_candidate1_terminal_negative_v1.json",
        "sha256": "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8",
        "size_bytes": 3989,
        "status": "REJECTED_TRAIN_GATE",
    },
    "candidate2_terminal": {
        "path": "research/f1_aug30_train18_candidate2_terminal_negative_v1.json",
        "sha256": "ecdbe871364044214fc46e5b5e59095cc9b38cefe9678292f1a8462dffdf3515",
        "size_bytes": 7001,
        "status": "REJECTED_PREFIT_SCHEMA_INCOMPATIBILITY",
    },
    "candidate3_terminal": {
        "path": "research/f1_aug30_train18_candidate3_terminal_negative_v1.json",
        "sha256": "00449059f9e4cd989f3fbcb67e9fc8e7a499ad1dbca0a947deccd75d21019116",
        "size_bytes": 5518,
        "status": "REJECTED_PREFIT_CUSTODY_BOUNDARY_VIOLATION",
    },
    "candidate4_terminal": {
        "path": "research/f1_aug30_train18_candidate4_terminal_negative_v1.json",
        "sha256": "10d6bd14ad10dfd1ace464e551cb79f65abeb3c12bc0094fb2695f61a1aa2a50",
        "size_bytes": 5384,
        "status": "REJECTED_PREFIT_SOURCE_SCHEMA_INCOMPATIBILITY",
    },
    "candidate5_terminal": {
        "path": CANDIDATE5_TERMINAL_PATH,
        "sha256": CANDIDATE5_TERMINAL_SHA256,
        "size_bytes": CANDIDATE5_TERMINAL_SIZE_BYTES,
        "status": CANDIDATE5_TERMINAL_STATUS,
    },
}

KNOWN_BINDINGS = {
    "candidate1_terminal_sha256": PREDECESSOR_LINEAGE["candidate1_terminal"]["sha256"],
    "candidate2_terminal_sha256": PREDECESSOR_LINEAGE["candidate2_terminal"]["sha256"],
    "candidate3_terminal_sha256": PREDECESSOR_LINEAGE["candidate3_terminal"]["sha256"],
    "candidate4_terminal_sha256": PREDECESSOR_LINEAGE["candidate4_terminal"]["sha256"],
    "candidate5_terminal_sha256": CANDIDATE5_TERMINAL_SHA256,
    "candidate5_terminal_disposition_sha256": CANDIDATE5_DISPOSITION_SHA256,
    "sealed_split_sha256": "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185",
    "train_allowlist_sha256": "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0",
    "original_preregistration_sha256": "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a",
    "custody_label_attestation_sha256": "fe9313bbd09de945fc4da0c238a6b1f2769979e78cccc2abff29ca434dbf84ad",
    "custody_cache_attestation_sha256": "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71",
    "custody_staged_validator_pass_sha256": "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433",
}

BINDING_KEYS = frozenset(
    {
        *KNOWN_BINDINGS,
        "candidate6_preregistration_sha256",
        "candidate6_runtime_source_sha256",
        "candidate6_optimizer_source_sha256",
        "candidate6_runtime_tests_sha256",
        "candidate6_optimizer_tests_sha256",
        "candidate6_preflight_source_sha256",
        "candidate6_preflight_tests_sha256",
        "candidate6_preflight_report_sha256",
        "candidate6_preflight_lock_sha256",
        "candidate6_preflight_read_ledger_sha256",
        "evaluator_source_sha256",
        "fit_audit_core_sha256",
        "train_read_ledger_sha256",
        "train_label_projection_sha256",
        "train_cache_manifest_sha256",
        "train_source_cache_binding_sha256",
    }
)

OWNERSHIP_ANGLE_DEG = 5.0
OWNERSHIP_TRANSVERSE_OFFSET_MM = 4.0
OWNERSHIP_INCUMBENT_SPAN_OVERLAP = 0.60
OWNERSHIP_CANDIDATE_SPAN_OVERLAP = 0.60
OWNERSHIP_ANGLE_NUMERIC_MARGIN_DEG = 1.0e-5
OWNERSHIP_LINEAR_NUMERIC_MARGIN_MM = 1.0e-9
OWNERSHIP_OVERLAP_NUMERIC_MARGIN = 1.0e-12
MAX_ABS_COORDINATE_MM = 1_000_000.0
MIN_CONDITIONED_LINE_LENGTH_MM = 1.0e-3
MAX_CONDITIONED_LINE_LENGTH_MM = 2_000_000.0
MAX_ADDITIONS_PER_CASE = 1
MAX_ADDED_FALSE_POSITIVES_TRAIN18 = 2
FIXED_ELIGIBILITY_INPUTS = ("finite_line_geometry", "source_kind")
DETERMINISTIC_TIE_BREAK_INPUTS = (
    "canonical_line_endpoints",
    "source_kind",
)
DETERMINISTIC_TIE_BREAK_DIRECTION = "CANONICAL_JSON_BYTES_ASCENDING"
RESERVED_BEND_ID_PREFIX = "C6R-"
RESERVED_SOURCE_HYPOTHESIS_ID_PREFIX = "C6G-"

NUMERIC_RULE_THRESHOLDS = {
    "line_length_mm": (5.0, 10.0, 20.0, 40.0, 80.0),
    "score": (0.0, 0.25, 0.5, 1.0, 2.0),
    "finite_support_fraction": (0.25, 0.5, 0.75, 0.9),
    "endpoint_support_fraction": (0.25, 0.5, 0.75, 0.9),
    "two_sided_transition_score": (0.25, 0.5, 0.75, 0.9),
    "normal_second_eigenvalue": (0.01, 0.05, 0.1),
}
SOURCE_KIND_RULE_VALUES = (
    "strict_blind_owned",
    "bend_hypothesis",
    "fragment_merge",
    "cylindrical_standard",
    "cylindrical_wide",
    "plane_lattice",
    "scan_mesh_adjacency",
)
ALLOWED_SOURCE_KINDS = frozenset(
    {
        *SOURCE_KIND_RULE_VALUES,
        "control_support_strip",
        "endpoint_connector",
        "evidence_trace",
    }
)
NUMERIC_FEATURES = frozenset(NUMERIC_RULE_THRESHOLDS)
CATEGORICAL_FEATURES = frozenset({"source_kind"})
ALLOWED_RULE_FEATURES = NUMERIC_FEATURES | CATEGORICAL_FEATURES


def _fixed_rule_space() -> tuple[dict[str, Any], ...]:
    rules: list[dict[str, Any]] = []
    for feature, thresholds in NUMERIC_RULE_THRESHOLDS.items():
        for threshold in thresholds:
            rules.append(
                {"feature": feature, "operator": "ge", "threshold": threshold}
            )
    for value in SOURCE_KIND_RULE_VALUES:
        rules.append({"feature": "source_kind", "operator": "eq", "value": value})
    return tuple(rules)


FIXED_RULE_SPACE = _fixed_rule_space()
if len(FIXED_RULE_SPACE) != 32:
    raise RuntimeError("Candidate6 fixed rule space must contain exactly 32 rules")

REQUIRED_SEMANTIC_FLAGS = {
    "source_only": True,
    "research_only": True,
    "labels_used": False,
    "cad_used": False,
    "part_identity_used": False,
    "expected_count_used": False,
}
OPTIONAL_FALSE_SEMANTIC_FLAGS = frozenset(
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
TOP_LEVEL_SEMANTIC_GUARDS = (
    frozenset(REQUIRED_SEMANTIC_FLAGS) | OPTIONAL_FALSE_SEMANTIC_FLAGS
)
POOL_INFERENCE_CONTRACT = {
    "cad_exposed": False,
    "expected_count_exposed": False,
    "manual_truth_exposed": False,
    "part_identifier_exposed": False,
}

FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "cad",
        "drawing",
        "expected",
        "file",
        "filename",
        "gold",
        "holdout",
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
        "holdoutstatus",
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

REQUIRED_PREDICTION_STRING_FIELDS = (
    "bend_id",
    "selection_route",
    "source_hypothesis_id",
    "source_kind",
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
    "line_length_mm": "line_length_feature_used",
    "score": "score_feature_used",
    "finite_support_fraction": "finite_support_fraction_feature_used",
    "endpoint_support_fraction": "endpoint_support_fraction_feature_used",
    "two_sided_transition_score": "two_sided_transition_score_feature_used",
    "normal_second_eigenvalue": "normal_second_eigenvalue_feature_used",
    "source_kind": "source_kind_feature_used",
}

MODEL_KEYS = frozenset(
    {
        "additive_only",
        "attempt_ordinal",
        "bindings",
        "deterministic_tie_break_direction",
        "deterministic_tie_break_inputs",
        "freeze_status",
        "holdout_score_opened",
        "external_launcher_authentication_required",
        "fixed_eligibility_inputs",
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
        "raw_candidates_ignored",
        "rule",
        "schema_version",
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
        "train18_allowlist_exact",
        "train_read_ledger_exact",
        "true_positive_nondecrease_every_case",
        "v51_prediction_values_and_order_preserved",
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


def _strict_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric and non-boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


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
    tokens = _key_tokens(key)
    compact = re.sub(r"[^a-z0-9]+", "", key.lower())
    truth_class_alias = bool(
        {"true", "positive"}.issubset(tokens)
        or {"false", "positive"}.issubset(tokens)
        or {"true", "negative"}.issubset(tokens)
        or {"false", "negative"}.issubset(tokens)
        or tokens.intersection({"tp", "fp", "tn", "fn"})
    )
    return bool(
        tokens.intersection(FORBIDDEN_KEY_TOKENS)
        or compact in FORBIDDEN_COMPACT_KEYS
        or {"file", "name"}.issubset(tokens)
        or truth_class_alias
    )


def _safe_false_guard(path: str, key: str, value: Any) -> bool:
    if path == "V51 semantic artifact" and key in TOP_LEVEL_SEMANTIC_GUARDS:
        return value is False or key in {"source_only", "research_only"}
    if path == "candidate pool.inference_contract" and key in POOL_INFERENCE_CONTRACT:
        return value is False
    return False


def _validate_no_forbidden_keys(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            if not _safe_false_guard(path, key, child) and _forbidden_key(key):
                raise ValueError(f"{path} exposes forbidden key: {key}")
            _validate_no_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_forbidden_keys(child, path=f"{path}[{index}]")


def _point(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-coordinate JSON array")
    point = tuple(
        _strict_number(coordinate, name=f"{name}[{index}]")
        for index, coordinate in enumerate(value)
    )
    if any(abs(coordinate) > MAX_ABS_COORDINATE_MM for coordinate in point):
        raise ValueError(f"{name} exceeds the conditioned physical coordinate domain")
    return point


def _vector_subtract(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))  # type: ignore[return-value]


def _vector_add(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(float(left[index]) + float(right[index]) for index in range(3))  # type: ignore[return-value]


def _scale(value: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(coordinate) * scalar for coordinate in value)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _canonical_endpoints(
    first: Sequence[float], second: Sequence[float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if len(first) != 3 or len(second) != 3:
        raise ValueError("line endpoints must be three-dimensional")
    left = tuple(
        _strict_number(value, name=f"line endpoint[0][{index}]")
        for index, value in enumerate(first)
    )
    right = tuple(
        _strict_number(value, name=f"line endpoint[1][{index}]")
        for index, value in enumerate(second)
    )
    if any(
        abs(coordinate) > MAX_ABS_COORDINATE_MM
        for point in (left, right)
        for coordinate in point
    ):
        raise ValueError("line endpoints exceed the conditioned coordinate domain")
    length = _norm(_vector_subtract(right, left))
    if not MIN_CONDITIONED_LINE_LENGTH_MM <= length <= MAX_CONDITIONED_LINE_LENGTH_MM:
        raise ValueError("line geometry leaves the conditioned physical length domain")
    return (left, right) if left <= right else (right, left)


def _endpoints_equivalent(
    left: tuple[tuple[float, float, float], tuple[float, float, float]],
    right: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> bool:
    return all(
        abs(left[index][axis] - right[index][axis]) <= 1e-6
        for index in range(2)
        for axis in range(3)
    )


def _endpoints_from_row(
    row: Mapping[str, Any], *, name: str, require_prediction_lines: bool = False
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    representations: list[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = []
    if row.get("span_endpoints") is not None:
        span = row["span_endpoints"]
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f"{name}.span_endpoints must be finite 2x3 geometry")
        representations.append(
            _canonical_endpoints(
                _point(span[0], name=f"{name}.span_endpoints[0]"),
                _point(span[1], name=f"{name}.span_endpoints[1]"),
            )
        )
    line_present = row.get("line_start") is not None or row.get("line_end") is not None
    if line_present:
        if row.get("line_start") is None or row.get("line_end") is None:
            raise ValueError(f"{name} line_start/line_end must be jointly present")
        representations.append(
            _canonical_endpoints(
                _point(row["line_start"], name=f"{name}.line_start"),
                _point(row["line_end"], name=f"{name}.line_end"),
            )
        )
    anchor_present = any(
        row.get(field) is not None for field in ("anchor", "axis_direction", "span_mm")
    )
    if anchor_present:
        if any(
            row.get(field) is None for field in ("anchor", "axis_direction", "span_mm")
        ):
            raise ValueError(f"{name} anchor/axis/span must be jointly present")
        anchor = _point(row["anchor"], name=f"{name}.anchor")
        axis = _point(row["axis_direction"], name=f"{name}.axis_direction")
        span_mm = _strict_number(row["span_mm"], name=f"{name}.span_mm")
        axis_norm = _norm(axis)
        if axis_norm <= 1e-12 or span_mm <= 1e-9:
            raise ValueError(f"{name} anchor/axis/span geometry is degenerate")
        half = _scale(axis, 0.5 * span_mm / axis_norm)
        representations.append(
            _canonical_endpoints(
                _vector_subtract(anchor, half), _vector_add(anchor, half)
            )
        )
    if require_prediction_lines and not line_present:
        raise ValueError(f"{name} requires line_start and line_end")
    if not representations:
        return None
    reference = representations[0]
    if any(not _endpoints_equivalent(reference, value) for value in representations[1:]):
        raise ValueError(f"{name} contains conflicting line representations")
    return reference


def _validate_text(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a bounded nonempty string")
    return value


def _validate_semantic_bytes(
    data: bytes, *, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_bytes(data) != _require_sha(expected_sha256, name="V51 semantic SHA"):
        raise ValueError("V51 semantic bytes differ from their external binding")
    semantic = _parse_json_bytes(
        data, name="V51 semantic artifact", canonical_required=False
    )
    for field, expected in REQUIRED_SEMANTIC_FLAGS.items():
        if field not in semantic or semantic[field] is not expected:
            raise ValueError(f"V51 semantic artifact requires {field}={expected!r}")
    for field in OPTIONAL_FALSE_SEMANTIC_FLAGS:
        if field in semantic and semantic[field] is not False:
            raise ValueError(f"V51 semantic artifact requires optional {field}=false")
    _validate_no_forbidden_keys(semantic, path="V51 semantic artifact")
    raw_predictions = semantic.get("predictions")
    if not isinstance(raw_predictions, list):
        raise ValueError("V51 semantic predictions must be a list")
    predictions: list[dict[str, Any]] = []
    bend_ids: set[str] = set()
    for index, raw in enumerate(raw_predictions):
        if not isinstance(raw, dict):
            raise ValueError(f"V51 prediction {index} must be an object")
        for field in REQUIRED_PREDICTION_STRING_FIELDS:
            _validate_text(raw.get(field), name=f"V51 prediction {index}.{field}")
        bend_id = str(raw["bend_id"])
        if bend_id.startswith(RESERVED_BEND_ID_PREFIX):
            raise ValueError("V51 prediction uses the reserved Candidate6 bend-id namespace")
        if str(raw["source_hypothesis_id"]).startswith(
            RESERVED_SOURCE_HYPOTHESIS_ID_PREFIX
        ):
            raise ValueError(
                "V51 prediction uses the reserved Candidate6 source-id namespace"
            )
        if bend_id in bend_ids:
            raise ValueError(f"duplicate V51 prediction bend_id: {bend_id}")
        bend_ids.add(bend_id)
        if "selector_score" in raw and raw["selector_score"] is not None:
            _strict_number(raw["selector_score"], name=f"prediction {bend_id}.selector_score")
        _endpoints_from_row(
            raw, name=f"V51 prediction {bend_id}", require_prediction_lines=True
        )
        predictions.append(dict(raw))
    return semantic, predictions


def _candidate_source_kind(row: Mapping[str, Any]) -> str | None:
    aliases: dict[str, str] = {}
    for field in ("representative_source", "candidate_source", "source_kind"):
        raw = row.get(field)
        if raw is not None and raw != "":
            aliases[field] = _validate_text(raw, name=f"line object {field}")
    semantic_aliases = {
        value
        for field, value in aliases.items()
        if not (field == "candidate_source" and value == "cross_source_line_object")
    }
    if len(semantic_aliases) > 1:
        raise ValueError("line object source-kind aliases disagree")
    if not semantic_aliases:
        return None
    value = next(iter(semantic_aliases))
    return value if value in ALLOWED_SOURCE_KINDS else None


def _optional_feature(row: Mapping[str, Any], name: str) -> float | None:
    if name not in row or row[name] is None:
        return None
    value = _strict_number(row[name], name=f"line object.{name}")
    if name in {
        "endpoint_support_fraction",
        "finite_support_fraction",
        "two_sided_transition_score",
    } and not 0.0 <= value <= 1.0:
        raise ValueError(f"line object.{name} must be in the physical [0,1] domain")
    if name == "normal_second_eigenvalue" and value < 0.0:
        raise ValueError(
            "line object.normal_second_eigenvalue must be nonnegative"
        )
    return value


def _candidate_from_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    endpoints = _endpoints_from_row(row, name="line object")
    source_kind = _candidate_source_kind(row)
    if endpoints is None or source_kind is None:
        return None
    length = _norm(_vector_subtract(endpoints[1], endpoints[0]))
    features: dict[str, float | str | None] = {
        "line_length_mm": length,
        "source_kind": source_kind,
    }
    for feature in NUMERIC_FEATURES - {"line_length_mm"}:
        features[feature] = _optional_feature(row, feature)
    identity_projection = {
        "line_end": list(endpoints[1]),
        "line_start": list(endpoints[0]),
        "source_kind": source_kind,
    }
    digest = sha256_bytes(canonical_bytes(identity_projection))
    return {
        "digest": digest,
        "features": features,
        "identity_projection": identity_projection,
    }


def _validate_pool_bytes(
    data: bytes, *, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_bytes(data) != _require_sha(expected_sha256, name="candidate pool SHA"):
        raise ValueError("candidate pool bytes differ from their external binding")
    pool = _parse_json_bytes(data, name="candidate pool", canonical_required=False)
    if (
        pool.get("schema_version") != 3
        or pool.get("algorithm") != "f1_uniform_scan_only_candidate_pool_v3"
        or pool.get("inference_contract") != POOL_INFERENCE_CONTRACT
        or pool.get("source_point_serialization") is not False
        or _strict_int(pool.get("working_point_budget"), name="working_point_budget")
        != 200000
    ):
        raise ValueError("candidate pool source-only header mismatch")
    _validate_no_forbidden_keys(pool, path="candidate pool")
    raw_line_objects = pool.get("line_objects")
    if not isinstance(raw_line_objects, list):
        raise ValueError("candidate pool line_objects must be a list")
    candidates: list[dict[str, Any]] = []
    for row in raw_line_objects:
        if not isinstance(row, dict):
            raise ValueError("candidate pool line object must be an object")
        candidate = _candidate_from_row(row)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda value: canonical_bytes(value["identity_projection"]))
    return pool, candidates


def fixed_rule_space() -> tuple[dict[str, Any], ...]:
    return tuple(dict(rule) for rule in FIXED_RULE_SPACE)


def validate_rule(rule: Mapping[str, Any]) -> None:
    if not isinstance(rule, Mapping):
        raise ValueError("Candidate6 rule must be an object")
    feature = rule.get("feature")
    if feature not in ALLOWED_RULE_FEATURES:
        raise ValueError(f"Candidate6 unsupported rule feature: {feature}")
    if feature in NUMERIC_FEATURES:
        if set(rule) != {"feature", "operator", "threshold"} or rule.get("operator") != "ge":
            raise ValueError("Candidate6 numeric rule must use exact ge schema")
        _strict_number(rule.get("threshold"), name="Candidate6 rule threshold")
    else:
        if set(rule) != {"feature", "operator", "value"} or rule.get("operator") != "eq":
            raise ValueError("Candidate6 categorical rule must use exact eq schema")
        _validate_text(rule.get("value"), name="Candidate6 rule value")
    encoded = canonical_bytes(dict(rule))
    if encoded not in {canonical_bytes(value) for value in FIXED_RULE_SPACE}:
        raise ValueError("Candidate6 rule is outside the frozen 32-rule grid")


def _rule_matches(
    rule: Mapping[str, Any], features: Mapping[str, float | str | None]
) -> bool:
    value = features[str(rule["feature"])]
    if value is None:
        return False
    if rule["operator"] == "eq":
        return value == rule["value"]
    return float(value) >= float(rule["threshold"])


def _candidate_rank(rule: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    value = candidate["features"][rule["feature"]]
    geometry_key = canonical_bytes(candidate["identity_projection"])
    if rule["operator"] == "ge":
        return (-float(value), geometry_key)
    return (geometry_key,)


def line_pair_metrics(
    incumbent: tuple[tuple[float, float, float], tuple[float, float, float]],
    candidate: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> dict[str, float]:
    incumbent = _canonical_endpoints(incumbent[0], incumbent[1])
    candidate = _canonical_endpoints(candidate[0], candidate[1])
    incumbent_midpoint = _scale(_vector_add(incumbent[0], incumbent[1]), 0.5)
    candidate_midpoint = _scale(_vector_add(candidate[0], candidate[1]), 0.5)
    incumbent_delta = _vector_subtract(incumbent[1], incumbent[0])
    candidate_delta = _vector_subtract(candidate[1], candidate[0])
    incumbent_length = _norm(incumbent_delta)
    candidate_length = _norm(candidate_delta)
    if incumbent_length <= 1e-9 or candidate_length <= 1e-9:
        raise ValueError("ownership lines must be nondegenerate")
    incumbent_axis = _scale(incumbent_delta, 1.0 / incumbent_length)
    candidate_axis = _scale(candidate_delta, 1.0 / candidate_length)
    raw_dot = max(-1.0, min(1.0, _dot(incumbent_axis, candidate_axis)))
    angle = math.degrees(math.acos(abs(raw_dot)))
    if raw_dot < 0.0:
        candidate_axis = _scale(candidate_axis, -1.0)
    midpoint_delta = _vector_subtract(candidate_midpoint, incumbent_midpoint)
    incumbent_residual = _vector_subtract(
        midpoint_delta, _scale(incumbent_axis, _dot(midpoint_delta, incumbent_axis))
    )
    candidate_residual = _vector_subtract(
        midpoint_delta, _scale(candidate_axis, _dot(midpoint_delta, candidate_axis))
    )
    transverse_offset = max(_norm(incumbent_residual), _norm(candidate_residual))
    incumbent_interval = sorted(
        _dot(_vector_subtract(point, incumbent_midpoint), incumbent_axis)
        for point in incumbent
    )
    candidate_interval = sorted(
        _dot(_vector_subtract(point, incumbent_midpoint), incumbent_axis)
        for point in candidate
    )
    overlap = max(
        0.0,
        min(incumbent_interval[1], candidate_interval[1])
        - max(incumbent_interval[0], candidate_interval[0]),
    )
    result = {
        "axis_angle_deg": angle,
        "candidate_span_overlap": overlap / candidate_length,
        "incumbent_span_overlap": overlap / incumbent_length,
        "transverse_offset_mm": transverse_offset,
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("ownership geometry produced a non-finite derived metric")
    return result


def _ownership_conflict(
    candidate: tuple[tuple[float, float, float], tuple[float, float, float]],
    incumbents: Sequence[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> bool:
    for incumbent in incumbents:
        metrics = line_pair_metrics(incumbent, candidate)
        if (
            metrics["axis_angle_deg"]
            <= OWNERSHIP_ANGLE_DEG + OWNERSHIP_ANGLE_NUMERIC_MARGIN_DEG
            and metrics["transverse_offset_mm"]
            <= OWNERSHIP_TRANSVERSE_OFFSET_MM + OWNERSHIP_LINEAR_NUMERIC_MARGIN_MM
            and metrics["incumbent_span_overlap"]
            >= OWNERSHIP_INCUMBENT_SPAN_OVERLAP - OWNERSHIP_OVERLAP_NUMERIC_MARGIN
            and metrics["candidate_span_overlap"]
            >= OWNERSHIP_CANDIDATE_SPAN_OVERLAP - OWNERSHIP_OVERLAP_NUMERIC_MARGIN
        ):
            return True
    return False


def _select_addition(
    predictions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
) -> dict[str, Any] | None:
    validate_rule(rule)
    incumbents = [
        _endpoints_from_row(
            prediction,
            name=f"V51 prediction {index}",
            require_prediction_lines=True,
        )
        for index, prediction in enumerate(predictions)
    ]
    incumbent_lines = [value for value in incumbents if value is not None]
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        if not _rule_matches(rule, candidate["features"]):
            continue
        projection = candidate["identity_projection"]
        endpoints = (
            tuple(projection["line_start"]),
            tuple(projection["line_end"]),
        )
        if _ownership_conflict(endpoints, incumbent_lines):
            continue
        eligible.append(dict(candidate))
    if not eligible:
        return None
    eligible.sort(key=lambda value: _candidate_rank(rule, value))
    return eligible[0]


def _addition_prediction(candidate: Mapping[str, Any]) -> dict[str, Any]:
    projection = candidate["identity_projection"]
    digest = str(candidate["digest"])
    return {
        "bend_id": f"{RESERVED_BEND_ID_PREFIX}{digest[:20]}",
        "line_end": list(projection["line_end"]),
        "line_start": list(projection["line_start"]),
        "selection_route": "candidate6_conservative_recall_recovery_v1",
        "selector_score": None,
        "source_hypothesis_id": (
            f"{RESERVED_SOURCE_HYPOTHESIS_ID_PREFIX}{digest[:20]}"
        ),
        "source_kind": str(projection["source_kind"]),
    }


def _ownership_contract() -> dict[str, Any]:
    return {
        "angle_deg_max": OWNERSHIP_ANGLE_DEG,
        "candidate_span_overlap_min": OWNERSHIP_CANDIDATE_SPAN_OVERLAP,
        "conditioned_geometry_domain": {
            "absolute_coordinate_mm_max": MAX_ABS_COORDINATE_MM,
            "line_length_mm_max": MAX_CONDITIONED_LINE_LENGTH_MM,
            "line_length_mm_min": MIN_CONDITIONED_LINE_LENGTH_MM,
        },
        "geometry": "EVALUATOR_ALIGNED_FINITE_LINE",
        "incumbent_span_overlap_min": OWNERSHIP_INCUMBENT_SPAN_OVERLAP,
        "numeric_decision_equivalence": (
            "CONSERVATIVE_SUPPRESSION_WITHIN_DECLARED_STDLIB_EPSILON_BAND"
        ),
        "numeric_margins": {
            "angle_deg": OWNERSHIP_ANGLE_NUMERIC_MARGIN_DEG,
            "linear_mm": OWNERSHIP_LINEAR_NUMERIC_MARGIN_MM,
            "overlap_fraction": OWNERSHIP_OVERLAP_NUMERIC_MARGIN,
        },
        "symmetric_transverse_offset": True,
        "transverse_offset_mm_max": OWNERSHIP_TRANSVERSE_OFFSET_MM,
    }


def _validate_candidate5_terminal(
    data: bytes, *, expected_sha256: str
) -> dict[str, Any]:
    if expected_sha256 != CANDIDATE5_TERMINAL_SHA256:
        raise ValueError("Candidate5 terminal differs from the audited binding")
    if len(data) != CANDIDATE5_TERMINAL_SIZE_BYTES or sha256_bytes(data) != expected_sha256:
        raise ValueError("Candidate5 terminal byte binding mismatch")
    terminal = _parse_json_bytes(data, name="Candidate5 terminal", canonical_required=True)
    if (
        terminal.get("schema_version") != 1
        or terminal.get("kind") != TERMINAL_KIND
        or terminal.get("candidate_id") != 5
        or terminal.get("attempt_ordinal") != 5
        or terminal.get("status") != CANDIDATE5_TERMINAL_STATUS
        or terminal.get("candidate5_continuation_authorized") is not False
        or terminal.get("holdout_execution_authorized") is not False
        or terminal.get("holdout_access_count") != 0
        or terminal.get("holdout_score_opened") is not False
        or terminal.get("promotion_eligible") is not False
        or terminal.get("post_terminal_candidate5_model_or_threshold_mutation")
        != "FORBIDDEN"
    ):
        raise ValueError("Candidate5 terminal status contract mismatch")
    return terminal


def _validate_candidate5_disposition(
    data: bytes, *, expected_sha256: str
) -> dict[str, Any]:
    if expected_sha256 != CANDIDATE5_DISPOSITION_SHA256:
        raise ValueError("Candidate5 disposition differs from the audited binding")
    if len(data) != CANDIDATE5_DISPOSITION_SIZE_BYTES or sha256_bytes(data) != expected_sha256:
        raise ValueError("Candidate5 disposition byte binding mismatch")
    value = _parse_json_bytes(data, name="Candidate5 disposition", canonical_required=True)
    if (
        value.get("schema_version") != 1
        or value.get("kind") != DISPOSITION_KIND
        or value.get("status") != "CONFIRMED_REJECTED_TRAIN_GATE"
        or value.get("candidate_id") != 5
        or value.get("attempt_ordinal") != 5
        or value.get("candidate5_continuation_authorized") is not False
        or value.get("candidate6_engineering_successor_authorized") is not True
        or value.get("holdout_access_count") != 0
        or value.get("holdout_score_opened") is not False
        or value.get("candidate5_terminal_binding") != PREDECESSOR_LINEAGE["candidate5_terminal"]
    ):
        raise ValueError("Candidate5 disposition status contract mismatch")
    return value


def _validate_bindings(
    value: Any, *, verified_runtime_source_sha256: str
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != BINDING_KEYS:
        raise ValueError("Candidate6 binding schema mismatch")
    bindings = {
        key: _require_sha(value[key], name=f"bindings.{key}")
        for key in sorted(BINDING_KEYS)
    }
    for key, expected in KNOWN_BINDINGS.items():
        if bindings[key] != expected:
            raise ValueError(f"Candidate6 immutable binding mismatch: {key}")
    verified = _require_sha(
        verified_runtime_source_sha256, name="verified Candidate6 runtime source SHA"
    )
    if bindings["candidate6_runtime_source_sha256"] != verified:
        raise ValueError("Candidate6 runtime source binding mismatch")
    return bindings


def _validate_pass(
    data: bytes,
    *,
    expected_sha256: str,
    expected_rule: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
    verified_runtime_source_sha256: str,
) -> None:
    expected = _require_sha(expected_sha256, name="Candidate6 TRAIN PASS SHA")
    if sha256_bytes(data) != expected:
        raise ValueError("Candidate6 TRAIN PASS byte binding mismatch")
    artifact = _parse_json_bytes(data, name="Candidate6 TRAIN PASS", canonical_required=True)
    if set(artifact) != PASS_KEYS:
        raise ValueError("Candidate6 TRAIN PASS schema mismatch")
    if (
        artifact["schema_version"] != 1
        or artifact["kind"] != PASS_KIND
        or artifact["status"] != "PASS"
        or artifact["attempt_ordinal"] != 6
        or artifact["training_gate_passed"] is not True
        or artifact["holdout_execution_authorized"] is not False
        or artifact["holdout_score_opened"] is not False
        or artifact["pre_holdout_execution_anchor_created"] is not False
        or artifact["gate_result_value_free"] is not True
        or artifact["label_derived_metric_values_present"] is not False
        or canonical_bytes(artifact["selected_rule"])
        != canonical_bytes(dict(expected_rule))
    ):
        raise ValueError("Candidate6 TRAIN PASS status/rule mismatch")
    bindings = _validate_bindings(
        artifact["bindings"],
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    if bindings != dict(expected_bindings):
        raise ValueError("Candidate6 model/PASS binding mismatch")
    checks = artifact["checks"]
    if not isinstance(checks, dict) or set(checks) != GATE_CHECK_KEYS:
        raise ValueError("Candidate6 TRAIN PASS check schema mismatch")
    if any(checks[key] is not True for key in GATE_CHECK_KEYS):
        raise ValueError("Candidate6 TRAIN PASS contains a failed gate")


def _validate_model(
    model: Mapping[str, Any],
    *,
    pass_bytes: bytes,
    pass_sha256: str,
    verified_runtime_source_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(model, Mapping) or set(model) != MODEL_KEYS:
        raise ValueError("Candidate6 model schema mismatch")
    if (
        model["schema_version"] != 1
        or model["kind"] != MODEL_KIND
        or model["profile"] != PROFILE
        or model["attempt_ordinal"] != 6
        or model["training_gate_passed"] is not True
        or model["freeze_status"] != "FROZEN_AFTER_TRAIN_GATE_PASS"
        or model["deterministic_tie_break_direction"]
        != DETERMINISTIC_TIE_BREAK_DIRECTION
        or model["deterministic_tie_break_inputs"]
        != list(DETERMINISTIC_TIE_BREAK_INPUTS)
        or model["external_launcher_authentication_required"] is not True
        or model["fixed_eligibility_inputs"] != list(FIXED_ELIGIBILITY_INPUTS)
        or model["holdout_score_opened"] is not False
        or model["pre_holdout_execution_anchor_created"] is not False
        or model["post_holdout_model_or_threshold_mutation"]
        != "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
        or model["source_only"] is not True
        or model["additive_only"] is not True
        or model["line_objects_only"] is not True
        or model["raw_candidates_ignored"] is not True
        or model["preserve_v51_prediction_values_and_order"] is not True
        or model["missing_evidence_abstains"] is not True
        or model["max_additions_per_case"] != MAX_ADDITIONS_PER_CASE
        or model["max_added_false_positives_train18"]
        != MAX_ADDED_FALSE_POSITIVES_TRAIN18
        or model["ownership_contract"] != _ownership_contract()
        or any(model[field] is not False for field in FALSE_INFERENCE_DECLARATIONS)
    ):
        raise ValueError("Candidate6 model status/source contract mismatch")
    rule = model["rule"]
    validate_rule(rule)
    expected_features = [rule["feature"]]
    if rule["feature"] != "source_kind":
        expected_features.append("source_kind")
    if model["inference_features"] != expected_features:
        raise ValueError(
            "Candidate6 model must disclose its selected feature and source eligibility"
        )
    for feature, declaration in FEATURE_DECLARATION_KEYS.items():
        expected_used = rule["feature"] == feature or feature == "source_kind"
        if model[declaration] is not expected_used:
            raise ValueError("Candidate6 feature declaration mismatch")
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
        raise ValueError("Candidate6 TRAIN PASS binding mismatch")
    _validate_pass(
        pass_bytes,
        expected_sha256=pass_sha256,
        expected_rule=rule,
        expected_bindings=bindings,
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    return rule


def preflight_candidate6_source_bytes(
    v51_semantic_bytes: bytes,
    pool_bytes: bytes,
    *,
    v51_semantic_sha256: str,
    pool_sha256: str,
) -> dict[str, Any]:
    """Exercise label-incapable semantic, pool, rule, and ownership paths."""

    _semantic, predictions = _validate_semantic_bytes(
        v51_semantic_bytes, expected_sha256=v51_semantic_sha256
    )
    _pool, candidates = _validate_pool_bytes(pool_bytes, expected_sha256=pool_sha256)
    original_prefix = canonical_bytes(predictions)
    incumbent_lines = [
        _endpoints_from_row(
            prediction,
            name=f"V51 prediction {index}",
            require_prediction_lines=True,
        )
        for index, prediction in enumerate(predictions)
    ]
    for incumbent in incumbent_lines:
        if incumbent is None:
            raise RuntimeError("Candidate6 preflight lost incumbent geometry")
        for candidate in candidates:
            projection = candidate["identity_projection"]
            candidate_line = (
                tuple(projection["line_start"]),
                tuple(projection["line_end"]),
            )
            metrics = line_pair_metrics(incumbent, candidate_line)
            runtime_threshold_match = bool(
                metrics["axis_angle_deg"] <= OWNERSHIP_ANGLE_DEG
                and metrics["transverse_offset_mm"] <= OWNERSHIP_TRANSVERSE_OFFSET_MM
                and metrics["incumbent_span_overlap"]
                >= OWNERSHIP_INCUMBENT_SPAN_OVERLAP
                and metrics["candidate_span_overlap"]
                >= OWNERSHIP_CANDIDATE_SPAN_OVERLAP
            )
            if runtime_threshold_match and not _ownership_conflict(
                candidate_line, [incumbent]
            ):
                raise RuntimeError(
                    "Candidate6 runtime ownership-boundary implication failed"
                )
    for rule in FIXED_RULE_SPACE:
        validate_rule(rule)
        if rule["operator"] == "ge":
            sentinel = {feature: None for feature in ALLOWED_RULE_FEATURES}
            sentinel[rule["feature"]] = rule["threshold"]
            if not _rule_matches(rule, sentinel):
                raise RuntimeError("Candidate6 numeric rule direction is inconsistent")
            sentinel[rule["feature"]] = None
            if _rule_matches(rule, sentinel):
                raise RuntimeError("Candidate6 missing numeric evidence did not abstain")
        else:
            sentinel = {feature: None for feature in ALLOWED_RULE_FEATURES}
            sentinel["source_kind"] = rule["value"]
            if not _rule_matches(rule, sentinel):
                raise RuntimeError("Candidate6 categorical rule direction is inconsistent")
        selected = _select_addition(predictions, candidates, rule)
        output = list(predictions)
        if selected is not None:
            output.append(_addition_prediction(selected))
        if canonical_bytes(output[: len(predictions)]) != original_prefix:
            raise RuntimeError("Candidate6 preflight changed a V51 prediction")
    identical = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    separated = ((0.0, 5.0, 0.0), (10.0, 5.0, 0.0))
    if not _ownership_conflict(identical, [identical]):
        raise RuntimeError("Candidate6 ownership duplicate self-check failed")
    if _ownership_conflict(separated, [identical]):
        raise RuntimeError("Candidate6 4mm ownership separation self-check failed")
    return {
        "all_fixed_rule_forms_exercised": True,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "candidate_metric_computed": False,
        "candidate_rule_selected": False,
        "candidate_source_rows_sanitized": True,
        "deterministic_geometry_source_tie_break_exercised": True,
        "fixed_allowed_source_kind_eligibility_exercised": True,
        "generic_pool_accounting_ignored": True,
        "holdout_score_opened": False,
        "input_pool_sha256": pool_sha256,
        "input_v51_semantic_sha256": v51_semantic_sha256,
        "kind": PREFLIGHT_RESULT_KIND,
        "labels_used": False,
        "missing_evidence_abstention_exercised": True,
        "model_fit_performed": False,
        "runtime_threshold_match_implies_conservative_conflict_for_bound_pairs": True,
        "ownership_geometry_exercised": True,
        "raw_candidates_ignored": True,
        "recursive_forbidden_key_validation_exercised": True,
        "schema_version": SCHEMA_VERSION,
        "source_only": True,
        "status": "SOURCE_COMPATIBLE",
        "v51_prediction_values_and_order_preserved": True,
    }


def apply_candidate6(
    v51_semantic_bytes: bytes,
    pool_bytes: bytes,
    model: Mapping[str, Any],
    train_gate_pass_bytes: bytes,
    candidate5_terminal_bytes: bytes,
    candidate5_disposition_bytes: bytes,
    *,
    v51_semantic_sha256: str,
    pool_sha256: str,
    train_gate_pass_sha256: str,
    candidate5_terminal_sha256: str,
    candidate5_disposition_sha256: str,
    verified_runtime_source_sha256: str,
) -> dict[str, Any]:
    """Append at most one ownership-safe recovered line after all seals pass."""

    _validate_candidate5_terminal(
        candidate5_terminal_bytes, expected_sha256=candidate5_terminal_sha256
    )
    _validate_candidate5_disposition(
        candidate5_disposition_bytes,
        expected_sha256=candidate5_disposition_sha256,
    )
    rule = _validate_model(
        model,
        pass_bytes=train_gate_pass_bytes,
        pass_sha256=train_gate_pass_sha256,
        verified_runtime_source_sha256=verified_runtime_source_sha256,
    )
    bindings = model["bindings"]
    if (
        bindings["candidate5_terminal_sha256"] != candidate5_terminal_sha256
        or bindings["candidate5_terminal_disposition_sha256"]
        != candidate5_disposition_sha256
    ):
        raise ValueError("Candidate6 model does not bind supplied Candidate5 lineage")
    _semantic, predictions = _validate_semantic_bytes(
        v51_semantic_bytes, expected_sha256=v51_semantic_sha256
    )
    _pool, candidates = _validate_pool_bytes(pool_bytes, expected_sha256=pool_sha256)
    original_prefix = canonical_bytes(predictions)
    selected = _select_addition(predictions, candidates, rule)
    output = list(predictions)
    if selected is not None:
        output.append(_addition_prediction(selected))
    output_ids = [str(prediction["bend_id"]) for prediction in output]
    if len(output_ids) != len(set(output_ids)):
        raise RuntimeError("Candidate6 produced a duplicate prediction bend_id")
    if canonical_bytes(output[: len(predictions)]) != original_prefix:
        raise RuntimeError("Candidate6 changed V51 prediction values/order")
    result = {
        "addition_made": selected is not None,
        "additive_only": True,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "candidate5_terminal_disposition_sha256": candidate5_disposition_sha256,
        "candidate5_terminal_sha256": candidate5_terminal_sha256,
        "candidate6_model_sha256": sha256_bytes(canonical_bytes(dict(model))),
        "deterministic_tie_break_direction": DETERMINISTIC_TIE_BREAK_DIRECTION,
        "deterministic_tie_break_inputs": list(DETERMINISTIC_TIE_BREAK_INPUTS),
        "external_launcher_authentication_required": True,
        "fixed_eligibility_inputs": list(FIXED_ELIGIBILITY_INPUTS),
        "holdout_score_opened": False,
        "input_pool_sha256": pool_sha256,
        "input_v51_semantic_sha256": v51_semantic_sha256,
        "kind": RESULT_KIND,
        "labels_used": False,
        "line_objects_only": True,
        "max_additions_per_case": MAX_ADDITIONS_PER_CASE,
        "missing_evidence_abstains": True,
        "ownership_contract": _ownership_contract(),
        "predictions": output,
        "profile": PROFILE,
        "promotion_eligible": False,
        "raw_candidates_ignored": True,
        "research_only": True,
        "rule": dict(rule),
        "scan_identity_authority": "ATTESTED_CACHE_MANIFEST_ONLY",
        "schema_version": SCHEMA_VERSION,
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
    result["candidate6_result_sha256"] = sha256_bytes(canonical_bytes(result))
    return result
