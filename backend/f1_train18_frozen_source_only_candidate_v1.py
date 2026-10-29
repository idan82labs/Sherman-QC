"""Portable source-only inference for the frozen Aug-31 TRAIN18 candidate.

This module is intentionally filesystem- and truth-blind.  It accepts only an
already-produced V51 semantic result, its scan-only candidate pool, and a
portable JSON model.  It has no path, case identity, filename, expected-count,
CAD, drawing, label, or scorer input.

The companion truth-aware optimizer lives in
``scripts/optimize_f1_train18_source_only_v1.py``.  Keeping inference here makes
it possible to run a future pseudo-unseen executor in a process that cannot
import or reach the training labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
MODEL_KIND = "sherman_qc_f1_train18_frozen_source_only_candidate_v1"
RESULT_KIND = "sherman_qc_f1_train18_frozen_source_only_result_v1"

REQUIRED_FALSE_INPUT_FLAGS = (
    "labels_used",
    "expected_count_used",
    "cad_used",
    "drawing_used",
    "part_identity_used",
)

# Deliberately excludes all candidate/case/point/support count fields.  Fractions
# are retained because they are normalized source measurements rather than
# expected or observed object counts.
NUMERIC_FEATURES = (
    "selector_score",
    "score",
    "confidence",
    "finite_support_fraction",
    "endpoint_support_fraction",
    "normal_second_eigenvalue",
    "normal_spread_deg_scan",
    "tight_normal_second_eigenvalue_r2mm",
    "tight_to_wide_normal_second_ratio",
    "two_sided_transition_score",
    "two_sided_normal_angle_deg",
    "two_sided_support_balance",
    "two_sided_min_support_fraction",
    "two_sided_normal_coherence",
    "median_line_distance_mm",
    "p90_line_distance_mm",
    "transverse_angular_span_deg",
    "min_surface_gap",
    "angle_deg",
    "radius_mm",
    "span_mm",
)

DIMENSIONLESS_FEATURES = (
    "finite_support_fraction",
    "endpoint_support_fraction",
    "normal_second_eigenvalue",
    "tight_normal_second_eigenvalue_r2mm",
    "tight_to_wide_normal_second_ratio",
    "two_sided_transition_score",
    "two_sided_support_balance",
    "two_sided_min_support_fraction",
    "two_sided_normal_coherence",
)

SOURCE_KINDS = (
    "plane_lattice",
    "cylindrical_standard",
    "cylindrical_wide",
    "scan_mesh_adjacency",
    "strict_blind_owned",
    "bend_hypothesis",
    "endpoint_connector",
    "fragment_merge",
    "control_support_strip",
    "source_consensus_midpoint",
    "v41_fragment_tolerant_parent_bridge_sidecar",
    "paired_return_completion",
    "other",
)

ORIGINS = ("incumbent", "raw", "axis_span")
ANCHOR_MODES = ("native", "center", "low_endpoint", "high_endpoint")
REPRESENTATIONS = ("absolute_quality", "case_relative_quality", "intrinsic_quality")
UNIVERSES = (
    "incumbent_only",
    "raw_reselect",
    "hybrid_raw",
    "axis_span_reselect",
    "hybrid_axis_span",
    "hybrid_all",
)

FORBIDDEN_FEATURE_TOKENS = (
    "case_id",
    "filename",
    "file_name",
    "part_id",
    "cad",
    "expected",
    "bend_count",
    "candidate_count",
    "support_count",
    "member_count",
    "source_count",
    "point_count",
)


@dataclass(frozen=True)
class Observation:
    """One source-derived line hypothesis presented to the frozen selector."""

    identity: str
    source_hypothesis_id: str
    origin: str
    source_kind: str
    endpoints: tuple[tuple[float, float, float], tuple[float, float, float]]
    values: Mapping[str, float | None]
    anchor_mode: str = "native"
    span_ratio: float = 1.0
    fixed_span: bool = False


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_required(value: Any, *, name: str) -> float:
    number = _finite(value)
    if number is None:
        raise ValueError(f"{name} must be a finite number")
    return number


def _point(value: Any, *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite three-dimensional point")
    return point


def endpoints_from_record(row: Mapping[str, Any]) -> np.ndarray:
    """Return a validated 2x3 line from any supported source representation."""

    if row.get("span_endpoints") is not None:
        endpoints = np.asarray(row["span_endpoints"], dtype=np.float64)
    elif row.get("line_start") is not None and row.get("line_end") is not None:
        endpoints = np.asarray([row["line_start"], row["line_end"]], dtype=np.float64)
    else:
        anchor = _point(row.get("anchor"), name="anchor")
        axis = _point(row.get("axis_direction"), name="axis_direction")
        span = _finite_required(row.get("span_mm"), name="span_mm")
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12 or span <= 1e-9:
            raise ValueError("anchor/axis/span line is degenerate")
        axis = axis / norm
        endpoints = np.asarray([anchor - axis * span / 2.0, anchor + axis * span / 2.0])
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("line endpoints must be finite 2x3 coordinates")
    if float(np.linalg.norm(endpoints[1] - endpoints[0])) <= 1e-9:
        raise ValueError("line endpoints are degenerate")
    return endpoints


def _canonical_axis(value: np.ndarray) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or not np.isfinite(axis).all() or norm <= 1e-12:
        raise ValueError("axis must be finite and nondegenerate")
    axis = axis / norm
    pivot = int(np.argmax(np.abs(axis)))
    return -axis if axis[pivot] < 0.0 else axis


def _line_tuple(endpoints: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        tuple(float(value) for value in endpoints[0]),
        tuple(float(value) for value in endpoints[1]),
    )


def _source_kind(row: Mapping[str, Any]) -> str:
    value = str(
        row.get("source_kind")
        or row.get("representative_source")
        or row.get("candidate_source")
        or "other"
    )
    return value if value in SOURCE_KINDS[:-1] else "other"


def _values(*rows: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in NUMERIC_FEATURES:
        result[name] = None
        for row in reversed(rows):
            value = _finite(row.get(name))
            if value is not None:
                result[name] = value
                break
    return result


def _candidate_maps(pool: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if int(pool.get("working_point_budget") or 0) != 200000:
        raise ValueError("candidate pool point-budget contract mismatch")
    raw_candidates = pool.get("candidates")
    line_objects = pool.get("line_objects")
    if not isinstance(raw_candidates, list) or not isinstance(line_objects, list):
        raise ValueError("candidate pool arrays are missing")
    merged: dict[str, dict[str, Any]] = {}
    for collection in (raw_candidates, line_objects):
        for raw in collection:
            if not isinstance(raw, dict):
                raise ValueError("candidate pool rows must be objects")
            token = str(raw.get("candidate_id") or "")
            if not token:
                raise ValueError("candidate pool row has no candidate_id")
            merged.setdefault(token, {}).update(dict(raw))
    rows = [dict(row) for row in line_objects]
    if len({str(row.get("candidate_id") or "") for row in rows}) != len(rows):
        raise ValueError("line-object candidate IDs must be unique")
    return merged, rows


def _raw_observations(pool: Mapping[str, Any]) -> list[Observation]:
    merged, rows = _candidate_maps(pool)
    observations: list[Observation] = []
    for row in sorted(rows, key=lambda value: str(value.get("candidate_id") or "")):
        token = str(row["candidate_id"])
        combined = merged[token]
        endpoints = endpoints_from_record(combined)
        observations.append(
            Observation(
                identity=f"raw:{token}",
                source_hypothesis_id=token,
                origin="raw",
                source_kind=_source_kind(combined),
                endpoints=_line_tuple(endpoints),
                values=_values(combined),
            )
        )
    return observations


def _incumbent_observations(
    v51_result: Mapping[str, Any], pool: Mapping[str, Any]
) -> list[Observation]:
    merged, _rows = _candidate_maps(pool)
    predictions = v51_result.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("V51 predictions must be a list")
    observations: list[Observation] = []
    seen: set[str] = set()
    for ordinal, raw in enumerate(predictions):
        if not isinstance(raw, dict):
            raise ValueError("V51 prediction rows must be objects")
        prediction_id = str(raw.get("bend_id") or "")
        if not prediction_id or prediction_id in seen:
            raise ValueError("V51 prediction bend IDs must be unique and non-empty")
        seen.add(prediction_id)
        hypothesis = str(raw.get("source_hypothesis_id") or "")
        source = merged.get(hypothesis, {})
        endpoints = endpoints_from_record(raw)
        length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        synthetic = {"span_mm": length}
        observations.append(
            Observation(
                identity=f"incumbent:{ordinal:05d}:{prediction_id}",
                source_hypothesis_id=hypothesis or prediction_id,
                origin="incumbent",
                source_kind=_source_kind(raw if _source_kind(raw) != "other" else source),
                endpoints=_line_tuple(endpoints),
                values=_values(source, raw, synthetic),
            )
        )
    return observations


def _axis_span_observations(
    raw_observations: Sequence[Observation],
    *,
    span_scales: Sequence[float],
    fixed_spans_mm: Sequence[float],
) -> list[Observation]:
    result: list[Observation] = []
    for base in raw_observations:
        endpoints = np.asarray(base.endpoints, dtype=np.float64)
        axis = _canonical_axis(endpoints[1] - endpoints[0])
        projections = [float(np.dot(point, axis)) for point in endpoints]
        low = endpoints[int(np.argmin(projections))]
        high = endpoints[int(np.argmax(projections))]
        center = np.mean(endpoints, axis=0)
        base_span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        span_specs: dict[float, tuple[float, bool]] = {}
        for value in fixed_spans_mm:
            span = _finite_required(value, name="fixed span")
            if span <= 1e-9:
                raise ValueError("fixed spans must be positive")
            span_specs[round(span, 9)] = (span, True)
        for value in span_scales:
            scale = _finite_required(value, name="span scale")
            if scale <= 1e-9:
                raise ValueError("span scales must be positive")
            span = base_span * scale
            span_specs.setdefault(round(span, 9), (span, False))
        for _key in sorted(span_specs):
            span, fixed = span_specs[_key]
            lines = {
                "center": (center - axis * span / 2.0, center + axis * span / 2.0),
                "low_endpoint": (low, low + axis * span),
                "high_endpoint": (high - axis * span, high),
            }
            for mode in ("center", "low_endpoint", "high_endpoint"):
                start, end = lines[mode]
                identity_payload = {
                    "base": base.identity,
                    "span_mm": round(span, 9),
                    "anchor_mode": mode,
                    "line_start": start.tolist(),
                    "line_end": end.tolist(),
                }
                token = sha256_bytes(canonical_bytes(identity_payload))[:24]
                values = dict(base.values)
                values["span_mm"] = span
                result.append(
                    Observation(
                        identity=f"axis_span:{token}",
                        source_hypothesis_id=base.source_hypothesis_id,
                        origin="axis_span",
                        source_kind=base.source_kind,
                        endpoints=_line_tuple(np.asarray([start, end], dtype=np.float64)),
                        values=values,
                        anchor_mode=mode,
                        span_ratio=span / base_span,
                        fixed_span=fixed,
                    )
                )
    result.sort(key=lambda row: row.identity)
    return result


def build_observations(
    v51_result: Mapping[str, Any],
    pool: Mapping[str, Any],
    *,
    universe: str,
    span_scales: Sequence[float] = (),
    fixed_spans_mm: Sequence[float] = (),
) -> tuple[list[Observation], list[Observation]]:
    """Build a policy universe plus raw rows used for relative normalization."""

    if universe not in UNIVERSES:
        raise ValueError(f"unsupported observation universe: {universe}")
    raw = _raw_observations(pool)
    incumbents = _incumbent_observations(v51_result, pool)
    variants: list[Observation] = []
    if universe in {"axis_span_reselect", "hybrid_axis_span", "hybrid_all"}:
        variants = _axis_span_observations(
            raw, span_scales=span_scales, fixed_spans_mm=fixed_spans_mm
        )
    if universe == "incumbent_only":
        selected = incumbents
    elif universe == "raw_reselect":
        selected = raw
    elif universe == "hybrid_raw":
        selected = incumbents + raw
    elif universe == "axis_span_reselect":
        selected = variants
    elif universe == "hybrid_axis_span":
        selected = incumbents + variants
    else:
        selected = incumbents + raw + variants
    identities = [row.identity for row in selected]
    if len(set(identities)) != len(identities):
        raise RuntimeError("observation identities are not unique")
    return selected, raw


def _percentile_maps(raw: Sequence[Observation]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in NUMERIC_FEATURES:
        pairs = sorted(
            (float(row.values[name]), row.identity)
            for row in raw
            if row.values.get(name) is not None
        )
        mapping: dict[str, float] = {}
        if pairs:
            index = 0
            while index < len(pairs):
                end = index + 1
                while end < len(pairs) and pairs[end][0] == pairs[index][0]:
                    end += 1
                percentile = ((index + end - 1) / 2.0 + 0.5) / len(pairs)
                for _value, identity in pairs[index:end]:
                    mapping[identity] = percentile
                index = end
        result[name] = mapping
    return result


def feature_names(representation: str) -> tuple[str, ...]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unsupported representation: {representation}")
    if representation == "absolute_quality":
        core = tuple(f"abs_{name}" for name in NUMERIC_FEATURES)
        missing = tuple(f"missing_{name}" for name in NUMERIC_FEATURES)
    elif representation == "case_relative_quality":
        core = tuple(f"percentile_{name}" for name in NUMERIC_FEATURES)
        missing = tuple(f"missing_{name}" for name in NUMERIC_FEATURES)
    else:
        core = tuple(f"intrinsic_{name}" for name in DIMENSIONLESS_FEATURES)
        missing = tuple(f"missing_{name}" for name in DIMENSIONLESS_FEATURES)
    categorical = (
        *(f"source_{name}" for name in SOURCE_KINDS),
        *(f"origin_{name}" for name in ORIGINS),
        *(f"anchor_{name}" for name in ANCHOR_MODES),
        "variant_span_ratio",
        "variant_abs_log_span_ratio",
        "variant_fixed_span",
    )
    names = tuple(core) + tuple(missing) + tuple(categorical)
    lowered = [name.lower() for name in names]
    for name in lowered:
        if any(token in name for token in FORBIDDEN_FEATURE_TOKENS):
            raise RuntimeError(f"forbidden feature name generated: {name}")
        if "count" in name:
            raise RuntimeError(f"count feature generated: {name}")
    return names


def feature_matrix(
    observations: Sequence[Observation],
    raw_reference: Sequence[Observation],
    *,
    representation: str,
) -> np.ndarray:
    """Materialize a finite-or-NaN source-only feature matrix."""

    names = feature_names(representation)
    percentiles = _percentile_maps(raw_reference)
    raw_by_hypothesis = {
        row.source_hypothesis_id: row.identity for row in raw_reference
    }
    rows: list[list[float]] = []
    numeric_names = (
        NUMERIC_FEATURES if representation != "intrinsic_quality" else DIMENSIONLESS_FEATURES
    )
    for observation in observations:
        vector: list[float] = []
        for name in numeric_names:
            value = observation.values.get(name)
            if representation == "case_relative_quality":
                reference_identity = raw_by_hypothesis.get(observation.source_hypothesis_id)
                relative = (
                    percentiles[name].get(reference_identity)
                    if reference_identity is not None
                    else None
                )
                vector.append(float(relative) if relative is not None else math.nan)
            else:
                vector.append(float(value) if value is not None else math.nan)
        vector.extend(
            1.0 if observation.values.get(name) is None else 0.0
            for name in numeric_names
        )
        vector.extend(float(observation.source_kind == name) for name in SOURCE_KINDS)
        vector.extend(float(observation.origin == name) for name in ORIGINS)
        vector.extend(float(observation.anchor_mode == name) for name in ANCHOR_MODES)
        vector.extend(
            (
                float(observation.span_ratio),
                abs(math.log(max(float(observation.span_ratio), 1e-12))),
                float(observation.fixed_span),
            )
        )
        if len(vector) != len(names):
            raise RuntimeError("feature vector length does not match feature contract")
        rows.append(vector)
    if not rows:
        return np.empty((0, len(names)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def validate_source_input(v51_result: Mapping[str, Any]) -> None:
    """Enforce the same source-only boundary in training and frozen inference."""

    if v51_result.get("source_only") is not True or v51_result.get("research_only") is not True:
        raise ValueError("V51 input must be source-only research output")
    for field in REQUIRED_FALSE_INPUT_FLAGS:
        if v51_result.get(field) not in (None, False):
            raise ValueError(f"V51 reports forbidden inference input: {field}")


def _validate_classifier(classifier: Mapping[str, Any], width: int) -> None:
    medians = classifier.get("imputer_medians")
    if not isinstance(medians, list) or len(medians) != width:
        raise ValueError("classifier imputer width mismatch")
    if any(_finite(value) is None for value in medians):
        raise ValueError("classifier imputer contains non-finite values")
    kind = classifier.get("kind")
    if kind == "constant":
        probability = _finite(classifier.get("positive_probability"))
        if probability is None or not 0.0 <= probability <= 1.0:
            raise ValueError("constant classifier probability is invalid")
    elif kind == "logistic":
        for name in ("mean", "scale", "coefficients"):
            values = classifier.get(name)
            if not isinstance(values, list) or len(values) != width:
                raise ValueError(f"logistic {name} width mismatch")
            if any(_finite(value) is None for value in values):
                raise ValueError(f"logistic {name} contains non-finite values")
        if _finite(classifier.get("intercept")) is None:
            raise ValueError("logistic intercept is invalid")
    elif kind == "tree":
        _validate_tree(classifier.get("tree"), width)
    elif kind == "forest":
        trees = classifier.get("trees")
        if not isinstance(trees, list) or not trees:
            raise ValueError("forest must contain trees")
        for tree in trees:
            _validate_tree(tree, width)
    else:
        raise ValueError(f"unsupported classifier kind: {kind}")


def _validate_tree(tree: Any, width: int) -> None:
    if not isinstance(tree, dict):
        raise ValueError("portable tree is missing")
    keys = ("children_left", "children_right", "feature_index", "threshold", "positive_probability")
    arrays = [tree.get(name) for name in keys]
    if any(not isinstance(values, list) for values in arrays):
        raise ValueError("portable tree arrays are missing")
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("portable tree arrays have inconsistent lengths")
    for node, feature in enumerate(tree["feature_index"]):
        index = int(feature)
        left = int(tree["children_left"][node])
        right = int(tree["children_right"][node])
        if left == right == -1:
            if index not in (-2, -1):
                raise ValueError("portable tree leaf feature sentinel is invalid")
        elif not 0 <= index < width:
            raise ValueError("portable tree internal feature index is invalid")


def validate_model(model: Mapping[str, Any]) -> dict[str, Any]:
    if model.get("schema_version") != SCHEMA_VERSION or model.get("kind") != MODEL_KIND:
        raise ValueError("frozen model schema/kind mismatch")
    if model.get("freeze_status") != "FROZEN_BEFORE_HOLDOUT_SCORE":
        raise ValueError("model is not frozen for pseudo-unseen execution")
    if model.get("source_only") is not True:
        raise ValueError("model is not declared source-only")
    if model.get("holdout_score_opened") is not False:
        raise ValueError("model must be sealed before any holdout score is opened")
    binding_fields = (
        "split_sha256",
        "train_allowlist_sha256",
        "preregistration_sha256",
        "label_projection_attestation_sha256",
        "train_cache_manifest_sha256",
        "train_cache_manifest_attestation_sha256",
        "train_source_cache_binding_sha256",
        "optimizer_code_sha256",
        "runtime_code_sha256",
        "fit_audit_sha256",
    )
    for field in binding_fields:
        value = str(model.get(field) or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"model binding is missing or invalid: {field}")
    for field in ("case_id_feature_used", "filename_feature_used", "cad_feature_used", "count_feature_used"):
        if model.get(field) is not False:
            raise ValueError(f"model must explicitly disable {field}")
    representation = str(model.get("representation") or "")
    universe = str(model.get("universe") or "")
    if universe not in UNIVERSES:
        raise ValueError("model universe is invalid")
    expected_names = feature_names(representation)
    if tuple(model.get("feature_names") or ()) != expected_names:
        raise ValueError("model feature contract mismatch")
    profile = model.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("model profile is missing")
    required = {
        "probability_threshold",
        "output_cap",
        "duplicate_angle_deg",
        "duplicate_offset_mm",
        "duplicate_span_overlap",
        "span_scales",
        "fixed_spans_mm",
    }
    if not required.issubset(profile):
        raise ValueError("model profile is incomplete")
    threshold = _finite(profile["probability_threshold"])
    if threshold is None or not 0.0 <= threshold <= 1.0:
        raise ValueError("probability threshold is invalid")
    output_cap = profile["output_cap"]
    if (
        isinstance(output_cap, bool)
        or not isinstance(output_cap, int)
        or output_cap <= 0
    ):
        raise ValueError("output cap is invalid")
    _validate_classifier(model.get("classifier") or {}, len(expected_names))
    return dict(profile)


def _impute(matrix: np.ndarray, medians: Sequence[Any]) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] != len(medians):
        raise ValueError("feature matrix width mismatch")
    result = np.asarray(matrix, dtype=np.float64).copy()
    fill = np.asarray([_finite_required(value, name="imputer median") for value in medians])
    rows, columns = np.where(~np.isfinite(result))
    if len(rows):
        result[rows, columns] = fill[columns]
    if not np.isfinite(result).all():
        raise ValueError("imputation did not produce finite features")
    return result


def _tree_probability(row: np.ndarray, tree: Mapping[str, Any]) -> float:
    node = 0
    node_count = len(tree["children_left"])
    for _visited in range(node_count + 1):
        left = int(tree["children_left"][node])
        right = int(tree["children_right"][node])
        if left == right == -1:
            probability = _finite_required(
                tree["positive_probability"][node], name="tree leaf probability"
            )
            if not 0.0 <= probability <= 1.0:
                raise ValueError("tree leaf probability is outside [0,1]")
            return probability
        feature = int(tree["feature_index"][node])
        threshold = _finite_required(tree["threshold"][node], name="tree threshold")
        node = left if row[feature] <= threshold else right
        if node < 0 or node >= node_count:
            raise ValueError("tree child index is invalid")
    raise ValueError("tree contains a cycle")


def classifier_probabilities(matrix: np.ndarray, classifier: Mapping[str, Any]) -> np.ndarray:
    values = _impute(matrix, classifier["imputer_medians"])
    kind = str(classifier["kind"])
    if kind == "constant":
        probability = _finite_required(
            classifier["positive_probability"], name="constant probability"
        )
        return np.full(len(values), probability, dtype=np.float64)
    if kind == "logistic":
        mean = np.asarray(classifier["mean"], dtype=np.float64)
        scale = np.asarray(classifier["scale"], dtype=np.float64)
        coefficients = np.asarray(classifier["coefficients"], dtype=np.float64)
        if np.any(scale <= 0.0):
            raise ValueError("logistic scale values must be positive")
        logits = ((values - mean) / scale) @ coefficients + float(classifier["intercept"])
        logits = np.clip(logits, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-logits))
    if kind == "tree":
        return np.asarray([_tree_probability(row, classifier["tree"]) for row in values])
    if kind == "forest":
        return np.mean(
            np.asarray(
                [
                    [_tree_probability(row, tree) for row in values]
                    for tree in classifier["trees"]
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
    raise ValueError(f"unsupported classifier kind: {kind}")


def _pair_metrics(first: Observation, second: Observation) -> dict[str, float]:
    left = np.asarray(first.endpoints, dtype=np.float64)
    right = np.asarray(second.endpoints, dtype=np.float64)
    left_delta = left[1] - left[0]
    right_delta = right[1] - right[0]
    left_length = float(np.linalg.norm(left_delta))
    right_length = float(np.linalg.norm(right_delta))
    left_axis = left_delta / left_length
    right_axis = right_delta / right_length
    dot = float(np.clip(np.dot(left_axis, right_axis), -1.0, 1.0))
    angle = math.degrees(math.acos(abs(dot)))
    if dot < 0.0:
        right_axis = -right_axis
    left_midpoint = np.mean(left, axis=0)
    right_midpoint = np.mean(right, axis=0)
    delta = right_midpoint - left_midpoint
    offset = max(
        float(np.linalg.norm(delta - left_axis * np.dot(delta, left_axis))),
        float(np.linalg.norm(delta - right_axis * np.dot(delta, right_axis))),
    )
    left_interval = sorted(float(np.dot(point - left_midpoint, left_axis)) for point in left)
    right_interval = sorted(float(np.dot(point - left_midpoint, left_axis)) for point in right)
    overlap = max(
        0.0,
        min(left_interval[1], right_interval[1])
        - max(left_interval[0], right_interval[0]),
    )
    return {
        "axis_angle_deg": angle,
        "transverse_offset_mm": offset,
        "left_span_overlap": overlap / left_length,
        "right_span_overlap": overlap / right_length,
    }


def _duplicate(first: Observation, second: Observation, profile: Mapping[str, Any]) -> bool:
    metrics = _pair_metrics(first, second)
    overlap = _finite_required(profile["duplicate_span_overlap"], name="duplicate overlap")
    return bool(
        metrics["axis_angle_deg"]
        <= _finite_required(profile["duplicate_angle_deg"], name="duplicate angle")
        and metrics["transverse_offset_mm"]
        <= _finite_required(profile["duplicate_offset_mm"], name="duplicate offset")
        and metrics["left_span_overlap"] >= overlap
        and metrics["right_span_overlap"] >= overlap
    )


def select_observations(
    observations: Sequence[Observation],
    probabilities: Sequence[float],
    profile: Mapping[str, Any],
) -> list[tuple[Observation, float]]:
    if len(observations) != len(probabilities):
        raise ValueError("probability and observation lengths differ")
    threshold = _finite_required(profile["probability_threshold"], name="probability threshold")
    cap = int(profile["output_cap"])
    ranked = [
        (round(float(probability), 12), row)
        for row, probability in zip(observations, probabilities)
        if math.isfinite(float(probability)) and round(float(probability), 12) >= threshold
    ]
    origin_priority = {"incumbent": 0, "raw": 1, "axis_span": 2}
    ranked.sort(
        key=lambda item: (
            -item[0],
            origin_priority[item[1].origin],
            -float(np.linalg.norm(np.diff(np.asarray(item[1].endpoints), axis=0))),
            item[1].identity,
        )
    )
    selected: list[tuple[Observation, float]] = []
    for probability, row in ranked:
        if any(_duplicate(row, incumbent, profile) for incumbent, _score in selected):
            continue
        selected.append((row, probability))
        if len(selected) >= cap:
            break
    return selected


def predictions_from_selection(
    selected: Iterable[tuple[Observation, float]], *, model_sha256: str
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for observation, probability in selected:
        payload = {
            "identity": observation.identity,
            "line_start": list(observation.endpoints[0]),
            "line_end": list(observation.endpoints[1]),
            "model_sha256": model_sha256,
        }
        token = sha256_bytes(canonical_bytes(payload))[:24]
        predictions.append(
            {
                "bend_id": f"T18F-{token}",
                "line_start": list(observation.endpoints[0]),
                "line_end": list(observation.endpoints[1]),
                "selection_route": "frozen_train18_source_only_candidate_v1",
                "selector_score": round(float(probability), 12),
                "source_hypothesis_id": observation.source_hypothesis_id,
                "source_kind": observation.source_kind,
                "source_origin": observation.origin,
                "anchor_mode": observation.anchor_mode,
            }
        )
    predictions.sort(key=lambda row: row["bend_id"])
    return predictions


def compose_frozen_candidate(
    v51_result: Mapping[str, Any],
    pool: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen train-only model without any truth or identity input."""

    validate_source_input(v51_result)
    profile = validate_model(model)
    observations, raw = build_observations(
        v51_result,
        pool,
        universe=str(model["universe"]),
        span_scales=tuple(float(value) for value in profile["span_scales"]),
        fixed_spans_mm=tuple(float(value) for value in profile["fixed_spans_mm"]),
    )
    matrix = feature_matrix(
        observations,
        raw,
        representation=str(model["representation"]),
    )
    probabilities = classifier_probabilities(matrix, model["classifier"])
    selected = select_observations(observations, probabilities, profile)
    model_sha = sha256_bytes(canonical_bytes(model))
    predictions = predictions_from_selection(selected, model_sha256=model_sha)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "profile": str(model.get("profile_name") or "TRAIN18_FROZEN_V1"),
        "predictions": predictions,
        "candidate_count": len(predictions),
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "expected_count_used": False,
        "cad_used": False,
        "drawing_used": False,
        "part_identity_used": False,
        "case_id_used": False,
        "filename_used": False,
        "count_feature_used": False,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "frozen_before_holdout_score": True,
        "frozen_model_sha256": model_sha,
        "inference_audit": {
            "representation": model["representation"],
            "universe": model["universe"],
            "observation_count": len(observations),
            "eligible_count": int(
                sum(
                    round(float(value), 12)
                    >= float(profile["probability_threshold"])
                    for value in probabilities
                )
            ),
            "emitted_count": len(predictions),
            "fixed_output_cap": int(profile["output_cap"]),
        },
    }
