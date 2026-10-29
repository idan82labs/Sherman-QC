"""Locked numerical configuration for PWTS_REPRESENTATION_PREFLIGHT_V1."""

from __future__ import annotations

import hashlib
import json
import copy
from types import MappingProxyType
from typing import Any


EXPECTED_CONFIG_SHA256 = "5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49"

_CONFIG: dict[str, Any] = {
    "case_tokens": [f"C{i:03d}" for i in range(1, 38)],
    "destructive_negative": {
        "arms": ["DELETE_LEFT", "DELETE_RIGHT"],
        "freeze_trace_geometry": True,
        "recompute_membership_and_certificate": True,
        "required_first_rejection": {
            "DELETE_LEFT": "LEFT_SUPPORT_INSUFFICIENT",
            "DELETE_RIGHT": "RIGHT_SUPPORT_INSUFFICIENT",
        },
    },
    "execution": {
        "aggregate_cpu_seconds": 133200,
        "aggregate_serialized_bytes": 2147483648,
        "aggregate_wall_seconds": 177600,
        "blas_threads": 1,
        "cpu_only": True,
        "float_error_mode": "raise",
        "fresh_process_per_case_variant": True,
        "per_case_serialized_bytes": 100663296,
        "per_process_cpu_seconds": 900,
        "per_process_peak_rss_bytes": 12884901888,
        "per_process_scratch_bytes": 4294967296,
        "per_process_wall_seconds": 1200,
        "pythonhashseed": 0,
        "semantic_variants": ["BASE_A", "BASE_B", "ORDER_PERMUTED", "INDEPENDENT_VALIDATOR"],
        "workers": 1,
    },
    "input": {
        "coordinate_dtype": "float64",
        "deduplicate_exact_canonical_xyz": True,
        "drop_nonfinite": True,
        "maximum_sample_points": 200000,
        "minimum_unique_points": 128,
        "normalize_negative_zero": True,
        "point_id_domain": "PWTS_POINT_V1",
        "properties": ["x", "y", "z"],
        "sample_rank_domain": "PWTS_SAMPLE_V1",
    },
    "normal_field": {
        "eigenvalue_order": "ascending",
        "include_query_point": True,
        "knn_scales": [16, 32, 64],
        "linearity_floor_lambda1_over_lambda2": 0.02,
        "normal_sign_rule": "largest_abs_component_positive",
        "surface_variation_ceiling": 0.25,
    },
    "numeric_canonicalization": {
        "float_quantum": 1e-10,
        "validator_abs_tolerance": 1e-8,
        "validator_rel_tolerance": 1e-8,
    },
    "preflight_id": "PWTS_REPRESENTATION_PREFLIGHT_V1",
    "schema_version": 1,
    "strip_certificate": {
        "arc_bins": 8,
        "inner_radius_over_trace_h": 1.5,
        "maximum_median_surface_variation": 0.12,
        "maximum_p90_surface_variation": 0.25,
        "minimum_abs_side_cosine": 0.35,
        "minimum_bilaterally_covered_bins": 6,
        "minimum_covered_bins_per_side": 6,
        "minimum_normal_second_moment_coherence": 0.8,
        "minimum_points_per_covered_bin": 3,
        "minimum_side_count_ratio": 0.25,
        "minimum_side_normal_separation_rad": 0.13962634015954636,
        "minimum_support_points_per_side": 32,
        "minimum_trace_median_response_rad": 0.10471975511965978,
        "minimum_trace_minus_side_median_response_rad": 0.03490658503988659,
        "outer_radius_over_trace_h": 6.0,
    },
    "structural_caps": {
        "maximum_certificate_rows_per_case": 1024,
        "maximum_seed_rows_per_case": 1024,
        "maximum_strip_membership_rows_per_case": 1000000,
        "maximum_trace_node_rows_per_case": 131072,
        "maximum_trace_rows_per_case": 1024,
    },
    "tangent_trace": {
        "extension_knn": 32,
        "extension_radius_over_local_h": 1.25,
        "maximum_nodes": 257,
        "maximum_step_over_trace_h": 1.5,
        "minimum_arc_length_over_trace_h": 6.0,
        "minimum_forward_cosine": 0.7071067811865476,
        "minimum_nodes": 9,
        "minimum_tangent_alignment_cosine": 0.8191520442889918,
        "minimum_weighted_neighbors": 8,
        "tangent_anisotropy_min": 0.65,
        "tangent_knn": 64,
    },
    "transition_field": {
        "local_max_knn": 16,
        "seed_geomean_angle_rad": 0.10471975511965978,
        "seed_min_each_scale_angle_rad": 0.05235987755982989,
        "seed_scale_balance_min": 0.35,
        "trace_geomean_angle_rad": 0.06981317007977318,
        "trace_min_each_scale_angle_rad": 0.03490658503988659,
        "trace_scale_balance_min": 0.25,
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return the ruling's canonical JSON encoding (with no trailing newline)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


CONFIG_SHA256 = hashlib.sha256(canonical_json_bytes(_CONFIG)).hexdigest()
CONFIG_HASH_MATCHES_RULING = CONFIG_SHA256 == EXPECTED_CONFIG_SHA256


def require_authorized_config_hash() -> None:
    """Fail closed: the printed object and stated digest must agree before execution."""
    if not CONFIG_HASH_MATCHES_RULING:
        raise RuntimeError(
            "PWTS ruling configuration is internally inconsistent: "
            f"stated={EXPECTED_CONFIG_SHA256}, computed={CONFIG_SHA256}"
        )

def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def get_config_copy() -> dict[str, Any]:
    """Return an isolated plain-object copy of the locked configuration."""
    return copy.deepcopy(_CONFIG)


CONFIG = _deep_freeze(_CONFIG)
