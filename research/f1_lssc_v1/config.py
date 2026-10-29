"""Immutable constants for the authorized LSSC-V1 causal preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class LSSCConfig:
    schema_version: int = 1
    architecture_token: str = "LOCAL_SCREW_SLIPPAGE_COHERENCE_V1"
    short_token: str = "LSSC_V1"
    authorization_token: str = (
        "AUTHORIZE_LOCAL_SCREW_SLIPPAGE_COHERENCE_V1_C006_FROZEN_"
        "SOURCE_ONLY_CAUSAL_PREFLIGHT_V1"
    )
    input_min_points: int = 1024
    input_max_points: int = 200_000
    spacing_neighbor_rank: int = 4
    normal_neighbor_count: int = 24
    normal_radius_max_s: float = 4.0
    normal_min_neighbors: int = 16
    normal_surface_ratio_min: float = 0.12
    normal_thickness_ratio_max: float = 0.25
    chart_neighbor_count: int = 16
    chart_edge_length_max_s: float = 2.5
    chart_normal_abs_dot_min: float = 0.8191520442889918
    anchor_cap: int = 12_000
    anchor_floor: int = 2_048
    anchor_divisor: int = 24
    patch_radii_over_s: tuple[float, ...] = (6.0, 10.0, 16.0, 24.0)
    patch_min_points: int = 48
    patch_max_points: int = 4_096
    total_patch_membership_max: int = 32_000_000
    eigen_floor: float = 1e-12
    cylindrical_lambda2_max: float = 0.004
    cylindrical_lambda3_min: float = 0.008
    cylindrical_lambda4_min: float = 0.15
    translation_leak_max: float = 0.08
    translation_rotation_axis_abs_dot_min: float = 0.9902680687415704
    rotation_fraction_min: float = 0.20
    axis_foot_distance_rho_max: float = 3.25
    planar_lambda3_max: float = 0.004
    planar_lambda4_min: float = 0.20
    planar_rotation_rank_ratio_max: float = 0.05
    planar_translation_second_singular_min: float = 0.90
    radial_min_s: float = 0.5
    tail_distance_rho_min: float = 1.40
    tail_distance_rho_max: float = 4.00
    tail_axial_fraction_max: float = 0.35
    tail_transverse_rho_min: float = 1.25
    tail_transverse_fraction_min: float = 0.70
    tail_candidates_per_side: int = 32
    tail_pair_axial_rho_max: float = 0.75
    tail_normal_axis_abs_dot_max: float = 0.20791169081775931
    tail_dihedral_min_degrees: float = 25.0
    tail_dihedral_max_degrees: float = 90.0
    tail_cross_axis_abs_dot_min: float = 0.9781476007338057
    vote_neighbor_distance_s: float = 8.0
    vote_axis_angle_max_degrees: float = 8.0
    vote_line_separation_max_s: float = 2.5
    vote_radius_ratio_max: float = 2.0
    vote_dihedral_difference_max_degrees: float = 15.0
    vote_axial_test_min_s: float = 2.0
    vote_axial_fraction_min: float = 0.75
    vote_neighbor_cap: int = 24
    component_min_votes: int = 8
    component_span_min_s: float = 24.0
    component_rms_max_s: float = 1.5
    component_p90_max_s: float = 2.5
    component_median_axis_deviation_max_degrees: float = 4.0
    component_p90_axis_deviation_max_degrees: float = 8.0
    component_dihedral_iqr_max_degrees: float = 15.0
    component_tail_cross_p90_max_degrees: float = 12.0
    component_radius_ratio_max: float = 2.0
    axial_bin_width_s: float = 4.0
    occupied_bin_fraction_min: float = 0.65
    maximum_empty_run_s: float = 8.0
    line_likeness_min: float = 12.0
    native_image_pixels: int = 4096
    randomness_enabled: bool = False
    gpu_enabled: bool = False
    target_capability_granted: bool = False
    source_point_serialization: bool = False

    def as_frozen_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["patch_radii_over_s"] = list(self.patch_radii_over_s)
        value.update(
            {
                "arithmetic": "IEEE754_FLOAT64",
                "coordinate_duplicate_rule": "EXACT_XYZ_AFTER_NEGATIVE_ZERO_NORMALIZATION",
                "canonical_order": "LEXICOGRAPHIC_XYZ",
                "normal_orientation": "UNORIENTED_OUTER_PRODUCT_INVARIANT",
                "architecture_lineage": "POST_CFAC_TERMINAL_LOCAL_SLIPPAGE_OBJECT",
            }
        )
        return value


CONFIG = LSSCConfig()
CONFIG_SHA256 = sha_json(CONFIG.as_frozen_dict())


def validate_config(config: LSSCConfig = CONFIG) -> None:
    if config.patch_radii_over_s != (6.0, 10.0, 16.0, 24.0):
        raise ValueError("LSSC multiscale schedule changed")
    if config.cylindrical_lambda2_max != 0.004 or config.planar_lambda3_max != 0.004:
        raise ValueError("LSSC slippage eigengates changed")
    if config.randomness_enabled or config.gpu_enabled:
        raise ValueError("LSSC randomness/GPU must remain disabled")
    if config.target_capability_granted or config.source_point_serialization:
        raise ValueError("LSSC target access and source serialization remain closed")


validate_config()
