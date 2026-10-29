"""Immutable constants for the authorized LEPG-V1 one-shot preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class LEPGConfig:
    schema_version: int = 1
    architecture_token: str = "LINEAR_EXTRUSION_PROFILE_GRAMMAR_V1"
    authorization_token: str = "AUTHORIZE_LINEAR_EXTRUSION_PROFILE_GRAMMAR_V1_SOURCE_FREE_THEN_C006_FROZEN_ONE_SHOT"
    max_source_points: int = 200_000
    min_source_points: int = 2_048
    knn_count: int = 33
    signature_ranks: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    signature_quantum_s: float = 0.25
    anchor_limit: int = 16_384
    signature_query_neighbours: int = 64
    signature_candidates_retained: int = 24
    pairs_evaluated_per_anchor: int = 12
    pair_distance_min_s: float = 6.0
    pair_distance_max_s: float = 64.0
    patch_neighbour_count: int = 12
    patch_match_tolerance_s: float = 1.25
    patch_required_matches_each_direction: int = 9
    accepted_pair_hard_cap: int = 196_608
    direction_seed_limit: int = 256
    direction_mean_shift_half_angle_degrees: float = 4.0
    direction_mean_shift_iterations: int = 3
    direction_merge_angle_degrees: float = 2.0
    direction_min_pairs: int = 48
    direction_min_unique_endpoints: int = 96
    direction_candidate_limit: int = 24
    axial_bin_width_s: float = 3.0
    profile_cell_size_s: float = 0.5
    eligible_axial_bin_min_points: int = 64
    consensus_cell_min_axial_bins: int = 6
    consensus_cell_min_center_span_s: float = 15.0
    consensus_profile_min_cells: int = 96
    consensus_profile_hard_cap: int = 50_000
    profile_anchor_limit: int = 512
    profile_neighbour_limit: int = 48
    profile_neighbour_distance_min_s: float = 3.0
    profile_neighbour_distance_max_s: float = 24.0
    triplets_per_anchor: int = 96
    triplet_min_sine: float = math.sin(math.radians(12.0))
    circle_radius_min_s: float = 4.0
    circle_radius_max_s: float = 128.0
    center_vote_cell_size_s: float = 1.0
    center_vote_min: int = 6
    center_candidate_limit: int = 64
    radial_histogram_bin_s: float = 0.5
    radial_peak_limit: int = 8
    shell_assignment_tolerance_s: float = 1.25
    shell_min_profile_cells: int = 24
    skin_gap_min_s: float = 2.0
    skin_gap_max_s: float = 24.0
    common_center_refit_iterations: int = 3
    radial_rms_max_s: float = 0.75
    angle_bin_degrees: float = 3.0
    shortest_interval_mass_fraction: float = 0.90
    arc_span_min_degrees: float = 25.0
    arc_span_max_degrees: float = 170.0
    shell_endpoint_disagreement_max_degrees: float = 6.0
    arc_outside_mass_fraction_max: float = 0.12
    arc_angular_bin_coverage_min: float = 0.70
    ray_line_tolerance_s: float = 1.0
    ray_start_distance_s: float = 2.0
    ray_bin_length_s: float = 1.0
    ray_required_length_min_s: float = 8.0
    ray_required_length_max_s: float = 20.0
    ray_required_length_inner_fraction: float = 0.50
    ray_bin_coverage_min: float = 0.70
    shell_contour_component_fraction_min: float = 0.80
    per_bin_arc_coverage_min: float = 0.50
    per_bin_ray_coverage_min: float = 0.50
    passing_axial_bins_min: int = 6
    passing_axial_center_span_s: float = 15.0
    passing_bin_fraction_min: float = 0.60
    duplicate_direction_degrees: float = 2.0
    duplicate_line_distance_s: float = 2.0
    duplicate_radius_difference_s: float = 2.0
    duplicate_endpoint_difference_degrees: float = 6.0
    final_axis_hard_cap: int = 16
    native_image_pixels: int = 4096
    randomness_enabled: bool = False
    gpu_enabled: bool = False
    source_point_serialization: bool = False
    target_capability_granted: bool = False

    def as_frozen_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["signature_ranks"] = list(self.signature_ranks)
        value.update({
            "arithmetic": "IEEE754_FLOAT64",
            "coordinate_duplicate_rule": "EXACT_XYZ",
            "canonical_order": "LEXICOGRAPHIC_XYZ_BYTES",
            "normals": "FORBIDDEN",
            "architecture_lineage": "POST_LSSC_TERMINAL_RAW_COORDINATE_EXTRUSION_PROFILE",
        })
        return value


CONFIG = LEPGConfig()
CONFIG_SHA256 = sha_json(CONFIG.as_frozen_dict())


def validate_config(config: LEPGConfig = CONFIG) -> None:
    if config.signature_ranks != (1, 2, 4, 8, 16, 32):
        raise ValueError("LEPG signature ranks changed")
    if config.direction_mean_shift_iterations != 3 or config.common_center_refit_iterations != 3:
        raise ValueError("LEPG iteration count changed")
    if config.randomness_enabled or config.gpu_enabled:
        raise ValueError("LEPG randomness/GPU must remain disabled")
    if config.source_point_serialization or config.target_capability_granted:
        raise ValueError("LEPG source serialization and target access remain closed")


validate_config()
