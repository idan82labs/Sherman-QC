"""Immutable constants for the authorized AMHC-V1 C006 one-shot preflight."""

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


CAMERAS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
FULL_LAYERS = ("raw-high-turn", "retained-components")
NUISANCE_LEVELS = ("NONE", "MINOR", "MATERIAL", "DOMINANT", "UNCERTAIN")


@dataclass(frozen=True, slots=True)
class AMHCConfig:
    schema_version: int = 1
    architecture_token: str = "ANTIPODAL_MIDSURFACE_HINGE_CURRENT_V1"
    attempt_id: str = "AMHC_V1_C006_FROZEN_SOURCE_ONLY_ONE_SHOT_PREFLIGHT"
    authorization_token: str = (
        "AUTHORIZE_ANTIPODAL_MIDSURFACE_HINGE_CURRENT_V1_C006_"
        "FROZEN_SOURCE_ONLY_ONE_SHOT_PREFLIGHT"
    )
    raw_source_sha256: str = "050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97"
    raw_source_bytes: int = 15_015_773
    raw_source_relative: str = (
        "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/"
        "35705010_A/scans/35705010_A_X.ply"
    )
    sample_count: int = 200_000
    normal_neighbor_count: int = 24
    normal_radius_h: float = 4.0
    normal_minimum_neighbors: int = 12
    normal_lambda_ratio_maximum: float = 0.35
    thickness_anchor_cap: int = 8_192
    pilot_j_minimum: int = 26
    pilot_j_maximum: int = 96
    pilot_octave_denominator: int = 16
    query_neighbor_count: int = 8
    query_accept_radius_h: float = 1.5
    axial_low_h: float = 3.0
    axial_high_h: float = 64.0
    lateral_maximum_h: float = 1.25
    normal_pair_maximum_degrees: float = 20.0
    anchor_axis_maximum_degrees: float = 15.0
    peer_axis_maximum_degrees: float = 25.0
    thickness_log2_tolerance: float = 0.125
    pilot_minimum_hits: int = 512
    pilot_mode_fraction_minimum: float = 0.35
    reciprocal_minimum_pairs: int = 5_000
    reciprocal_participation_minimum: float = 0.08
    midpoint_sixth_neighbor_maximum_h: float = 5.0
    graph_neighbor_count: int = 12
    graph_length_low_g: float = 0.5
    graph_length_high_g: float = 2.5
    graph_length_maximum_h: float = 8.0
    graph_director_edge_dot_maximum: float = 0.82
    graph_long_edge_g: float = 1.5
    graph_midpoint_fill_g: float = 0.75
    fan_minimum_directions: int = 8
    fan_sector_count: int = 8
    fan_minimum_occupied_sectors: int = 6
    fan_maximum_gap_degrees: float = 150.0
    erosion_hops: int = 2
    core_minimum_midpoints: int = 4_000
    core_coverage_minimum: float = 0.20
    active_angle_minimum_degrees: float = 4.0
    active_robust_sigma_multiplier: float = 6.0
    active_sigma_floor_degrees: float = 0.25
    active_fraction_maximum: float = 0.05
    component_minimum_edges: int = 32
    component_minimum_diameter_g: float = 24.0
    line_support_radius_g: float = 4.0
    line_support_ratio_minimum: float = 9.0
    line_support_fraction_minimum: float = 0.75
    component_cap: int = 256
    polyline_cap: int = 128
    polyline_vertex_cap: int = 256
    cpu_workers_per_role: int = 1
    global_wall_seconds: int = 300
    peak_rss_ceiling_bytes: int = 16 * 1024**3
    semantic_payload_ceiling_bytes: int = 32 * 1024**2
    visual_package_ceiling_bytes: int = 256 * 1024**2
    authoritative_png_count: int = 30
    global_image_pixels: int = 4_096
    crop_image_pixels: int = 2_048
    rejected_crop_count: int = 8
    retained_crop_count: int = 8
    block_size: int = 4_096
    randomness_enabled: bool = False
    gpu_enabled: bool = False
    target_capability_granted: bool = False
    source_point_serialization: bool = False

    def as_frozen_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "cameras": list(CAMERAS),
                "full_layers": list(FULL_LAYERS),
                "length_bins_g": [[0.5, 1.0], [1.0, 1.75], [1.75, 2.5]],
                "normal_signs": "PROJECTIVE_ONLY_NO_GLOBAL_ORIENTATION",
                "pilot_t_values": "h*2**(j/16),j=26..96",
                "pilot_choice": "MIN_LATERAL_THEN_GRID_RESIDUAL_THEN_CANONICAL_INDEX",
                "percentiles": "CANONICAL_ASCENDING_NEAREST_RANK",
                "role_order": ["BASE_A", "ORDER_PERMUTED", "INDEPENDENT_VALIDATOR"],
                "provenance": "GPT56_PRO_AUTHORIZED_ACTION_FIRST_SOURCE_ONLY_PREFLIGHT",
            }
        )
        return value


CONFIG = AMHCConfig()
CONFIG_SHA256 = sha_json(CONFIG.as_frozen_dict())


def validate_config(config: AMHCConfig = CONFIG) -> None:
    if config.sample_count != 200_000:
        raise ValueError("AMHC sample cardinality changed")
    if config.authoritative_png_count != 6 * 2 + 8 + 8 + 2:
        raise ValueError("AMHC visual cardinality changed")
    if config.randomness_enabled or config.gpu_enabled or config.cpu_workers_per_role != 1:
        raise ValueError("AMHC must remain deterministic single-CPU and GPU-free")
    if config.target_capability_granted or config.source_point_serialization:
        raise ValueError("AMHC target access and durable source serialization remain closed")


validate_config()
