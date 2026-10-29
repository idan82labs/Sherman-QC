"""Immutable constants for authorized CFAC-V1 source-free/C006 testing."""

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
class CFACConfig:
    schema_version: int = 1
    architecture_token: str = "CONCENTRIC_FOCAL_AXIS_COLLAPSE_V1"
    short_token: str = "CFAC_V1"
    authorization_token: str = (
        "AUTHORIZE_CONCENTRIC_FOCAL_AXIS_COLLAPSE_V1_IMPLEMENT_AND_EXECUTE_"
        "C006_FROZEN_SOURCE_ONLY_PREFLIGHT"
    )
    attempt_id: str = "CFAC_V1_C006_FROZEN_SOURCE_ONLY_PREFLIGHT"
    raw_source_sha256: str = "050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97"
    raw_source_bytes: int = 15_015_773
    raw_source_relative: str = (
        "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/"
        "35705010_A/scans/35705010_A_X.ply"
    )
    sample_count: int = 200_000
    minimum_unique_points: int = 50_000
    scale_neighbor_rank: int = 4
    jet_neighbor_count: int = 48
    tangent_sector_count: int = 12
    tangent_minimum_sectors: int = 10
    jet_d48_maximum_s: float = 5.5
    vertical_rms_maximum_s: float = 0.08
    normal_matrix_condition_maximum: float = 1e8
    gradient_norm_maximum: float = 0.35
    radius_minimum_s: float = 2.5
    radius_maximum_s: float = 64.0
    anisotropy_maximum: float = 0.18
    focal_vote_fraction_maximum: float = 0.60
    focal_vote_absolute_maximum: int = 120_000
    focal_dense_sixth_maximum_s: float = 1.25
    focal_coherence_radius_s: float = 2.0
    focal_coherence_minimum: float = 0.90
    graph_target_offsets_s: tuple[float, ...] = (-3.0, -1.5, 1.5, 3.0)
    graph_target_miss_maximum_s: float = 1.25
    graph_axis_disagreement_maximum_degrees: float = 15.0
    graph_perpendicular_maximum_s: float = 1.5
    provisional_minimum_votes: int = 256
    provisional_span_minimum_s: float = 16.0
    provisional_span_radius_multiplier: float = 2.5
    provisional_line_fraction_minimum: float = 0.96
    provisional_transverse_rms_maximum_s: float = 1.5
    provisional_collapse_density_minimum: float = 8.0
    reassociation_distance_s: float = 2.0
    reassociation_axis_degrees: float = 15.0
    reassociation_axial_expansion_s: float = 2.0
    radial_split_minimum_points: int = 64
    radial_split_minimum_fraction: float = 0.20
    radial_two_mode_gain_minimum: float = 0.70
    radial_separation_minimum_s: float = 2.0
    radial_mad_floor_s: float = 0.35
    radial_mad_radius_fraction: float = 0.06
    axial_bin_count: int = 8
    axial_minimum_dual_bins: int = 6
    axial_minimum_mode_points: int = 16
    axial_gap_mad_fraction: float = 0.20
    arc_width_minimum_degrees: float = 20.0
    arc_width_maximum_degrees: float = 220.0
    arc_width_agreement_degrees: float = 15.0
    arc_endpoint_agreement_degrees: float = 15.0
    continuation_offset_minimum_degrees: float = -25.0
    continuation_offset_maximum_degrees: float = 25.0
    continuation_offset_step_degrees: float = 2.5
    continuation_axial_sites: int = 8
    continuation_tangent_distances_s: tuple[float, ...] = (3.0, 6.0, 9.0, 12.0)
    continuation_site_miss_maximum_s: float = 1.25
    continuation_planar_curvature_maximum: float = 1.0 / 80.0
    continuation_normal_degrees: float = 15.0
    continuation_occupancy_minimum: float = 0.75
    continuation_endpoint_agreement_degrees: float = 7.5
    graph_edge_cap: int = 4_000_000
    provisional_component_cap: int = 256
    final_component_cap: int = 32
    wall_seconds: int = 290
    source_free_wall_seconds: int = 240
    peak_rss_ceiling_bytes: int = int(15.5 * 1024**3)
    source_free_rss_ceiling_bytes: int = 14 * 1024**3
    artifact_ceiling_bytes: int = 96 * 1024**2
    authoritative_png_count: int = 3
    native_image_pixels: int = 4096
    block_size: int = 2048
    randomness_enabled: bool = False
    gpu_enabled: bool = False
    target_capability_granted: bool = False
    source_point_serialization: bool = False

    def as_frozen_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["graph_target_offsets_s"] = list(self.graph_target_offsets_s)
        value["continuation_tangent_distances_s"] = list(
            self.continuation_tangent_distances_s
        )
        value.update(
            {
                "arithmetic": "IEEE754_FLOAT64",
                "knn_ties": "DISTANCE_THEN_CANONICAL_ROW_INDEX",
                "normal_orientation": "LOCAL_ARBITRARY_FOCAL_MAP_ORIENTATION_INVARIANT",
                "principal_curvature_order": "DOMINANT_BY_ABSOLUTE_MAGNITUDE",
                "architecture_lineage": "POST_AMHC_TERMINAL_DISTINCT_FOCAL_SPACE_OBJECT",
            }
        )
        return value


CONFIG = CFACConfig()
CONFIG_SHA256 = sha_json(CONFIG.as_frozen_dict())


def validate_config(config: CFACConfig = CONFIG) -> None:
    if config.sample_count != 200_000 or config.jet_neighbor_count != 48:
        raise ValueError("CFAC cardinality or jet neighborhood changed")
    if config.radius_minimum_s != 2.5 or config.radius_maximum_s != 64.0:
        raise ValueError("CFAC clarified radius gate changed")
    if config.anisotropy_maximum != 0.18:
        raise ValueError("CFAC clarified anisotropy gate changed")
    if config.randomness_enabled or config.gpu_enabled:
        raise ValueError("CFAC randomness/GPU must remain disabled")
    if config.target_capability_granted or config.source_point_serialization:
        raise ValueError("CFAC target access and source serialization remain closed")


validate_config()
