"""Immutable constants from the authorized GCDF-V1 ruling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


ATTRITION_TOKENS = (
    "DUPLICATE_QUANTIZATION", "EDGE_NONMUTUAL", "EDGE_RADIUS_HIGH",
    "EDGE_REDUNDANCY_LOW", "EDGE_TANGENCY_NUMERIC_BOUNDARY", "EDGE_TANGENCY_HIGH",
    "PAIR_SCALE_LOW", "PAIR_SCALE_HIGH", "GLOBAL_DEFICIT_LOW",
    "LOCAL_PROFILE_NOT_EVALUABLE", "LOCAL_PEAK_LOW", "PEAK_CONCENTRATION_LOW",
    "VOTE_LINK_ISOLATED", "VOTE_COUNT_LOW", "UNIQUE_PEAK_COUNT_LOW",
    "ENDPOINT_DIVERSITY_LOW", "ENDPOINT_DOMINANCE_HIGH", "SPINE_EXTENT_LOW",
    "SPINE_BRANCHING_HIGH", "SPINE_CHORD_RATIO_LOW", "CAUSAL_RADIAL_SUPPORT_LOW",
    "CAUSAL_SPLIT_LOW", "CAUSAL_REDUCTION_LOW", "WRONG_SIGN_CONTROL_HIGH",
    "PASS", "NOT_PRESENT", "NOT_EVALUABLE",
)


@dataclass(frozen=True, slots=True)
class GCDFConfig:
    schema_version: int = 1
    architecture_family_token: str = "GEODESIC_CHORD_DEFICIT_FLUX_V1"
    primitive_token: str = "LANDMARK_SHORTEST_PATH_CHORD_DEFICIT_BACKPROJECTION_V1"
    attempt_id: str = "GCDF_V1_C006_FROZEN_SOURCE_CAUSAL_PREFLIGHT_V1"
    raw_source_sha256: str = "050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97"
    sample_count: int = 200_000
    quantization_bits: int = 28
    metric_fraction_bits: int = 16
    minimum_unique_nodes: int = 198_000
    graph_neighbors: int = 24
    tangent_neighbors: int = 32
    density_neighbor_rank: int = 8
    edge_radius_multiple: int = 3
    reliable_gap_ppm: int = 20_032
    unreliable_gap_ppm: int = 19_968
    tangent_admit_q24: int = 14_092_797
    tangent_boundary_high_q24: int = 14_092_924
    active_component_minimum_nodes: int = 512
    active_mass_minimum_ppm: int = 950_000
    active_component_maximum: int = 8
    active_mean_degree_minimum: int = 4
    tangency_boundary_maximum_ppm: int = 50_000
    landmark_count: int = 160
    pair_scale_low_multiple: int = 64
    pair_scale_high_multiple: int = 512
    pair_coverage_minimum: int = 64
    global_deficit_minimum_ppm: int = 3_000
    local_half_window_multiple: int = 24
    profile_stride_multiple: int = 4
    local_side_low_multiple: int = 20
    local_side_high_multiple: int = 28
    local_peak_minimum_ppm: int = 5_000
    concentration_floor_ppm: int = 1_000
    peak_concentration_minimum_ppm: int = 450_000
    accepted_path_minimum: int = 24
    accepted_fraction_minimum_ppm: int = 2_500
    accepted_fraction_maximum_ppm: int = 300_000
    vote_link_radius_multiple: int = 16
    vote_link_edge_cap: int = 64
    vote_minimum: int = 24
    unique_peak_minimum: int = 12
    endpoint_diversity_minimum: int = 12
    endpoint_dominance_maximum_ppm: int = 250_000
    spine_extent_minimum_multiple: int = 64
    spine_branching_maximum_ppm: int = 350_000
    spine_chord_ratio_minimum_ppm: int = 920_000
    candidate_maximum: int = 32
    causal_radial_minimum_multiple: int = 8
    causal_fold_minimum: int = 6
    causal_median_minimum_ppm: int = 500_032
    causal_vote_minimum_ppm: int = 350_032
    causal_wrong_sign_maximum_ppm: int = 199_968
    peak_rss_ceiling_bytes: int = 6_442_450_944
    role_wall_ceiling_seconds: int = 300
    renderer_wall_ceiling_seconds: int = 120
    randomness_enabled: bool = False
    target_capability_granted: bool = False
    target_capability_opened: bool = False
    target_process_launched: bool = False

    def as_frozen_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["attrition_tokens"] = list(ATTRITION_TOKENS)
        value["provenance"] = "PREREGISTERED_ENGINEERING_CONSTANTS_NOT_LITERATURE_ESTABLISHED"
        value["minimum_gate_equality_passes"] = True
        value["maximum_gate_equality_passes"] = True
        value["median"] = "LOWER_MEDIAN_AFTER_CANONICAL_ASCENDING_SORT"
        return value


CONFIG = GCDFConfig()
CONFIG_SHA256 = sha_json(CONFIG.as_frozen_dict())


def validate_config(config: GCDFConfig = CONFIG) -> None:
    if config.sample_count != 200_000 or config.landmark_count != 160:
        raise ValueError("GCDF cardinality contract changed")
    if (config.graph_neighbors, config.tangent_neighbors, config.density_neighbor_rank) != (24, 32, 8):
        raise ValueError("GCDF neighborhood contract changed")
    if config.target_capability_granted or config.target_capability_opened or config.target_process_launched:
        raise ValueError("GCDF target capability must remain closed")
    if config.randomness_enabled:
        raise ValueError("GCDF randomness is prohibited")
    if len(ATTRITION_TOKENS) != len(set(ATTRITION_TOKENS)):
        raise ValueError("GCDF attrition tokens must be unique")


validate_config()
