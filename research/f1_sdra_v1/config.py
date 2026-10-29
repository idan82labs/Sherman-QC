"""Immutable constants authorized for the single SDRA-V1 C006 preflight."""

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


K_VALUES = (24, 48, 96, 192, 384)
ACTIVE_SET_ORDER = (("a",), ("c",), ("b",), ("a", "c"), ("a", "b"), ("c", "b"), ("a", "c", "b"))
CAMERAS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
GEOMETRY_LAYERS = (
    "complete-source", "q24", "q48", "q96", "q192", "q384",
    "fitted-c", "model-gain-g", "candidate-mask", "supported-candidates-d",
)
EXECUTION_STATES = ("OBSERVED_COMPLETE", "OBSERVED_PARTIAL", "NOT_EVALUABLE")
ENTITY_STATES = ("PRESENT", "NOT_PRESENT", "NOT_APPLICABLE")


@dataclass(frozen=True, slots=True)
class SDRAConfig:
    schema_version: int = 1
    architecture_family_token: str = "SPHERICAL_DIRECTION_RANK_ASYMPTOTICS_V1"
    primitive_direction_token: str = "ANCHOR_CENTERED_UNIT_DIRECTION_SECOND_MOMENT_V1"
    primitive_model_token: str = "R_NEG2_R_ZERO_R_POS2_NONNEGATIVE_MODEL_COMPARISON_V1"
    primitive_support_token: str = "CANDIDATE_CODIMENSION_GROWTH_AUDIT_V1"
    attempt_id: str = "SDRA_V1_C006_FROZEN_SOURCE_CAUSAL_PREFLIGHT_V1"
    authorization_token: str = "AUTHORIZE_F1_POST_GCDF_SINGLE_ARCHITECTURE_PREFLIGHT_V1"
    raw_source_sha256: str = "050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97"
    raw_source_bytes: int = 15_015_773
    sample_count: int = 200_000
    quantization_bits: int = 30
    matrix_fraction_bits: int = 48
    metric_fraction_bits: int = 32
    block_size: int = 4_096
    morton_bits_per_axis: int = 6
    sentinel_count: int = 8_192
    sentinel_scale_eligible_minimum: int = 6_554
    sentinel_candidate_maximum: int = 2_048
    full_scale_eligible_minimum: int = 180_000
    candidate_minimum: int = 1
    candidate_maximum: int = 30_000
    supported_fraction_numerator: int = 1
    supported_fraction_denominator: int = 2
    successive_x_ratio_low_q32: int = 5 * (1 << 32) // 4
    successive_x_ratio_high_q32: int = 4 * (1 << 32)
    x384_low_q32: int = 6 * (1 << 32)
    x384_high_q32: int = 40 * (1 << 32)
    mu1_minimum_q32: int = 5 * (1 << 32) // 64
    c_low_q32: int = 1 * (1 << 32) // 64
    c_high_q32: int = 17 * (1 << 32) // 64
    gain_minimum_q32: int = 1 * (1 << 32) // 2
    error_maximum_q32: int = 1 * (1 << 32) // 5
    phi_minimum_q32: int = 3 * (1 << 32) // 8
    supported_m48_minimum: int = 2
    support_median_d_maximum_q32: int = 8 * (1 << 32) // 5
    support_p90_d_maximum_q32: int = 19 * (1 << 32) // 10
    cpu_workers_per_role: int = 1
    peak_rss_ceiling_bytes: int = 12_884_901_888
    producer_wall_seconds: int = 150
    validator_wall_seconds: int = 300
    renderer_wall_seconds: int = 90
    orchestration_wall_seconds: int = 720
    semantic_payload_ceiling_bytes: int = 100_663_296
    visual_package_ceiling_bytes: int = 805_306_368
    authoritative_png_count: int = 77
    global_image_pixels: int = 4_096
    context_image_pixels: int = 2_048
    context_region_count: int = 12
    context_morton_bits_per_axis: int = 6
    context_cell_expansion_r384_multiple: int = 4
    framing_margin_ppm: int = 20_000
    randomness_enabled: bool = False
    gpu_enabled: bool = False
    target_capability_granted: bool = False
    source_point_serialization: bool = False

    def as_frozen_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({
            "k_values": list(K_VALUES),
            "active_set_order": [list(row) for row in ACTIVE_SET_ORDER],
            "cameras": list(CAMERAS),
            "geometry_layers": list(GEOMETRY_LAYERS),
            "execution_states": list(EXECUTION_STATES),
            "entity_states": list(ENTITY_STATES),
            "rounding": "IEEE754_NEAREST_EVEN_THEN_FIXED_POINT_NEAREST_EVEN",
            "knn_rank": ["SQUARED_INTEGER_DISTANCE", "POINT_ID"],
            "percentiles": "CANONICAL_ASCENDING_NEAREST_RANK",
            "color_normalization": "FINITE_NONZERO_P01_P99_CLIP_OR_EXACT_BINARY",
            "provenance": "PROFESSOR_AUTHORIZED_SHERMAN_SPECIFIC_FALSIFIABLE_HYPOTHESIS",
        })
        return value


CONFIG = SDRAConfig()
CONFIG_SHA256 = sha_json(CONFIG.as_frozen_dict())


def validate_config(config: SDRAConfig = CONFIG) -> None:
    if K_VALUES != (24, 48, 96, 192, 384):
        raise ValueError("SDRA cumulative neighborhood sequence changed")
    if config.sample_count != 200_000 or config.sentinel_count != 8_192:
        raise ValueError("SDRA cardinality contract changed")
    if config.authoritative_png_count != 6 * 10 + 12 + 5:
        raise ValueError("SDRA authoritative visual cardinality changed")
    if config.target_capability_granted or config.source_point_serialization:
        raise ValueError("SDRA target/source serialization capability must remain closed")
    if config.randomness_enabled or config.gpu_enabled or config.cpu_workers_per_role != 1:
        raise ValueError("SDRA must remain deterministic single-CPU and GPU-free")
    if len(ACTIVE_SET_ORDER) != len(set(ACTIVE_SET_ORDER)):
        raise ValueError("SDRA active-set order must be unique")


validate_config()
