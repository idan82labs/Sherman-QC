"""Immutable engineering constants for the authorized PGRR-V1 tranche."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class PGRRConfig:
    schema_version: int = 1
    architecture_family_token: str = "PROJECTIVE_GAUSS_RULING_RIBBON_V1"
    primitive_token: str = "ANTIPODAL_GREAT_CIRCLE_ARC_RULED_PREIMAGE"
    tranche_id: str = "PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_V1"
    sample_cap: int = 200_000
    first_neighbor_count: int = 48
    normal_k_small: int = 24
    normal_k_large: int = 48
    density_neighbor_rank: int = 12
    normal_condition_ratio: float = 2.0 ** -12
    normal_scale_agreement_radians: float = math.radians(10.0)
    icosphere_subdivision_level: int = 3
    projective_pole_count: int = 321
    pole_band_radians: float = math.radians(12.0)
    pole_minimum_eligible_points: int = 64
    pole_minimum_angular_span_radians: float = math.radians(20.0)
    mutual_neighbor_count: int = 12
    edge_maximum_angular_gap_radians: float = math.radians(7.5)
    component_minimum_points: int = 64
    component_minimum_span_radians: float = math.radians(20.0)
    component_maximum_span_radians: float = math.radians(165.0)
    nominal_angular_bin_width_radians: float = math.radians(5.0)
    minimum_points_per_angular_bin: int = 4
    longitudinal_scale_multiple: float = 8.0
    longitudinal_column_count: int = 8
    minimum_longitudinal_columns: int = 6
    minimum_bin_column_occupancy: int = 4
    ruling_axis_radians: float = math.radians(15.0)
    ruling_elongation_ratio: float = 4.0
    raw_ribbon_maximum: int = 512
    deduplicated_ribbon_maximum: int = 128
    duplicate_axis_radians: float = math.radians(8.0)
    duplicate_jaccard_numerator: int = 80
    duplicate_jaccard_denominator: int = 100
    formal_near_miss_density_fraction: float = 0.25
    visual_subject_maximum: int = 128
    source_point_maximum: int = 200_000
    stored_neighbor_reference_maximum: int = 9_600_000
    mutual_edge_maximum: int = 1_200_000
    c006_target_count: int = 73
    frozen_target_count: int = 4_788
    handoff_row_maximum: int = 9_344
    angular_bin_maximum: int = 33
    peak_rss_ceiling_bytes: int = 6_442_450_944
    total_machine_wall_ceiling_seconds: int = 1_800
    scientific_serialization_ceiling_bytes: int = 67_108_864
    visual_package_ceiling_bytes: int = 335_544_320
    log_ceiling_bytes: int = 16_777_216
    total_output_ceiling_bytes: int = 419_430_400
    numeric_boundary_ulp_window: int = 32
    high_precision_bits: int = 160
    random_enabled: bool = False
    gpu_enabled: bool = False
    ranking_enabled: bool = False
    top_k_enabled: bool = False
    parameter_sweep_enabled: bool = False
    source_point_serialization_enabled: bool = False

    def as_frozen_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["provenance"] = (
            "PROPOSED_ENGINEERING_CONSTANTS_NOT_LITERATURE_ESTABLISHED"
        )
        value["angles_serialized_as_binary64_radians"] = True
        return value


CONFIG = PGRRConfig()
CONFIG_SHA256 = sha_json(CONFIG.as_frozen_dict())


def validate_config(config: PGRRConfig = CONFIG) -> None:
    if config.sample_cap != 200_000 or config.source_point_maximum != 200_000:
        raise ValueError("PGRR sample cap changed")
    if config.first_neighbor_count != 48 or config.normal_k_large != 48:
        raise ValueError("PGRR normal-neighborhood contract changed")
    if config.normal_k_small != 24 or config.mutual_neighbor_count != 12:
        raise ValueError("PGRR neighborhood constants changed")
    if config.projective_pole_count != 321:
        raise ValueError("PGRR pole count changed")
    if config.raw_ribbon_maximum != 512 or config.deduplicated_ribbon_maximum != 128:
        raise ValueError("PGRR selectivity ceilings changed")
    if config.c006_target_count * config.deduplicated_ribbon_maximum != 9_344:
        raise ValueError("PGRR handoff cardinality proof changed")
    if any(
        (
            config.random_enabled,
            config.gpu_enabled,
            config.ranking_enabled,
            config.top_k_enabled,
            config.parameter_sweep_enabled,
            config.source_point_serialization_enabled,
        )
    ):
        raise ValueError("PGRR prohibited capability enabled")


validate_config()
