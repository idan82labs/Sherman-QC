from __future__ import annotations

import copy
import hashlib
import subprocess
import sys

import numpy as np

from research.pwts_v1.independent_validator import (
    INVALID_STATUS,
    PREFLIGHT_ID,
    VALID_STATUS,
    ValidationPolicy,
    _canonical_vector,
    _compact_array_digest,
    derive_expected_criteria,
    forbidden_imports,
    row_key,
    reconstruct_case_from_raw,
    sha256_json,
    sorted_id_digest,
    table_digest,
    validate_evidence,
)
from research.pwts_v1.config import CONFIG_SHA256, get_config_copy
from research.pwts_v1.schema import SCHEMA_SHA256
from research.pwts_v1.semantic_core import build_representation, parse_ply_xyz


CONFIG_SHA = "5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49"
SCHEMA_SHA = "1" * 64
SOURCE_MANIFEST_SHA = "2" * 64
CASES = ("C001", "C002")


def _pid(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sort(evidence, table):
    evidence[table].sort(key=lambda row: row_key(table, row))


def make_real_like_compact_evidence():
    """A copied-table-shaped fixture; it contains no scan or synthetic geometry."""

    evidence = {
        "preflight_id": PREFLIGHT_ID,
        "schema_version": 1,
        "terminal_status": VALID_STATUS,
        "config_sha256": CONFIG_SHA,
        "schema_sha256": SCHEMA_SHA,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "case_manifest": [],
        "field_summary": [],
        "seed_primitive": [],
        "trace_primitive": [],
        "trace_node": [],
        "strip_membership": [],
        "strip_summary": [],
        "certificate": [],
        "destructive_negative": [],
        "rejection_census": [],
        "equivalence": [],
        "resource": [],
        "criteria": [],
        "safety": {
            "gold_truth": False,
            "expected_counts": False,
            "manual_labels": False,
            "cad_reference": False,
            "prior_f1_candidates": False,
            "lpf_semantics": False,
            "mesh_topology": False,
            "solver": False,
            "tuned_tracers": False,
            "heatmap_acceptance": False,
            "learning": False,
            "ownership_births": False,
            "replays": False,
            "synthetic_rescue": False,
            "runtime_main": False,
            "production": False,
            "benchmark_reporting": False,
            "import_reports": {
                "runner": ["hashlib", "json"],
                "semantic_core": ["math", "numpy"],
                "independent_validator": ["ast", "hashlib", "json"],
            },
        },
    }

    for index, token in enumerate(CASES, start=1):
        raw_sha = f"{index + 2:x}" * 64
        evidence["case_manifest"].append(
            {
                "case_token": token,
                "raw_path": f"locked/{token}.ply",
                "raw_byte_count": 1000 + index,
                "raw_ply_sha256": raw_sha,
                "observed_raw_ply_sha256": raw_sha,
                "ply_header_sha256": f"{index + 4:x}" * 64,
                "parsed_vertex_count": 130,
                "finite_coordinate_count": 130,
                "exact_duplicate_count": 2,
                "ignored_property_name_digest": f"{index + 6:x}" * 64,
                "canonical_coordinate_multiset_digest": f"{index + 8:x}" * 64,
                "sample_count": 128,
                "sampled_point_id_digest": f"{index + 10:x}" * 64,
                "validator_sampled_point_id_digest": f"{index + 10:x}" * 64,
                "config_sha256": CONFIG_SHA,
                "schema_sha256": SCHEMA_SHA,
                "source_points_serialized": False,
            }
        )
        active = token == "C001"
        evidence["field_summary"].append(
            {
                "case_token": token,
                "sample_count": 128,
                "normal_valid_count_k16": 128,
                "normal_valid_count_k32": 128,
                "normal_valid_count_k64": 128,
                "seed_eligible_count": 1 if active else 0,
                "trace_eligible_count": 64 if active else 0,
                "deterministic_local_maximum_count": 1 if active else 0,
                "dense_knn_digest": f"{index + 12:x}" * 64,
                "dense_normal_digest": f"{index + 14:x}" * 64,
                "dense_response_digest": f"{index + 1:x}" * 64,
            }
        )
        for role in ("BASE_A", "BASE_B", "ORDER_PERMUTED", "INDEPENDENT_VALIDATOR"):
            evidence["resource"].append(
                {
                    "case_token": token,
                    "process_role": role,
                    "cpu_seconds": 1.0,
                    "wall_seconds": 2.0,
                    "process_tree_peak_rss_bytes": 50_000_000,
                    "scratch_bytes": 1_000,
                    "serialized_bytes": 2_000,
                    "monitor_available": True,
                    "full_lifetime_accounting": True,
                    "reported_passing": True,
                }
            )

    trace_id = "trace-001"
    certificate_id = "certificate-001"
    evidence["seed_primitive"].append(
        {
            "case_token": "C001",
            "seed_point_id": "seed-point-001",
            "d1_rad": 0.12,
            "d2_rad": 0.13,
            "response_rad": 0.1248999599,
            "scale_balance": 0.9,
            "tangent_support_count": 12,
            "tangent_anisotropy": 0.8,
            "trace_id": trace_id,
            "valid": True,
            "first_rejection_code": "VALID",
        }
    )
    for ordinal in range(9):
        evidence["trace_node"].append(
            {
                "case_token": "C001",
                "trace_id": trace_id,
                "ordinal": ordinal,
                "point_id": _pid(f"T{ordinal:03d}"),
                "response_rad": 0.2,
                "tangent_x": 1.0,
                "tangent_y": 0.0,
                "tangent_z": 0.0,
            }
        )
    evidence["trace_primitive"].append(
        {
            "case_token": "C001",
            "trace_id": trace_id,
            "seed_point_id": "seed-point-001",
            "node_count": 9,
            "h_t": 1.0,
            "normalized_arc_length": 8.0,
            "maximum_normalized_step": 1.0,
            "median_response_rad": 0.2,
            "maximum_turn_rad": 0.1,
            "node_table_digest": sorted_id_digest(
                row["point_id"] for row in evidence["trace_node"]
            ),
            "valid": True,
            "first_rejection_code": "VALID",
        }
    )
    for side, prefix, cosine in (("left", "L", -0.8), ("right", "R", 0.8)):
        for number in range(32):
            evidence["strip_membership"].append(
                {
                    "case_token": "C001",
                    "trace_id": trace_id,
                    "side": side,
                    "point_id": _pid(f"{prefix}{number:03d}"),
                    "nearest_segment_ordinal": min(number // 4, 7),
                    "arc_bin": number // 4,
                    "rho_over_h_t": 2.0,
                    "signed_side_cosine": cosine,
                    "response_rad": 0.1,
                    "surface_variation": 0.05,
                }
            )
        evidence["strip_summary"].append(
            {
                "case_token": "C001",
                "trace_id": trace_id,
                "side": side,
                "count": 32,
                "covered_bins": 8,
                "normal_coherence": 0.9,
                "dominant_normal_x": 0.0,
                "dominant_normal_y": 0.0 if side == "left" else 0.3,
                "dominant_normal_z": 1.0,
                "median_surface_variation": 0.05,
                "p90_surface_variation": 0.05,
                "median_response_rad": 0.1,
                "sorted_support_digest": sorted_id_digest(
                    _pid(f"{prefix}{number:03d}") for number in range(32)
                ),
            }
        )
    evidence["certificate"].append(
        {
            "case_token": "C001",
            "certificate_id": certificate_id,
            "trace_id": trace_id,
            "trace_valid": True,
            "membership_intersection_count": 0,
            "left_support_count": 32,
            "right_support_count": 32,
            "left_covered_bins": 8,
            "right_covered_bins": 8,
            "bilaterally_covered_bins": 8,
            "side_count_ratio": 1.0,
            "left_normal_coherence": 0.9,
            "right_normal_coherence": 0.9,
            "left_median_surface_variation": 0.05,
            "left_p90_surface_variation": 0.05,
            "right_median_surface_variation": 0.05,
            "right_p90_surface_variation": 0.05,
            "side_normal_separation_rad": 0.3,
            "trace_median_response_rad": 0.2,
            "left_median_response_rad": 0.1,
            "right_median_response_rad": 0.1,
            "gate_trace_valid": True,
            "gate_membership_disjoint": True,
            "gate_left_support": True,
            "gate_right_support": True,
            "gate_left_coverage": True,
            "gate_right_coverage": True,
            "gate_bilateral_cocoverage": True,
            "gate_side_count_balance": True,
            "gate_left_normal_coherence": True,
            "gate_right_normal_coherence": True,
            "gate_left_surface_variation": True,
            "gate_right_surface_variation": True,
            "gate_side_normal_separation": True,
            "gate_trace_response": True,
            "gate_transition_contrast": True,
            "valid": True,
            "first_rejection_code": "VALID",
        }
    )
    for arm, side, code in (
        ("DELETE_LEFT", "left", "LEFT_SUPPORT_INSUFFICIENT"),
        ("DELETE_RIGHT", "right", "RIGHT_SUPPORT_INSUFFICIENT"),
    ):
        prefix = "L" if side == "left" else "R"
        evidence["destructive_negative"].append(
            {
                "case_token": "C001",
                "certificate_id": certificate_id,
                "deletion_arm": arm,
                "trace_id": trace_id,
                "deleted_id_digest": sorted_id_digest(
                    _pid(f"{prefix}{number:03d}") for number in range(32)
                ),
                "deleted_id_count": 32,
                "rebuilt_left_support_count": 0 if side == "left" else 32,
                "rebuilt_right_support_count": 0 if side == "right" else 32,
                "rebuilt_valid": False,
                "first_rejection_code": code,
                "raw_id_deletion_applied": True,
            }
        )

    denominator_stages = {
        "C001": {
            "all_sampled_points": 128,
            "normal_valid_k16": 128,
            "normal_valid_k32": 128,
            "normal_valid_k64": 128,
            "seed_eligible_points": 1,
            "deterministic_local_maxima": 1,
            "seed_primitives": 1,
            "valid_traces": 1,
            "certificate_candidates": 1,
            "valid_certificates": 1,
            "destructive_negative_arms": 2,
        },
        "C002": {
            "all_sampled_points": 128,
            "normal_valid_k16": 128,
            "normal_valid_k32": 128,
            "normal_valid_k64": 128,
            "seed_eligible_points": 0,
            "deterministic_local_maxima": 0,
            "seed_primitives": 0,
            "valid_traces": 0,
            "certificate_candidates": 0,
            "valid_certificates": 0,
            "destructive_negative_arms": 0,
        },
    }
    for token, stages in denominator_stages.items():
        for stage, denominator in stages.items():
            evidence["rejection_census"].append(
                {
                    "case_token": token,
                    "stage": stage,
                    "rejection_code": "VALID" if denominator else "EMPTY_POPULATION",
                    "numerator": denominator,
                    "denominator": denominator,
                }
            )

    semantic_tables = (
        "case_manifest",
        "field_summary",
        "seed_primitive",
        "trace_primitive",
        "trace_node",
        "strip_membership",
        "strip_summary",
        "certificate",
        "destructive_negative",
        "rejection_census",
    )
    for table in semantic_tables:
        _sort(evidence, table)
    for token in CASES:
        for comparison, other in (
            ("BASE_A_VS_BASE_B", "BASE_B"),
            ("BASE_A_VS_ORDER_PERMUTED", "ORDER_PERMUTED"),
            ("BASE_A_VS_INDEPENDENT_VALIDATOR", "INDEPENDENT_VALIDATOR"),
        ):
            counts = {}
            digests = {}
            for table in semantic_tables:
                case_rows = [row for row in evidence[table] if row.get("case_token") == token]
                counts[table] = {"BASE_A": len(case_rows), other: len(case_rows)}
                digest = table_digest(case_rows)
                digests[table] = {"BASE_A": digest, other: digest}
            evidence["equivalence"].append(
                {
                    "case_token": token,
                    "comparison": comparison,
                    "per_table_row_counts": counts,
                    "per_table_digests": digests,
                    "semantic_equivalence": True,
                    "first_mismatch": None,
                }
            )
    for table in (
        "case_manifest",
        "field_summary",
        "seed_primitive",
        "trace_primitive",
        "trace_node",
        "strip_membership",
        "strip_summary",
        "certificate",
        "destructive_negative",
        "rejection_census",
        "equivalence",
        "resource",
    ):
        _sort(evidence, table)
    evidence["criteria"] = derive_expected_criteria(
        evidence,
        ValidationPolicy(
            expected_configuration_sha256=CONFIG_SHA,
            expected_schema_sha256=SCHEMA_SHA,
            expected_source_manifest_sha256=SOURCE_MANIFEST_SHA,
            expected_case_tokens=CASES,
        ),
    )
    return evidence


def test_independent_validator_accepts_complete_real_like_tables():
    evidence = make_real_like_compact_evidence()
    result = validate_evidence(
        evidence,
        policy=ValidationPolicy(
            expected_configuration_sha256=CONFIG_SHA,
            expected_schema_sha256=SCHEMA_SHA,
            expected_source_manifest_sha256=SOURCE_MANIFEST_SHA,
            expected_case_tokens=CASES,
        ),
    )
    assert result.valid, result.errors
    assert result.terminal_status == VALID_STATUS
    assert result.derived["valid_certificate_count"] == 1
    assert result.derived["destructive_negative_arm_count"] == 2


def test_independent_validator_fails_closed_on_noncanonical_and_forged_gate():
    evidence = make_real_like_compact_evidence()
    evidence["strip_membership"].reverse()
    cert = evidence["certificate"][0]
    cert["left_support_count"] = 0
    cert["valid"] = True
    result = validate_evidence(
        evidence,
        policy=ValidationPolicy(
            expected_configuration_sha256=CONFIG_SHA,
            expected_schema_sha256=SCHEMA_SHA,
            expected_source_manifest_sha256=SOURCE_MANIFEST_SHA,
            expected_case_tokens=CASES,
        ),
    )
    assert not result.valid
    assert result.terminal_status == INVALID_STATUS
    assert any("noncanonical row order" in error for error in result.errors)
    assert any("certificate left count mismatch" in error for error in result.errors)


def test_import_boundary_detects_semantic_imports_without_importing_them():
    assert forbidden_imports("import json\nfrom backend import zero_shot_bend_profile") == (
        "backend",
    )
    assert forbidden_imports("import ast\nimport hashlib\n") == ()


def test_importing_independent_validator_does_not_transitively_load_semantic_core():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import research.pwts_v1.independent_validator; "
                "assert 'research.pwts_v1.semantic_core' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_independent_dense_digest_is_invariant_to_signed_zero():
    positive_zero = np.asarray(
        [[0.0, 1.0, 0.0], [2.0, 0.0, -3.0]], dtype=np.float64
    )
    negative_zero = positive_zero.copy()
    negative_zero[0, 0] = -0.0
    negative_zero[0, 2] = -0.0
    negative_zero[1, 1] = -0.0

    assert _compact_array_digest(negative_zero) == _compact_array_digest(positive_zero)
    assert np.signbit(negative_zero[[0, 0, 1], [0, 2, 1]]).all()


def test_independent_canonical_vector_normalizes_exact_zero_without_mutating_input():
    vector = np.asarray([-0.0, -2.0, 0.0], dtype=np.float64)
    canonical = _canonical_vector(vector)

    assert canonical.tolist() == [0.0, 2.0, 0.0]
    assert not np.signbit(canonical).any()
    assert np.signbit(vector[0])
    assert vector[1] == -2.0


def test_evidence_input_is_not_mutated():
    evidence = make_real_like_compact_evidence()
    before = copy.deepcopy(evidence)
    validate_evidence(
        evidence,
        policy=ValidationPolicy(
            expected_configuration_sha256=CONFIG_SHA,
            expected_schema_sha256=SCHEMA_SHA,
            expected_source_manifest_sha256=SOURCE_MANIFEST_SHA,
            expected_case_tokens=CASES,
        ),
    )
    assert evidence == before


def test_independent_raw_reconstruction_matches_flat_reference_tables_bytewise():
    lines = [
        "ply", "format ascii 1.0", "element vertex 144",
        "property float x", "property float y", "property float z", "end_header",
        *(f"{i}.0 {j}.0 0.0" for i in range(12) for j in range(12)),
    ]
    raw = ("\n".join(lines) + "\n").encode("ascii")
    points, header, parsed_count, ignored = parse_ply_xyz(raw)
    binding = {
        "raw_path": "locked/C001.ply",
        "raw_ply_sha256": hashlib.sha256(raw).hexdigest(),
        "observed_raw_ply_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_byte_count": len(raw),
        "ply_header_sha256": hashlib.sha256(header).hexdigest(),
        "parsed_vertex_count": parsed_count,
        "ignored_property_name_digest": sha256_json(sorted(ignored)),
        "schema_sha256": SCHEMA_SHA256,
    }
    reference = build_representation(
        points, "C001", source_binding=binding
    ).tables
    independent = reconstruct_case_from_raw(
        case_token="C001",
        raw_bytes=raw,
        config=get_config_copy(),
        config_sha256=CONFIG_SHA256,
        schema_sha256=SCHEMA_SHA256,
        source_binding=binding,
    )
    assert independent == reference


def test_independent_raw_reconstruction_matches_valid_wedge_tables_bytewise():
    side = 25
    half = side // 2
    lines = [
        "ply", "format ascii 1.0", f"element vertex {side * side}",
        "property double x", "property double y", "property double z", "end_header",
        *(
            f"{i * 0.5:.6f} {j * 0.5:.6f} {abs(j * 0.5):.6f}"
            for i in range(side) for j in range(-half, half + 1)
        ),
    ]
    raw = ("\n".join(lines) + "\n").encode("ascii")
    points, header, parsed_count, ignored = parse_ply_xyz(raw)
    binding = {
        "raw_path": "locked/C001.ply",
        "raw_ply_sha256": hashlib.sha256(raw).hexdigest(),
        "observed_raw_ply_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_byte_count": len(raw),
        "ply_header_sha256": hashlib.sha256(header).hexdigest(),
        "parsed_vertex_count": parsed_count,
        "ignored_property_name_digest": sha256_json(sorted(ignored)),
        "schema_sha256": SCHEMA_SHA256,
    }
    reference = build_representation(
        points, "C001", source_binding=binding
    ).tables
    independent = reconstruct_case_from_raw(
        case_token="C001",
        raw_bytes=raw,
        config=get_config_copy(),
        config_sha256=CONFIG_SHA256,
        schema_sha256=SCHEMA_SHA256,
        source_binding=binding,
    )
    assert sum(row["valid"] is True for row in reference["certificate"]) > 0
    assert independent == reference


def test_nonrecursive_criteria_projection_binds_all_seven_rows():
    evidence = make_real_like_compact_evidence()
    rows = evidence["criteria"]
    complete = next(
        row for row in rows
        if row["criterion_id"] == "COMPLETE_TABLES_PRESENT_AND_CANONICAL"
    )
    no_points = next(
        row for row in rows
        if row["criterion_id"] == "NO_SOURCE_POINT_SERIALIZATION"
    )
    assert complete["source_table_digest"] == no_points["source_table_digest"]

    changed = copy.deepcopy(evidence)
    changed["criteria"][0]["result"] = False
    result = validate_evidence(
        changed,
        policy=ValidationPolicy(
            expected_configuration_sha256=CONFIG_SHA,
            expected_schema_sha256=SCHEMA_SHA,
            expected_source_manifest_sha256=SOURCE_MANIFEST_SHA,
            expected_case_tokens=CASES,
        ),
    )
    assert not result.valid
    assert any("canonical seven" in error for error in result.errors)
