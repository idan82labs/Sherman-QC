from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from research.f1_pgrr_v1.config import CONFIG, CONFIG_SHA256, validate_config
from research.f1_pgrr_v1 import geometry
from research.f1_pgrr_v1.independent_validator import poles as independent_poles
from research.f1_pgrr_v1.independent_validator import validate_against_producer
from research.f1_pgrr_v1.tamper_suite import assert_tamper_suite
from research.f1_ciip_v1.identifiers import point_id_sha256
from scripts.run_f1_pgrr_v1_source_free_import_regression import run_regression


def test_locked_configuration_contract() -> None:
    validate_config()
    assert CONFIG_SHA256 == "97fb3a44affa34048e663eba446f4f11e1c3f6170b0c799ca71e44291bb32bb7"
    assert CONFIG.sample_cap == 200_000
    assert CONFIG.projective_pole_count == 321
    assert CONFIG.c006_target_count * CONFIG.deduplicated_ribbon_maximum == 9_344
    assert CONFIG.peak_rss_ceiling_bytes == 6_442_450_944


def test_projective_pole_tables_match_independent_regeneration() -> None:
    producer = geometry.projective_poles()
    validator = independent_poles()
    assert producer.shape == (321, 3)
    assert np.array_equal(producer, validator)
    assert np.allclose(np.linalg.norm(producer, axis=1), 1.0, rtol=0.0, atol=4e-16)
    for row in producer:
        first = next(value for value in row if value != 0.0)
        assert first > 0.0


def test_analytic_symmetric_eigensolver_matches_lapack() -> None:
    matrices = []
    for index in range(1, 17):
        basis = np.asarray(
            [
                [1.0 + index, 0.1 * index, -0.03 * index],
                [0.1 * index, 3.0 + 0.5 * index, 0.07 * index],
                [-0.03 * index, 0.07 * index, 8.0 + index],
            ],
            dtype=np.float64,
        )
        matrices.append(basis)
    matrix = np.stack(matrices)
    values, vectors = geometry._analytic_eigh_smallest(matrix)
    expected, _expected_vectors = np.linalg.eigh(matrix)
    assert np.allclose(values, expected, rtol=2e-13, atol=2e-13)
    residual = np.einsum("nij,nj->ni", matrix, vectors) - values[:, 0, None] * vectors
    assert float(np.max(np.linalg.norm(residual, axis=1))) < 1e-11


def test_projective_theta_is_antipodally_invariant() -> None:
    axis = np.asarray([0.0, 0.0, 1.0])
    normals = np.asarray(
        [[1.0, 0.0, 0.0], [math.sqrt(0.5), math.sqrt(0.5), 0.0]],
        dtype=np.float64,
    )
    first = geometry._projective_theta(normals, axis)
    second = geometry._projective_theta(-normals, axis)
    delta = np.minimum(np.abs(first - second), math.pi - np.abs(first - second))
    assert np.max(delta) <= 2 * np.finfo(float).eps


def test_mutual_graph_keeps_only_reciprocal_neighbors() -> None:
    neighbors = np.tile(np.arange(48, dtype=np.uint32), (50, 1))
    for row in range(50):
        neighbors[row] = np.asarray(
            [value for value in range(50) if value != row][:48], dtype=np.uint32
        )
    left, right = geometry._mutual_graph(neighbors)
    assert len(left) <= CONFIG.mutual_edge_maximum
    assert np.all(left < right)
    assert list(zip(left[:10], right[:10], strict=True)) == sorted(
        zip(left[:10], right[:10], strict=True)
    )


def test_mandatory_tamper_regressions_all_reject() -> None:
    assert all(assert_tamper_suite().values())


def test_candidate_worker_has_no_target_capability_import() -> None:
    source = Path("scripts/run_f1_pgrr_v1_candidate_worker.py").read_text(encoding="utf-8")
    prohibited = (
        "frozen_handoff",
        "f1_v2_cross_parent",
        "mapping_preflight",
        "ordinary_batch",
        "target ledger",
    )
    assert all(token not in source for token in prohibited)


def test_exact_unresolved_worker_import_regression_is_source_free() -> None:
    result = run_regression()
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["command_record"]["launch_argv0"] == str(Path(__import__("sys").executable))
    assert result["environment_identity"]["parent_sys_prefix"] == result["environment_identity"]["child_sys_prefix"]
    assert result["environment_identity"]["parent_sys_base_prefix"] == result["environment_identity"]["child_sys_base_prefix"]


def test_semantic_consensus_rejects_membership_exchange() -> None:
    producer = {
        "configuration_sha256": "a",
        "source_sha256": "b",
        "point_count": 2,
        "canonical_point_order_sha256": "c",
        "stable_point_commitment_sha256": "d",
        "pole_table_sha256": "e",
        "coarse_possible_pole_count": 1,
        "raw_ribbons": [
            {"pole_id": 0, "component_min_point_id": "p", "membership_sha256": "m1", "member_count": 64, "bin_count": 4},
            {"pole_id": 1, "component_min_point_id": "q", "membership_sha256": "m2", "member_count": 64, "bin_count": 4},
        ],
        "deduplicated_ribbons": [],
        "decision": "PGRR_V1_CANDIDATES_FROZEN_PENDING_HANDOFF",
    }
    independent = {
        "configuration_sha256": "a",
        "source_sha256": "b",
        "point_count": 2,
        "canonical_point_order_sha256": "c",
        "stable_point_commitment_sha256": "d",
        "pole_table_sha256": "e",
        "coarse_possible_pole_count": 1,
        "raw_ribbons_semantic": list(reversed(producer["raw_ribbons"])),
        "deduplicated_ribbons_semantic": [],
        "decision": "PGRR_V1_CANDIDATES_FROZEN_PENDING_HANDOFF",
    }
    assert validate_against_producer(producer, independent)["semantic_consensus"] is False


def test_source_free_analytic_fixture_reaches_a_terminal_representation_state() -> None:
    xyz = np.asarray(
        [
            (float(x), float(y), 0.02 * float((x + y) % 3))
            for x in range(12)
            for y in range(12)
        ],
        dtype="<f8",
    )
    point_ids = np.empty((len(xyz), 32), dtype=np.uint8)
    raw = xyz.view(np.uint8).reshape(len(xyz), 24)
    for index in range(len(xyz)):
        point_ids[index] = np.frombuffer(point_id_sha256(bytes(raw[index])), dtype=np.uint8)
    sample = geometry.Sample(
        xyz=xyz,
        point_ids=point_ids,
        multiplicities=np.ones(len(xyz), dtype=np.uint32),
        canonical_point_order_sha256="fixture",
        unique_record_count=len(xyz),
    )
    field = geometry.build_normal_field(sample)
    result = geometry.build_candidates(sample, field)
    assert result["decision"] in {
        "PGRR_V1_NO_GAUSS_ARC_MASS_ARCHITECTURE_STOP",
        "PGRR_V1_NO_RULED_PREIMAGE_ARCHITECTURE_STOP",
        "PGRR_V1_NONSELECTIVE_REPRESENTATION_ARCHITECTURE_STOP",
        "PGRR_V1_CANDIDATES_FROZEN_PENDING_HANDOFF",
    }
    assert result["target_access_before_candidate_freeze"] is False
    assert result["source_points_serialized"] is False
