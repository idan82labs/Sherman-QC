from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from research.pwts_v1.config import (
    CONFIG,
    CONFIG_HASH_MATCHES_RULING,
    CONFIG_SHA256,
    EXPECTED_CONFIG_SHA256,
    canonical_json_bytes,
    get_config_copy,
)
from research.pwts_v1.schema import SCHEMA, assert_coordinate_free
from research.pwts_v1.semantic_core import (
    COORDINATE_MULTISET_DOMAIN,
    FieldState,
    CanonicalPoints,
    RepresentationResult,
    TraceGeometry,
    _build_deletions,
    build_certificate,
    build_memberships,
    build_trace,
    build_representation,
    canonicalize_points,
    deterministic_record_permutation,
    exact_knn,
    parse_ply_xyz,
)
import research.pwts_v1.semantic_core as semantic_core_module


def test_locked_configuration_hash_and_deep_immutability() -> None:
    assert CONFIG_HASH_MATCHES_RULING
    assert CONFIG_SHA256 == EXPECTED_CONFIG_SHA256
    assert hashlib.sha256(canonical_json_bytes(get_config_copy())).hexdigest() == EXPECTED_CONFIG_SHA256
    with pytest.raises(TypeError):
        CONFIG["input"]["maximum_sample_points"] = 1
    with pytest.raises(TypeError):
        CONFIG["case_tokens"][0] = "BROKEN"


def test_canonical_points_are_order_invariant_exact_and_binary_bound() -> None:
    records = np.array(
        [[0.0, -0.0, 1.0], [5.0, 2.0, 3.0], [0.0, 0.0, 1.0], [2.0, 8.0, 4.0], [math.nan, 0, 0]],
        dtype=np.float64,
    )
    first = canonicalize_points(records, maximum_points=2)
    second = canonicalize_points(records[::-1], maximum_points=2)
    assert first.point_ids == second.point_ids
    assert np.array_equal(first.points, second.points)
    assert first.parsed_count == 5
    assert first.finite_count == 4
    assert first.unique_count == 3
    assert first.exact_duplicate_count == 1
    assert first.sample_count == 2

    finite_packed = []
    for row in records[:4]:
        row = row.copy()
        row[row == 0.0] = 0.0
        finite_packed.append(struct.pack("<ddd", *row))
    expected_multiset = hashlib.sha256(
        COORDINATE_MULTISET_DOMAIN + b"".join(sorted(finite_packed))
    ).hexdigest()
    assert first.coordinate_multiset_digest == expected_multiset
    assert first.sampled_point_id_digest == hashlib.sha256(
        b"".join(sorted(bytes.fromhex(value) for value in first.point_ids))
    ).hexdigest()


def test_record_permutation_and_exact_knn_tie_break_are_deterministic() -> None:
    records = np.array([[i, i % 3, i % 5] for i in range(70)], dtype=np.float64)
    permuted_a = deterministic_record_permutation(records)
    permuted_b = deterministic_record_permutation(records)
    assert np.array_equal(permuted_a, permuted_b)
    assert not np.array_equal(permuted_a, records)

    # Query point has six equal-distance axis neighbors; k=4 must select by ID.
    points = np.vstack((np.zeros((1, 3)), np.eye(3), -np.eye(3), np.arange(1, 65)[:, None] * [10, 11, 12]))
    point_ids = tuple(hashlib.sha256(struct.pack("<Q", i)).digest() for i in range(len(points)))
    indices, distances = exact_knn(points, point_ids, k=4)
    expected = sorted(range(1, 7), key=lambda i: point_ids[i])[:4]
    assert indices[0].tolist() == expected
    assert distances[0].tolist() == [1.0] * 4


def _bilateral_fixture() -> tuple[object, FieldState, TraceGeometry]:
    raw = []
    trace_xyz = [(float(i), 0.0, 0.0) for i in range(9)]
    raw.extend(trace_xyz)
    for arc_bin in range(8):
        x = arc_bin + 0.5
        for distance in (2.0, 2.5, 3.0, 3.5):
            raw.append((x, -distance, 0.0))
            raw.append((x, distance, 0.0))
    # More than 64 unique points are needed by the representation contract.
    raw.extend((20.0 + i, 20.0, float(i % 2)) for i in range(70))
    canonical = canonicalize_points(np.asarray(raw, dtype=np.float64))
    n = canonical.sample_count
    lookup = {tuple(row): i for i, row in enumerate(canonical.points)}
    trace_nodes = [lookup[row] for row in trace_xyz]

    normals = np.zeros((n, 3, 3), dtype=np.float64)
    normals[:, :, 2] = 1.0
    response = np.full(n, 0.01, dtype=np.float64)
    variation = np.full((n, 3), 0.01, dtype=np.float64)
    for index in trace_nodes:
        response[index] = 0.15
    angle = math.radians(20.0)
    for row, index in lookup.items():
        if row[1] > 0 and row[0] < 9:
            normals[index, 0] = (0.0, math.sin(angle), math.cos(angle))
    tangent = np.zeros((n, 3), dtype=np.float64)
    tangent[:, 0] = 1.0
    state = FieldState(
        knn=np.zeros((n, 64), dtype=np.int64),
        knn_dist=np.ones((n, 64)),
        normals=normals,
        variation=variation,
        normal_valid=np.ones((n, 3), dtype=bool),
        d1=np.zeros(n), d2=np.zeros(n), response=response, balance=np.ones(n),
        seed_eligible=np.zeros(n, dtype=bool), trace_eligible=np.ones(n, dtype=bool),
        tangent=tangent, tangent_valid=np.ones(n, dtype=bool),
        tangent_support=np.full(n, 8), tangent_anisotropy=np.ones(n),
        h=np.ones(n), local_maxima=np.zeros(n, dtype=bool),
    )
    trace_points = canonical.points[trace_nodes]
    vectors = np.diff(trace_points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    geometry = TraceGeometry(
        trace_id="11" * 32,
        seed_index=trace_nodes[4],
        node_indices=trace_nodes,
        h_t=1.0,
        valid=True,
        first_rejection_code="VALID",
        segment_starts=trace_points[:-1],
        segment_vectors=vectors,
        segment_lengths=lengths,
        segment_g=np.tile((0.0, 1.0, 0.0), (8, 1)),
        cumulative_lengths=np.arange(9, dtype=np.float64),
    )
    return canonical, state, geometry


def test_strips_certificate_and_frozen_trace_deletions_are_causal() -> None:
    canonical, state, geometry = _bilateral_fixture()
    membership = build_memberships("C001", geometry, canonical, state)
    assert len(membership) == 64
    assert sum(row["side"] == "left" for row in membership) == 32
    assert sum(row["side"] == "right" for row in membership) == 32
    certificate, summaries = build_certificate("C001", geometry, membership, canonical, state)
    assert certificate["valid"] is True
    assert certificate["first_rejection_code"] == "VALID"
    assert [row["covered_bins"] for row in summaries] == [8, 8]

    arms = _build_deletions("C001", certificate, geometry, membership, canonical, state)
    assert [(row["deletion_arm"], row["first_rejection_code"]) for row in arms] == [
        ("DELETE_LEFT", "LEFT_SUPPORT_INSUFFICIENT"),
        ("DELETE_RIGHT", "RIGHT_SUPPORT_INSUFFICIENT"),
    ]
    assert arms[0]["rebuilt_left_support_count"] == 0
    assert arms[1]["rebuilt_right_support_count"] == 0
    assert all(row["rebuilt_valid"] is False and row["raw_id_deletion_applied"] for row in arms)


def test_trace_uses_one_symmetric_combined_257_node_cap() -> None:
    n = 401
    points = np.column_stack((np.arange(n, dtype=np.float64) * 0.5, np.zeros(n), np.zeros(n)))
    ids_bytes = tuple(hashlib.sha256(struct.pack("<Q", i)).digest() for i in range(n))
    canonical = CanonicalPoints(
        points=points,
        point_id_bytes=ids_bytes,
        point_ids=tuple(value.hex() for value in ids_bytes),
        parsed_count=n, finite_count=n, unique_count=n, exact_duplicate_count=0,
        sample_count=n, coordinate_multiset_digest="00" * 32,
        sampled_point_id_digest=hashlib.sha256(b"".join(sorted(ids_bytes))).hexdigest(),
    )
    knn = np.tile(np.arange(n, dtype=np.int64)[:, None], (1, 64))
    for i in range(n):
        if i > 0:
            knn[i, 0] = i - 1
        if i + 1 < n:
            knn[i, 1] = i + 1
    tangent = np.zeros((n, 3)); tangent[:, 0] = 1.0
    normals = np.zeros((n, 3, 3)); normals[:, :, 2] = 1.0
    state = FieldState(
        knn=knn, knn_dist=np.ones((n, 64)), normals=normals,
        variation=np.zeros((n, 3)), normal_valid=np.ones((n, 3), dtype=bool),
        d1=np.ones(n), d2=np.ones(n), response=np.ones(n), balance=np.ones(n),
        seed_eligible=np.ones(n, dtype=bool), trace_eligible=np.ones(n, dtype=bool),
        tangent=tangent, tangent_valid=np.ones(n, dtype=bool),
        tangent_support=np.full(n, 64), tangent_anisotropy=np.ones(n), h=np.ones(n),
        local_maxima=np.zeros(n, dtype=bool),
    )
    geometry, row = build_trace("C001", n // 2, canonical, state)
    assert len(geometry.node_indices) == 257
    assert row["node_count"] == 257
    assert geometry.valid is False
    assert geometry.first_rejection_code == "TRACE_NODE_CAP_EXCEEDED"


def test_full_semantic_tables_are_permutation_equivalent_and_coordinate_free() -> None:
    grid = np.asarray([(x, y, 0.0) for x in range(13) for y in range(13)], dtype=np.float64)
    first = build_representation(grid, "C001").compact_payload("BASE_A")
    second = build_representation(grid[::-1], "C001").compact_payload("BASE_B")
    assert first["tables"] == second["tables"]
    assert first["source_points_serialized"] is False
    assert_coordinate_free(first)
    text = json.dumps(first, sort_keys=True)
    assert '"source_points"' not in text
    assert '"sampled_points"' not in text

    field_summary = first["tables"]["field_summary"][0]
    assert field_summary["sample_count"] == len(grid)
    required_stages = {
        "all_sampled_points",
        "normal_valid_k16",
        "normal_valid_k32",
        "normal_valid_k64",
        "seed_eligible_points",
        "deterministic_local_maxima",
        "seed_primitives",
        "valid_traces",
        "certificate_candidates",
        "valid_certificates",
        "destructive_negative_arms",
    }
    census = first["tables"]["rejection_census"]
    for stage in required_stages:
        rows = [row for row in census if row["stage"] == stage]
        assert rows, stage
        denominators = {row["denominator"] for row in rows}
        assert len(denominators) == 1
        denominator = denominators.pop()
        assert sum(row["numerator"] for row in rows) == denominator
        if denominator == 0:
            assert rows == [{
                "case_token": "C001",
                "stage": stage,
                "rejection_code": "EMPTY_POPULATION",
                "numerator": 0,
                "denominator": 0,
            }]


def test_ascii_ply_parser_consumes_only_xyz_and_tracks_ignored_properties() -> None:
    raw = (
        b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
        b"property float z\nproperty uchar red\nelement face 1\nproperty list uchar int vertex_indices\n"
        b"end_header\n1 2 3 9\n4 5 6 8\n3 0 1 0\n"
    )
    points, header, count, ignored = parse_ply_xyz(raw)
    assert count == 2
    assert np.array_equal(points, [[1, 2, 3], [4, 5, 6]])
    assert header.endswith(b"end_header\n")
    assert ignored == ["vertex:red", "face:vertex_indices"]


def test_schema_declares_raw_binary_id_digest_and_no_xyz_serialization() -> None:
    assert SCHEMA["id_digest_rule"] == "sha256_raw_32_byte_ids_ascending_no_framing"
    assert SCHEMA["source_points_serialized"] is False
    payload = RepresentationResult("C001", {"certificate": []}).compact_payload()
    assert payload["source_points_serialized"] is False


def test_semantic_cli_verifies_file_and_canonical_locks_independently(tmp_path: Path) -> None:
    points = [(float(x), float(y), 0.0) for x in range(13) for y in range(13)]
    body = "".join(f"{x} {y} {z}\n" for x, y, z in points)
    raw = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "end_header\n"
        f"{body}"
    ).encode("ascii")
    raw_path = tmp_path / "C001.ply"
    raw_path.write_bytes(raw)
    header = raw[: raw.index(b"end_header") + len(b"end_header\n")]

    manifest_content = {
        "preflight_id": "PWTS_REPRESENTATION_PREFLIGHT_V1",
        "kind": "pwts_source_lock_v1",
        "semantic_work_performed": False,
        "sampling_performed": False,
        "source_points_serialized": False,
        "cases": [{
            "case_token": "C001",
            "raw_path": str(raw_path),
            "raw_byte_count": len(raw),
            "raw_ply_sha256": hashlib.sha256(raw).hexdigest(),
            "ply_header_sha256": hashlib.sha256(header).hexdigest(),
            "parsed_vertex_count": len(points),
            "finite_coordinate_count": len(points),
            "exact_duplicate_count": 0,
            "ignored_property_name_digest": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
            "canonical_coordinate_multiset_sha256": "00" * 32,
        }],
    }
    input_manifest_sha = hashlib.sha256(canonical_json_bytes(manifest_content)).hexdigest()
    manifest = {**manifest_content, "input_manifest_sha256": input_manifest_sha}
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_file_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest_file_sha != input_manifest_sha

    # Pretty formatting intentionally makes the file hash distinct from the
    # canonical configuration digest while preserving identical semantics.
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text(json.dumps(get_config_copy(), indent=2) + "\n", encoding="utf-8")
    configuration_file_sha = hashlib.sha256(configuration_path.read_bytes()).hexdigest()
    assert configuration_file_sha != EXPECTED_CONFIG_SHA256

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA, indent=2) + "\n", encoding="utf-8")
    runner_path = Path(__file__).resolve().parents[1] / "scripts" / "run_pwts_representation_preflight_v1.py"
    executable_path = Path(semantic_core_module.__file__).resolve()
    output_dir = tmp_path / "output"
    scratch_dir = tmp_path / "scratch"
    command = [
        sys.executable, str(executable_path),
        "--role", "BASE_A",
        "--case-token", "C001",
        "--raw-ply", str(raw_path),
        "--source-manifest", str(manifest_path),
        "--expected-source-manifest-file-sha256", manifest_file_sha,
        "--expected-input-manifest-sha256", input_manifest_sha,
        "--configuration", str(configuration_path),
        "--expected-configuration-file-sha256", configuration_file_sha,
        "--expected-configuration-sha256", EXPECTED_CONFIG_SHA256,
        "--schema", str(schema_path),
        "--expected-schema-sha256", hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "--expected-executable-sha256", hashlib.sha256(executable_path.read_bytes()).hexdigest(),
        "--runner-source", str(runner_path),
        "--expected-runner-sha256", hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        "--output-dir", str(output_dir),
        "--scratch-dir", str(scratch_dir),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output_dir / "case_payload.json").read_text(encoding="utf-8"))
    assert payload["case_token"] == "C001"
    assert payload["process_role"] == "BASE_A"
    assert payload["source_points_serialized"] is False

    bad_source_command = list(command)
    source_hash_index = bad_source_command.index("--expected-source-manifest-file-sha256") + 1
    bad_source_command[source_hash_index] = "ff" * 32
    failed_source = subprocess.run(bad_source_command, check=False, capture_output=True, text=True)
    assert failed_source.returncode != 0
    assert "source manifest file SHA mismatch" in failed_source.stderr

    bad_projection_command = list(command)
    manifest_content_hash_index = bad_projection_command.index("--expected-input-manifest-sha256") + 1
    bad_projection_command[manifest_content_hash_index] = "ee" * 32
    failed_projection = subprocess.run(
        bad_projection_command, check=False, capture_output=True, text=True
    )
    assert failed_projection.returncode != 0
    assert "input manifest content SHA mismatch" in failed_projection.stderr

    bad_config_file_command = list(command)
    config_file_hash_index = bad_config_file_command.index(
        "--expected-configuration-file-sha256"
    ) + 1
    bad_config_file_command[config_file_hash_index] = "dd" * 32
    failed_config_file = subprocess.run(
        bad_config_file_command, check=False, capture_output=True, text=True
    )
    assert failed_config_file.returncode != 0
    assert "configuration file SHA mismatch" in failed_config_file.stderr

    bad_config = dict(get_config_copy())
    bad_config["schema_version"] = 2
    configuration_path.write_text(json.dumps(bad_config, indent=2) + "\n", encoding="utf-8")
    config_file_hash_index = command.index("--expected-configuration-file-sha256") + 1
    bad_config_command = list(command)
    bad_config_command[config_file_hash_index] = hashlib.sha256(configuration_path.read_bytes()).hexdigest()
    failed_config = subprocess.run(bad_config_command, check=False, capture_output=True, text=True)
    assert failed_config.returncode != 0
    assert "configuration content SHA mismatch" in failed_config.stderr
