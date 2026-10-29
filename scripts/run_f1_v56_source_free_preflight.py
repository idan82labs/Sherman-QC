#!/usr/bin/env python3
"""Run the frozen source-free CTMLR V1 fixture suite before precommit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_canonical_turn_median_line_reconstruction_v56 import (
    PRECOMMIT_TOKEN,
    PROFILE,
    MeshData,
    SeedSchemaError,
    collapse_exact_duplicates,
    compose_v56,
    materialize_seed,
)


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_fold(
    angle_degrees: float,
    *,
    radius: float,
    tail: float = 22.0,
    step: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(angle_degrees)
    before = np.arange(-tail, -0.5 * step, step)
    points = [[float(value), 0.0] for value in before]
    if radius <= 0.0:
        points.append([0.0, 0.0])
        locus = np.asarray((0.0, 0.0), dtype=np.float64)
        start = np.asarray((0.0, 0.0), dtype=np.float64)
    else:
        count = max(9, int(math.ceil(radius * angle / step)) + 1)
        phi = np.linspace(-0.5 * math.pi, -0.5 * math.pi + angle, count)
        arc = np.column_stack((radius * np.cos(phi), radius + radius * np.sin(phi)))
        points.extend(arc.tolist())
        middle_phi = -0.5 * math.pi + 0.5 * angle
        locus = np.asarray((radius * math.cos(middle_phi), radius + radius * math.sin(middle_phi)), dtype=np.float64)
        start = arc[-1]
    direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
    for distance in np.arange(step, tail + 0.5 * step, step):
        points.append((start + distance * direction).tolist())
    result = np.asarray(points, dtype=np.float64)
    keep = np.concatenate(([True], np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-12))
    return result[keep], locus


def _profile_plane(*, tail: float = 22.0, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(-tail, tail + 0.5 * step, step)
    return np.column_stack((values, np.zeros_like(values))), np.asarray((0.0, 0.0))


def _profile_distributed(*, radius: float = 22.0, angle_degrees: float = 90.0, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(angle_degrees)
    count = int(math.ceil(radius * angle / step)) + 1
    phi = np.linspace(-0.5 * math.pi, -0.5 * math.pi + angle, count)
    points = np.column_stack((radius * np.cos(phi), radius + radius * np.sin(phi)))
    middle_phi = -0.5 * math.pi + 0.5 * angle
    locus = np.asarray((radius * math.cos(middle_phi), radius + radius * math.sin(middle_phi)))
    return points, locus


def _extrude(profile: np.ndarray, *, half_span: float = 40.0, axial_step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    axial = np.arange(-half_span, half_span + 0.5 * axial_step, axial_step)
    vertices = np.asarray([[x, yz[0], yz[1]] for x in axial for yz in profile], dtype=np.float64)
    width = len(profile)
    faces: list[list[int]] = []
    for i in range(len(axial) - 1):
        for j in range(width - 1):
            a = i * width + j
            b = (i + 1) * width + j
            c = (i + 1) * width + j + 1
            d = i * width + j + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
    return vertices, np.asarray(faces, dtype=np.int64)


def _combine(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for part_vertices, part_faces in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        offset += len(part_vertices)
    return np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0)


def _seed(y: float, z: float, *, span: float = 70.0, schema: str = "S") -> dict[str, Any]:
    start = [-0.5 * span, y, z]
    end = [0.5 * span, y, z]
    if schema == "E":
        return {"line_start": start, "line_end": end}
    if schema == "A":
        return {"anchor": [0.0, y, z], "axis_direction": [1.0, 0.0, 0.0], "span_mm": span}
    return {"span_endpoints": [start, end]}


def _run_case(mesh: MeshData, seeds: list[dict[str, Any]], *, v52: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    v52_rows = list(v52 or [])
    v52_result = {
        "predictions": v52_rows,
        "source_only": True,
        "research_only": True,
    }
    return compose_v56(v52_result, {"line_objects": seeds}, mesh, include_visual_audit=True)


def _line_errors(row: dict[str, Any], truth_yz: np.ndarray) -> tuple[float, float]:
    start = np.asarray(row["line_start"], dtype=np.float64)
    end = np.asarray(row["line_end"], dtype=np.float64)
    direction = (end - start) / np.linalg.norm(end - start)
    direction_error = math.degrees(math.acos(min(max(abs(float(direction[0])), -1.0), 1.0)))
    midpoint = 0.5 * (start + end)
    transverse = float(np.linalg.norm(midpoint[1:] - truth_yz))
    return direction_error, transverse


def run_suite() -> dict[str, Any]:
    fixture_results: list[dict[str, Any]] = []

    base_radius = 1.0
    base_profile, base_locus = _profile_fold(90.0, radius=base_radius)
    base_mesh = MeshData.from_arrays(*_extrude(base_profile))
    b, h = base_mesh.bbox_diagonal, base_mesh.median_mesh_edge
    span = 70.0
    e = {"line_start": [-35.0, *base_locus.tolist()], "line_end": [35.0, *base_locus.tolist()]}
    s = {"span_endpoints": [e["line_start"], e["line_end"]]}
    a = {"anchor": [0.0, *base_locus.tolist()], "axis_direction": [1.0, 0.0, 0.0], "span_mm": span}
    canonical = [
        materialize_seed(row, bbox_diagonal=b, median_mesh_edge=h, origin_class="POOL", origin_file_ordinal=1, row_ordinal=index)
        for index, row in enumerate((e, s, a))
    ]
    assert all(np.array_equal(canonical[0].line_start, item.line_start) and np.array_equal(canonical[0].line_end, item.line_end) for item in canonical[1:])
    reversed_seed = materialize_seed(
        {"line_start": e["line_end"], "line_end": e["line_start"]},
        bbox_diagonal=b, median_mesh_edge=h, origin_class="POOL", origin_file_ordinal=1, row_ordinal=3,
    )
    negated_seed = materialize_seed(
        {"anchor": a["anchor"], "axis_direction": [-1.0, 0.0, 0.0], "span_mm": span},
        bbox_diagonal=b, median_mesh_edge=h, origin_class="POOL", origin_file_ordinal=1, row_ordinal=4,
    )
    assert np.array_equal(canonical[0].line_start, reversed_seed.line_start)
    assert np.array_equal(canonical[0].line_start, negated_seed.line_start)
    redundant = {**e, **s, **a}
    materialize_seed(redundant, bbox_diagonal=b, median_mesh_edge=h, origin_class="POOL", origin_file_ordinal=1, row_ordinal=5)
    inconsistent = {**redundant, "anchor": [0.0, float(base_locus[0] + 2.0), float(base_locus[1])]}
    try:
        materialize_seed(inconsistent, bbox_diagonal=b, median_mesh_edge=h, origin_class="POOL", origin_file_ordinal=1, row_ordinal=6)
        raise AssertionError("inconsistent redundant encoding was accepted")
    except SeedSchemaError as error:
        assert error.reason == "INCONSISTENT_REDUNDANT_ANCHOR"
    bad_rows = (
        ({"line_start": [0.0, 0.0, math.nan], "line_end": [1.0, 0.0, 0.0]}, "INVALID_LINE_START"),
        ({"anchor": [0.0, 0.0, 0.0], "axis_direction": [0.0, 0.0, 0.0], "span_mm": 1.0}, "ZERO_OR_NONFINITE_DIRECTION"),
        ({"anchor": [0.0, 0.0, 0.0], "axis_direction": [1.0, 0.0, 0.0], "span_mm": 0.0}, "ZERO_OR_NONFINITE_SPAN"),
        ({"anchor": [0.0, 0.0, 0.0], "axis_direction": [1.0, 0.0, 0.0], "span_mm": -1.0}, "ZERO_OR_NONFINITE_SPAN"),
        ({"span_endpoints": [[0.0, 0.0, 0.0]]}, "MALFORMED_SPAN_ENDPOINTS"),
    )
    for index, (row, reason) in enumerate(bad_rows, start=10):
        try:
            materialize_seed(row, bbox_diagonal=b, median_mesh_edge=h, origin_class="POOL", origin_file_ordinal=1, row_ordinal=index)
            raise AssertionError(f"bad row accepted: {reason}")
        except SeedSchemaError as error:
            assert error.reason == reason
    representatives, multiplicities = collapse_exact_duplicates(canonical, base_mesh.constants)
    assert len(representatives) == 1 and multiplicities == [3]
    fixture_results.append({"name": "schema_contract", "status": "PASS"})

    def assert_case(name: str, mesh: MeshData, seeds: list[dict[str, Any]], expected: int, truths: list[np.ndarray] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        result, visual = _run_case(mesh, seeds)
        assert len(result["predictions"]) == expected, (name, len(result["predictions"]), result["canonical_turn_median_line_reconstruction_audit"]["candidate_rejection_counts"])
        errors = []
        if truths:
            remaining = list(truths)
            for row in result["predictions"]:
                candidates = [(_line_errors(row, truth), index) for index, truth in enumerate(remaining)]
                best, index = min(candidates, key=lambda value: (value[0][1], value[0][0], value[1]))
                direction_error, transverse_error = best
                assert direction_error <= 1.0, (name, direction_error)
                assert transverse_error <= mesh.constants.transverse_center_threshold, (name, transverse_error, mesh.constants.transverse_center_threshold)
                errors.append({"direction_error_degrees": direction_error, "transverse_error": transverse_error})
                remaining.pop(index)
        fixture_results.append({"name": name, "status": "PASS", "output_count": expected, "errors": errors})
        return result, visual

    sharp_profile, sharp_locus = _profile_fold(90.0, radius=0.0)
    sharp_mesh = MeshData.from_arrays(*_extrude(sharp_profile))
    assert_case("centered_sharp_90", sharp_mesh, [_seed(*sharp_locus)], 1, [sharp_locus])

    rounded_profile, rounded_locus = _profile_fold(60.0, radius=2.0)
    rounded_mesh = MeshData.from_arrays(*_extrude(rounded_profile))
    assert_case("centered_rounded_60", rounded_mesh, [_seed(*rounded_locus)], 1, [rounded_locus])

    plane_profile, plane_locus = _profile_plane()
    plane_mesh = MeshData.from_arrays(*_extrude(plane_profile))
    assert_case("single_plane", plane_mesh, [_seed(*plane_locus)], 0)

    offset = 0.40 * base_radius
    offset_seed = _seed(float(base_locus[0] + offset), float(base_locus[1]))
    offset_result, _ = assert_case("offset_recentering", base_mesh, [offset_seed], 1, [base_locus])
    original_midpoint = np.asarray(offset_seed["span_endpoints"], dtype=np.float64).mean(axis=0)
    output_midpoint = np.asarray((offset_result["predictions"][0]["line_start"], offset_result["predictions"][0]["line_end"]), dtype=np.float64).mean(axis=0)
    assert not np.allclose(original_midpoint, output_midpoint, atol=1e-12, rtol=0.0)

    separation = 5.0
    # A 3-4-5 anti-diagonal translation keeps the two equal L-like profiles
    # disjoint, while the midpoint seed remains inside both W tubes and the
    # proposals remain beyond the duplicate distance.
    separation_vector = np.asarray((3.0, -4.0))
    left_vertices, left_faces = _extrude(base_profile - 0.5 * separation_vector)
    right_vertices, right_faces = _extrude(base_profile + 0.5 * separation_vector)
    double_mesh = MeshData.from_arrays(*_combine([(left_vertices, left_faces), (right_vertices, right_faces)]))
    left_locus = base_locus - 0.5 * separation_vector
    right_locus = base_locus + 0.5 * separation_vector
    midpoint_locus = 0.5 * (left_locus + right_locus)
    _, ambiguity_visual = assert_case("equidistant_two_fold_ambiguity", double_mesh, [_seed(*midpoint_locus)], 0)
    assert ambiguity_visual["seeds"][0]["decision"] == "REJECT_MULTIPLE_TURN_EVENTS"
    assert_case("two_seed_two_fold", double_mesh, [_seed(*left_locus), _seed(*right_locus)], 2, [left_locus, right_locus])

    distributed_profile, distributed_locus = _profile_distributed()
    distributed_mesh = MeshData.from_arrays(*_extrude(distributed_profile))
    assert_case("distributed_cylinder", distributed_mesh, [_seed(*distributed_locus)], 0)

    boundary_profile = plane_profile[plane_profile[:, 0] <= 0.0]
    boundary_mesh = MeshData.from_arrays(*_extrude(boundary_profile))
    assert_case("open_boundary_without_fold", boundary_mesh, [_seed(0.0, 0.0)], 0)

    diagonal = {"span_endpoints": [[-35.0, -18.0, 0.0], [35.0, 18.0, 0.0]]}
    assert_case("diagonal_cross_panel", plane_mesh, [diagonal], 0)

    wide_separation = 8.0
    wide_vector = np.asarray((0.6 * wide_separation, -0.8 * wide_separation))
    left_vertices, left_faces = _extrude(base_profile - 0.5 * wide_vector)
    right_vertices, right_faces = _extrude(base_profile + 0.5 * wide_vector)
    parallel_mesh = MeshData.from_arrays(*_combine([(left_vertices, left_faces), (right_vertices, right_faces)]))
    left_locus = base_locus - 0.5 * wide_vector
    right_locus = base_locus + 0.5 * wide_vector
    assert wide_separation >= 0.075 * parallel_mesh.bbox_diagonal
    assert_case("parallel_folds_0p08B", parallel_mesh, [_seed(*left_locus), _seed(*right_locus)], 2, [left_locus, right_locus])

    hold_row = {"bend_id": "V52-HOLD", "line_start": [-35.0, *base_locus.tolist()], "line_end": [35.0, *base_locus.tolist()]}
    hold_result, _ = _run_case(base_mesh, [], v52=[hold_row])
    assert hold_result["predictions"] == [hold_row]
    fixture_results.append({"name": "byte_preserving_v52_hold", "status": "PASS", "output_count": 1})

    replay_first, _ = _run_case(base_mesh, [_seed(*base_locus)])
    replay_second, _ = _run_case(base_mesh, [_seed(*base_locus)])
    for payload in (replay_first, replay_second):
        audit = payload["canonical_turn_median_line_reconstruction_audit"]
        audit.pop("wall_ms", None)
        audit.pop("peak_rss_bytes", None)
    assert _canonical(replay_first) == _canonical(replay_second)
    fixture_results.append({"name": "canonical_primary_replay_identity", "status": "PASS"})

    try:
        _canonical({"line_start": [0.0, math.nan, 0.0]})
        raise AssertionError("nonfinite output serialization was accepted")
    except ValueError:
        pass
    fixture_results.append({"name": "nonfinite_output_rejected", "status": "PASS"})

    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v56_source_free_preflight",
        "profile": PROFILE,
        "precommit_token": PRECOMMIT_TOKEN,
        "implementation_sha256": {
            "backend": _sha(ROOT / "backend/f1_canonical_turn_median_line_reconstruction_v56.py"),
            "source_free_preflight": _sha(Path(__file__).resolve()),
        },
        "fixture_count": len(fixture_results),
        "fixtures": fixture_results,
        "status": "PASS",
    }
    first = _canonical(payload)
    second = _canonical(json.loads(first))
    assert first == second
    payload["canonical_sha256"] = hashlib.sha256(first).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run_suite()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(result))
    print(json.dumps({"status": result["status"], "fixture_count": result["fixture_count"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
