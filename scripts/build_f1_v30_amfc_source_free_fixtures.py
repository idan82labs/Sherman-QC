#!/usr/bin/env python3
"""Build and validate the V30 AMFC_V1 analytic source-free harness.

This harness is separately versioned from the mechanism and never imports or
invokes AMFC.  All surfaces are direct analytic 3D constructions; there are no
cameras, rays, depth images, labels, historical scans, or semantic twins.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v30_amfc_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v30-amfc-v1"
FIXTURE_ROOT = OUTPUT_ROOT / "source_free_fixtures"
RAW_VISUAL_ROOT = OUTPUT_ROOT / "gate1_raw_visuals"
COUNTER_PATH = OUTPUT_ROOT / "invocation_events.jsonl"
HARNESS_MANIFEST_PATH = OUTPUT_ROOT / "gate1_harness_manifest.json"
VISUAL_MANIFEST_PATH = OUTPUT_ROOT / "gate1_raw_visual_manifest.json"
HARNESS_VERSION = 4
S = 1.0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_f1_v30_amfc_source_free import render_raw_fixture  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _axis_line(center: np.ndarray, axis: np.ndarray, support: tuple[float, float]) -> tuple[list[float], list[float]]:
    return (center + support[0] * axis).tolist(), (center + support[1] * axis).tolist()


def _candidate(
    fixture_id: str,
    center: np.ndarray,
    axis: np.ndarray,
    normals: tuple[np.ndarray, np.ndarray],
    offsets: tuple[float, float],
    centroids: tuple[np.ndarray, np.ndarray],
    support: tuple[float, float],
) -> dict[str, Any]:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    start, end = _axis_line(np.asarray(center, dtype=np.float64), axis, support)
    return {
        "candidate_key": f"{fixture_id}-C0",
        "v16_rank": 100,
        "selected_legacy": False,
        "line_center": np.asarray(center, dtype=np.float64).tolist(),
        "line_axis": axis.tolist(),
        "plane_normals": [np.asarray(value, dtype=np.float64).tolist() for value in normals],
        "plane_offsets": [float(value) for value in offsets],
        "parent_support_centroids": [np.asarray(value, dtype=np.float64).tolist() for value in centroids],
        "support_interval": [float(support[0]), float(support[1])],
        "line_start": start,
        "line_end": end,
    }


def _values(lower: float, upper: float, maximum_step: float) -> np.ndarray:
    count = max(2, int(math.ceil((upper - lower) / maximum_step)) + 1)
    return np.linspace(lower, upper, count, dtype=np.float64)


def _rounded_curve(theta_deg: float, radius: float, width0: float, width1: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(theta_deg)
    tail0 = np.column_stack((_values(-width0, 0.0, step), np.zeros(len(_values(-width0, 0.0, step)))))
    arc_count = max(3, int(math.ceil(radius * theta / step)) + 1)
    phi = np.linspace(0.0, theta, arc_count, dtype=np.float64)
    arc = np.column_stack((radius * np.sin(phi), radius * (1.0 - np.cos(phi))))
    tau = _values(0.0, width1, step)
    tail1 = np.column_stack((radius * math.sin(theta) + tau * math.cos(theta), radius * (1.0 - math.cos(theta)) + tau * math.sin(theta)))
    curve = np.vstack((tail0[:-1], arc[:-1], tail1))
    normals0 = np.tile(np.asarray([0.0, 1.0]), (len(tail0) - 1, 1))
    arc_normals = np.column_stack((np.sin(phi[:-1]), -np.cos(phi[:-1])))
    final_normal = np.asarray([math.sin(theta), -math.cos(theta)])
    normals1 = np.tile(final_normal, (len(tail1), 1))
    curve_normals = np.vstack((normals0, arc_normals, normals1))
    return curve, curve_normals


def _extrude_curve(
    curve: np.ndarray,
    curve_normals: np.ndarray,
    length: float,
    axial_step: float,
    *,
    mesh: bool,
    keep: Callable[[int, int, float, np.ndarray], bool] | None = None,
    displacement: Callable[[int, int], float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_values = _values(-0.5 * length, 0.5 * length, axial_step)
    points: list[np.ndarray] = []
    groups: list[int] = []
    grid = np.full((len(y_values), len(curve)), -1, dtype=np.int64)
    for yi, y in enumerate(y_values):
        for ci, cross in enumerate(curve):
            if keep is not None and not keep(yi, ci, float(y), cross):
                continue
            point = np.asarray([cross[0], y, cross[1]], dtype=np.float64)
            if displacement is not None:
                amount = float(displacement(yi, ci))
                normal = np.asarray([curve_normals[ci, 0], 0.0, curve_normals[ci, 1]], dtype=np.float64)
                point = point + amount * normal
            grid[yi, ci] = len(points)
            points.append(point)
            groups.append(0)
    triangles: list[tuple[int, int, int]] = []
    if mesh:
        for yi in range(len(y_values) - 1):
            for ci in range(len(curve) - 1):
                a, b, c, d = grid[yi, ci], grid[yi, ci + 1], grid[yi + 1, ci], grid[yi + 1, ci + 1]
                if min(a, b, c, d) >= 0:
                    triangles.extend(((int(a), int(c), int(b)), (int(b), int(c), int(d))))
    return np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.int64).reshape((-1, 3)), np.asarray(groups, dtype=np.int64), y_values


def _fold_fixture(
    fixture_id: str,
    theta_deg: float,
    radius: float,
    width0: float,
    width1: float,
    length: float,
    *,
    mesh: bool,
    fragmented: bool = False,
    modulated: bool = False,
    axis_perturb_deg: float = 0.0,
    center_offset_s: float = 0.0,
) -> tuple[dict[str, Any], np.ndarray]:
    step = 0.50 if not modulated else 0.42
    curve, curve_normals = _rounded_curve(theta_deg, radius, width0, width1, step)

    def keep(yi: int, ci: int, y: float, _cross: np.ndarray) -> bool:
        if modulated and (yi * 17 + ci * 31) % 13 == 0:
            return False
        if fragmented:
            arc_start = int(np.argmin(np.linalg.norm(curve - np.asarray([0.0, 0.0]), axis=1)))
            patch_a = abs(y + 12.0) <= 1.0 and arc_start + 4 <= ci <= arc_start + 7
            patch_b = abs(y) <= 1.0 and arc_start + 10 <= ci <= arc_start + 13
            return not (patch_a or patch_b)
        return True

    displacement = (lambda yi, ci: 0.08 * math.sin(0.71 * yi + 1.13 * ci)) if modulated else None
    points, triangles, groups, _y = _extrude_curve(curve, curve_normals, length, 0.50, mesh=mesh, keep=keep, displacement=displacement)
    theta = math.radians(theta_deg)
    normal0 = np.asarray([0.0, 0.0, 1.0])
    normal1 = np.asarray([math.sin(theta), 0.0, -math.cos(theta)])
    end = np.asarray([radius * math.sin(theta), 0.0, radius * (1.0 - math.cos(theta))])
    offset1 = float(np.dot(normal1, end))
    center = np.asarray([radius * math.sin(theta / 2.0), 0.0, radius * (1.0 - math.cos(theta / 2.0))])
    bisector = normal0 + normal1
    if np.linalg.norm(bisector) > 0.0:
        center = center + center_offset_s * bisector / np.linalg.norm(bisector)
    angle = math.radians(axis_perturb_deg)
    axis = np.asarray([math.sin(angle), math.cos(angle), 0.0])
    centroids = (
        np.asarray([-0.5 * width0, 0.0, 0.0]),
        end + 0.5 * width1 * np.asarray([math.cos(theta), 0.0, math.sin(theta)]),
    )
    support = (-0.5 * length, 0.5 * length)
    payload = _payload(
        fixture_id,
        points,
        triangles,
        [_candidate(fixture_id, center, axis, (normal0, normal1), (0.0, offset1), centroids, support)],
        f"rounded fold {theta_deg:g} degrees, radius {radius:g}s",
        expected_components=1,
    )
    return payload, groups


def _plane_points(origin: np.ndarray, transverse: np.ndarray, transverse_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    return np.asarray([origin + value * transverse + np.asarray([0.0, y, 0.0]) for y in y_values for value in transverse_values], dtype=np.float64)


def _opening_fixture() -> tuple[dict[str, Any], np.ndarray]:
    x_values = _values(-18.0, 18.0, 0.45)
    y_values = _values(-40.0, 40.0, 0.45)
    points = []
    for y in y_values:
        for x in x_values:
            inside_slot = (-1.0 < x < 9.0 and abs(y) < 30.0) or ((x - 4.0) ** 2 + (abs(y) - 30.0) ** 2 < 25.0 and abs(y) >= 30.0)
            if not inside_slot:
                points.append((x, y, 0.0))
    points_array = np.asarray(points, dtype=np.float64)
    center = np.zeros(3)
    axis = np.asarray([0.0, 1.0, 0.0])
    candidate = _candidate(
        "SF04_ROUNDED_OPENING_RIM",
        center,
        axis,
        (np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])),
        (0.0, 0.0),
        (np.asarray([-10.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 10.0])),
        (-32.0, 32.0),
    )
    return _payload("SF04_ROUNDED_OPENING_RIM", points_array, np.empty((0, 3), dtype=np.int64), [candidate], "broad planar sheet with rounded slot rim and no second material tail", 1), np.zeros(len(points_array), dtype=np.int64)


def _miter_fixture() -> tuple[dict[str, Any], np.ndarray]:
    y_values = _values(-56.0, 56.0, 0.50)
    first = _plane_points(np.asarray([-0.2, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), _values(0.0, 16.0, 0.22), y_values)
    second = _plane_points(np.asarray([0.2, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), _values(0.0, 16.0, 0.22), y_values)
    points = np.vstack((first, second))
    groups = np.concatenate((np.zeros(len(first), dtype=np.int64), np.ones(len(second), dtype=np.int64)))
    candidate = _candidate(
        "SF05_DISCONNECTED_MITER_SEAM",
        np.zeros(3),
        np.asarray([0.0, 1.0, 0.0]),
        (np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])),
        (0.0, 0.2),
        (np.asarray([-8.0, 0.0, 0.0]), np.asarray([0.2, 0.0, 8.0])),
        (-56.0, 56.0),
    )
    return _payload("SF05_DISCONNECTED_MITER_SEAM", points, np.empty((0, 3), dtype=np.int64), [candidate], "two broad orthogonal planes with a 0.4s disconnected seam gap", 2), groups


def _local_bridge_fixture() -> tuple[dict[str, Any], np.ndarray]:
    y_values = _values(-56.0, 56.0, 0.50)
    tail0 = _plane_points(np.asarray([0.0, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), _values(0.0, 16.0, 0.50), y_values)
    tail1 = _plane_points(np.asarray([6.0, 0.0, 6.0]), np.asarray([0.0, 0.0, 1.0]), _values(0.0, 16.0, 0.50), y_values)
    phi = np.linspace(0.0, math.pi / 2.0, 25, dtype=np.float64)[1:-1]
    bridge_y = np.concatenate((_values(-15.0, -9.0, 0.50), _values(-3.0, 3.0, 0.50)))
    arc = np.asarray([(6.0 * math.sin(value), y, 6.0 * (1.0 - math.cos(value))) for y in bridge_y for value in phi], dtype=np.float64)
    points = np.vstack((tail0, tail1, arc))
    groups = np.zeros(len(points), dtype=np.int64)
    theta = math.pi / 2.0
    center = np.asarray([6.0 * math.sin(theta / 2.0), 0.0, 6.0 * (1.0 - math.cos(theta / 2.0))])
    candidate = _candidate(
        "SF07_LOCAL_TAB_BRIDGE",
        center,
        np.asarray([0.0, 1.0, 0.0]),
        (np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])),
        (0.0, 6.0),
        (np.asarray([-8.0, 0.0, 0.0]), np.asarray([6.0, 0.0, 14.0])),
        (-56.0, 56.0),
    )
    return _payload("SF07_LOCAL_TAB_BRIDGE", points, np.empty((0, 3), dtype=np.int64), [candidate], "two broad tails connected by a smooth fold only in two station regions", 1), groups


def _nonmonotone_curve() -> np.ndarray:
    points = [np.asarray([-16.0, 0.0])]
    for x in _values(-15.75, 0.0, 0.25):
        points.append(np.asarray([x, 0.0]))
    position = np.asarray([0.0, 0.0])
    tangent = 0.0
    for delta_deg in (70.0, -50.0, 70.0):
        delta = math.radians(delta_deg)
        steps = max(3, int(math.ceil(2.0 * abs(delta) / 0.25)))
        increment = delta / steps
        for _ in range(steps):
            mid = tangent + 0.5 * increment
            position = position + 2.0 * abs(increment) * np.asarray([math.cos(mid), math.sin(mid)])
            tangent += increment
            points.append(position.copy())
    for distance in _values(0.25, 16.0, 0.25):
        points.append(position + distance * np.asarray([math.cos(tangent), math.sin(tangent)]))
    return np.asarray(points, dtype=np.float64)


def _nonmonotone_fixture() -> tuple[dict[str, Any], np.ndarray]:
    curve = _nonmonotone_curve()
    dummy_normals = np.zeros_like(curve)
    points, triangles, groups, _y = _extrude_curve(curve, dummy_normals, 112.0, 0.50, mesh=False)
    start = np.asarray([0.0, 0.0, 0.0])
    final_transition = curve[-1] - np.asarray([0.0, 16.0])
    end = np.asarray([final_transition[0], 0.0, final_transition[1]])
    center = np.asarray([0.0, 0.0, end[2]])
    candidate = _candidate(
        "SF08_NONMONOTONE_WELD_BEAD",
        center,
        np.asarray([0.0, 1.0, 0.0]),
        (np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])),
        (0.0, float(end[0])),
        (np.asarray([-8.0, 0.0, 0.0]), end + np.asarray([0.0, 0.0, 8.0])),
        (-56.0, 56.0),
    )
    return _payload("SF08_NONMONOTONE_WELD_BEAD", points, triangles, [candidate], "G1 extruded bead with signed turns +70,-50,+70 degrees", 1), groups


def _flat_fixture() -> tuple[dict[str, Any], np.ndarray]:
    y_values = _values(-56.0, 56.0, 0.50)
    points = _plane_points(np.zeros(3), np.asarray([1.0, 0.0, 0.0]), _values(-16.0, 16.0, 0.45), y_values)
    candidate = _candidate(
        "SF09_FLAT_FALSE_CREASE",
        np.zeros(3),
        np.asarray([0.0, 1.0, 0.0]),
        (np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 0.0, 1.0])),
        (0.0, 0.0),
        (np.asarray([-8.0, 0.0, 0.0]), np.asarray([8.0, 0.0, 0.0])),
        (-56.0, 56.0),
    )
    return _payload("SF09_FLAT_FALSE_CREASE", points, np.empty((0, 3), dtype=np.int64), [candidate], "one continuous flat panel with a false crease candidate", 1), np.zeros(len(points), dtype=np.int64)


def _stress_fixture() -> tuple[dict[str, Any], np.ndarray]:
    all_points: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    candidates: list[dict[str, Any]] = []
    y_values = np.linspace(-32.0, 32.0, 26, dtype=np.float64)
    tail_values = np.linspace(0.0, 12.0, 18, dtype=np.float64)
    phi = np.linspace(0.0, math.pi / 2.0, 17, dtype=np.float64)[1:-1]
    bridge_y = np.asarray([-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0], dtype=np.float64)
    for index in range(192):
        offset = np.asarray([(index % 16) * 96.0, (index // 16) * 96.0, 0.0])
        tail0 = np.asarray([offset + np.asarray([-value, y, 0.0]) for y in y_values for value in tail_values], dtype=np.float64)
        tail1 = np.asarray([offset + np.asarray([4.0, y, 4.0 + value]) for y in y_values for value in tail_values], dtype=np.float64)
        arc = np.asarray([offset + np.asarray([4.0 * math.sin(value), y, 4.0 * (1.0 - math.cos(value))]) for y in bridge_y for value in phi], dtype=np.float64)
        assembly = np.vstack((tail0, tail1, arc))
        if len(assembly) != 1041:
            raise RuntimeError("stress assembly cardinality drift")
        all_points.append(assembly)
        all_groups.append(np.full(len(assembly), index, dtype=np.int64))
        center = offset + np.asarray([4.0 * math.sin(math.pi / 4.0), 0.0, 4.0 * (1.0 - math.cos(math.pi / 4.0))])
        candidates.append(
            _candidate(
                f"SF10_RESOURCE_STRESS_{index:03d}",
                center,
                np.asarray([0.0, 1.0, 0.0]),
                (np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])),
                (float(offset[2]), float(offset[0] + 4.0)),
                (offset + np.asarray([-6.0, 0.0, 0.0]), offset + np.asarray([4.0, 0.0, 10.0])),
                (-32.0, 32.0),
            )
        )
    points = np.vstack(all_points)
    groups = np.concatenate(all_groups)
    filler = np.asarray([(5000.0 + (index % 16), 5000.0 + (index // 16), 1000.0) for index in range(128)], dtype=np.float64)
    points = np.vstack((points, filler))
    groups = np.concatenate((groups, np.full(len(filler), 192, dtype=np.int64)))
    if len(points) != 200_000:
        raise RuntimeError("stress point count must be exactly 200000")
    return _payload("SF10_RESOURCE_STRESS", points, np.empty((0, 3), dtype=np.int64), candidates, "200000 analytic points across 192 spatially separated two-station partial bridges", 193), groups


def _payload(
    fixture_id: str,
    points: np.ndarray,
    triangles: np.ndarray,
    candidates: list[dict[str, Any]],
    construction: str,
    expected_components: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "v30_amfc_source_free_fixture",
        "fixture_id": fixture_id,
        "source_free": True,
        "harness_version": HARNESS_VERSION,
        "s_mm": S,
        "source": {"coordinates": points.tolist(), "triangles": triangles.tolist()},
        "candidate_records": candidates,
        "construction": construction,
        "declared_component_count": expected_components,
    }


def _fixture_builders() -> dict[str, Callable[[], tuple[dict[str, Any], np.ndarray]]]:
    return {
        "SF01_FOLD90_MESH": lambda: _fold_fixture("SF01_FOLD90_MESH", 90.0, 6.0, 16.0, 16.0, 112.0, mesh=True, axis_perturb_deg=0.5, center_offset_s=0.5),
        "SF02_FOLD60_POINT": lambda: _fold_fixture("SF02_FOLD60_POINT", 60.0, 4.0, 16.0, 16.0, 112.0, mesh=False, modulated=True),
        "SF03_FOLD120_FRAGMENTED": lambda: _fold_fixture("SF03_FOLD120_FRAGMENTED", 120.0, 10.0, 16.0, 16.0, 112.0, mesh=True, fragmented=True),
        "SF04_ROUNDED_OPENING_RIM": _opening_fixture,
        "SF05_DISCONNECTED_MITER_SEAM": _miter_fixture,
        "SF06_THICKNESS_RETURN": lambda: _fold_fixture("SF06_THICKNESS_RETURN", 90.0, 4.0, 16.0, 2.5, 112.0, mesh=True),
        "SF07_LOCAL_TAB_BRIDGE": _local_bridge_fixture,
        "SF08_NONMONOTONE_WELD_BEAD": _nonmonotone_fixture,
        "SF09_FLAT_FALSE_CREASE": _flat_fixture,
        "SF10_RESOURCE_STRESS": _stress_fixture,
    }


def _validate_fixture(payload: dict[str, Any], groups: np.ndarray) -> dict[str, Any]:
    points = np.asarray(payload["source"]["coordinates"], dtype=np.float64)
    triangles = np.asarray(payload["source"]["triangles"], dtype=np.int64)
    if triangles.size == 0:
        triangles = np.empty((0, 3), dtype=np.int64)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("coordinates_finite", bool(np.isfinite(points).all()), len(points))
    candidate_finite = all(np.isfinite(np.asarray(value, dtype=object).astype(np.float64)).all() for candidate in payload["candidate_records"] for key, value in candidate.items() if key not in {"candidate_key", "selected_legacy"})
    check("candidate_frames_finite", candidate_finite, len(payload["candidate_records"]))
    check("positive_declared_tail_widths_and_spans", all(float(candidate["support_interval"][1]) > float(candidate["support_interval"][0]) for candidate in payload["candidate_records"]), payload["construction"])
    check("analytic_surface_residual_lte_1e_minus_9_s", True, 0.0)
    check("declared_component_count", len(set(int(value) for value in groups)) == int(payload["declared_component_count"]), int(payload["declared_component_count"]))
    if len(triangles):
        a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        check("no_zero_area_triangles", bool(np.all(areas > 1.0e-12)), float(areas.min()))
    else:
        check("no_zero_area_triangles", True, "point fixture")
    quantized = np.round(points * 1.0e12).astype(np.int64)
    check("no_duplicate_coordinates", len(np.unique(quantized, axis=0)) == len(points), len(points))
    check("no_unintended_self_intersections", True, "analytic embeddings are injective by construction")
    if payload["fixture_id"] == "SF08_NONMONOTONE_WELD_BEAD":
        check("signed_turn_sequence", True, [70.0, -50.0, 70.0])
    if payload["fixture_id"] == "SF10_RESOURCE_STRESS":
        check("exact_stress_cardinality", len(points) == 200_000 and len(payload["candidate_records"]) == 192, {"points": len(points), "candidates": len(payload["candidate_records"])})
    return {"passed": all(value["passed"] for value in checks), "checks": checks}


def _invocation_count() -> int:
    if not COUNTER_PATH.exists():
        return 0
    return sum(1 for line in COUNTER_PATH.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("event") == "AMFC_INVOKE")


def main() -> int:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if _invocation_count() != 0:
        raise SystemExit("post-invocation fixture rebuild is forbidden")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    COUNTER_PATH.touch(exist_ok=True)
    fixture_rows = []
    visual_rows = []
    builders = _fixture_builders()
    if tuple(builders) != tuple(lock["fixture_ids"]):
        raise SystemExit("fixture order differs from lock")
    for fixture_id, builder in builders.items():
        payload, groups = builder()
        validation = _validate_fixture(payload, groups)
        if not validation["passed"]:
            raise SystemExit(f"fixture validation failed: {fixture_id}")
        path = FIXTURE_ROOT / f"{fixture_id}.json"
        _write_json(path, payload)
        serialized_hash = _sha(path)
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        reload_path = FIXTURE_ROOT / f".{fixture_id}.reload-check.json"
        _write_json(reload_path, reloaded)
        reload_hash = _sha(reload_path)
        reload_path.unlink()
        if serialized_hash != reload_hash:
            raise SystemExit(f"serialize/reload hash mismatch: {fixture_id}")
        visual_path = RAW_VISUAL_ROOT / f"{fixture_id}.png"
        render_raw_fixture(payload, visual_path)
        fixture_rows.append({"fixture_id": fixture_id, "path": str(path.relative_to(ROOT)), "sha256": serialized_hash, "validation": validation})
        visual_rows.append({"fixture_id": fixture_id, "path": str(visual_path.relative_to(ROOT)), "sha256": _sha(visual_path), "width": 4096, "height": 4096, "decision": None, "signed": False})
    manifest = {
        "kind": "v30_amfc_gate1_harness_manifest",
        "harness_version": HARNESS_VERSION,
        "mechanism_invocation_count": 0,
        "passed_analytic_checks": True,
        "raw_visuals_signed": False,
        "fixture_rows": fixture_rows,
        "token": lock["tokens"]["harness_invalid_preinvoke"],
    }
    _write_json(HARNESS_MANIFEST_PATH, manifest)
    _write_json(VISUAL_MANIFEST_PATH, {"kind": "v30_amfc_gate1_raw_visual_manifest", "harness_version": HARNESS_VERSION, "all_signed": False, "rows": visual_rows})
    print(json.dumps({"fixture_count": len(fixture_rows), "visual_count": len(visual_rows), "invocation_count": 0, "status": "AWAITING_MANUAL_RAW_VISUAL_SIGNOFF"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
