#!/usr/bin/env python3
"""EXP_CAD_0028 NTRC-V1 source-only six-bend pairwise measurement.

The scientific contract is frozen in ``EXP_CAD_0028_PREREGISTRATION_V1.json``.
This module deliberately has no evaluator, label, expected-count, filename-role,
or quality-mapping input.  It exposes the independently auditable topology,
five-parameter guard-only nuisance fit, signed normal-turn statistic, pair gate,
and exact robustness-row construction.  Real source execution remains locked
until a separate preflight binding records the immutable runner and test hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXP17_PATH = ROOT / "scripts" / "run_exp_cad_0017_native_roundbend_patch_axes.py"
EXP25_PATH = ROOT / "scripts" / "run_exp_cad_0025_analytic_brep_carrier_orientation.py"
EXP27_PATH = ROOT / "scripts" / "run_exp_cad_0027_fixed_frame_differential_bend_conformance.py"


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EXP17 = _import(EXP17_PATH, "exp28_exp17")
EXP25 = _import(EXP25_PATH, "exp28_exp25")
EXP27 = _import(EXP27_PATH, "exp28_exp27")
EXP21 = EXP25.EXP21

EXPERIMENT_ID = "EXP_CAD_0028"
ARCHITECTURE = "NORMAL_TURN_RIGID_CONSENSUS_V1"
EXPECTED_CASE_TOKENS = ("N009", "N010")
EXPECTED_PREREGISTRATION_SHA256 = (
    "10b96fa48b05bed6848ce837839c945cc02054c91c8c83691f44ba50bf0d7b84"
)
EXPECTED_IMPORTED_SHA256 = {
    "EXP17": "be2db1631d692482a5ff87ce6e1560aa294066da64fb65afe195bd75918bd005",
    "EXP25": "09c28f30d8653b00fd77639245718e1b81243dbd4ab0bc8378a5ec0daa295cc7",
    "EXP27": "2751c57e7bcf2a42cd3d1dcccbbfdde43aa189cea5e1183c93fd1f7f7c1fc658",
}

EXPECTED_WARNING_CODES = (
    "config.incoherent-body-refs",
    "config.incomplete-snapshot",
    "config.inferred-without-native",
    "feature.native-kind-retained",
    "feature.typed-operand-incomplete",
    "tessellation.face-ownership-unresolved",
)

AXIS_CROSS_MAX = 1e-8
COAXIAL_DISTANCE_MM = 0.02
RADIUS_TOLERANCE_MM = 0.002
ENDPOINT_TOLERANCE_MM = 0.05
THICKNESS_TOLERANCE_MM = 0.02
EXPECTED_RADII_MM = (1.0, 2.5)
EXPECTED_EVENT_COUNT = 6
EXPECTED_ARC_DEG = 90.0
EXPECTED_ARC_TOLERANCE_DEG = 1e-6
AXIAL_BIN_COUNT = 8
ANGULAR_BIN_COUNT = 8
MIN_POINTS_PER_CELL = 3
MIN_OBSERVED_AXIAL_BINS = 6
MIN_CONTIGUOUS_AXIAL_BINS = 4
MAX_RADIAL_NORMAL_ERROR_DEG = 25.0
MIN_RADIAL_SIGN_CONSISTENCY = 0.95
GUARD_MIN_POINTS = 64
GUARD_MIN_AXIAL_BINS = 6
FIT_ITERATIONS = 3
FIT_CONDITION_MAX = 1e4
SATURATION_FRACTION = 0.98
PAIR_MARGIN_FLOOR_DEG = 0.25
PAIR_RATIO_FLOOR_DEG = 0.25
PAIR_RATIO_MIN = 1.25
CASE_SKIN_SIGN_NEUTRAL_DEG = 0.25
MIN_DIRECTIONAL_EVENTS = 2
VOXEL_FRACTIONS = (0.1, 0.125, 0.15)
PARTITIONS = ("FULL", "RAW_EVEN", "RAW_ODD", "SPATIAL_PARITY_0", "SPATIAL_PARITY_1")


def sha256_path(path: Path) -> str:
    return EXP21.sha256_path(path)


def normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return value / length


def canonical_unoriented_axis(axis: np.ndarray) -> np.ndarray:
    value = normalized(axis)
    for component in value:
        if abs(float(component)) <= 1e-12:
            continue
        return -value if component < 0.0 else value
    raise ValueError("axis has no canonical nonzero component")


def canonical_axis_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic right-handed (axis, u, v)."""

    a = canonical_unoriented_axis(axis)
    candidates = np.eye(3)
    seed = candidates[int(np.argmin(np.abs(candidates @ a)))]
    u = normalized(seed - float(np.dot(seed, a)) * a)
    v = normalized(np.cross(a, u))
    return a, u, v


def _xyz(value: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([value[name] for name in "xyz"], dtype=np.float64)


def _index(model: Mapping[str, Any], names: Sequence[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        rows = model.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"missing CADIR collection {name}")
        indexed = {str(row["id"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"duplicate CADIR identifier in {name}")
        result[name] = indexed
    return result


def _custom_property(cadir: Mapping[str, Any], name: str) -> Any:
    return EXP17.V1._custom_property(cadir, name)


def _active_shell_face_ids(model: Mapping[str, Any]) -> tuple[str, str, str, set[str]]:
    idx = _index(model, ("bodies", "regions", "shells", "faces"))
    active = [row for row in model.get("configurations", []) if row.get("active") is True]
    if len(active) != 1 or len(active[0].get("bodies", [])) != 1:
        raise ValueError("selector requires exactly one active configuration and solid body")
    body_id = str(active[0]["bodies"][0])
    body = idx["bodies"].get(body_id)
    if body is None:
        raise ValueError("active configuration body reference is unresolved")
    region_ids = [str(value) for value in body.get("regions", [])]
    if not region_ids:
        raise ValueError("active body has no regions")
    shell_owners: list[tuple[str, str]] = []
    for region_id in region_ids:
        region = idx["regions"].get(region_id)
        if region is None or str(region.get("body")) != body_id:
            raise ValueError("active region ownership is unresolved")
        shell_owners.extend((region_id, str(value)) for value in region.get("shells", []))
    shell_ids = [shell_id for _region_id, shell_id in shell_owners]
    if len(shell_ids) != 1:
        raise ValueError("selector requires exactly one shell total; zero-shell regions are allowed")
    owning_region_id = shell_owners[0][0]
    shell = idx["shells"].get(shell_ids[0])
    if shell is None or str(shell.get("region")) != owning_region_id:
        raise ValueError("active shell ownership is unresolved")
    faces = {str(value) for value in shell.get("faces", [])}
    if not faces or any(value not in idx["faces"] for value in faces):
        raise ValueError("active shell has incomplete face ownership")
    return body_id, owning_region_id, shell_ids[0], faces


def _edge_endpoints(edge: Mapping[str, Any], idx: Mapping[str, Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for field in ("start", "end"):
        vertex = idx["vertices"].get(str(edge.get(field)))
        if vertex is None:
            raise ValueError("edge endpoint vertex is unresolved")
        point = idx["points"].get(str(vertex.get("point")))
        if point is None:
            raise ValueError("edge endpoint point is unresolved")
        values.append(_xyz(point["position"]))
    return values[0], values[1]


def _selected_cylinder_face(
    model: Mapping[str, Any], face_id: str, idx: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    face = idx["faces"][face_id]
    loops = [str(value) for value in face.get("loops", [])]
    if len(loops) != 1:
        raise ValueError(f"selected cylinder {face_id} must have exactly one loop")
    loop = idx["loops"].get(loops[0])
    if loop is None or str(loop.get("face")) != face_id:
        raise ValueError(f"selected cylinder {face_id} loop ownership is unresolved")
    surface = idx["surfaces"].get(str(face.get("surface")))
    if surface is None or surface.get("geometry", {}).get("kind") != "cylinder":
        raise ValueError(f"selected face {face_id} is not an analytic cylinder")
    geometry = surface["geometry"]
    axis = canonical_unoriented_axis(_xyz(geometry["axis"]))
    origin = _xyz(geometry["origin"])
    foot = origin - float(np.dot(origin, axis)) * axis
    line_trims: list[dict[str, Any]] = []
    circle_trims: list[dict[str, Any]] = []
    boundary_points: list[np.ndarray] = []
    for coedge_id in loop.get("coedges", []):
        coedge = idx["coedges"].get(str(coedge_id))
        if coedge is None or str(coedge.get("owner_loop")) != loops[0]:
            raise ValueError(f"selected cylinder {face_id} coedge ownership is unresolved")
        edge = idx["edges"].get(str(coedge.get("edge")))
        if edge is None:
            raise ValueError(f"selected cylinder {face_id} edge is unresolved")
        curve = idx["curves"].get(str(edge.get("curve")))
        if curve is None:
            raise ValueError(f"selected cylinder {face_id} curve is unresolved")
        first, second = _edge_endpoints(edge, idx)
        boundary_points.extend((first, second))
        kind = curve.get("geometry", {}).get("kind")
        if kind == "line":
            edge_axis = normalized(second - first)
            if float(np.linalg.norm(np.cross(edge_axis, axis))) > AXIS_CROSS_MAX:
                raise ValueError(f"selected cylinder {face_id} line trim is not axial")
            radial = idx["coedges"].get(str(coedge.get("radial_next")))
            adjacent_loop = idx["loops"].get(str(radial.get("owner_loop"))) if radial else None
            adjacent_face = idx["faces"].get(str(adjacent_loop.get("face"))) if adjacent_loop else None
            adjacent_surface = idx["surfaces"].get(str(adjacent_face.get("surface"))) if adjacent_face else None
            if adjacent_surface is None or adjacent_surface.get("geometry", {}).get("kind") != "plane":
                raise ValueError(f"selected cylinder {face_id} axial trim lacks planar radial-next owner")
            line_trims.append({
                "edge_id": str(edge["id"]),
                "adjacent_face_id": str(adjacent_face["id"]),
                "start_mm": first,
                "end_mm": second,
            })
        elif kind == "circle":
            circle = curve["geometry"]
            circle_axis = canonical_unoriented_axis(_xyz(circle["axis"]))
            circle_center = _xyz(circle["center"])
            circle_radius = float(circle["radius"])
            center_radial = circle_center - foot
            center_radial -= float(np.dot(center_radial, axis)) * axis
            if (
                float(np.linalg.norm(np.cross(circle_axis, axis))) > AXIS_CROSS_MAX
                or abs(circle_radius - float(geometry["radius"])) > 1e-6
                or float(np.linalg.norm(center_radial)) > 1e-6
            ):
                raise ValueError(f"selected cylinder {face_id} circular end trim is not coaxial")
            circle_trims.append({
                "edge_id": str(edge["id"]),
                "start_mm": first,
                "end_mm": second,
                "center_mm": circle_center,
            })
        else:
            raise ValueError(f"selected cylinder {face_id} has unsupported trim kind {kind}")
    if len(line_trims) != 2 or len(circle_trims) != 2:
        raise ValueError(f"selected cylinder {face_id} must have two line and two circular trims")
    if len({row["edge_id"] for row in line_trims + circle_trims}) != 4:
        raise ValueError(f"selected cylinder {face_id} trim edges are not distinct")
    radial_units = []
    for row in line_trims:
        midpoint = (row["start_mm"] + row["end_mm"]) / 2.0
        radial = midpoint - foot
        radial -= float(np.dot(radial, axis)) * axis
        row["radial_unit"] = normalized(radial)
        radial_units.append(row["radial_unit"])
    line_trims.sort(key=lambda row: tuple(np.round(row["radial_unit"], 12)))
    radial_units = [row["radial_unit"] for row in line_trims]
    minor_arc_deg = math.degrees(
        math.acos(float(np.clip(np.dot(radial_units[0], radial_units[1]), -1.0, 1.0)))
    )
    if abs(minor_arc_deg - EXPECTED_ARC_DEG) > EXPECTED_ARC_TOLERANCE_DEG:
        raise ValueError(f"selected cylinder {face_id} minor arc is not 90 degrees")
    parameters = np.asarray(boundary_points, dtype=np.float64) @ axis
    centre_parameters = sorted(float(np.dot(row["center_mm"], axis)) for row in circle_trims)
    if not np.allclose(
        centre_parameters,
        [float(np.min(parameters)), float(np.max(parameters))],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"selected cylinder {face_id} circular trims are not its finite axial ends")
    return {
        "face_id": face_id,
        "surface_id": str(surface["id"]),
        "axis": axis,
        "foot": foot,
        "radius_mm": float(geometry["radius"]),
        "t_min_mm": float(np.min(parameters)),
        "t_max_mm": float(np.max(parameters)),
        "line_trims": line_trims,
        "circle_trims": circle_trims,
        "guard_radial_units": radial_units,
        "minor_arc_deg": minor_arc_deg,
    }


def _native_parameter_token(feature: Mapping[str, Any], name: str) -> Optional[str]:
    candidates: list[Any] = [feature.get(name)]
    for container_name in ("parameters", "dimensions"):
        container = feature.get(container_name)
        if isinstance(container, Mapping):
            candidates.append(container.get(name))
        elif isinstance(container, list):
            candidates.extend(
                row.get("value")
                for row in container
                if isinstance(row, Mapping) and str(row.get("name")) == name
            )
    for value in candidates:
        if isinstance(value, Mapping):
            value = value.get("value")
        if value is not None:
            return str(value).strip()
    return None


_D1_RADIUS = re.compile(r"^R([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")
_D4_RAD = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)rad$")


def validate_roundbend_native_parameters(feature: Mapping[str, Any]) -> dict[str, Any]:
    d1 = _native_parameter_token(feature, "D1")
    d4 = _native_parameter_token(feature, "D4")
    match = _D1_RADIUS.fullmatch(d1 or "")
    if match is None or not math.isfinite(float(match.group(1))):
        raise ValueError("RoundBend D1 is not an explicit finite native radius token")
    d4_value = None
    if d4 is not None:
        match4 = _D4_RAD.fullmatch(d4)
        if match4 is None or not math.isfinite(float(match4.group(1))):
            raise ValueError("RoundBend D4 is present without an explicit finite rad suffix")
        d4_value = float(match4.group(1))
    return {"D1_token": d1, "D1_radius": float(match.group(1)), "D4_token": d4, "D4_rad": d4_value}


def select_geometry_consensus_events(cadir: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed six-feature/six-independent-cylinder-pair selector."""

    model = cadir.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("CADIR model is missing")
    body_id, region_id, shell_id, shell_faces = _active_shell_face_ids(model)
    arenas = cadir.get("native", {}).get("sldprt", {}).get("arenas", {})
    features = [
        row for row in arenas.get("features", [])
        if isinstance(row, Mapping)
        and row.get("kind") == "RoundBend"
        and row.get("suppressed") is False
    ]
    if len(features) != EXPECTED_EVENT_COUNT:
        raise ValueError("active unsuppressed RoundBend count is not six")
    names = [str(row.get("name")) for row in features]
    if len(set(names)) != EXPECTED_EVENT_COUNT:
        raise ValueError("active RoundBend names are not unique")
    feature_parameter_audit = [validate_roundbend_native_parameters(row) for row in features]
    if any(abs(row["D1_radius"] - 1.0) > RADIUS_TOLERANCE_MM for row in feature_parameter_audit):
        raise ValueError("active RoundBend D1 radius is not 1 mm within 0.002 mm")
    stale_bends = int(_custom_property(cadir, "Bends"))
    if stale_bends != 0:
        raise ValueError("42637000_B Bends property no longer has frozen stale value zero")
    thickness = float(_custom_property(cadir, "Sheet Metal Thickness"))
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("sheet thickness is invalid")

    idx = _index(
        model,
        ("faces", "loops", "coedges", "edges", "vertices", "points", "surfaces", "curves"),
    )
    cylinders = []
    for face_id in sorted(shell_faces):
        face = idx["faces"][face_id]
        surface = idx["surfaces"].get(str(face.get("surface")))
        if surface is None or surface.get("geometry", {}).get("kind") != "cylinder":
            continue
        radius = float(surface["geometry"]["radius"])
        if min(abs(radius - expected) for expected in EXPECTED_RADII_MM) > RADIUS_TOLERANCE_MM:
            continue
        cylinders.append(_selected_cylinder_face(model, face_id, idx))
    if len(cylinders) != 2 * EXPECTED_EVENT_COUNT:
        raise ValueError("active shell does not contain exactly twelve eligible 1/2.5 mm cylinders")

    candidates: list[tuple[int, int, float, float]] = []
    for first_index, first in enumerate(cylinders):
        for second_index in range(first_index + 1, len(cylinders)):
            second = cylinders[second_index]
            axis_cross = float(np.linalg.norm(np.cross(first["axis"], second["axis"])))
            line_distance = float(np.linalg.norm(first["foot"] - second["foot"]))
            radii = sorted((float(first["radius_mm"]), float(second["radius_mm"])))
            radius_levels_match = all(
                abs(value - expected) <= RADIUS_TOLERANCE_MM
                for value, expected in zip(radii, EXPECTED_RADII_MM)
            )
            gap_error = abs((radii[1] - radii[0]) - thickness)
            endpoint_error = max(
                abs(float(first["t_min_mm"]) - float(second["t_min_mm"])),
                abs(float(first["t_max_mm"]) - float(second["t_max_mm"])),
            )
            if (
                axis_cross <= AXIS_CROSS_MAX
                and line_distance <= COAXIAL_DISTANCE_MM
                and radius_levels_match
                and gap_error <= THICKNESS_TOLERANCE_MM
                and endpoint_error <= ENDPOINT_TOLERANCE_MM
            ):
                candidates.append((first_index, second_index, gap_error, endpoint_error))
    degrees = {index: 0 for index in range(len(cylinders))}
    for first, second, _gap, _endpoint in candidates:
        degrees[first] += 1
        degrees[second] += 1
    involved = {index for pair in candidates for index in pair[:2]}
    if len(candidates) != EXPECTED_EVENT_COUNT or len(involved) != 2 * EXPECTED_EVENT_COUNT:
        raise ValueError("geometry did not yield exactly six independent cylinder pairs")
    if any(degrees[index] != 1 for index in involved):
        raise ValueError("cylinder-pair geometry is ambiguous or reuses a surface")

    raw_events = []
    for first_index, second_index, gap_error, endpoint_error in candidates:
        skins = sorted((cylinders[first_index], cylinders[second_index]), key=lambda row: row["radius_mm"])
        axis = canonical_unoriented_axis(skins[0]["axis"])
        for first_radial, second_radial in zip(
            skins[0]["guard_radial_units"], skins[1]["guard_radial_units"]
        ):
            if float(np.linalg.norm(first_radial - second_radial)) > 1e-6:
                raise ValueError("inner/outer skin guard radial endpoints disagree")
        foot = (skins[0]["foot"] + skins[1]["foot"]) / 2.0
        t_min = (skins[0]["t_min_mm"] + skins[1]["t_min_mm"]) / 2.0
        t_max = (skins[0]["t_max_mm"] + skins[1]["t_max_mm"]) / 2.0
        start = foot + axis * t_min
        end = foot + axis * t_max
        raw_events.append({
            "axis": axis,
            "axis_foot_mm": foot,
            "line_start_mm": start,
            "line_end_mm": end,
            "span_mm": float(t_max - t_min),
            "skins": skins,
            "canonical_guard_radial_units": [
                normalized(first + second)
                for first, second in zip(skins[0]["guard_radial_units"], skins[1]["guard_radial_units"])
            ],
            "minor_arc_deg": 90.0,
            "radius_gap_error_mm": gap_error,
            "endpoint_error_mm": endpoint_error,
        })
    raw_events.sort(key=lambda row: (
        tuple(np.round(row["axis_foot_mm"], 8)),
        tuple(np.round(row["line_start_mm"], 8)),
        tuple(np.round(row["line_end_mm"], 8)),
        tuple(skin["surface_id"] for skin in row["skins"]),
    ))
    events = []
    for ordinal, row in enumerate(raw_events, 1):
        events.append({
            **row,
            "event_id": f"NTRC_EVENT_{ordinal:02d}",
            "sheet_thickness_mm": thickness,
        })
    return {
        "selector": "SIX_ACTIVE_ROUNDBENDS_AND_SIX_INDEPENDENT_UNIQUE_COAXIAL_CYLINDER_PAIRS_V1",
        "proposal_status": "GEOMETRY_CONSENSUS_PROPOSAL_ONLY",
        "active_body_id": body_id,
        "active_region_id": region_id,
        "active_shell_id": shell_id,
        "active_roundbend_names": names,
        "active_roundbend_native_parameter_audit": feature_parameter_audit,
        "active_roundbend_count": len(features),
        "stale_bends_property_audit_only": stale_bends,
        "sheet_thickness_mm": thickness,
        "event_count": len(events),
        "events": events,
    }


def _face_neighbors(face_id: str, idx: Mapping[str, Mapping[str, Any]]) -> set[str]:
    neighbors: set[str] = set()
    face = idx["faces"][face_id]
    for loop_id in face.get("loops", []):
        loop = idx["loops"].get(str(loop_id))
        if loop is None or str(loop.get("face")) != face_id:
            raise ValueError(f"face/loop ownership is incomplete for {face_id}")
        for coedge_id in loop.get("coedges", []):
            coedge = idx["coedges"].get(str(coedge_id))
            radial = idx["coedges"].get(str(coedge.get("radial_next"))) if coedge else None
            radial_loop = idx["loops"].get(str(radial.get("owner_loop"))) if radial else None
            if radial_loop is not None:
                neighbor = str(radial_loop.get("face"))
                if neighbor != face_id:
                    neighbors.add(neighbor)
    return neighbors


def _plane(face_id: str, idx: Mapping[str, Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    face = idx["faces"].get(face_id)
    surface = idx["surfaces"].get(str(face.get("surface"))) if face else None
    if surface is None or surface.get("geometry", {}).get("kind") != "plane":
        raise ValueError(f"face {face_id} is not an analytic plane")
    geometry = surface["geometry"]
    return _xyz(geometry["origin"]), normalized(_xyz(geometry["normal"]))


def _face_edges(face_id: str, idx: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    face = idx["faces"][face_id]
    for loop_id in face.get("loops", []):
        loop = idx["loops"].get(str(loop_id))
        if loop is None or str(loop.get("face")) != face_id:
            raise ValueError(f"face/loop ownership is incomplete for {face_id}")
        for coedge_id in loop.get("coedges", []):
            coedge = idx["coedges"].get(str(coedge_id))
            edge = idx["edges"].get(str(coedge.get("edge"))) if coedge else None
            if edge is None:
                raise ValueError(f"face edge is unresolved for {face_id}")
            first, second = _edge_endpoints(edge, idx)
            radial = idx["coedges"].get(str(coedge.get("radial_next")))
            radial_loop = idx["loops"].get(str(radial.get("owner_loop"))) if radial else None
            rows.append({
                "edge_id": str(edge["id"]),
                "start_mm": first,
                "end_mm": second,
                "neighbor_face_id": str(radial_loop.get("face")) if radial_loop else None,
            })
    return rows


def reconstruct_connected_coplanar_panel(
    model: Mapping[str, Any],
    *,
    seed_face_id: str,
    active_shell_faces: set[str],
    seam_edge_id: str,
    seam_start_mm: np.ndarray,
    seam_end_mm: np.ndarray,
    event_axis: np.ndarray,
    seam_radial_unit: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct one finite topology-connected coplanar tangent panel."""

    idx = _index(model, ("faces", "loops", "coedges", "edges", "vertices", "points", "surfaces"))
    if seed_face_id not in active_shell_faces:
        raise ValueError("tangent guard seed is outside the unique active shell")
    origin, normal = _plane(seed_face_id, idx)
    if float(np.dot(normal, seam_radial_unit)) < 0.0:
        normal = -normal
    component: set[str] = set()
    frontier = [seed_face_id]
    while frontier:
        face_id = frontier.pop()
        if face_id in component:
            continue
        candidate_origin, candidate_normal = _plane(face_id, idx)
        if (
            float(np.linalg.norm(np.cross(candidate_normal, normal))) > 1e-8
            or abs(float(np.dot(candidate_origin - origin, normal))) > 1e-6
        ):
            continue
        component.add(face_id)
        for neighbor in sorted(_face_neighbors(face_id, idx)):
            if neighbor not in component and neighbor in active_shell_faces:
                try:
                    neighbor_origin, neighbor_normal = _plane(neighbor, idx)
                except ValueError:
                    continue
                if (
                    float(np.linalg.norm(np.cross(neighbor_normal, normal))) <= 1e-8
                    and abs(float(np.dot(neighbor_origin - origin, normal))) <= 1e-6
                ):
                    frontier.append(neighbor)
    if not component:
        raise ValueError("connected coplanar tangent panel is empty")

    boundary_segments = []
    all_vertices = []
    for face_id in sorted(component):
        for edge in _face_edges(face_id, idx):
            all_vertices.extend((edge["start_mm"], edge["end_mm"]))
            if edge["neighbor_face_id"] not in component:
                boundary_segments.append(edge)
    if seam_edge_id not in {row["edge_id"] for row in boundary_segments}:
        raise ValueError("cylinder seam is not a finite boundary of its tangent panel")
    non_event_segments = [row for row in boundary_segments if row["edge_id"] != seam_edge_id]
    if not non_event_segments:
        raise ValueError("tangent panel has no non-event finite trims")
    axis = canonical_unoriented_axis(event_axis)
    seam_midpoint = (np.asarray(seam_start_mm) + np.asarray(seam_end_mm)) / 2.0
    vertices = np.asarray(all_vertices, dtype=np.float64)
    transverse = vertices - seam_midpoint
    transverse -= (transverse @ axis)[:, None] * axis
    transverse -= (transverse @ normal)[:, None] * normal
    candidate = np.mean(transverse, axis=0)
    if float(np.linalg.norm(candidate)) <= 1e-8:
        candidate = np.cross(normal, axis)
    interior = normalized(candidate)
    coordinates = transverse @ interior
    if float(np.max(coordinates)) < float(-np.min(coordinates)):
        interior = -interior
        coordinates = -coordinates
    return {
        "seed_face_id": seed_face_id,
        "component_face_ids": sorted(component),
        "plane_origin_mm": origin,
        "plane_normal": normal,
        "seam_edge_id": seam_edge_id,
        "seam_midpoint_mm": seam_midpoint,
        "seam_start_mm": np.asarray(seam_start_mm, dtype=np.float64),
        "seam_end_mm": np.asarray(seam_end_mm, dtype=np.float64),
        "interior_direction": interior,
        "interior_extent_mm": float(np.max(coordinates)),
        "non_event_boundary_segments": non_event_segments,
    }


def build_event_panels(
    model: Mapping[str, Any], event: Mapping[str, Any], active_shell_faces: set[str]
) -> list[dict[str, Any]]:
    panels = []
    for skin_index, skin in enumerate(event["skins"]):
        for guard_index, trim in enumerate(skin["line_trims"]):
            panel = reconstruct_connected_coplanar_panel(
                model,
                seed_face_id=str(trim["adjacent_face_id"]),
                active_shell_faces=active_shell_faces,
                seam_edge_id=str(trim["edge_id"]),
                seam_start_mm=np.asarray(trim["start_mm"]),
                seam_end_mm=np.asarray(trim["end_mm"]),
                event_axis=np.asarray(event["axis"]),
                seam_radial_unit=np.asarray(trim["radial_unit"]),
            )
            panels.append({**panel, "skin_index": skin_index, "guard_index": guard_index})
    if len(panels) != 4 or len({row["seed_face_id"] for row in panels}) != 4:
        raise ValueError("event does not expose four unique topology-owned tangent panels")
    return panels


def cad_guard_actionability_preflight(
    extraction: Mapping[str, Any],
    panels_by_event: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """CAD-only feasibility gate; it must run before either physical scan is loaded."""

    events = list(extraction["events"])
    if len(events) != EXPECTED_EVENT_COUNT or len(panels_by_event) != EXPECTED_EVENT_COUNT:
        raise ValueError("CAD guard preflight requires all six geometry-consensus events")
    rows = []
    for event, panels in zip(events, panels_by_event):
        if len(panels) != 4:
            raise ValueError("CAD guard preflight requires four panels per event")
        thickness = float(event["sheet_thickness_mm"])
        minimum_extent = 2.0 * thickness
        for panel in panels:
            extent = float(panel["interior_extent_mm"])
            actionable = bool(extent + 1e-12 >= minimum_extent)
            rows.append({
                "event_id": str(event["event_id"]),
                "skin": "INNER" if int(panel["skin_index"]) == 0 else "OUTER",
                "guard_index": int(panel["guard_index"]),
                "seed_face_id": str(panel["seed_face_id"]),
                "connected_component_face_ids": list(panel["component_face_ids"]),
                "interior_extent_mm": extent,
                "minimum_extent_mm": minimum_extent,
                "actionable": actionable,
                "failure_reason": None if actionable else "INSUFFICIENT_FINITE_PANEL_FOR_1T_INWARD_PLUS_1T_TRIM_EROSION",
            })
    unavailable = [row for row in rows if not row["actionable"]]
    return {
        "schema": "EXP_CAD_0028_NTRC_CAD_GUARD_ACTIONABILITY_PREFLIGHT_V1",
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "proposal_status": extraction["proposal_status"],
        "event_count": len(events),
        "required_guard_count": len(rows),
        "actionable_guard_count": len(rows) - len(unavailable),
        "unavailable_guard_count": len(unavailable),
        "all_four_guards_per_event_available": len(unavailable) == 0,
        "status": "PASS" if not unavailable else "TERMINAL_FOUR_FINITE_GUARDS_UNAVAILABLE",
        "source_scans_loaded": False,
        "evaluator_roles_opened": False,
        "rows": rows,
    }


def _point_segment_distances(points: np.ndarray, segments: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not segments:
        return np.full(len(points), math.inf)
    minimum = np.full(len(points), math.inf, dtype=np.float64)
    for segment in segments:
        first = np.asarray(segment["start_mm"], dtype=np.float64)
        second = np.asarray(segment["end_mm"], dtype=np.float64)
        delta = second - first
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-16:
            distance = np.linalg.norm(points - first, axis=1)
        else:
            parameter = np.clip(((points - first) @ delta) / denominator, 0.0, 1.0)
            closest = first + parameter[:, None] * delta
            distance = np.linalg.norm(points - closest, axis=1)
        minimum = np.minimum(minimum, distance)
    return minimum


def event_endpoint_erosion_mm(event: Mapping[str, Any]) -> float:
    thickness = float(event["sheet_thickness_mm"])
    span = float(event["span_mm"])
    erosion = max(3.0 * thickness, 0.10 * span)
    if span - 2.0 * erosion < 8.0 * thickness:
        raise ValueError("event has insufficient interior span after frozen endpoint erosion")
    return erosion


def sample_four_guard_supports(
    points: np.ndarray,
    normals: np.ndarray,
    event: Mapping[str, Any],
    panels: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select fixed, finite, eroded, uniquely owned support for all four guards."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    thickness = float(event["sheet_thickness_mm"])
    axis = canonical_unoriented_axis(np.asarray(event["axis"]))
    start = np.asarray(event["line_start_mm"], dtype=np.float64)
    span = float(event["span_mm"])
    erosion = event_endpoint_erosion_mm(event)
    axial = (points - start) @ axis
    plane_residuals = np.column_stack(
        [np.abs((points - np.asarray(panel["plane_origin_mm"])) @ np.asarray(panel["plane_normal"])) for panel in panels]
    )
    supports = []
    audits = []
    for panel_index, panel in enumerate(panels):
        residual = plane_residuals[:, panel_index]
        alternatives = np.min(np.delete(plane_residuals, panel_index, axis=1), axis=1)
        inward = (points - np.asarray(panel["seam_midpoint_mm"])) @ np.asarray(panel["interior_direction"])
        normal_dot = np.abs(normals @ np.asarray(panel["plane_normal"]))
        normal_error = np.degrees(np.arccos(np.clip(normal_dot, 0.0, 1.0)))
        base = (
            (residual <= 0.5 * thickness)
            & ((alternatives - residual) >= 0.25 * thickness)
            & (axial >= erosion)
            & (axial <= span - erosion)
            & (inward >= 1.0 * thickness)
            & (inward <= 6.0 * thickness)
            & (inward <= float(panel["interior_extent_mm"]) - 1.0 * thickness)
            & (normal_error <= 25.0)
        )
        ids = np.flatnonzero(base)
        trim_distance = _point_segment_distances(points[ids], panel["non_event_boundary_segments"])
        ids = ids[trim_distance >= 1.0 * thickness]
        bins = np.floor((axial[ids] - erosion) / (span - 2.0 * erosion) * AXIAL_BIN_COUNT).astype(np.int64)
        bins = np.clip(bins, 0, AXIAL_BIN_COUNT - 1)
        counts = np.bincount(bins, minlength=AXIAL_BIN_COUNT)
        median_error = float(np.median(normal_error[ids])) if len(ids) else None
        actionable = bool(
            len(ids) >= GUARD_MIN_POINTS
            and np.count_nonzero(counts > 0) >= GUARD_MIN_AXIAL_BINS
            and median_error is not None
            and median_error <= 25.0
        )
        audits.append({
            "panel_index": panel_index,
            "point_count": int(len(ids)),
            "axial_bin_counts": counts.tolist(),
            "median_normal_error_deg": median_error,
            "actionable": actionable,
            "selected_point_ids": ids,
        })
        supports.append({
            "points_mm": points[ids],
            "plane_normal": np.asarray(panel["plane_normal"]),
            "plane_origin_mm": np.asarray(panel["plane_origin_mm"]),
            "axial_bin_counts": counts,
        })
    return supports, audits


def canonical_voxel_aggregate(
    points: np.ndarray, normals: np.ndarray, voxel_mm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid + normalized summed-unit-normal, in canonical voxel order."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points/normals must be matching Nx3 arrays")
    if not math.isfinite(voxel_mm) or voxel_mm <= 0.0:
        raise ValueError("voxel size must be positive")
    lengths = np.linalg.norm(normals, axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & np.isfinite(lengths)
        & (lengths > 1e-12)
    )
    raw_ids = np.flatnonzero(valid).astype(np.int64)
    points = points[valid]
    normals = normals[valid] / lengths[valid, None]
    if not len(points):
        return points, normals, raw_ids
    keys = np.floor(points / voxel_mm).astype(np.int64)
    order = np.lexsort((raw_ids, keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    points = points[order]
    normals = normals[order]
    raw_ids = raw_ids[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1))]
    stops = np.r_[starts[1:], len(keys)]
    out_points = []
    out_normals = []
    representative_ids = []
    for start, stop in zip(starts, stops):
        vector = np.sum(normals[start:stop], axis=0)
        length = float(np.linalg.norm(vector))
        if not math.isfinite(length) or length <= 1e-12:
            continue
        out_points.append(np.mean(points[start:stop], axis=0))
        out_normals.append(vector / length)
        representative_ids.append(int(np.min(raw_ids[start:stop])))
    return (
        np.asarray(out_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(out_normals, dtype=np.float64).reshape(-1, 3),
        np.asarray(representative_ids, dtype=np.int64),
    )


def partition_mask(
    aligned_points: np.ndarray,
    original_raw_ids: np.ndarray,
    partition: str,
    *,
    body_bbox_min_mm: np.ndarray,
    thickness_mm: float,
) -> np.ndarray:
    points = np.asarray(aligned_points, dtype=np.float64)
    raw_ids = np.asarray(original_raw_ids, dtype=np.int64)
    if len(points) != len(raw_ids):
        raise ValueError("raw identity count mismatch")
    if partition == "FULL":
        return np.ones(len(points), dtype=bool)
    if partition == "RAW_EVEN":
        return raw_ids % 2 == 0
    if partition == "RAW_ODD":
        return raw_ids % 2 == 1
    if partition in ("SPATIAL_PARITY_0", "SPATIAL_PARITY_1"):
        cell = thickness_mm / 8.0
        keys = np.floor((points - np.asarray(body_bbox_min_mm)) / cell).astype(np.int64)
        parity = np.mod(np.sum(keys, axis=1), 2)
        return parity == int(partition[-1])
    raise ValueError(f"unknown partition {partition}")


def rodrigues(rotation_vector_rad: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector_rad, dtype=np.float64)
    theta = float(np.linalg.norm(vector))
    if theta <= 1e-15:
        return np.eye(3)
    axis = vector / theta
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(theta) * cross + (1.0 - math.cos(theta)) * (cross @ cross)


def apply_event_transform(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    center_mm: np.ndarray,
    translation_mm: np.ndarray,
    rotation_vector_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = rodrigues(rotation_vector_rad)
    center = np.asarray(center_mm, dtype=np.float64)
    transformed_points = (np.asarray(points) - center) @ rotation.T + center + translation_mm
    transformed_normals = np.asarray(normals) @ rotation.T
    return transformed_points, transformed_normals


def solve_guard_only_nuisance(
    guard_samples: Sequence[Mapping[str, Any]],
    *,
    event_axis: np.ndarray,
    event_center_mm: np.ndarray,
    thickness_mm: float,
) -> dict[str, Any]:
    """Fit [tau_u,tau_v,omega_axis,omega_u,omega_v] with equal guard mass."""

    if len(guard_samples) != 4:
        raise ValueError("nuisance fit requires exactly four topology-owned guards")
    axis, u, v = canonical_axis_frame(event_axis)
    center = np.asarray(event_center_mm, dtype=np.float64)
    translation_bound = thickness_mm / 8.0
    rotation_bound = math.radians(0.25)
    parameter_bounds = np.asarray(
        [translation_bound, translation_bound, rotation_bound, rotation_bound, rotation_bound]
    )
    frozen = []
    for guard in guard_samples:
        points = np.asarray(guard["points_mm"], dtype=np.float64)
        if len(points) < GUARD_MIN_POINTS:
            return {"status": "ABSTAIN", "reason": "GUARD_POINT_COUNT", "saturated": False}
        axial_counts = np.asarray(guard.get("axial_bin_counts", np.ones(AXIAL_BIN_COUNT)), dtype=np.int64)
        if axial_counts.shape != (AXIAL_BIN_COUNT,) or np.count_nonzero(axial_counts > 0) < GUARD_MIN_AXIAL_BINS:
            return {"status": "ABSTAIN", "reason": "GUARD_AXIAL_OCCUPANCY", "saturated": False}
        frozen.append({
            "points_mm": points.copy(),
            "plane_normal": normalized(np.asarray(guard["plane_normal"], dtype=np.float64)),
            "plane_origin_mm": np.asarray(guard["plane_origin_mm"], dtype=np.float64),
        })

    def system(current: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """Fixed-membership residual/Jacobian; membership is never reselected."""

        translation = current[0] * u + current[1] * v
        rotation_vector = current[2] * axis + current[3] * u + current[4] * v
        all_design: list[np.ndarray] = []
        all_residual: list[np.ndarray] = []
        all_base_weight: list[np.ndarray] = []
        counts: list[int] = []
        for guard in frozen:
            points = guard["points_mm"]
            normal = guard["plane_normal"]
            origin = guard["plane_origin_mm"]
            moved, _ = apply_event_transform(
                points,
                np.broadcast_to(normal, points.shape),
                center_mm=center,
                translation_mm=translation,
                rotation_vector_rad=rotation_vector,
            )
            residual = (moved - origin) @ normal
            lever = moved - center
            rotational = np.column_stack(
                (
                    np.cross(np.broadcast_to(axis, lever.shape), lever) @ normal,
                    np.cross(np.broadcast_to(u, lever.shape), lever) @ normal,
                    np.cross(np.broadcast_to(v, lever.shape), lever) @ normal,
                )
            )
            design = np.column_stack(
                (
                    np.full(len(points), float(np.dot(normal, u))),
                    np.full(len(points), float(np.dot(normal, v))),
                    rotational,
                )
            )
            all_design.append(design)
            all_residual.append(residual)
            # Equal total mass per guard regardless of support density.
            all_base_weight.append(np.full(len(points), 0.25 / len(points)))
            counts.append(len(points))
        return np.vstack(all_design), np.concatenate(all_residual), np.concatenate(all_base_weight), counts

    def huber_objective(residual: np.ndarray, base_weight: np.ndarray) -> float:
        delta = thickness_mm / 8.0
        absolute = np.abs(residual)
        loss = np.where(absolute <= delta, 0.5 * residual ** 2, delta * (absolute - 0.5 * delta))
        return float(np.sum(base_weight * loss))

    parameter = np.zeros(5, dtype=np.float64)
    iteration_rows = []
    for iteration in range(FIT_ITERATIONS):
        design, residual, base_weight, counts = system(parameter)
        scale = thickness_mm / 8.0
        huber = np.ones(len(residual), dtype=np.float64)
        outside = np.abs(residual) > scale
        huber[outside] = scale / np.abs(residual[outside])
        weights = np.empty(len(residual), dtype=np.float64)
        offset = 0
        for count in counts:
            local = base_weight[offset:offset + count] * huber[offset:offset + count]
            total = float(np.sum(local))
            if total <= 0.0 or not math.isfinite(total):
                return {"status": "ABSTAIN", "reason": "GUARD_HUBER_WEIGHT_ZERO", "saturated": False}
            weights[offset:offset + count] = 0.25 * local / total
            offset += count
        normalized_design = design * parameter_bounds[None, :]
        weighted_design = normalized_design * np.sqrt(weights[:, None])
        weighted_rhs = -residual * np.sqrt(weights)
        singular = np.linalg.svd(weighted_design, compute_uv=False)
        rank_tolerance = 1e-8 * singular[0] if len(singular) else math.inf
        rank = int(np.count_nonzero(singular > rank_tolerance))
        condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else math.inf
        if rank != 5 or not math.isfinite(condition) or condition > FIT_CONDITION_MAX:
            return {
                "status": "ABSTAIN",
                "reason": "NORMALIZED_DESIGN_RANK_OR_CONDITION",
                "rank": rank,
                "condition": condition if math.isfinite(condition) else None,
                "saturated": False,
            }
        normalized_step, *_ = np.linalg.lstsq(weighted_design, weighted_rhs, rcond=None)
        step = normalized_step * parameter_bounds
        candidate = parameter + step
        _candidate_design, candidate_residual, candidate_base_weight, _ = system(candidate)
        objective_before = huber_objective(residual, base_weight)
        objective_after = huber_objective(candidate_residual, candidate_base_weight)
        if objective_after > objective_before * (1.0 + 1e-12):
            return {
                "status": "ABSTAIN",
                "reason": "HUBER_OBJECTIVE_INCREASE",
                "objective_before": objective_before,
                "objective_after": objective_after,
                "saturated": False,
            }
        parameter = candidate
        fraction = np.abs(parameter / parameter_bounds)
        iteration_rows.append({
            "iteration": iteration + 1,
            "guard_point_counts": counts,
            "guard_weight_totals": [0.25, 0.25, 0.25, 0.25],
            "rank": rank,
            "condition": condition,
            "rank_tolerance": rank_tolerance,
            "maximum_abs_normalized_parameter": float(np.max(fraction)),
            "huber_objective_before": objective_before,
            "huber_objective_after": objective_after,
            "weighted_residual_rms_mm": float(np.sqrt(np.average(residual ** 2, weights=weights))),
        })
    normalized_parameter = parameter / parameter_bounds
    saturated = bool(np.any(np.abs(normalized_parameter) >= SATURATION_FRACTION))
    translation = parameter[0] * u + parameter[1] * v
    rotation_vector = parameter[2] * axis + parameter[3] * u + parameter[4] * v
    return {
        "status": "ABSTAIN" if saturated else "FIT",
        "reason": "PARAMETER_SATURATION" if saturated else None,
        "saturated": saturated,
        "parameter_order": ["TAU_U", "TAU_V", "OMEGA_AXIS", "OMEGA_U", "OMEGA_V"],
        "parameter": parameter.tolist(),
        "normalized_parameter": normalized_parameter.tolist(),
        "translation_mm": translation.tolist(),
        "axial_translation_mm_exact": float(np.dot(translation, axis)),
        "rotation_vector_rad": rotation_vector.tolist(),
        "rotation_vector_deg": np.degrees(rotation_vector).tolist(),
        "iterations": iteration_rows,
    }


def longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def longest_true_run_bounds(mask: np.ndarray) -> tuple[int, int]:
    """Non-wrapping longest run; ties resolve to the lowest start index."""

    best_start = best_stop = 0
    current_start: Optional[int] = None
    for index, value in enumerate(np.r_[np.asarray(mask, dtype=bool), False]):
        if value and current_start is None:
            current_start = index
        elif not value and current_start is not None:
            if index - current_start > best_stop - best_start:
                best_start, best_stop = current_start, index
            current_start = None
    return best_start, best_stop


def signed_azimuth(
    vector: np.ndarray, first_radial: np.ndarray, second_radial: np.ndarray, axis: np.ndarray
) -> float:
    axis = normalized(axis)
    first = normalized(first_radial - float(np.dot(first_radial, axis)) * axis)
    second = normalized(second_radial - float(np.dot(second_radial, axis)) * axis)
    direction = 1.0 if float(np.dot(axis, np.cross(first, second))) >= 0.0 else -1.0
    tangent = normalized(direction * np.cross(axis, first))
    projected = np.asarray(vector, dtype=np.float64)
    projected = normalized(projected - float(np.dot(projected, axis)) * axis)
    return float(math.atan2(float(np.dot(projected, tangent)), float(np.dot(projected, first))))


def skin_turn_statistic(
    cell_normals: np.ndarray,
    cell_counts: np.ndarray,
    *,
    cad_radial_cell_vectors: np.ndarray,
    axis: np.ndarray,
) -> dict[str, Any]:
    """Compute cylinder-normal turn E and Q with one global skin flip."""

    normals = np.asarray(cell_normals, dtype=np.float64)
    counts = np.asarray(cell_counts, dtype=np.int64)
    cad_radial = np.asarray(cad_radial_cell_vectors, dtype=np.float64)
    if normals.shape != (AXIAL_BIN_COUNT, ANGULAR_BIN_COUNT, 3) or counts.shape != (
        AXIAL_BIN_COUNT, ANGULAR_BIN_COUNT
    ):
        raise ValueError("skin cell grid shape mismatch")
    if cad_radial.shape != (ANGULAR_BIN_COUNT, 3):
        raise ValueError("CAD radial cell vector shape mismatch")
    axis = normalized(axis)
    cad_radial = np.asarray([normalized(row) for row in cad_radial])
    lengths = np.linalg.norm(normals, axis=2)
    supported_cells = (
        (counts >= MIN_POINTS_PER_CELL)
        & np.isfinite(normals).all(axis=2)
        & np.isfinite(lengths)
        & (lengths > 1e-12)
    )
    unit = np.zeros_like(normals)
    unit[supported_cells] = normals[supported_cells] / lengths[supported_cells, None]
    dot_grid = np.sum(unit * cad_radial[None, :, :], axis=2)
    supported_dots = dot_grid[supported_cells]
    median_raw_dot = float(np.median(supported_dots)) if len(supported_dots) else 0.0
    if median_raw_dot == 0.0 or not math.isfinite(median_raw_dot):
        global_flip = 0
    else:
        global_flip = 1 if median_raw_dot > 0.0 else -1
    oriented_full = global_flip * unit
    oriented_dot_grid = global_flip * dot_grid
    projected = oriented_full - np.sum(oriented_full * axis[None, None, :], axis=2)[:, :, None] * axis
    projected_lengths = np.linalg.norm(projected, axis=2)
    turn_supported_cells = supported_cells & np.isfinite(projected_lengths) & (projected_lengths > 1e-12)
    oriented = np.zeros_like(projected)
    oriented[turn_supported_cells] = projected[turn_supported_cells] / projected_lengths[turn_supported_cells, None]
    sign_consistency = (
        float(np.mean(oriented_dot_grid[supported_cells] > 0.0)) if np.any(supported_cells) else 0.0
    )
    median_normal_error = (
        float(np.median(np.degrees(np.arccos(np.clip(oriented_dot_grid[supported_cells], -1.0, 1.0)))))
        if np.any(supported_cells)
        else None
    )
    eligible = np.zeros(AXIAL_BIN_COUNT, dtype=bool)
    errors = np.full(AXIAL_BIN_COUNT, np.nan, dtype=np.float64)
    for axial_index in range(AXIAL_BIN_COUNT):
        observed_angular = turn_supported_cells[axial_index]
        start, stop = longest_true_run_bounds(observed_angular)
        if int(np.count_nonzero(observed_angular)) < 6 or stop - start < 4:
            continue
        observed_turn = 0.0
        cad_turn = 0.0
        for angular_index in range(start, stop - 1):
            first = oriented[axial_index, angular_index]
            second = oriented[axial_index, angular_index + 1]
            observed_raw = math.atan2(
                float(np.dot(axis, np.cross(first, second))),
                float(np.dot(first, second)),
            )
            cad_first = cad_radial[angular_index]
            cad_second = cad_radial[angular_index + 1]
            cad_raw = math.atan2(
                float(np.dot(axis, np.cross(cad_first, cad_second))),
                float(np.dot(cad_first, cad_second)),
            )
            if abs(cad_raw) <= 1e-12:
                eligible[axial_index] = False
                observed_turn = math.nan
                break
            cad_direction = 1.0 if cad_raw > 0.0 else -1.0
            observed_turn += cad_direction * observed_raw
            cad_turn += abs(cad_raw)
        if not math.isfinite(observed_turn):
            continue
        difference = observed_turn - cad_turn
        errors[axial_index] = math.degrees(math.atan2(math.sin(difference), math.cos(difference)))
        eligible[axial_index] = True
    observed_count = int(np.count_nonzero(eligible))
    contiguous = longest_true_run(eligible)
    actionable = bool(
        global_flip != 0
        and
        observed_count >= MIN_OBSERVED_AXIAL_BINS
        and contiguous >= MIN_CONTIGUOUS_AXIAL_BINS
        and sign_consistency >= MIN_RADIAL_SIGN_CONSISTENCY
        and median_normal_error is not None
        and median_normal_error <= MAX_RADIAL_NORMAL_ERROR_DEG
    )
    eligible_errors = errors[eligible & np.isfinite(errors)]
    signed_median = float(np.median(eligible_errors)) if actionable else None
    q_value = float(np.median(np.abs(eligible_errors))) if actionable else None
    return {
        "status": "MEASURED" if actionable else "ABSTAIN",
        "global_normal_flip": global_flip,
        "median_raw_radial_dot_before_global_flip": median_raw_dot,
        "observed_axial_bins": eligible.tolist(),
        "observed_axial_bin_count": observed_count,
        "longest_contiguous_observed_run": contiguous,
        "radial_normal_sign_consistency": sign_consistency,
        "median_radial_normal_error_deg": median_normal_error,
        "signed_error_deg_by_axial_bin": [float(value) if math.isfinite(value) else None for value in errors],
        "signed_median_error_deg": signed_median,
        "Q_median_absolute_signed_error_deg": q_value,
        "Q_is_abs_of_signed_median": False,
    }


def cad_radial_cell_vectors(event: Mapping[str, Any]) -> np.ndarray:
    axis = canonical_unoriented_axis(np.asarray(event["axis"], dtype=np.float64))
    first, second = [normalized(np.asarray(row, dtype=np.float64)) for row in event["canonical_guard_radial_units"]]
    theta = math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))
    if abs(math.degrees(theta) - EXPECTED_ARC_DEG) > EXPECTED_ARC_TOLERANCE_DEG:
        raise ValueError("event CAD radial endpoints no longer define a 90-degree minor arc")
    direction = 1.0 if float(np.dot(axis, np.cross(first, second))) >= 0.0 else -1.0
    fractions = 0.15 + 0.10 * np.arange(ANGULAR_BIN_COUNT)
    return np.asarray([rodrigues(direction * fraction * theta * axis) @ first for fraction in fractions])


def measure_event(
    points: np.ndarray,
    normals: np.ndarray,
    event: Mapping[str, Any],
    panels: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit guards, then score both cylinder skins on fixed interior cells."""

    supports, guard_audits = sample_four_guard_supports(points, normals, event, panels)
    if not all(row["actionable"] for row in guard_audits):
        return {
            "event_id": event["event_id"],
            "status": "ABSTAIN",
            "reason": "FOUR_GUARD_ACTIONABILITY",
            "guard_audits": [{k: v for k, v in row.items() if k != "selected_point_ids"} for row in guard_audits],
            "skins": [],
        }, {"guard_audits": guard_audits}
    center = (np.asarray(event["line_start_mm"]) + np.asarray(event["line_end_mm"])) / 2.0
    fit = solve_guard_only_nuisance(
        supports,
        event_axis=np.asarray(event["axis"]),
        event_center_mm=center,
        thickness_mm=float(event["sheet_thickness_mm"]),
    )
    if fit["status"] != "FIT":
        return {
            "event_id": event["event_id"],
            "status": "ABSTAIN",
            "reason": "GUARD_ONLY_NUISANCE_FIT",
            "guard_fit": fit,
            "guard_audits": [{k: v for k, v in row.items() if k != "selected_point_ids"} for row in guard_audits],
            "skins": [],
        }, {"guard_audits": guard_audits}
    moved_points, moved_normals = apply_event_transform(
        points,
        normals,
        center_mm=center,
        translation_mm=np.asarray(fit["translation_mm"]),
        rotation_vector_rad=np.asarray(fit["rotation_vector_rad"]),
    )
    moved_normals /= np.maximum(np.linalg.norm(moved_normals, axis=1)[:, None], 1e-12)
    axis = canonical_unoriented_axis(np.asarray(event["axis"]))
    start = np.asarray(event["line_start_mm"])
    span = float(event["span_mm"])
    thickness = float(event["sheet_thickness_mm"])
    erosion = event_endpoint_erosion_mm(event)
    axial = (moved_points - start) @ axis
    bases = []
    for skin in event["skins"]:
        foot = np.asarray(skin["foot"])
        relative = moved_points - foot
        radial = relative - (relative @ axis)[:, None] * axis
        radial_distance = np.linalg.norm(radial, axis=1)
        radial_unit = radial / np.maximum(radial_distance[:, None], 1e-12)
        angles, theta = EXP27.angle_coordinates(
            radial_unit,
            np.asarray(event["canonical_guard_radial_units"][0]),
            np.asarray(event["canonical_guard_radial_units"][1]),
            axis,
        )
        fractions = angles / theta
        normal_error = np.degrees(
            np.arccos(np.clip(np.abs(np.sum(moved_normals * radial_unit, axis=1)), 0.0, 1.0))
        )
        residual = radial_distance - float(skin["radius_mm"])
        base = (
            (axial >= erosion)
            & (axial <= span - erosion)
            & (fractions >= 0.10)
            & (fractions <= 0.90)
            & (np.abs(residual) <= 0.35 * thickness)
            & (normal_error <= MAX_RADIAL_NORMAL_ERROR_DEG)
        )
        bases.append({
            "radial_unit": radial_unit,
            "radial_residual_mm": residual,
            "angular_fraction": fractions,
            "normal_error_deg": normal_error,
            "base_mask": base,
        })
    costs = np.column_stack([
        np.where(row["base_mask"], np.abs(row["radial_residual_mm"]), np.inf) for row in bases
    ])
    owners = np.argmin(costs, axis=1)
    finite_owner = np.isfinite(np.min(costs, axis=1))
    cad_radials = cad_radial_cell_vectors(event)
    skin_results = []
    skin_arrays = []
    for skin_index, base in enumerate(bases):
        selected = finite_owner & (owners == skin_index) & base["base_mask"]
        selected_ids = np.flatnonzero(selected)
        axial_bins = np.floor(
            (axial[selected_ids] - erosion) / (span - 2.0 * erosion) * AXIAL_BIN_COUNT
        ).astype(np.int64)
        angular_bins = np.floor(
            (base["angular_fraction"][selected_ids] - 0.10) / 0.80 * ANGULAR_BIN_COUNT
        ).astype(np.int64)
        axial_bins = np.clip(axial_bins, 0, AXIAL_BIN_COUNT - 1)
        angular_bins = np.clip(angular_bins, 0, ANGULAR_BIN_COUNT - 1)
        counts = np.zeros((AXIAL_BIN_COUNT, ANGULAR_BIN_COUNT), dtype=np.int64)
        modes = np.zeros((AXIAL_BIN_COUNT, ANGULAR_BIN_COUNT, 3), dtype=np.float64)
        for axial_index in range(AXIAL_BIN_COUNT):
            for angular_index in range(ANGULAR_BIN_COUNT):
                local = (axial_bins == axial_index) & (angular_bins == angular_index)
                ids = selected_ids[local]
                counts[axial_index, angular_index] = len(ids)
                if len(ids) < MIN_POINTS_PER_CELL:
                    continue
                vector = np.sum(moved_normals[ids], axis=0)
                if float(np.linalg.norm(vector)) > 1e-12:
                    modes[axial_index, angular_index] = normalized(vector)
        statistic = skin_turn_statistic(
            modes,
            counts,
            cad_radial_cell_vectors=cad_radials,
            axis=axis,
        )
        skin_results.append({
            "skin": "INNER" if skin_index == 0 else "OUTER",
            "cylinder_face_id": str(event["skins"][skin_index]["face_id"]),
            "radius_mm": float(event["skins"][skin_index]["radius_mm"]),
            "owned_point_count": int(len(selected_ids)),
            **statistic,
        })
        skin_arrays.append({
            "selected_point_ids": selected_ids,
            "cell_counts": counts,
            "cell_normal_modes": modes,
            "radial_residual_mm": base["radial_residual_mm"],
            "angular_fraction": base["angular_fraction"],
        })
    status = "MEASURED_BOTH_SKINS" if all(row["status"] == "MEASURED" for row in skin_results) else "ABSTAIN"
    sign_gate = (
        case_skin_signed_error_consensus([row["signed_median_error_deg"] for row in skin_results])
        if status == "MEASURED_BOTH_SKINS"
        else {"passes": False, "reason": "SKIN_MEASUREMENT_ABSTENTION"}
    )
    if not sign_gate["passes"]:
        status = "ABSTAIN"
    return {
        "event_id": event["event_id"],
        "status": status,
        "reason": None if status == "MEASURED_BOTH_SKINS" else sign_gate["reason"],
        "endpoint_erosion_mm_each_end": erosion,
        "remaining_interior_span_mm": span - 2.0 * erosion,
        "guard_fit": fit,
        "guard_audits": [{k: v for k, v in row.items() if k != "selected_point_ids"} for row in guard_audits],
        "within_case_skin_signed_error_consensus": sign_gate,
        "skins": skin_results,
    }, {
        "moved_points": moved_points,
        "moved_normals": moved_normals,
        "guard_audits": guard_audits,
        "skin_arrays": skin_arrays,
        "cad_radial_cell_vectors": cad_radials,
    }


def robust_sigma(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) != 39 or not np.isfinite(array).all():
        raise ValueError("adaptive margin requires exactly 39 finite signed D values")
    median = float(np.median(array))
    return 1.4826 * float(np.median(np.abs(array - median)))


def case_skin_signed_error_consensus(signed_median_errors_deg: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(signed_median_errors_deg, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        return {"passes": False, "reason": "MISSING_OR_NONFINITE_SKIN_SIGNED_ERROR"}
    active = np.abs(values) > CASE_SKIN_SIGN_NEUTRAL_DEG
    both_neutral = bool(np.all(~active))
    both_directional_same_sign = bool(np.all(active) and np.sign(values[0]) == np.sign(values[1]))
    passes = both_neutral or both_directional_same_sign
    if passes:
        reason = None
    elif np.count_nonzero(active) == 1:
        reason = "MIXED_NEUTRAL_AND_DIRECTIONAL_SKINS"
    else:
        reason = "OPPOSING_INNER_OUTER_SIGNED_ERROR"
    return {
        "passes": passes,
        "reason": reason,
        "neutral_threshold_deg": CASE_SKIN_SIGN_NEUTRAL_DEG,
        "signed_median_errors_deg": values.tolist(),
        "neutral_skin_mask": (~active).tolist(),
    }


def paired_event_direction(
    first_skin_q: Sequence[float],
    second_skin_q: Sequence[float],
    variant_d_by_skin: Sequence[Sequence[float]],
    *,
    first_skin_signed_median_e: Sequence[float],
    second_skin_signed_median_e: Sequence[float],
) -> dict[str, Any]:
    """Gate two skins independently; preserve scan-swap antisymmetry."""

    first = np.asarray(first_skin_q, dtype=np.float64)
    second = np.asarray(second_skin_q, dtype=np.float64)
    if first.shape != (2,) or second.shape != (2,):
        raise ValueError("each event requires inner and outer Q for both scans")
    d = first - second
    first_consensus = case_skin_signed_error_consensus(first_skin_signed_median_e)
    second_consensus = case_skin_signed_error_consensus(second_skin_signed_median_e)
    rows = []
    for skin_index in range(2):
        sigma = robust_sigma(variant_d_by_skin[skin_index])
        margin = max(PAIR_MARGIN_FLOOR_DEG, 3.0 * sigma)
        larger = max(float(first[skin_index]), float(second[skin_index]))
        smaller = min(float(first[skin_index]), float(second[skin_index]))
        ratio = (larger + PAIR_RATIO_FLOOR_DEG) / (smaller + PAIR_RATIO_FLOOR_DEG)
        rows.append({
            "skin": "INNER" if skin_index == 0 else "OUTER",
            "Q_n009_deg": float(first[skin_index]),
            "Q_n010_deg": float(second[skin_index]),
            "D_n009_minus_n010_deg": float(d[skin_index]),
            "variant_robust_sigma_deg": sigma,
            "required_margin_deg": margin,
            "ratio": ratio,
            "qualifies": bool(abs(float(d[skin_index])) >= margin and ratio >= PAIR_RATIO_MIN),
        })
    same_sign = bool(d[0] != 0.0 and d[1] != 0.0 and np.sign(d[0]) == np.sign(d[1]))
    directional = bool(
        first_consensus["passes"]
        and second_consensus["passes"]
        and same_sign
        and all(row["qualifies"] for row in rows)
    )
    sign = int(np.sign(d[0])) if directional else 0
    return {
        "status": "DIRECTIONAL" if directional else "ABSTAIN",
        "direction_sign_n009_minus_n010": sign,
        "direction": "N009_MORE_NONCONFORMING" if sign > 0 else (
            "N010_MORE_NONCONFORMING" if sign < 0 else "ABSTAIN"
        ),
        "both_skin_D_sign_agreement": same_sign,
        "n009_within_case_skin_signed_error_consensus": first_consensus,
        "n010_within_case_skin_signed_error_consensus": second_consensus,
        "skins": rows,
    }


def pair_direction(event_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signs = [int(row.get("direction_sign_n009_minus_n010", 0)) for row in event_rows]
    positive = sum(value > 0 for value in signs)
    negative = sum(value < 0 for value in signs)
    directional = (positive >= MIN_DIRECTIONAL_EVENTS and negative == 0) or (
        negative >= MIN_DIRECTIONAL_EVENTS and positive == 0
    )
    sign = 1 if directional and positive else (-1 if directional else 0)
    return {
        "status": "DIRECTIONAL" if directional else "ABSTAIN",
        "direction_sign_n009_minus_n010": sign,
        "direction": "N009_MORE_NONCONFORMING" if sign > 0 else (
            "N010_MORE_NONCONFORMING" if sign < 0 else "ABSTAIN"
        ),
        "positive_event_count": positive,
        "negative_event_count": negative,
        "abstaining_event_count": len(signs) - positive - negative,
    }


def robustness_direction_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require all 39 frozen rows to reproduce the central pair direction."""

    expected_ids = {row["variant_id"] for row in robustness_variants()}
    actual_ids = {str(row.get("variant_id")) for row in rows}
    if len(rows) != 39 or actual_ids != expected_ids:
        raise ValueError("robustness direction gate requires the exact frozen 39-ID inventory")
    central_rows = [row for row in rows if row.get("variant_id") == "V0.125_FULL"]
    if len(central_rows) != 1:
        raise ValueError("robustness matrix lacks exactly one central row")
    central = central_rows[0]
    central_sign = int(central.get("direction_sign_n009_minus_n010", 0))
    invariant = bool(
        central.get("status") == "DIRECTIONAL"
        and central_sign in (-1, 1)
        and all(
            row.get("status") == "DIRECTIONAL"
            and int(row.get("direction_sign_n009_minus_n010", 0)) == central_sign
            for row in rows
        )
    )
    return {
        "status": "PASS" if invariant else "FAIL_TERMINAL_ABSTAIN",
        "direction_invariant": invariant,
        "central_direction_sign_n009_minus_n010": central_sign,
        "row_count": len(rows),
    }


def robustness_variants() -> list[dict[str, Any]]:
    rows = []
    for voxel_fraction in VOXEL_FRACTIONS:
        for partition in PARTITIONS:
            rows.append({
                "variant_id": f"V{voxel_fraction:g}_{partition}",
                "voxel_thickness_fraction": voxel_fraction,
                "partition": partition,
                "perturbed_member": None,
                "pose_kind": None,
            })
    for member in EXPECTED_CASE_TOKENS:
        for sign in (-1, 1):
            for axis_index, axis_name in enumerate("XYZ"):
                rows.append({
                    "variant_id": f"POSE_{member}_T{axis_name}_{sign:+d}",
                    "voxel_thickness_fraction": 0.125,
                    "partition": "FULL",
                    "perturbed_member": member,
                    "pose_kind": "TRANSLATION",
                    "source_frame_axis_index": axis_index,
                    "magnitude_thickness_fraction": sign / 16.0,
                })
                rows.append({
                    "variant_id": f"POSE_{member}_R{axis_name}_{sign:+d}",
                    "voxel_thickness_fraction": 0.125,
                    "partition": "FULL",
                    "perturbed_member": member,
                    "pose_kind": "ROTATION",
                    "source_frame_axis_index": axis_index,
                    "magnitude_deg": sign * 0.125,
                    "rotation_center": "WHOLE_PART_CAD_CENTROID",
                })
    if len(rows) != 39 or len({row["variant_id"] for row in rows}) != 39:
        raise AssertionError("frozen robustness inventory must contain 39 unique rows")
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def validate_decoder_report(report: Mapping[str, Any]) -> dict[str, Any]:
    decoded = report.get("decode_report")
    if not isinstance(decoded, Mapping):
        raise ValueError("cadmpeg decode_report is missing")
    losses = decoded.get("losses")
    if not isinstance(losses, list):
        raise ValueError("cadmpeg loss inventory is missing")
    codes = tuple(sorted(
        str(row.get("code", {}).get("code"))
        for row in losses
        if isinstance(row, Mapping)
    ))
    severities = [str(row.get("severity")) for row in losses if isinstance(row, Mapping)]
    if (
        decoded.get("geometry_transferred") is not True
        or decoded.get("container_only") is not False
        or len(losses) != 6
        or codes != EXPECTED_WARNING_CODES
        or any(value != "warning" for value in severities)
    ):
        raise ValueError("cadmpeg decoder provenance differs from the frozen contract")
    return {
        "geometry_transferred": True,
        "container_only": False,
        "warning_count": 6,
        "warning_codes": list(codes),
        "error_count": 0,
    }


def validate_proper_rigid(transform: np.ndarray) -> dict[str, Any]:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("registration transform is not a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-10):
        raise ValueError("registration transform has invalid homogeneous row")
    rotation = value[:3, :3]
    determinant = float(np.linalg.det(rotation))
    orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-8) or orthogonality > 1e-8:
        raise ValueError("registration transform is not proper rigid unit scale")
    return {"determinant": determinant, "orthogonality_max_abs_error": orthogonality}


def unique_active_body_bounds(
    model: Mapping[str, Any], active_shell_faces: set[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = _index(model, ("faces", "loops", "coedges", "edges", "vertices", "points"))
    values = []
    for face_id in sorted(active_shell_faces):
        for edge in _face_edges(face_id, idx):
            values.extend((edge["start_mm"], edge["end_mm"]))
    if not values:
        raise ValueError("active shell has no finite B-rep vertices")
    array = np.unique(np.round(np.asarray(values, dtype=np.float64), 12), axis=0)
    return np.min(array, axis=0), np.max(array, axis=0), np.mean(array, axis=0)


def load_frozen_case(
    case_token: str,
    *,
    scan_path: Path,
    exp15_case_path: Path,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    if case_token not in EXPECTED_CASE_TOKENS:
        raise ValueError("EXP28 case token is outside the frozen pair")
    if sha256_path(scan_path) != frozen["raw_scan_sha256"][case_token]:
        raise ValueError(f"{case_token} source scan hash mismatch")
    if sha256_path(exp15_case_path) != frozen["exp15_semantic_sha256"][case_token]:
        raise ValueError(f"{case_token} EXP15 semantic hash mismatch")
    semantic = json.loads(exp15_case_path.read_text(encoding="utf-8"))
    if (
        semantic.get("case_token") != case_token
        or semantic.get("source_cad_sha256") != frozen["source_sldprt_sha256"]
        or semantic.get("scan_sha256") != frozen["raw_scan_sha256"][case_token]
        or semantic.get("alignment_abstained") is not False
        or semantic.get("scale_factor") != 1.0
        or semantic.get("evaluator_data_opened") is not False
        or semantic.get("gold_or_expected_counts_used") is not False
    ):
        raise ValueError(f"{case_token} frozen EXP15 semantic contract failed")
    transform = np.asarray(semantic["raw_scan_to_reference_transform"], dtype=np.float64)
    rigid_audit = validate_proper_rigid(transform)
    cloud = EXP27.o3d.io.read_point_cloud(str(scan_path))
    if not cloud.has_points() or not cloud.has_normals():
        raise ValueError(f"{case_token} source scan lacks points or normals")
    raw_points = np.asarray(cloud.points, dtype=np.float64)
    raw_normals = np.asarray(cloud.normals, dtype=np.float64)
    lengths = np.linalg.norm(raw_normals, axis=1)
    valid = (
        np.isfinite(raw_points).all(axis=1)
        & np.isfinite(raw_normals).all(axis=1)
        & np.isfinite(lengths)
        & (lengths > 1e-12)
    )
    raw_ids = np.flatnonzero(valid).astype(np.int64)
    raw_points = raw_points[valid]
    raw_normals = raw_normals[valid] / lengths[valid, None]
    return {
        "case_token": case_token,
        "raw_points": raw_points,
        "raw_normals": raw_normals,
        "raw_ids": raw_ids,
        "raw_to_cad": transform,
        "semantic_sha256": sha256_path(exp15_case_path),
        "scan_sha256": sha256_path(scan_path),
        "rigid_audit": rigid_audit,
    }


def prepare_variant_case(
    case: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    thickness_mm: float,
    body_bbox_min_mm: np.ndarray,
    whole_part_cad_centroid_mm: np.ndarray,
) -> dict[str, Any]:
    raw_points = np.asarray(case["raw_points"], dtype=np.float64)
    raw_normals = np.asarray(case["raw_normals"], dtype=np.float64)
    transform = np.asarray(case["raw_to_cad"], dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    if variant.get("perturbed_member") == case["case_token"]:
        source_center = (np.asarray(whole_part_cad_centroid_mm) - translation) @ rotation
        axis_index = int(variant["source_frame_axis_index"])
        source_axis = np.eye(3)[axis_index]
        if variant["pose_kind"] == "TRANSLATION":
            raw_points = raw_points + source_axis * (
                float(variant["magnitude_thickness_fraction"]) * thickness_mm
            )
        elif variant["pose_kind"] == "ROTATION":
            pose_rotation = rodrigues(math.radians(float(variant["magnitude_deg"])) * source_axis)
            raw_points = (raw_points - source_center) @ pose_rotation.T + source_center
            raw_normals = raw_normals @ pose_rotation.T
        else:
            raise ValueError("unknown frozen pose perturbation kind")
    aligned_points = raw_points @ rotation.T + translation
    aligned_normals = raw_normals @ rotation.T
    selected = partition_mask(
        aligned_points,
        np.asarray(case["raw_ids"]),
        str(variant["partition"]),
        body_bbox_min_mm=np.asarray(body_bbox_min_mm),
        thickness_mm=thickness_mm,
    )
    voxel_mm = float(variant["voxel_thickness_fraction"]) * thickness_mm
    capped_points, capped_normals, representatives = canonical_voxel_aggregate(
        aligned_points[selected], aligned_normals[selected], voxel_mm
    )
    return {
        "points": capped_points,
        "normals": capped_normals,
        "raw_aligned_points": aligned_points,
        "raw_aligned_normals": aligned_normals,
        "partition_input_count": int(np.count_nonzero(selected)),
        "voxel_point_count": int(len(capped_points)),
        "representative_identity_sha256": hashlib.sha256(
            np.ascontiguousarray(representatives, dtype="<i8").tobytes()
        ).hexdigest(),
    }


def validate_preregistration(path: Path) -> dict[str, Any]:
    actual = sha256_path(path)
    if actual != EXPECTED_PREREGISTRATION_SHA256:
        raise ValueError(f"EXP28 preregistration hash mismatch: {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("EXP28 preregistration identity mismatch")
    if payload.get("source_execution_state_at_freeze") != "NOT_RUN":
        raise ValueError("EXP28 preregistration source state changed")
    return payload


def validate_imported_bindings() -> dict[str, str]:
    actual = {
        "EXP17": sha256_path(EXP17_PATH),
        "EXP25": sha256_path(EXP25_PATH),
        "EXP27": sha256_path(EXP27_PATH),
    }
    if actual != EXPECTED_IMPORTED_SHA256:
        raise ValueError(f"EXP28 imported helper binding mismatch: {actual}")
    return actual


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    preregistration = validate_preregistration(args.preregistration)
    imported = validate_imported_bindings()
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "imported_code_sha256": imported,
        "robustness_row_count": len(robustness_variants()),
        "topology_core_implemented": True,
        "nuisance_core_implemented": True,
        "signed_statistic_core_implemented": True,
        "real_source_execution_authorized": False,
        "blocked_pending": [
            "INDEPENDENT_PREREGISTRATION_AND_CORE_AUDIT",
            "SOURCE_EXECUTION_ORCHESTRATOR_AND_NATIVE_RENDERER",
            "SEPARATE_PREFLIGHT_BINDING",
        ],
        "evaluator_roles_opened": False,
        "production_promotion_authorized": bool(preregistration["authorizations"]["production_promotion"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
