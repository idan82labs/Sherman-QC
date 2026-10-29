"""Conservative CAD-only flange previews for operator bend-mark review.

This is a visual association, not a manufacturing-event proof or scan result.
Only a single, unambiguous pair of planar flange skins adjacent to a native
inner/outer bend cylinder pair is emitted. Unsupported geometry abstains.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from backend.cad_import import import_cad_file
from scripts.extract_step_topology_bend_axes import extract_step_bend_axes


MAX_CAD_BYTES = 5_000_000
MAX_CANDIDATES = 32
MAX_FACES = 1024
MAX_TRIANGLES = 200_000
MAX_OVERLAY_TRIANGLES = 10_000


class FlangePreviewUnavailable(ValueError):
    """The CAD cannot support a bounded native-face preview."""


def _source_faces(path: Path, expected_count: int):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() <= 0:
        raise FlangePreviewUnavailable("STEP_READ_UNAVAILABLE")
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_FACE)
    faces = []
    while explorer.More():
        if len(faces) >= MAX_FACES:
            raise FlangePreviewUnavailable("FACE_BUDGET_EXCEEDED")
        faces.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    if len(faces) != expected_count:
        raise FlangePreviewUnavailable("FACE_ENUMERATION_MISMATCH")
    return faces


def _area_and_triangles(mesh: Any, face_record: dict[str, Any]) -> tuple[float, np.ndarray]:
    start, stop = face_record["triangle_range"]
    triangles = np.asarray(mesh.faces[start:stop], dtype=np.int32)
    if len(triangles) == 0:
        return 0.0, triangles
    points = np.asarray(mesh.vertices[triangles], dtype=np.float64)
    area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1).sum() / 2
    return float(area), triangles


def _flange_face_indices(
    axis: dict[str, Any], faces: list[Any], native: dict[str, Any],
    tessellation: dict[str, Any], mesh: Any, edge_owners: dict[str, set[int]],
) -> list[int] | None:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane

    # The axis extractor is one-based; the importer and face sidecar are zero-based.
    bend_faces = {index - 1 for index in axis["face_indices"]}
    if len(bend_faces) != 2 or any(index < 0 or index >= len(faces) for index in bend_faces):
        return None
    expected_radii = sorted(float(value) for value in axis["radii_mm"])
    source_axis = np.asarray([axis["line_start"], axis["line_end"]], dtype=np.float64)
    source_direction = source_axis[1] - source_axis[0]
    source_direction /= np.linalg.norm(source_direction)
    actual_radii = []
    for index in bend_faces:
        surface = BRepAdaptor_Surface(faces[index])
        if surface.GetType() != GeomAbs_Cylinder:
            return None
        cylinder = surface.Cylinder()
        actual_radii.append(float(cylinder.Radius()))
        native_axis = cylinder.Axis()
        direction = native_axis.Direction()
        native_direction = np.array([direction.X(), direction.Y(), direction.Z()])
        point = native_axis.Location()
        native_point = np.array([point.X(), point.Y(), point.Z()])
        if abs(float(np.dot(source_direction, native_direction))) < 0.99999:
            return None
        if np.linalg.norm(np.cross(native_point - source_axis[0], source_direction)) > 0.01:
            return None
    if len(expected_radii) != 2 or not np.allclose(sorted(actual_radii), expected_radii, atol=1e-5):
        return None

    adjacent: set[int] = set()
    for index in bend_faces:
        for use in native["face_records"][index]["edge_uses"]:
            if not use["degenerate"]:
                adjacent.update(edge_owners[use["edge_id"]])
    adjacent.difference_update(bend_faces)
    planar = []
    for index in sorted(adjacent):
        if BRepAdaptor_Surface(faces[index]).GetType() != GeomAbs_Plane:
            continue
        area, triangles = _area_and_triangles(mesh, tessellation["face_records"][index])
        if area > 1.0 and len(triangles) > 0 and np.isfinite(area):
            planar.append((index, area))

    # In a plain sheet flange the adjacent skins form three scale-separated
    # pairs: wide web, smaller flange, and tiny thickness caps. If that pattern
    # is absent, do not guess which surface the bend creates.
    remaining = sorted(planar, key=lambda row: -row[1])
    groups: list[list[tuple[int, float]]] = []
    while remaining:
        reference = remaining[0][1]
        group = [row for row in remaining if abs(row[1] - reference) / reference <= 0.02]
        if len(group) != 2:
            return None
        groups.append(group)
        ids = {row[0] for row in group}
        remaining = [row for row in remaining if row[0] not in ids]
    if len(groups) < 2:
        return None
    web_area, flange_area = groups[0][0][1], groups[1][0][1]
    if web_area < 1.5 * flange_area:
        return None
    if len(groups) > 2 and flange_area < 5.0 * groups[2][0][1]:
        return None
    flange_indices = sorted(row[0] for row in groups[1])

    def plane(index: int) -> tuple[np.ndarray, np.ndarray]:
        native_plane = BRepAdaptor_Surface(faces[index]).Plane()
        direction = native_plane.Axis().Direction()
        location = native_plane.Location()
        return (
            np.array([direction.X(), direction.Y(), direction.Z()]),
            np.array([location.X(), location.Y(), location.Z()]),
        )

    flange_normal, flange_origin = plane(flange_indices[0])
    opposite_normal, opposite_origin = plane(flange_indices[1])
    web_normal, _ = plane(groups[0][0][0])
    thickness = float(axis["radius_separation_mm"])
    plane_separation = abs(float(np.dot(opposite_origin - flange_origin, flange_normal)))
    if (abs(float(np.dot(flange_normal, opposite_normal))) < 0.9999
            or abs(plane_separation - thickness) > max(0.05, 0.1 * thickness)
            or abs(float(np.dot(flange_normal, web_normal))) > math.cos(math.radians(5))):
        return None
    return flange_indices


def _native_face_edges(face: Any, record: dict[str, Any]) -> list[tuple[dict[str, Any], Any]] | None:
    """Replay the importer's cumulative child walk, preserving edge-use order."""
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS, TopoDS_Iterator

    edges = []

    def walk(shape: Any) -> None:
        if shape.ShapeType() == TopAbs_EDGE:
            edges.append(TopoDS.Edge_s(shape))
            return
        iterator = TopoDS_Iterator(shape, True, True)
        while iterator.More():
            walk(iterator.Value())
            iterator.Next()

    walk(face)
    uses = record["edge_uses"]
    return list(zip(uses, edges)) if len(uses) == len(edges) else None


def _line_endpoints(edge: Any) -> np.ndarray | None:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Line

    curve = BRepAdaptor_Curve(edge)
    if curve.GetType() != GeomAbs_Line:
        return None
    points = [curve.Value(parameter) for parameter in (curve.FirstParameter(), curve.LastParameter())]
    result = np.array([[point.X(), point.Y(), point.Z()] for point in points], dtype=np.float64)
    return result if np.isfinite(result).all() else None


def _straight_parallel_outer_depth(
    seam: np.ndarray, outer_edges: list[np.ndarray], axis_direction: np.ndarray,
) -> float | None:
    """Unambiguous in-plane tangent-to-opposite-rail distance, in CAD units.

    Require one nearly full-span straight outer rail. Chamfered ends may
    shorten it, but slots, taper and stepped/multiple rails must abstain.
    """
    direction = np.asarray(axis_direction, dtype=np.float64)
    seam = np.asarray(seam, dtype=np.float64)
    if (seam.shape != (2, 3) or direction.shape != (3,)
            or not np.isfinite(seam).all() or not np.isfinite(direction).all()):
        return None
    axis_norm = np.linalg.norm(direction)
    span = np.linalg.norm(seam[1] - seam[0])
    if axis_norm < 1e-9 or span < 1.0:
        return None
    direction /= axis_norm
    if abs(float(np.dot((seam[1] - seam[0]) / span, direction))) < 0.99999:
        return None
    seam_axial = sorted(float(np.dot(point, direction)) for point in seam)
    depths = []
    for edge in outer_edges:
        edge = np.asarray(edge, dtype=np.float64)
        if edge.shape != (2, 3) or not np.isfinite(edge).all():
            continue
        edge_vector = edge[1] - edge[0]
        edge_span = np.linalg.norm(edge_vector)
        if edge_span < 0.7 * span or abs(float(np.dot(edge_vector / edge_span, direction))) < 0.99999:
            continue
        axial = sorted(float(np.dot(point, direction)) for point in edge)
        if max(abs(axial[i] - seam_axial[i]) for i in (0, 1)) > 0.2 * span:
            continue
        offsets = [np.cross(point - seam[0], direction) for point in edge]
        if np.linalg.norm(offsets[0] - offsets[1]) > max(0.05, 0.001 * span):
            continue
        depth = float(np.linalg.norm(np.mean(offsets, axis=0)))
        if depth >= 0.1:
            depths.append(depth)
    return depths[0] if len(depths) == 1 else None


def _nominal_flange_depth(
    axis: dict[str, Any], flange_indices: list[int], faces: list[Any],
    native: dict[str, Any], edge_owners: dict[str, set[int]],
) -> float | None:
    """Measure both CAD flange skins; refuse ambiguous or unequal depths."""
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    bend_indices = {index - 1 for index in axis["face_indices"]}
    source_axis = np.asarray([axis["line_start"], axis["line_end"]], dtype=np.float64)
    axis_direction = source_axis[1] - source_axis[0]
    depths = []
    for face_index in flange_indices:
        face_edges = _native_face_edges(faces[face_index], native["face_records"][face_index])
        if face_edges is None:
            return None
        seams = [(use, edge) for use, edge in face_edges
                 if edge_owners.get(use["edge_id"], set()) - {face_index} <= bend_indices
                 and bool((edge_owners.get(use["edge_id"], set()) - {face_index}) & bend_indices)]
        if len(seams) != 1 or seams[0][0]["degenerate"]:
            return None
        seam_use, seam_edge = seams[0]
        bend_index = next(iter(edge_owners[seam_use["edge_id"]] - {face_index}))
        bend_edges = _native_face_edges(faces[bend_index], native["face_records"][bend_index])
        if bend_edges is None or sum(
            use["edge_id"] == seam_use["edge_id"] and edge.IsSame(seam_edge)
            for use, edge in bend_edges
        ) != 1:
            return None
        seam = _line_endpoints(seam_edge)
        if seam is None:
            return None
        outer_wire = BRepTools.OuterWire_s(faces[face_index])
        if outer_wire.IsNull():
            return None
        outer_explorer = TopExp_Explorer(outer_wire, TopAbs_EDGE)
        outer_edges = []
        while outer_explorer.More():
            outer_edges.append(TopoDS.Edge_s(outer_explorer.Current()))
            outer_explorer.Next()
        rails = []
        for use, edge in face_edges:
            if use["edge_id"] == seam_use["edge_id"] or use["degenerate"]:
                continue
            if not any(edge.IsSame(outer_edge) for outer_edge in outer_edges):
                continue
            endpoints = _line_endpoints(edge)
            if endpoints is not None:
                rails.append(endpoints)
        depth = _straight_parallel_outer_depth(seam, rails, axis_direction)
        if depth is None:
            return None
        depths.append(depth)
    if len(depths) != 2 or abs(depths[0] - depths[1]) > max(0.1, 0.01 * max(depths)):
        return None
    return float(sum(depths) / 2)


def _mesh_payload(mesh: Any, tessellation: dict[str, Any], face_indices: list[int]) -> dict[str, Any] | None:
    triangles = np.concatenate([
        _area_and_triangles(mesh, tessellation["face_records"][index])[1]
        for index in face_indices
    ])
    if len(triangles) == 0 or len(triangles) > MAX_OVERLAY_TRIANGLES:
        return None
    vertex_indices, inverse = np.unique(triangles.reshape(-1), return_inverse=True)
    vertices = np.asarray(mesh.vertices[vertex_indices], dtype=np.float32)
    if not np.isfinite(vertices).all():
        return None
    return {
        "vertices": vertices.reshape(-1).tolist(),
        "triangles": inverse.astype(np.int32).tolist(),
        "source_face_indices": face_indices,
        "association": "CAD_ADJACENT_PLANAR_FLANGE_SKINS",
    }


def _matching_axis(candidate: dict[str, Any], axes: list[dict[str, Any]]) -> int | None:
    try:
        line = np.asarray([candidate["line_start"], candidate["line_end"]], dtype=np.float64)
    except (TypeError, ValueError, KeyError):
        return None
    if line.shape != (2, 3) or not np.isfinite(line).all():
        return None
    matches = []
    for index, axis in enumerate(axes):
        native = np.asarray([axis["canonical_line_start"], axis["canonical_line_end"]], dtype=np.float64)
        error = min(
            max(np.linalg.norm(line[0] - native[0]), np.linalg.norm(line[1] - native[1])),
            max(np.linalg.norm(line[0] - native[1]), np.linalg.norm(line[1] - native[0])),
        )
        if error <= 0.75:
            matches.append(index)
    return matches[0] if len(matches) == 1 else None


def _unique_axis_matches(candidates: list[dict[str, Any]], axes: list[dict[str, Any]]) -> list[int | None]:
    matches = [_matching_axis(candidate, axes) for candidate in candidates]
    counts = Counter(index for index in matches if index is not None)
    return [index if index is not None and counts[index] == 1 else None for index in matches]


def build_flange_previews(
    cad_path: str | Path, expected_sha256: str, candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only CAD-bound unambiguous flange surfaces, otherwise abstain.

    Every returned surface shares the STEP coordinate frame with its candidate
    line; neither units nor physical bend order are certified here.
    """
    path = Path(cad_path)
    if path.suffix.lower() not in {".stp", ".step"} or not path.is_file():
        raise FlangePreviewUnavailable("STEP_SOURCE_REQUIRED")
    if path.stat().st_size > MAX_CAD_BYTES or len(candidates) > MAX_CANDIDATES:
        raise FlangePreviewUnavailable("PREVIEW_BUDGET_EXCEEDED")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise FlangePreviewUnavailable("CAD_HASH_MISMATCH")
    result = import_cad_file(str(path), 0.05, allow_ocp=True)
    if not result.success or result.mesh is None:
        raise FlangePreviewUnavailable("NATIVE_IMPORT_UNAVAILABLE")
    mesh = result.mesh
    native = mesh.native_topology or {}
    tessellation = mesh.tessellation_provenance or {}
    if (len(mesh.faces) > MAX_TRIANGLES
            or native.get("status") != "CAPTURED_IMPORT_LOCAL_INCIDENCE"
            or tessellation.get("mapping_status") != "FACE_OCCURRENCE_RANGES"):
        raise FlangePreviewUnavailable("NATIVE_FACE_MAPPING_UNAVAILABLE")
    faces = _source_faces(path, len(native["face_records"]))
    if len(tessellation["face_records"]) != len(faces):
        raise FlangePreviewUnavailable("FACE_RANGE_MISMATCH")
    edge_owners: dict[str, set[int]] = {}
    for row in native["face_records"]:
        for use in row["edge_uses"]:
            edge_owners.setdefault(use["edge_id"], set()).add(row["source_face_index"])
    axes = extract_step_bend_axes(path)["bend_axes"]
    previews = []
    axis_matches = _unique_axis_matches(candidates, axes)
    for candidate, index in zip(candidates, axis_matches):
        if index is None:
            continue
        face_indices = _flange_face_indices(axes[index], faces, native, tessellation, mesh, edge_owners)
        if face_indices is None:
            continue
        payload = _mesh_payload(mesh, tessellation, face_indices)
        if payload is None:
            continue
        try:
            depth = _nominal_flange_depth(axes[index], face_indices, faces, native, edge_owners)
        except (RuntimeError, ValueError, TypeError):
            # A malformed edge must not remove an otherwise valid visual preview.
            depth = None
        payload["nominal_flange_depth_mm"] = round(depth, 3) if depth is not None else None
        payload["depth_status"] = "CAD_TANGENT_TO_OUTER_EDGE" if depth is not None else "UNAVAILABLE_AMBIGUOUS_BOUNDARY"
        previews.append({"feature_id": candidate["feature_id"], **payload})
    return previews
