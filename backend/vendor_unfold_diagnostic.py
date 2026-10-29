"""Bounded diagnostic of the frozen V2 skin-boundary proof, never a preview.

This module does not change any V2 acceptance rule. File access occurs only
when diagnose_unfold_boundaries is explicitly called by a supervised caller.
No flat or partial mesh is produced, and PASS here is only a boundary-stage
diagnostic, not successful unfolding or manufacturing authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from backend import vendor_unfold_preview as engine


FROZEN_ENGINE_SHA256 = "f316f56530e2fd6e44f21aa28a9871f43c795714c4b15b652139ee2cfdca461a"
MAX_BOUNDARY_RECORDS = 128
MAX_RAIL_ATTEMPTS = 256
MAX_CERTIFICATE_RAIL_PAIRS = 128
MAX_EXPLICIT_BOUND_EVALUATIONS = 256
MAX_REPORTED_KNOTS = 32
MAX_DIAGNOSTIC_JSON_BYTES = 1_000_000


def _empty() -> dict[str, Any]:
    return {"schema_version": 1, "kind": "FROZEN_V2_SKIN_BOUNDARY_DIAGNOSTIC",
        "status": "UNAVAILABLE", "reason": None, "source_sha256": None,
        "engine_sha256": None, "diagnostic_sha256": None,
        "preview_status": "NOT_PRODUCED_DIAGNOSTIC_ONLY", "manufacturing_ready": False,
        "units": "assumed_mm", "units_verified": False,
        "vertices": [], "triangles": [], "bend_lines": [], "flange_surfaces": [],
        "engine_boundary_status": None, "engine_boundary_reason": None,
        "boundary_checks": [], "diagnostics": {"scope": "BOUNDARY_PROOF_ONLY_NOT_FULL_UNFOLD",
            "limits": {"cad_bytes": engine.MAX_CAD_BYTES, "faces": engine.MAX_FACES,
                "triangles": engine.MAX_TRIANGLES, "candidates": engine.MAX_CANDIDATES,
                "boundary_records": MAX_BOUNDARY_RECORDS, "rail_attempts": MAX_RAIL_ATTEMPTS,
                "certificate_rail_pairs": MAX_CERTIFICATE_RAIL_PAIRS,
                "explicit_bound_evaluations": MAX_EXPLICIT_BOUND_EVALUATIONS,
                "frozen_pair_check_max_internal_bounds": 2,
                "reported_knots_per_curve": MAX_REPORTED_KNOTS,
                "json_bytes": MAX_DIAGNOSTIC_JSON_BYTES}}}


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _curve(face: engine.SkinFace, edge: int) -> engine.NativeCurve | None:
    if edge in face.edge_curves:
        return face.edge_curves[edge]
    points = face.edges[edge]
    return engine.NativeCurve("line", points) if points is not None else None


def _curve_summary(curve: engine.NativeCurve | None) -> dict[str, Any] | None:
    if curve is None:
        return None
    row = {"kind": curve.kind, "pole_count": len(curve.poles),
           "straight_locus_proven": curve.straight_endpoints() is not None}
    if curve.kind == "spline":
        spans = np.diff(curve.knots)
        parameter_scale = max(abs(float(curve.knots[0])), abs(float(curve.knots[-1])),
                              float(curve.knots[-1] - curve.knots[0]))
        row.update({"degree": curve.degree, "clamped_nonperiodic_basis": curve.clamped_spline(),
            "span_count": len(spans), "own_span_min": float(spans.min()),
            "own_span_max": float(spans.max()), "native_parameter_scale": parameter_scale,
            "conditioning_span_floor": float(64 * np.finfo(float).eps * parameter_scale),
            "weight_min": float(curve.weights.min()), "weight_max": float(curve.weights.max()),
            "weight_ratio": float(curve.weights.max() / curve.weights.min()),
            "own_native_knots_prefix": curve.knots[:MAX_REPORTED_KNOTS].tolist(),
            "own_multiplicities_prefix": curve.multiplicities[:MAX_REPORTED_KNOTS].tolist(),
            "knot_report_truncated": len(curve.knots) > MAX_REPORTED_KNOTS})
    elif curve.kind == "circle":
        row.update({"radius": curve.radius, "u0": curve.u0, "u1": curve.u1,
                    "sweep_deg": math.degrees(curve.u1 - curve.u0)})
    return row


def _certificate(left: engine.NativeCurve | None, right: engine.NativeCurve | None,
                 offset: np.ndarray, counters: dict[str, int]) -> dict[str, Any]:
    row: dict[str, Any] = {"matched_by_frozen_engine": False, "reason": None}
    if left is None or right is None:
        row["reason"] = "NATIVE_CURVE_PROOF_UNAVAILABLE"
        return row
    row["left_curve"], row["right_curve"] = _curve_summary(left), _curve_summary(right)
    counters.setdefault("certificate_rail_pairs", 0)
    counters.setdefault("explicit_bound_evaluations", 0)
    engine._require(counters["certificate_rail_pairs"] < MAX_CERTIFICATE_RAIL_PAIRS,
                    "DIAGNOSTIC_CERTIFICATE_PAIR_BUDGET_EXCEEDED")
    counters["certificate_rail_pairs"] += 1
    try:
        row["matched_by_frozen_engine"] = bool(engine._translated_curves(left, right, offset))
    except engine.UnfoldUnavailable as exc:
        row["reason"] = str(exc)
        return row
    if left.straight_endpoints() is not None and right.straight_endpoints() is not None:
        row["proof_mode"] = "STRAIGHT_CONTROL_NET_LOCUS"
    elif left.kind == right.kind == "circle":
        row["proof_mode"] = "PHYSICAL_CIRCLE_LOCUS_BOUND"
    elif left.kind == right.kind == "spline":
        row["proof_mode"] = "OWN_NATIVE_KNOT_RATIONAL_BEZIER_CERTIFICATE"
        row["degree_equal"] = left.degree == right.degree
        row["span_count_equal"] = left.knots.shape == right.knots.shape
        row["forward_multiplicity_topology_equal"] = bool(np.array_equal(left.multiplicities, right.multiplicities))
        row["reverse_multiplicity_topology_equal"] = bool(np.array_equal(left.multiplicities, right.multiplicities[::-1]))
        for reverse, name in ((False, "forward"), (True, "reverse")):
            try:
                if left.degree != right.degree or not left.clamped_spline() or not right.clamped_spline():
                    row[name + "_certificate_reason"] = "DEGREE_OR_CLAMPING_MISMATCH"
                    row[name + "_deviation_bound"] = None
                    continue
                engine._require(counters["explicit_bound_evaluations"] < MAX_EXPLICIT_BOUND_EVALUATIONS,
                                "DIAGNOSTIC_EXPLICIT_BOUND_BUDGET_EXCEEDED")
                counters["explicit_bound_evaluations"] += 1
                bound = engine._spline_translation_bound(left, right, offset, reverse=reverse)
                row[name + "_deviation_bound"] = _finite(bound)
                row[name + "_certificate_reason"] = (
                    "CERTIFIED_WITHIN_GEOM_TOL" if bound <= engine.GEOM_TOL else
                    "GLOBAL_DEVIATION_BOUND_EXCEEDS_GEOM_TOL" if math.isfinite(bound) else
                    "TOPOLOGY_OR_RELATIVE_WEIGHT_BOUND_UNCERTIFIABLE")
            except engine.UnfoldUnavailable as exc:
                row[name + "_deviation_bound"] = None
                row[name + "_certificate_reason"] = str(exc)
    else:
        row["proof_mode"] = "CURVE_KIND_MISMATCH"
    if not row["matched_by_frozen_engine"] and row["reason"] is None:
        row["reason"] = "TRANSLATED_NATIVE_RAIL_NOT_CERTIFIED"
    return row


def _components(adjacency: dict[int, list[engine.BendLink]]) -> list[set[int]]:
    components, seen = [], set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        nodes, cylinders, stack = set(), set(), [start]
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            for link in adjacency[node]:
                cylinders.add(link.cylinder)
                stack.extend(index for index in link.planes if index not in nodes)
        seen.update(nodes)
        components.append(nodes | cylinders)
    return components


def _boundary_records(faces: list[engine.SkinFace], links: list[engine.BendLink],
                      adjacency: dict[int, list[engine.BendLink]], selected: list[engine.BendLink],
                      types: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    components = _components(adjacency)
    engine._require(len(components) == 2, "DIAGNOSTIC_REQUIRES_TWO_GRAPH_COMPONENTS")
    owners: dict[int, set[int]] = {}
    for face in faces:
        for edge in face.edges:
            owners.setdefault(edge, set()).add(face.index)
    first = selected[0]
    paired = next(link for link in links if link.feature_id == first.feature_id and link.cylinder != first.cylinder)
    thickness = abs(faces[first.cylinder].radius - faces[paired.cylinder].radius)
    rows, counters = [], {"boundary_records": 0, "rail_attempts": 0,
                          "certificate_rail_pairs": 0, "explicit_bound_evaluations": 0}
    for layer_index, layer in enumerate(components):
        opposite = components[1 - layer_index]
        for index in sorted(layer):
            face = faces[index]
            if face.kind != "plane":
                continue
            normal = engine._unit(face.normal)
            for edge in face.edges:
                neighbors = owners[edge] - {index}
                if len(neighbors) != 1:
                    continue
                neighbor_index = next(iter(neighbors))
                wall = faces[neighbor_index]
                if neighbor_index in layer or wall.kind not in ("cylinder", "extruded_boundary"):
                    continue
                counters["boundary_records"] += 1
                engine._require(counters["boundary_records"] <= MAX_BOUNDARY_RECORDS,
                                "DIAGNOSTIC_BOUNDARY_RECORD_BUDGET_EXCEEDED")
                axis_alignment = abs(float(np.dot(engine._unit(wall.axis), normal)))
                span = float(wall.v1 - wall.v0)
                row = {"skin_layer": layer_index, "skin_face_index": index, "skin_edge_id": edge,
                    "wall_face_index": neighbor_index, "wall_classified_kind": wall.kind,
                    "wall_native_surface_type": types.get(neighbor_index, {}).get("surface_type", "UNKNOWN"),
                    "skin_native_edge_curve_type": types.get(index, {}).get("edge_curve_types", {}).get(edge, "UNKNOWN"),
                    "wall_axis": engine._unit(wall.axis).tolist(), "skin_normal": normal.tolist(),
                    "wall_radius": wall.radius, "wall_u0": wall.u0, "wall_u1": wall.u1,
                    "wall_v0": wall.v0, "wall_v1": wall.v1, "wall_axial_span": span,
                    "known_thickness": thickness, "axis_normal_abs_dot": axis_alignment,
                    "axis_parallel_check": axis_alignment > 1. - engine.GEOM_TOL,
                    "span_thickness_residual": abs(span - thickness),
                    "span_matches_thickness_check": abs(span - thickness) <= engine.GEOM_TOL,
                    "skin_curve": _curve_summary(_curve(face, edge)), "opposite_rail_attempts": []}
                adjacent = set().union(*(owners[value] for value in wall.edges))
                translated = False
                for other_index in sorted(opposite & adjacent):
                    other = faces[other_index]
                    if other.kind != "plane":
                        continue
                    parallel = abs(float(np.dot(engine._unit(other.normal), normal)))
                    separation = float(np.dot(other.origin - face.origin, normal))
                    slab = (wall.vertices - face.origin) @ normal * math.copysign(1., separation)
                    for other_edge in sorted(set(other.edges) & set(wall.edges)):
                        counters["rail_attempts"] += 1
                        engine._require(counters["rail_attempts"] <= MAX_RAIL_ATTEMPTS,
                                        "DIAGNOSTIC_RAIL_ATTEMPT_BUDGET_EXCEEDED")
                        attempt = {"opposite_skin_face_index": other_index, "opposite_skin_edge_id": other_edge,
                            "opposite_native_edge_curve_type": types.get(other_index, {}).get("edge_curve_types", {}).get(other_edge, "UNKNOWN"),
                            "plane_normal_abs_dot": parallel,
                            "plane_parallel_check": parallel >= 1. - engine.GEOM_TOL,
                            "signed_plane_separation": separation,
                            "plane_separation_residual": abs(abs(separation) - thickness),
                            "plane_separation_matches_check": abs(abs(separation) - thickness) <= engine.GEOM_TOL,
                            "wall_slab_min": float(slab.min()), "wall_slab_max": float(slab.max()),
                            "wall_inside_thickness_slab_check": bool(slab.min() >= -engine.GEOM_TOL
                                                                     and slab.max() <= thickness + engine.GEOM_TOL)}
                        attempt["opposite_curve"] = _curve_summary(_curve(other, other_edge))
                        eligible = (attempt["plane_parallel_check"] and attempt["plane_separation_matches_check"]
                                    and attempt["wall_inside_thickness_slab_check"])
                        if eligible:
                            attempt["curve_certificate"] = _certificate(_curve(face, edge), _curve(other, other_edge),
                                                                       separation * normal, counters)
                            translated = bool(translated or attempt["curve_certificate"]["matched_by_frozen_engine"])
                        else:
                            attempt["curve_certificate"] = {"matched_by_frozen_engine": False,
                                                              "reason": "OPPOSITE_RAIL_GEOMETRY_PRECONDITION_FAILED"}
                        row["opposite_rail_attempts"].append(attempt)
                row["translated_opposite_rail_check"] = translated
                row["frozen_wall_boundary_checks_pass"] = bool(row["axis_parallel_check"]
                    and row["span_matches_thickness_check"] and translated)
                rows.append(row)
    return rows, counters


def _native_types(path: Path, faces: list[engine.SkinFace]) -> dict[int, dict[str, Any]]:
    """Metadata-only exact-snapshot read; replay the frozen edge-ID enumeration."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    reader = STEPControl_Reader()
    engine._require(reader.ReadFile(str(path)) == IFSelect_RetDone and reader.TransferRoots() > 0,
                    "DIAGNOSTIC_NATIVE_METADATA_READ_UNAVAILABLE")
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_FACE)
    result, edges, comparisons, uses = {}, [], 0, 0
    while explorer.More():
        index = len(result)
        engine._require(index < len(faces) and index < engine.MAX_FACES, "DIAGNOSTIC_NATIVE_FACE_ENUMERATION_MISMATCH")
        face = TopoDS.Face_s(explorer.Current())
        row = {"surface_type": str(BRepAdaptor_Surface(face).GetType()), "edge_curve_types": {}}
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            uses += 1
            engine._require(uses <= engine.MAX_EDGE_USES, "DIAGNOSTIC_NATIVE_EDGE_USE_BUDGET_EXCEEDED")
            edge = TopoDS.Edge_s(edge_explorer.Current())
            match = None
            for edge_id, previous in enumerate(edges):
                comparisons += 1
                engine._require(comparisons <= engine.MAX_EDGE_IDENTITY_COMPARISONS,
                                "DIAGNOSTIC_NATIVE_EDGE_IDENTITY_BUDGET_EXCEEDED")
                if edge.IsSame(previous):
                    match = edge_id
                    break
            if match is None:
                match = len(edges)
                edges.append(edge)
            row["edge_curve_types"][match] = str(BRepAdaptor_Curve(edge).GetType())
            edge_explorer.Next()
        engine._require(set(row["edge_curve_types"]) == set(faces[index].edges),
                        "DIAGNOSTIC_NATIVE_EDGE_ENUMERATION_MISMATCH")
        result[index] = row
        explorer.Next()
    engine._require(len(result) == len(faces), "DIAGNOSTIC_NATIVE_FACE_ENUMERATION_MISMATCH")
    return result


def diagnose_unfold_boundaries(cad_path: str | Path, expected_hash: str,
                               candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Observe the frozen boundary refusal through a source-bound bounded snapshot.

    The caller must supervise time/RSS. This does not run _develop_faces, does
    not inspect scans, and never emits AVAILABLE or any geometric preview.
    """
    payload = _empty()
    try:
        payload["engine_sha256"] = hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()
        payload["diagnostic_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        engine._require(payload["engine_sha256"] == FROZEN_ENGINE_SHA256, "FROZEN_ENGINE_SOURCE_HASH_MISMATCH")
        engine._require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None,
                        "INVALID_EXPECTED_CAD_HASH")
        engine._require(isinstance(candidates, list) and 0 < len(candidates) <= engine.MAX_CANDIDATES
                        and all(isinstance(row, dict) for row in candidates), "CANDIDATE_BUDGET_OR_SCHEMA_INVALID")
        path = Path(cad_path)
        engine._require(path.suffix.lower() in {".stp", ".step"} and path.is_file(), "STEP_SOURCE_REQUIRED")
        with path.open("rb") as source:
            data = source.read(engine.MAX_CAD_BYTES + 1)
        engine._require(0 < len(data) <= engine.MAX_CAD_BYTES, "CAD_BYTE_BUDGET_EXCEEDED")
        payload["source_sha256"] = hashlib.sha256(data).hexdigest()
        engine._require(payload["source_sha256"] == expected_hash, "CAD_HASH_MISMATCH")
        with tempfile.TemporaryDirectory(prefix="sherman-boundary-diagnostic-") as directory:
            snapshot = Path(directory) / "source.step"
            snapshot.write_bytes(data)
            faces, axes = engine._read_native(snapshot)
            links, adjacency = engine._skin_graph(faces, axes, candidates)
            root, selected = engine._select_skin(faces, links, adjacency, len(candidates))
            types = _native_types(snapshot, faces)
            payload["diagnostics"].update({"native_face_count": len(faces), "candidate_count": len(candidates),
                "candidate_feature_ids": [row["feature_id"] for row in candidates],
                "root_source_face_index": root,
                "complete_graph_component_face_indices": [sorted(component) for component in _components(adjacency)],
                "selected_cylinder_face_indices": sorted(link.cylinder for link in selected)})
            try:
                engine._verify_skin_boundary(faces, links, adjacency, selected)
                payload["engine_boundary_status"] = "PASS"
            except engine.UnfoldUnavailable as exc:
                payload["engine_boundary_status"] = "REFUSED"
                payload["engine_boundary_reason"] = str(exc)
            payload["boundary_checks"], counters = _boundary_records(faces, links, adjacency, selected, types)
            payload["diagnostics"]["diagnostic_counts"] = counters
            first_failed = next((i for i, row in enumerate(payload["boundary_checks"])
                                 if not row["frozen_wall_boundary_checks_pass"]), None)
            payload["diagnostics"]["first_failed_wall_record_index"] = first_failed
            payload["diagnostics"]["failure_localization_scope"] = "PLANAR_SKIN_TO_CURVED_BOUNDARY_WALL_ONLY"
            payload["status"] = "DIAGNOSTIC_ONLY"
        engine._require(len(json.dumps(payload, allow_nan=False, separators=(",", ":")).encode())
                        <= MAX_DIAGNOSTIC_JSON_BYTES, "DIAGNOSTIC_JSON_BUDGET_EXCEEDED")
    except engine.UnfoldUnavailable as exc:
        payload["status"], payload["reason"] = "UNAVAILABLE", str(exc)
        payload["boundary_checks"] = []
    except (OSError, ImportError, RuntimeError, TypeError, ValueError, KeyError, IndexError) as exc:
        payload["status"], payload["reason"] = "UNAVAILABLE", "NATIVE_BOUNDARY_DIAGNOSTIC_UNAVAILABLE"
        payload["boundary_checks"] = []
        payload["diagnostics"]["error_type"] = type(exc).__name__
    return payload


def diagnose_unfold_preview(cad_path: str | Path, expected_hash: str,
                            candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Runner-facing alias; diagnostic only, with no preview geometry."""
    return diagnose_unfold_boundaries(cad_path, expected_hash, candidates)
