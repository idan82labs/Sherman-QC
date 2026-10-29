"""Separately versioned finite-width visual CAD-skin development candidate.

V3 remains untouched. Its native importer, curve/surface ownership, two-skin
graph, boundary coverage, chart transforms, seam closure and whole-face overlap
functions are reused explicitly. Only the planar away-from-bend guard changes:
every native triangle is clipped to the finite expanded axial span by the pure
reviewed certificate. No triangles are removed, reconstructed, or filled.

This is selected CAD-skin visualization, not a K-factor/neutral-axis blank or
manufacturing qualification. Finite-width side proof concerns the supplied
native tessellation, not exact continuous curved trims. Native source access is
deferred until build_unfold_preview is explicitly called under a supervisor.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from backend import vendor_unfold_preview as legacy
from backend.vendor_unfold_finite_width import (
    FiniteWidthBudget, FiniteWidthUnavailable, certify_finite_width_flange,
)


MECHANISM_VERSION = 4
MODEL = legacy.MODEL
LEGACY_ENGINE_SOURCE_SHA256 = "8412778cee16d97eb70454cdc76904be939125c9fcacf4159d18c49e150f8f07"
FINITE_WIDTH_SOURCE_SHA256 = "11d8cc1b912797a9a14b15b0c64d900566f07eaa536bacd28428142ca55615a2"
FINITE_WIDTH_MODEL = "NATIVE_TESSELLATION_FINITE_AXIAL_SLAB"
UnfoldUnavailable = legacy.UnfoldUnavailable


def _envelope(source_hash: str | None, reason: str | None = None) -> dict[str, Any]:
    payload = legacy._envelope(source_hash, reason)
    payload["diagnostics"].update({
        "mechanism_version": MECHANISM_VERSION,
        "finite_width_model": FINITE_WIDTH_MODEL,
        "legacy_engine_source_sha256": LEGACY_ENGINE_SOURCE_SHA256,
        "finite_width_source_sha256": FINITE_WIDTH_SOURCE_SHA256,
    })
    payload["diagnostics"]["limitations"].extend([
        "FINITE_WIDTH_GUARD_IS_NATIVE_TESSELLATION_ONLY",
        "WHOLE_UNCHANGED_FACE_OVERLAP_REMAINS_REQUIRED",
        "TANGENT_SEAM_MUST_COVER_NATIVE_FINITE_AXIAL_SPAN",
    ])
    return payload


def unavailable_unfold_preview(reason: str) -> dict[str, Any]:
    """Explicit empty V4 envelope; never a V3 relabel or partial flat result."""
    return _envelope(None, reason)


def _side_certificate(face, cylinder, seam, signed_tangent, side, budget):
    try:
        return certify_finite_width_flange(
            face.vertices, face.triangles, cylinder_origin=cylinder.origin,
            cylinder_axis=cylinder.axis, tangent=signed_tangent,
            seam_endpoints=seam, axial_min=cylinder.v0, axial_max=cylinder.v1,
            side=side, budget=budget,
        )
    except FiniteWidthUnavailable as exc:
        raise UnfoldUnavailable(f"FINITE_WIDTH_{side.upper()}:{exc}") from exc


def _develop_faces(faces: list[legacy.SkinFace], axes: list[dict[str, Any]],
                   candidates: list[dict[str, Any]], source_hash: str) -> dict[str, Any]:
    """Pure chart seam; synthetic charts do not establish native solid validity."""
    links, adjacency = legacy._skin_graph(faces, axes, candidates)
    root, selected = legacy._select_skin(faces, links, adjacency, len(candidates))
    selected_ids = {link.cylinder for link in selected}
    selected_planes = {index for link in selected for index in link.planes}
    selected_edges = {edge for index in selected_planes for edge in faces[index].edges}
    # Exact V3 coplanar split refusal, followed by the unmodified all-face ledger.
    for face in faces:
        if face.kind != "plane" or face.index in selected_planes:
            continue
        if not selected_edges.intersection(face.edges):
            continue
        for index in selected_planes:
            plane = faces[index]
            if (set(plane.edges).intersection(face.edges)
                    and abs(np.dot(legacy._unit(face.normal), legacy._unit(plane.normal))) > 1.-legacy.GEOM_TOL
                    and abs(np.dot(face.origin-plane.origin, legacy._unit(plane.normal))) <= legacy.GEOM_TOL):
                raise UnfoldUnavailable("COPLANAR_SKIN_FACE_SPLITS_UNSUPPORTED")
    legacy._verify_skin_boundary(faces, links, adjacency, selected)
    root_face = faces[root]
    normal = legacy._unit(root_face.normal)
    first_link = next(link for link in adjacency[root] if link.cylinder in selected_ids)
    x_axis = legacy._unit(faces[first_link.cylinder].axis)
    rotation = np.stack((x_axis, legacy._unit(np.cross(normal, x_axis)), normal))
    translation = -rotation @ root_face.origin
    transforms = {root: (rotation, translation)}
    flat = {root: root_face.vertices @ rotation.T+translation}
    bend_lines, overlay_faces, done, stack = [], [], set(), [root]
    budget = FiniteWidthBudget()  # ONE shared budget, not one reset per side/bend.
    certificates = []
    while stack:
        parent = stack.pop()
        parent_rotation, parent_translation = transforms[parent]
        for link in adjacency[parent]:
            if link.cylinder not in selected_ids or link.cylinder in done:
                continue
            done.add(link.cylinder)
            child = next(index for index in link.planes if index != parent)
            legacy._require(child not in transforms, "CYCLIC_SKIN_GRAPH")
            cylinder = faces[link.cylinder]
            child_rotation, child_translation, delta, parent_u, tangent = legacy._child_transform(
                parent_rotation, parent_translation, cylinder, link.seams[parent], link.seams[child])
            sign = math.copysign(1., delta)
            parent_proof = _side_certificate(faces[parent], cylinder, link.seams[parent],
                                             tangent*sign, "parent", budget)
            child_tangent = legacy._rotation(cylinder.axis, delta) @ tangent
            child_proof = _side_certificate(faces[child], cylinder, link.seams[child],
                                            child_tangent*sign, "child", budget)
            certificates.append({"feature_id": link.feature_id,
                "cylinder_source_face_index": link.cylinder,
                "parent_source_face_index": parent, "child_source_face_index": child,
                "parent": parent_proof, "child": child_proof})
            child_flat = faces[child].vertices @ child_rotation.T+child_translation
            cylinder_flat = legacy._unwrap_cylinder(cylinder, parent_rotation, parent_translation, parent_u)
            # Same two-endpoint V3 native seam closure, without snapping inputs.
            for plane_id, plane_rotation, plane_translation, seam in (
                    (parent, parent_rotation, parent_translation, link.seams[parent]),
                    (child, child_rotation, child_translation, link.seams[child])):
                seam_u = float(legacy._cylinder_angles(cylinder, seam).mean())
                radial = math.cos(parent_u)*cylinder.x_axis+math.sin(parent_u)*cylinder.y_axis
                seam_axial = (seam-cylinder.origin) @ cylinder.axis
                arc_seam = (cylinder.origin+cylinder.radius*radial
                            + seam_axial[:, None]*cylinder.axis
                            + cylinder.radius*(seam_u-parent_u)*tangent)
                legacy._require(np.max(np.linalg.norm(
                    arc_seam @ parent_rotation.T+parent_translation
                    - (seam @ plane_rotation.T+plane_translation), axis=1)) <= legacy.GEOM_TOL,
                    "DEVELOPED_SEAM_CLOSURE_FAILED")
            transforms[child] = (child_rotation, child_translation)
            flat[child], flat[link.cylinder] = child_flat, cylinder_flat
            middle = cylinder.origin+cylinder.radius*(
                math.cos(parent_u)*cylinder.x_axis+math.sin(parent_u)*cylinder.y_axis)
            line = np.stack([
                middle+cylinder.v0*cylinder.axis+cylinder.radius*delta/2*tangent,
                middle+cylinder.v1*cylinder.axis+cylinder.radius*delta/2*tangent])
            line = line @ parent_rotation.T+parent_translation
            bend_lines.append({"feature_id": link.feature_id, "line_start": line[0].tolist(),
                               "line_end": line[1].tolist(), "bend_direction": "unknown",
                               "sweep_deg": math.degrees(abs(delta))})
            overlay_faces.append((link.feature_id, child))
            stack.append(child)
    legacy._require(len(done) == len(candidates), "INCOMPLETE_SELECTED_SKIN")
    for points in flat.values():
        legacy._require(np.isfinite(points).all() and np.max(np.abs(points[:, 2])) <= legacy.GEOM_TOL,
                        "DEVELOPED_SKIN_NOT_PLANAR")
        points[:, 2] = 0.
    for line in bend_lines:
        legacy._require(max(abs(line["line_start"][2]), abs(line["line_end"][2])) <= legacy.GEOM_TOL,
                        "DEVELOPED_BEND_LINE_NOT_PLANAR")
        line["line_start"][2] = line["line_end"][2] = 0.
    # Crucially: this consumes the ENTIRE native mesh, not clipped polygons.
    legacy._check_overlap(flat, faces)
    payload = _envelope(source_hash)
    vertex_offset = 0
    for index in sorted(flat):
        payload["vertices"].extend(flat[index].reshape(-1).tolist())
        payload["triangles"].extend((faces[index].triangles+vertex_offset).reshape(-1).tolist())
        vertex_offset += len(flat[index])
    payload["bend_lines"] = sorted(bend_lines, key=lambda row: row["feature_id"])
    # Construct overlays AFTER the same Z projection, retaining exact body XY
    # and the complete child triangulation; no pre-projection residual mismatch.
    payload["flange_surfaces"] = [
        {"feature_id": feature_id, "vertices": flat[index].reshape(-1).tolist(),
         "triangles": faces[index].triangles.reshape(-1).tolist()}
        for feature_id, index in sorted(overlay_faces)]
    payload["diagnostics"].update({"root_source_face_index": root,
        "selected_source_face_indices": sorted(flat), "selected_cylinder_face_indices": sorted(done),
        "candidate_count": len(candidates), "native_face_count": len(faces),
        "selected_skin_rule": "LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN",
        "seam_closure_checked": True, "native_surface_ownership_checked": True,
        "cross_face_overlap_checked": True, "native_surface_coverage_checked": True,
        "excluded_caps_and_hole_walls_checked": True,
        "direction_status": "UNKNOWN_NATIVE_NORMAL_SIGN_NOT_AUTHORITY",
        "finite_width_clipping_checked": True, "finite_width_native_seam_span_checked": True,
        "finite_width_proofs": sorted(certificates, key=lambda row: row["feature_id"]),
        "finite_width_work": budget.snapshot(),
        "whole_unchanged_mesh_overlap_checked": True})
    return payload


def build_unfold_preview(cad_path: str | Path, expected_hash: str,
                         candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Deferred bounded V3 byte/snapshot import feeding the V4 pure seam.

    The enclosing supervisor must pin this module, the exact legacy/helper
    bytes and native dependencies and enforce process-tree time/memory limits.
    Source-hash strings in diagnostics describe the required source pins, not
    an import-time cryptographic check or authority to access any physical CAD.
    """
    source_hash = None
    try:
        legacy._require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None,
                        "INVALID_EXPECTED_CAD_HASH")
        legacy._require(isinstance(candidates, list) and 0 < len(candidates) <= legacy.MAX_CANDIDATES
                        and all(isinstance(row, dict) for row in candidates), "CANDIDATE_BUDGET_OR_SCHEMA_INVALID")
        path = Path(cad_path)
        legacy._require(path.suffix.lower() in {".stp", ".step"} and path.is_file(), "STEP_SOURCE_REQUIRED")
        with path.open("rb") as source:
            data = source.read(legacy.MAX_CAD_BYTES+1)
        legacy._require(0 < len(data) <= legacy.MAX_CAD_BYTES, "CAD_BYTE_BUDGET_EXCEEDED")
        source_hash = hashlib.sha256(data).hexdigest()
        legacy._require(source_hash == expected_hash, "CAD_HASH_MISMATCH")
        with tempfile.TemporaryDirectory(prefix="sherman-unfold-v4-") as directory:
            snapshot = Path(directory)/"source.step"
            snapshot.write_bytes(data)
            faces, axes = legacy._read_native(snapshot)
            return _develop_faces(faces, axes, candidates, source_hash)
    except UnfoldUnavailable as exc:
        return _envelope(source_hash, str(exc))
    except (OSError, ImportError, RuntimeError, TypeError, ValueError, KeyError, IndexError) as exc:
        payload = _envelope(source_hash, "NATIVE_DEVELOPMENT_UNAVAILABLE")
        payload["diagnostics"]["error_type"] = type(exc).__name__
        return payload
