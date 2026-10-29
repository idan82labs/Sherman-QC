"""Pure native-edge coplanar flange chart stitching, separately versioned.

This is a topology normalization step, not a development or availability gate.
It reads no CAD/files, fits no planes, and removes no source mesh material.
Every face/vertex/triangle remains attributable through the returned ledger.
Only edges owned by two faces in the same proven planar component disappear
from that component's exterior boundary. Native cylinders remain singleton
charts, and the caller must still prove both skins, finite bend support, chart
closure, whole-mesh overlap and complete source material coverage.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend import vendor_unfold_preview as native


MECHANISM_VERSION = 1
MAX_STITCH_OVERLAP_COMPARISONS = native.MAX_OVERLAP_COMPARISONS


@dataclass
class StitchedCharts:
    faces: list[native.SkinFace]
    axes: list[dict[str, Any]]
    source_to_chart: dict[int, int]
    ledger: dict[str, Any]


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise native.UnfoldUnavailable(reason)


def _coplanar(left: native.SkinFace, right: native.SkinFace) -> bool:
    if left.kind != "plane" or right.kind != "plane":
        return False
    a, b = native._unit(left.normal), native._unit(right.normal)
    # Opposite analytic chart normals can represent the same support plane.
    # This says nothing about native face orientation or triangle winding.
    if min(np.linalg.norm(a-b), np.linalg.norm(a+b)) > native.GEOM_TOL:
        return False
    return (np.max(np.abs((right.vertices-left.origin) @ a)) <= native.GEOM_TOL
            and np.max(np.abs((left.vertices-right.origin) @ b)) <= native.GEOM_TOL)


def _consistent_rail(left: native.SkinFace, right: native.SkinFace, edge: int) -> bool:
    a, b = left.edges[edge], right.edges[edge]
    if a is not None and b is not None:
        a, b = np.asarray(a), np.asarray(b)
        return (a.shape == b.shape == (2, 3) and np.isfinite(a).all()
                and np.isfinite(b).all()
                and min(np.linalg.norm(a-b, axis=1).max(),
                        np.linalg.norm(a-b[::-1], axis=1).max()) <= native.GEOM_TOL
                and np.linalg.norm(a[1]-a[0]) > native.GEOM_TOL)
    if a is not None or b is not None:
        return False
    first, second = left.edge_curves.get(edge), right.edge_curves.get(edge)
    return (first is not None and second is not None
            and native._translated_curves(first, second, np.zeros(3)))


def _check_component_overlap(component: list[int], faces: list[native.SkinFace],
                             charged: list[int]) -> None:
    if len(component) < 2:
        return
    anchor = faces[component[0]]
    normal = native._unit(anchor.normal)
    # Select the coordinate axis least aligned with the normal, then project
    # to a proper orthonormal plane chart. No source coordinate is changed.
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    x = native._unit(np.cross(axis, normal))
    y = np.cross(normal, x)
    basis = np.stack((x, y))
    charts = {i: (faces[i].vertices-anchor.origin) @ basis.T for i in component}
    for offset, left_id in enumerate(component):
        left = charts[left_id][faces[left_id].triangles]
        low, high = left.min(axis=1), left.max(axis=1)
        for right_id in component[offset+1:]:
            right = charts[right_id][faces[right_id].triangles]
            rlow, rhigh = right.min(axis=1), right.max(axis=1)
            if np.any(high.max(axis=0) <= rlow.min(axis=0)) or np.any(
                    rhigh.max(axis=0) <= low.min(axis=0)):
                continue
            order = np.argsort(rlow[:, 0])
            sorted_low = rlow[order, 0]
            for i, triangle in enumerate(left):
                possible = order[:np.searchsorted(sorted_low, high[i, 0])]
                possible = possible[(rhigh[possible, 0] > low[i, 0])
                                    & (rhigh[possible, 1] > low[i, 1])
                                    & (rlow[possible, 1] < high[i, 1])]
                charged[0] += len(possible)
                _require(charged[0] <= MAX_STITCH_OVERLAP_COMPARISONS,
                         "COPLANAR_STITCH_OVERLAP_BUDGET_EXCEEDED")
                for j in possible:
                    _require(native._triangle_intersection_area(triangle, right[j]) <= native.GEOM_TOL**2,
                             "OVERLAPPING_COPLANAR_SOURCE_FACES")


def stitch_coplanar_native_faces(faces: list[native.SkinFace],
                                axes: list[dict[str, Any]]) -> StitchedCharts:
    """Return detached flange supernodes and an explicit raw-face lineage.

    Input face IDs must be dense import-local zero-based IDs; axis ownership
    uses the importer's one-based convention. Sharing positions alone never
    connects charts. Global native edge identity and degree <=2 are required.
    The original largest-root policy can be applied to total supernode area.
    """
    _require(isinstance(faces, list) and 0 < len(faces) <= native.MAX_FACES
             and [f.index for f in faces] == list(range(len(faces))),
             "COPLANAR_STITCH_FACE_ENUMERATION_INVALID")
    _require(sum(len(f.triangles) for f in faces) <= native.MAX_TRIANGLES
             and sum(len(f.vertices) for f in faces) <= 3*native.MAX_TRIANGLES,
             "COPLANAR_STITCH_GEOMETRY_BUDGET_EXCEEDED")
    owners: dict[int, list[int]] = {}
    for face in faces:
        native._validate_face(face)
        for edge in face.edges:
            _require(type(edge) is int and edge >= 0, "INVALID_NATIVE_EDGE_ID")
            owners.setdefault(edge, []).append(face.index)
    _require(sum(len(f.edges) for f in faces) <= native.MAX_EDGE_USES
             and all(len(value) <= 2 for value in owners.values()),
             "NON_MANIFOLD_EDGE_INCIDENCE")
    parent = list(range(len(faces)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    stitches = []
    for edge, ids in sorted(owners.items()):
        if len(ids) != 2 or not _coplanar(faces[ids[0]], faces[ids[1]]):
            continue
        _require(_consistent_rail(faces[ids[0]], faces[ids[1]], edge),
                 "COPLANAR_STITCH_SHARED_RAIL_MISMATCH")
        left, right = map(root, ids)
        parent[max(left, right)] = min(left, right)
        stitches.append({"edge_id": edge, "source_face_indices": sorted(ids)})
    groups: dict[int, list[int]] = {}
    for index in range(len(faces)):
        groups.setdefault(root(index), []).append(index)
    charts, mapping, records, charged = [], {}, [], [0]
    for ids in sorted(groups.values(), key=lambda group: group[0]):
        anchor = faces[ids[0]]
        # Pairwise near-coplanarity is not transitive; accumulated drift must
        # satisfy the SAME anchor plane before any grouping is returned.
        _require(all(_coplanar(anchor, faces[i]) for i in ids) if len(ids) > 1 else True,
                 "COPLANAR_STITCH_COMPONENT_PLANE_DRIFT")
        _check_component_overlap(ids, faces, charged)
        chart = copy.deepcopy(anchor)
        chart.index = len(charts)
        chart.edges, chart.edge_curves = {}, {}
        vertices, triangles, members, vertex_count, triangle_count = [], [], [], 0, 0
        inside = set(ids)
        removed = sorted(edge for edge, edge_owners in owners.items()
                         if len(edge_owners) == 2 and set(edge_owners).issubset(inside))
        for source_id in ids:
            face = faces[source_id]
            mapping[source_id] = chart.index
            vertices.append(face.vertices.copy())
            triangles.append(face.triangles.copy()+vertex_count)
            members.append({"source_face_index": source_id,
                "vertex_start": vertex_count, "vertex_count": len(face.vertices),
                "triangle_start": triangle_count, "triangle_count": len(face.triangles)})
            vertex_count += len(face.vertices)
            triangle_count += len(face.triangles)
            for edge, rail in face.edges.items():
                if edge in removed:
                    continue
                _require(edge not in chart.edges, "DUPLICATE_STITCHED_EXTERIOR_EDGE")
                chart.edges[edge] = None if rail is None else rail.copy()
                if edge in face.edge_curves:
                    chart.edge_curves[edge] = copy.deepcopy(face.edge_curves[edge])
        chart.vertices, chart.triangles = np.vstack(vertices), np.vstack(triangles)
        native._validate_face(chart)
        charts.append(chart)
        records.append({"chart_face_index": chart.index, "kind": chart.kind,
            "source_face_indices": ids, "members": members,
            "removed_internal_edge_ids": removed,
            "exterior_edge_ids": sorted(chart.edges)})
    remapped_axes = copy.deepcopy(axes)
    for axis in remapped_axes:
        indices = axis.get("face_indices")
        _require(type(indices) is list and len(indices) == 2
                 and all(type(i) is int and 1 <= i <= len(faces) for i in indices)
                 and len(set(indices)) == 2
                 and all(faces[i-1].kind == "cylinder" for i in indices),
                 "COPLANAR_STITCH_AXIS_OWNERSHIP_INVALID")
        axis["face_indices"] = [mapping[i-1]+1 for i in indices]
    return StitchedCharts(charts, remapped_axes, mapping, {
        "mechanism_version": MECHANISM_VERSION,
        "model": "NATIVE_EDGE_CONNECTED_COPLANAR_FLANGE_SUPERNODES",
        "native_face_count": len(faces), "chart_face_count": len(charts),
        "source_vertices_preserved": True, "source_triangles_preserved": True,
        "source_winding_preserved": True, "geometry_fitted": False,
        "manufacturing_ready": False, "stitches": stitches, "charts": records,
        "overlap_comparisons": charged[0],
        "required_external_gates": ["native_solid_validity_and_edge_identity",
            "two_complete_skin_layers_and_all_source_material_roles",
            "finite_width_seam_and_side_certificates", "chart_closure_and_orientation",
            "whole_developed_mesh_overlap"]})
