from __future__ import annotations

import numpy as np
import open3d as o3d

from backend.f1_paired_skin_ribbon_ownership_v18 import (
    MANDATORY_MASK,
    PROBE_RECORD_BYTES,
    NativeMeshProbeContext,
    ProbeRecord,
    SideProbe,
    _canonical_unit,
    collapse_ribbon_duplicates,
    probe_candidate,
)


def _extruded_polygon_mesh(polygon_yz, *, length=80.0, steps=32):
    polygon = np.asarray(polygon_yz, dtype=np.float64)
    xs = np.linspace(0.0, length, steps + 1)
    vertices = np.asarray(
        [[x, y, z] for x in xs for y, z in polygon], dtype=np.float64
    )
    width = len(polygon)
    triangles = []
    for xi in range(steps):
        for edge in range(width):
            nxt = (edge + 1) % width
            a = xi * width + edge
            b = xi * width + nxt
            c = (xi + 1) * width + nxt
            d = (xi + 1) * width + edge
            # Outward winding for a counter-clockwise yz polygon.
            triangles.extend(((a, b, c), (a, c, d)))
    # End caps are intentionally triangulated fans; they are real bounded
    # faces and must not be treated as principal skins by the probe.
    for xi, reverse in ((0, True), (steps, False)):
        base = xi * width
        for edge in range(1, width - 1):
            tri = (base, base + edge, base + edge + 1)
            triangles.append(tuple(reversed(tri)) if reverse else tri)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.compute_triangle_normals()
    return mesh


def _owner_by_normal_and_centroid(context, normal, centroid_axis, centroid_value):
    target = np.asarray(normal, dtype=np.float64)
    ranked = []
    for owner, surface in context.surfaces.items():
        cosine = abs(float(np.dot(surface.normal, target)))
        residual = abs(float(surface.centroid[centroid_axis]) - centroid_value)
        ranked.append((-cosine, residual, owner))
    return min(ranked)[2]


def _folded_ribbon_context():
    # L-shaped solid cross section, extruded along x.  The two long outer
    # skins meet at y=z=0; the two inner skins meet at y=z=2.
    mesh = _extruded_polygon_mesh(
        ((0.0, 0.0), (30.0, 0.0), (30.0, 2.0), (2.0, 2.0), (2.0, 30.0), (0.0, 30.0))
    )
    return NativeMeshProbeContext(mesh)


def _transform_mesh(mesh, rotation, translation):
    transformed = o3d.geometry.TriangleMesh(mesh)
    vertices = np.asarray(transformed.vertices)
    transformed.vertices = o3d.utility.Vector3dVector(vertices @ rotation.T + translation)
    transformed.compute_triangle_normals()
    return transformed


def test_fixed_record_is_exactly_224_bytes_and_canonical_direction():
    side = SideProbe(1, 3, MANDATORY_MASK | (1 << 1), (2, 2, 2, 2, 0), 30, True)
    record = ProbeRecord("c", (1, 2), (side, SideProbe(2, 4, side.mask, side.thickness, 30, True)))
    assert len(record.encode()) == PROBE_RECORD_BYTES
    np.testing.assert_array_equal(_canonical_unit((-1, 0, 0)), (1, 0, 0))


def test_finite_thickness_folded_ribbon_retains_exactly_one_canonical_ribbon():
    context = _folded_ribbon_context()
    owner_a = _owner_by_normal_and_centroid(context, (0, 0, 1), 2, 0.0)
    owner_b = _owner_by_normal_and_centroid(context, (0, 1, 0), 1, 0.0)
    row = {
        "candidate_id": "outer-fold",
        "surface1_id": owner_a,
        "surface2_id": owner_b,
        "span_endpoints": [[5.0, 0.0, 0.0], [75.0, 0.0, 0.0]],
    }
    forward = probe_candidate(context, row)
    reverse = probe_candidate(
        context,
        {**row, "candidate_id": "outer-fold-reverse", "surface1_id": owner_b, "surface2_id": owner_a},
    )
    assert forward.retain, forward.to_summary()
    assert reverse.retain, reverse.to_summary()
    assert len(collapse_ribbon_duplicates((reverse, forward))) == 1
    assert all(side.mask & MANDATORY_MASK == MANDATORY_MASK for side in forward.sides)
    assert len(set((*forward.near_owners, *(side.opposite_owner for side in forward.sides)))) == 4


def test_top_skin_to_thickness_perimeter_edge_is_rejected():
    context = _folded_ribbon_context()
    top = _owner_by_normal_and_centroid(context, (0, 0, 1), 2, 0.0)
    end = _owner_by_normal_and_centroid(context, (0, 1, 0), 1, 30.0)
    record = probe_candidate(
        context,
        {
            "candidate_id": "perimeter",
            "surface1_id": top,
            "surface2_id": end,
            "span_endpoints": [[5.0, 30.0, 0.0], [75.0, 30.0, 0.0]],
        },
    )
    assert not record.retain


def test_miter_without_conjugate_transition_is_rejected():
    # A triangular wedge has two broad meeting faces but no paired inner
    # transition.  It must not instantiate a four-owner ribbon.
    context = NativeMeshProbeContext(
        _extruded_polygon_mesh(((0.0, 0.0), (30.0, 0.0), (0.0, 30.0)))
    )
    broad = sorted(context.surfaces.values(), key=lambda surface: -surface.area)[:2]
    record = probe_candidate(
        context,
        {
            "candidate_id": "single-miter",
            "surface1_id": broad[0].surface_id,
            "surface2_id": broad[1].surface_id,
            "span_endpoints": [[5.0, 0.0, 0.0], [75.0, 0.0, 0.0]],
        },
    )
    assert not record.retain


def test_rigid_transforms_preserve_semantic_decision():
    source_mesh = _extruded_polygon_mesh(
        ((0.0, 0.0), (30.0, 0.0), (30.0, 2.0), (2.0, 2.0), (2.0, 30.0), (0.0, 30.0))
    )
    source_endpoints = np.asarray([[5.0, 0.0, 0.0], [75.0, 0.0, 0.0]])
    for degrees, translation in (
        (17.0, np.asarray((11.0, -7.0, 23.0))),
        (53.0, np.asarray((-19.0, 31.0, -5.0))),
        (101.0, np.asarray((7.0, 13.0, -29.0))),
    ):
        angle = np.radians(degrees)
        rotation = np.asarray(
            ((np.cos(angle), -np.sin(angle), 0.0),
             (np.sin(angle), np.cos(angle), 0.0),
             (0.0, 0.0, 1.0))
        )
        context = NativeMeshProbeContext(_transform_mesh(source_mesh, rotation, translation))
        transformed_endpoints = source_endpoints @ rotation.T + translation
        # Owners are selected by the transformed near-skin normals.
        top_normal = np.asarray((0.0, 0.0, 1.0)) @ rotation.T
        side_normal = np.asarray((0.0, 1.0, 0.0)) @ rotation.T
        owner_a = max(context.surfaces, key=lambda owner: abs(float(np.dot(context.surfaces[owner].normal, top_normal))))
        owner_b = max(
            (owner for owner in context.surfaces if owner != owner_a),
            key=lambda owner: abs(float(np.dot(context.surfaces[owner].normal, side_normal))),
        )
        record = probe_candidate(
            context,
            {
                "candidate_id": f"rigid-{degrees}",
                "surface1_id": owner_a,
                "surface2_id": owner_b,
                "span_endpoints": transformed_endpoints.tolist(),
            },
        )
        assert record.retain, record.to_summary()


def test_whole_component_face_orientation_reversal_preserves_decision():
    mesh = _extruded_polygon_mesh(
        ((0.0, 0.0), (30.0, 0.0), (30.0, 2.0), (2.0, 2.0), (2.0, 30.0), (0.0, 30.0))
    )
    triangles = np.asarray(mesh.triangles).copy()[:, ::-1]
    reversed_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy()),
        o3d.utility.Vector3iVector(triangles),
    )
    reversed_mesh.compute_triangle_normals()
    context = NativeMeshProbeContext(reversed_mesh)
    owner_a = _owner_by_normal_and_centroid(context, (0, 0, 1), 2, 0.0)
    owner_b = _owner_by_normal_and_centroid(context, (0, 1, 0), 1, 0.0)
    record = probe_candidate(
        context,
        {
            "candidate_id": "orientation-reversal",
            "surface1_id": owner_a,
            "surface2_id": owner_b,
            "span_endpoints": [[5.0, 0.0, 0.0], [75.0, 0.0, 0.0]],
        },
    )
    assert record.retain, record.to_summary()
