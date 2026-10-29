import numpy as np

from research.f1_deterministic_cad_planes_v1 import canonical_plane_keys, extract_planes


def test_opposite_normal_orientation_maps_to_same_plane_key():
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    keys = canonical_plane_keys(points, normals)
    assert np.array_equal(keys[0], keys[1])


def test_extract_planes_is_order_invariant_and_area_ranked():
    xy = np.array([[x, y, 0.0] for x in range(10) for y in range(10)], dtype=float)
    yz = np.array([[0.0, y, z] for y in range(5) for z in range(5)], dtype=float)
    points = np.vstack((xy, yz))
    normals = np.vstack((np.tile([0.0, 0.0, 1.0], (len(xy), 1)), np.tile([1.0, 0.0, 0.0], (len(yz), 1))))
    forward = extract_planes(points, normals, minimum_count=10)
    order = np.arange(len(points))[::-1]
    reverse = extract_planes(points[order], normals[order], minimum_count=10)
    assert [row["canonical_key"] for row in forward] == [row["canonical_key"] for row in reverse]
    assert forward[0]["bbox_area"] > forward[1]["bbox_area"]
