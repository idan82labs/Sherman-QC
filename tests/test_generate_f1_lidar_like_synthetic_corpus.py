import numpy as np

from scripts.generate_f1_lidar_like_synthetic_corpus import FAMILIES, TARGET_MAX_VERTICES, build_case


def test_all_synthetic_families_preserve_expected_bends_and_resource_bound():
    for family in FAMILIES:
        expected = len(family["profile"]) - 2
        for capture in range(2):
            mesh, truth, metadata = build_case(family, capture)
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            assert len(vertices) <= TARGET_MAX_VERTICES
            assert len(vertices) > 5_000
            assert len(triangles) > 5_000
            assert len(truth) == expected
            assert metadata["bend_count"] == expected
            assert 0.30 <= metadata["spacing_mm"] <= 0.85
            for row in truth:
                start = np.asarray(row["line_start"])
                end = np.asarray(row["line_end"])
                assert np.linalg.norm(end - start) > 100.0


def test_second_capture_contains_controlled_deviation():
    _mesh, _truth, metadata = build_case(FAMILIES[4], 1)
    assert metadata["deviation"]["kind"] == "controlled_vertex_offset"
    assert np.linalg.norm(metadata["deviation"]["offset_mm"]) >= 2.0
