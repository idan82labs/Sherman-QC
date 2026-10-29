import numpy as np

from scripts.generate_f1_psroc_corpus_matched_prospective_v1 import BUILDERS, FAMILIES, build_case


def test_all_six_fresh_families_build_finite_non_repetitive_geometry():
    assert len(BUILDERS) == len(FAMILIES) == 6
    names = set()
    for family in FAMILIES:
        mesh, truth, metadata = build_case(family, 17)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert triangles.ndim == 2 and triangles.shape[1] == 3
        assert len(vertices) > 100 and len(triangles) > 100
        assert np.isfinite(vertices).all()
        assert 2 <= len(truth) <= 4
        assert metadata["v5r3b_geometry_reused"] is False
        assert metadata["dense_repeated_vent_pattern"] is False
        assert metadata["broad_roll_negative"] is False
        assert metadata["family_name"] not in names
        names.add(metadata["family_name"])


def test_truth_lines_are_finite_nonzero_and_evaluator_only_by_contract():
    for family in FAMILIES:
        _mesh, truth, _metadata = build_case(family, 17)
        for row in truth:
            start = np.asarray(row["line_start"], dtype=float)
            end = np.asarray(row["line_end"], dtype=float)
            assert np.isfinite(start).all() and np.isfinite(end).all()
            assert np.linalg.norm(end - start) > 5.0
