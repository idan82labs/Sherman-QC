from __future__ import annotations

import numpy as np

from scripts.generate_f1_real_scan_affine_resample_robustness_v1 import (
    MAX_VERTICES,
    VARIANTS,
    affine_matrix,
    transform_lines,
    transform_points,
)


def test_affine_is_finite_invertible_and_near_identity_scale() -> None:
    for spec in VARIANTS:
        matrix = affine_matrix(spec)
        singular = np.linalg.svd(matrix, compute_uv=False)
        assert np.isfinite(matrix).all()
        assert abs(np.linalg.det(matrix)) > 0.8
        assert singular.min() > 0.93
        assert singular.max() < 1.08
        assert int(spec["target_vertices"]) <= MAX_VERTICES


def test_line_transform_matches_endpoint_transform() -> None:
    spec = VARIANTS[0]
    matrix = affine_matrix(spec)
    center = np.asarray((10.0, -5.0, 2.0))
    translation = np.asarray((3.0, 4.0, -8.0))
    prediction = {
        "bend_id": "B1",
        "line_start": [0.0, 1.0, 2.0],
        "line_end": [8.0, 4.0, -1.0],
    }
    transformed = transform_lines([prediction], matrix, center, translation)[0]
    expected = transform_points(
        np.asarray((prediction["line_start"], prediction["line_end"])), matrix, center, translation
    )
    assert transformed["line_id"] == "PT01"
    assert transformed["source_reference_id"] == "B1"
    np.testing.assert_allclose(transformed["line_start"], expected[0], atol=1e-9)
    np.testing.assert_allclose(transformed["line_end"], expected[1], atol=1e-9)
