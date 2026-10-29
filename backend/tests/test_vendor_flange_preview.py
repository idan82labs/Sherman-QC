"""Source-free safety checks for the optional CAD flange visual association."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.vendor_flange_preview import (
    FlangePreviewUnavailable, _matching_axis, _mesh_payload, _straight_parallel_outer_depth,
    _unique_axis_matches,
    build_flange_previews,
)


AXIS = {"canonical_line_start": [0, 0, 0], "canonical_line_end": [10, 0, 0]}


def test_line_bridge_is_order_independent_and_rejects_ambiguity():
    assert _matching_axis({"line_start": [10, 0, 0], "line_end": [0, 0, 0]}, [AXIS]) == 0
    assert _matching_axis({"line_start": [10, 1, 0], "line_end": [0, 1, 0]}, [AXIS]) is None
    assert _matching_axis({"line_start": [0, 0, 0], "line_end": [10, 0, 0]}, [AXIS, AXIS]) is None
    assert _matching_axis({"line_start": [float("nan"), 0, 0], "line_end": [10, 0, 0]}, [AXIS]) is None
    duplicate = {"line_start": [0, 0, 0], "line_end": [10, 0, 0]}
    assert _unique_axis_matches([duplicate, duplicate], [AXIS]) == [None, None]


def test_overlay_contains_only_selected_face_triangles():
    mesh = SimpleNamespace(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [4, 0, 0], [5, 0, 0], [4, 1, 0]]),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
    )
    sidecar = {"face_records": [{"triangle_range": [0, 1]}, {"triangle_range": [1, 2]}]}
    payload = _mesh_payload(mesh, sidecar, [1])
    assert payload["source_face_indices"] == [1]
    assert len(payload["triangles"]) == 3
    assert set(payload["vertices"][::3]) == {4.0, 5.0}


def test_hash_mismatch_refuses_import_before_geometry(tmp_path):
    path = tmp_path / "invented.stp"
    path.write_bytes(b"synthetic-not-step")
    with pytest.raises(FlangePreviewUnavailable, match="CAD_HASH_MISMATCH"):
        build_flange_previews(path, "0" * 64, [])


def test_nominal_depth_uses_tangent_not_neutral_axis_and_accepts_chamfered_ends():
    seam = np.array([[0., 0., 0.], [100., 0., 0.]])
    rail = np.array([[95., 17.7, 0.], [5., 17.7, 0.]])
    assert _straight_parallel_outer_depth(seam, [rail], np.array([1., 0., 0.])) == pytest.approx(17.7)
    # A canonical mid-bend axis at y=-3 would instead give a wrong 20.7 mm.


@pytest.mark.parametrize("rails", [
    [],
    [np.array([[0., 10., 0.], [100., 11., 0.]])],  # Taper: no constant height.
    [np.array([[0., 10., 0.], [50., 10., 0.]])],  # Partial trimmed outer edge.
    [np.array([[0., 10., 0.], [100., 10., 0.]]),
     np.array([[0., 20., 0.], [100., 20., 0.]])],  # Competing full rails.
    [np.array([[0., 0., 0.], [100., 0., 0.]])],  # Not an opposite edge.
])
def test_nominal_depth_abstains_without_unique_straight_full_outer_rail(rails):
    seam = np.array([[0., 0., 0.], [100., 0., 0.]])
    assert _straight_parallel_outer_depth(seam, rails, np.array([1., 0., 0.])) is None


def test_nominal_depth_rejects_unaligned_seam_or_nonfinite_geometry():
    assert _straight_parallel_outer_depth(
        np.array([[0., 0., 0.], [100., 1., 0.]]),
        [np.array([[0., 10., 0.], [100., 10., 0.]])],
        np.array([1., 0., 0.]),
    ) is None
    assert _straight_parallel_outer_depth(
        np.array([[0., 0., 0.], [100., 0., 0.]]),
        [np.array([[0., np.nan, 0.], [100., 10., 0.]])],
        np.array([1., 0., 0.]),
    ) is None
