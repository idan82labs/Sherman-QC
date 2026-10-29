from __future__ import annotations

import numpy as np

from scripts.f1_tftoc_v66_protocol import (
    ordinary_final_nms,
    order_winding_variant,
    pl_refinement_variant,
    similarity_variant,
)


def _row(route: str, start: list[float], end: list[float], score: float) -> dict:
    return {
        "bend_id": route,
        "route_hash": route,
        "better_first_key": [-score, route],
        "selector_score": score,
        "line_start": start,
        "line_end": end,
    }


def test_similarity_variant_applies_exact_locked_map_to_mesh_and_candidates():
    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (4.0, 6.0, 8.0)))
    rows = [_row("A", [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 1.0)]
    mapped, matrix, mapped_rows = similarity_variant(vertices, rows)
    diagonal = np.linalg.norm(np.asarray((4.0, 6.0, 8.0)))
    translation = np.asarray((8.0 * diagonal, -4.0 * diagonal, 2.0 * diagonal))
    expected = np.asarray((3.0, -1.0, -2.0)) * 0.8 + translation
    assert np.allclose(mapped[1], expected, atol=0.0, rtol=0.0)
    assert np.allclose(mapped_rows[0]["line_end"], expected, atol=0.0, rtol=0.0)
    assert mapped_rows[0]["route_hash"] == "A"
    assert np.allclose(matrix @ np.asarray((1.0, 2.0, 3.0)), expected - translation)


def test_order_winding_variant_preserves_coordinate_and_unoriented_face_sets():
    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    faces = np.asarray(((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)), dtype=np.int64)
    mapped_vertices, mapped_faces = order_winding_variant(vertices[::-1], 3 - faces[::-1])
    assert {tuple(row) for row in mapped_vertices} == {tuple(row) for row in vertices}
    original = {frozenset(tuple(vertices[index]) for index in face) for face in faces}
    observed = {frozenset(tuple(mapped_vertices[index]) for index in face) for face in mapped_faces}
    assert observed == original


def test_pl_refinement_is_exact_and_changes_only_selected_face_partition():
    vertices = np.asarray(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0)))
    faces = np.repeat(np.asarray(((0, 1, 2),), dtype=np.int64), 128, axis=0)
    mapped_vertices, mapped_faces, count = pl_refinement_variant(vertices, faces)
    assert count == 1
    assert len(mapped_vertices) == 4
    assert len(mapped_faces) == 130
    assert np.array_equal(mapped_vertices[-1], np.asarray((1.0, 1.0, 0.0)))


def test_ordinary_final_nms_is_better_first_and_called_as_one_stage():
    rows = [
        _row("LOW", [0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 0.2),
        _row("HIGH", [0.0, 1.0, 0.0], [100.0, 1.0, 0.0], 0.9),
        _row("DISTINCT", [0.0, 20.0, 0.0], [100.0, 20.0, 0.0], 0.5),
    ]
    accepted, audit = ordinary_final_nms(rows)
    assert [row["route_hash"] for row in accepted] == ["HIGH", "DISTINCT"]
    assert next(row for row in audit if row["route_hash"] == "LOW")["duplicate_of"] == "HIGH"
