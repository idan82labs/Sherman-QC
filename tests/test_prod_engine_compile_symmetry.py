"""E1 compile: stage symmetry test (plan §10 amendment 5) with mutate-and-check cases."""
from __future__ import annotations

import numpy as np
import pytest

from domains.production.engine.compile import compile_part
from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.dev_recipe import dev_recipe
from domains.production.engine.compile.symmetry import symmetry_test


def _compile(tmp, shape, info, sku):
    step = tp.write_step(shape, tmp / f"{sku}.step")
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku=sku)
    c = compile_part(recipe, step, tmp / f"out_{sku}").compiled
    assert c.status == "COMPILED", c.blocker_codes
    return c


def _rotations(stage, threshold=0.3):
    return [cand for cand in stage.symmetry.candidates if cand.kind == "ROTATION" and cand.rms_mm <= threshold]


def test_symmetric_channel_reports_the_bend_swap_as_ambiguous(tmp_path):
    shape, info = tp.u_part(base=50.0, h1=25.0, h2=25.0, width=40.0)
    c = _compile(tmp_path, shape, info, "SYN-USYM")
    final = c.stages[-1]
    b1, b2 = (b.bend_uid for b in c.bends)
    rots = _rotations(final)
    assert final.symmetry.is_symmetric and rots
    assert any(cand.bend_permutation == {b1: b2, b2: b1} for cand in rots)
    assert final.symmetry.ambiguous_bend_groups == (tuple(sorted((b1, b2))),)
    for cand in final.symmetry.candidates:
        r = np.array(cand.transform)[:3, :3]
        assert np.linalg.det(r) == pytest.approx(1.0 if cand.kind == "ROTATION" else -1.0, abs=1e-5)


def test_asymmetric_hat_has_no_rotation_symmetry_and_no_ambiguity(tmp_path):
    shape, info = tp.hat_part(width=40.0)
    c = _compile(tmp_path, shape, info, "SYN-HSYM")
    final = c.stages[-1]
    assert not _rotations(final)
    assert final.symmetry.ambiguous_bend_groups == ()
    best_rotation = min(cand.rms_mm for cand in final.symmetry.candidates if cand.kind == "ROTATION")
    assert best_rotation > 1.0  # far from the 0.3 mm threshold, not a borderline call


def test_an_off_centre_hole_breaks_the_channel_symmetry(tmp_path):
    shape, info = tp.u_part(base=50.0, h1=25.0, h2=25.0, width=40.0)
    holed = tp.cut_hole(shape, (25.0, 10.0, -1.0), (0, 0, 1), 10.0)  # base hole near one end
    c = _compile(tmp_path, holed, info, "SYN-UHOLE")
    final = c.stages[-1]
    assert not _rotations(final)
    assert final.symmetry.ambiguous_bend_groups == ()


def test_flat_blank_is_symmetric_without_bend_permutations(tmp_path):
    shape, info = tp.hat_part(width=40.0)
    c = _compile(tmp_path, shape, info, "SYN-HFLAT")
    s0 = c.stages[0]
    assert s0.symmetry.is_symmetric  # a rectangle blank maps onto itself
    assert all(not cand.bend_permutation for cand in s0.symmetry.candidates)
    assert s0.symmetry.ambiguous_bend_groups == ()


def test_symmetry_test_scores_holes_and_edges_as_misses():
    # a plate with one hole off-centre: mirror/rotation map points into the hole -> not "on surface"
    xs, ys = np.meshgrid(np.arange(0.25, 60, 0.5), np.arange(0.25, 40, 0.5))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    keep = np.linalg.norm(pts[:, :2] - [12.0, 10.0], axis=1) > 6.0
    top = pts[keep]
    bottom = top - [0.0, 0.0, 2.0]
    points = np.concatenate([top, bottom])
    normals = np.concatenate([np.tile([0, 0, 1.0], (len(top), 1)), np.tile([0, 0, -1.0], (len(bottom), 1))])
    res = symmetry_test(points, normals, {}, target_spacing=1.0, source_points=6000, iterations=8)
    plane_flip = [c for c in res["candidates"] if c["kind"] == "MIRROR" and c["rms_mm"] <= 0.3]
    assert plane_flip  # the mid-plane mirror (skins swap) is exact
    in_plane = [c for c in res["candidates"] if c["kind"] == "ROTATION" and c["rms_mm"] <= 0.3]
    assert not in_plane  # any in-plane rotation maps the hole onto material


def _polygon_plate(n_sides: int, radius: float, spacing: float = 0.5, thickness: float = 2.0):
    """Points and normals of a regular n-gon plate (both skins), in-plane orientation 7 deg off +X."""
    from matplotlib.path import Path as MplPath

    ang = np.radians(7.0) + 2 * np.pi * np.arange(n_sides) / n_sides
    poly = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    xs, ys = np.meshgrid(np.arange(-radius, radius, spacing), np.arange(-radius, radius, spacing))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
    inside = grid[MplPath(poly).contains_points(grid)]
    top = np.column_stack([inside, np.zeros(len(inside))])
    bottom = top - [0.0, 0.0, thickness]
    pts = np.concatenate([top, bottom])
    nrm = np.concatenate([np.tile([0.0, 0.0, 1.0], (len(top), 1)), np.tile([0.0, 0.0, -1.0], (len(bottom), 1))])
    return pts, nrm


@pytest.mark.parametrize("n_sides, order", [(3, 120.0), (6, 60.0), (4, 90.0)])
def test_rotations_of_order_3_4_and_6_are_found(n_sides, order):
    # an axis of order >= 3 forces two equal principal moments; the candidates then include the
    # 60 / 90 / 120 deg turns about the third axis (rule 1 only generated 90 deg turns there)
    from domains.production.engine.compile.geometry import rotation_angle_deg

    pts, nrm = _polygon_plate(n_sides, 30.0)
    res = symmetry_test(pts, nrm, {}, target_spacing=1.0, source_points=6000, iterations=8)
    hits = [c for c in res["candidates"] if c["kind"] == "ROTATION" and c["rms_mm"] <= 0.3]
    angles = sorted(round(rotation_angle_deg(c["transform"])) for c in hits)
    assert any(abs(a - order) <= 1.0 for a in angles), angles
