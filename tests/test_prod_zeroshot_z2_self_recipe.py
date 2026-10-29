"""Z2 self-recipe lane on synthetic bent sheets (no real scans, no labels)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

o3d = pytest.importorskip("open3d")


def _profile(t: float, ri: float, sweep_deg: float, la: float, lb: float, step: float = 0.3):
    """Mid-surface polyline (2D) of flange A (along -x to the bend), arc, flange B; with unit normals."""
    rm = ri + 0.5 * t
    pts, nrm = [], []
    for x in np.arange(la, 0, -step):
        pts.append((x, 0.0)); nrm.append((0.0, 1.0))
    phi = math.radians(sweep_deg)
    n_arc = max(4, int(rm * phi / step))
    for k in range(n_arc + 1):
        a = phi * k / n_arc
        # arc centre at (0, -rm); bend downward (convex side up/left)
        pts.append((-rm * math.sin(a), -rm + rm * math.cos(a)))
        nrm.append((-math.sin(a), math.cos(a)))
    end = np.array(pts[-1])
    d = np.array([-math.cos(phi), -math.sin(phi)])
    nb = np.array(nrm[-1])
    for s in np.arange(step, lb, step):
        p = end + d * s
        pts.append(tuple(p)); nrm.append(tuple(nb))
    return np.array(pts), np.array(nrm)


def _sheet_mesh(t=1.5, ri=2.0, sweep=90.0, la=40.0, lb=30.0, width=60.0, step=0.3):
    mid, n = _profile(t, ri, sweep, la, lb, step)
    outer = mid + 0.5 * t * n
    inner = mid - 0.5 * t * n
    poly = np.vstack([outer, inner[::-1]])            # closed cross-section
    zs = np.arange(0.0, width + 1e-9, step)
    m = len(poly)
    V = np.array([[p[0], p[1], z] for z in zs for p in poly])
    F = []
    for k in range(len(zs) - 1):
        for i in range(m):
            j = (i + 1) % m
            a, b, c, d = k * m + i, k * m + j, (k + 1) * m + j, (k + 1) * m + i
            F.append((a, b, c)); F.append((a, c, d))
    F = np.array(F)
    # orient outward: face normal should point away from the mid-surface
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    cen = V[F].mean(1)
    from scipy.spatial import cKDTree
    _, j = cKDTree(np.c_[mid, np.zeros(len(mid))][:, :2]).query(cen[:, :2])
    away = cen[:, :2] - mid[j]
    if np.mean(np.einsum("ij,ij->i", cr[:, :2], away) > 0) < 0.5:
        F = F[:, ::-1]
    return V, F, mid, n


def _apex_gold(t, ri, sweep, width):
    ro = ri + t
    a = math.radians(sweep) / 2
    # arc centre at (0, -rm); outer apex at radius ro on the bisector
    rm = ri + 0.5 * t
    p = np.array([-ro * math.sin(a), -rm + ro * math.cos(a)])
    return np.array([p[0], p[1], 0.0]), np.array([p[0], p[1], width])


@pytest.mark.parametrize("sweep", [90.0, 45.0])
def test_z2_finds_single_bend_at_outer_apex(tmp_path, sweep):
    from research.zeroshot.z2_self_recipe.run import run_scan
    from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines

    t, ri, width = 1.5, 2.0, 60.0
    V, F, _, _ = _sheet_mesh(t=t, ri=ri, sweep=sweep, width=width)
    rng = np.random.default_rng(0)
    V = V + rng.normal(0, 0.02, V.shape)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    p = tmp_path / "sheet.ply"
    o3d.io.write_triangle_mesh(str(p), mesh)
    payload, diag, scan = run_scan(p)
    assert abs(diag["t_mm"] - t) < 0.15
    lines = payload["owned_bend_regions"]
    assert len(lines) == 1, diag["edges"]
    a, b = _apex_gold(t, ri, sweep, width)
    gold = [{"bend_id": "G1", "line_start": a.tolist(), "line_end": b.tolist()}]
    m = match_lines(gold, lines, Thresholds())
    assert m["true_positive"] == 1
    off = m["matches"][0]
    assert abs(lines[0]["included_angle_deg"] - (180 - sweep)) < 1.0
    assert off["angle_deg"] < 1.0 if "angle_deg" in off else True


def test_z2_flat_sheet_has_no_bend(tmp_path):
    from research.zeroshot.z2_self_recipe.run import run_scan

    V, F, _, _ = _sheet_mesh(sweep=0.0001, la=40, lb=30)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    p = tmp_path / "flat.ply"
    o3d.io.write_triangle_mesh(str(p), mesh)
    payload, diag, _ = run_scan(p)
    assert payload["owned_bend_regions"] == []
