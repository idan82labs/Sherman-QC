"""D1 two-skin module: synthetic bent-sheet checks (no scan data, no engine imports)."""
from __future__ import annotations

import math

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from research.zeroshot.d1_two_skin import two_skin as ts  # noqa: E402
from research.zeroshot.d1_two_skin.custody import CustodyError, check_path  # noqa: E402


def _bent_sheet(t=2.0, r_in=2.0, included_deg=90.0, arm=40.0, length=60.0, step=0.25):
    """Closed-profile sheet with one bend, extruded along z (open ends). Returns mesh + truth."""
    beta = math.radians(180.0 - included_deg)            # bend (turn) angle
    rm = r_in + t / 2
    pts, nrm = [], []
    for x in np.arange(-arm, 0.0, step):                 # arm 1, travelling +x at y = rm
        pts.append((x, rm)); nrm.append((0.0, 1.0))
    n_arc = max(8, int(rm * beta / (step * 0.5)))
    for phi in np.linspace(math.pi / 2, math.pi / 2 - beta, n_arc):
        pts.append((rm * math.cos(phi), rm * math.sin(phi))); nrm.append((math.cos(phi), math.sin(phi)))
    phi_e = math.pi / 2 - beta
    d2 = np.array([math.sin(phi_e), -math.cos(phi_e)])   # travel direction of arm 2
    p_e = np.array([rm * math.cos(phi_e), rm * math.sin(phi_e)])
    for s in np.arange(step, arm, step):
        pts.append(tuple(p_e + s * d2)); nrm.append((math.cos(phi_e), math.sin(phi_e)))
    pts, nrm = np.array(pts), np.array(nrm)
    outer = pts + 0.5 * t * nrm
    inner = pts - 0.5 * t * nrm
    poly = np.vstack([outer, inner[::-1]])
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1]
    nz = int(length / step) + 1
    zs = np.linspace(0, length, nz)
    n = len(poly)
    V = np.array([(x, y, z) for z in zs for (x, y) in poly])
    F = []
    for k in range(nz - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            c, d = a + n, b + n
            F.append((a, b, d)); F.append((a, d, c))
    mesh = ts.mesh_from_arrays(V, np.array(F))
    phi_b = math.pi / 2 - beta / 2
    bis = np.array([math.cos(phi_b), math.sin(phi_b)])
    truth = {"t": t, "outer_apex": (r_in + t) * bis, "inner_apex": r_in * bis, "bisector": bis,
             "outer_sharp": (r_in + t) / math.sin(math.radians(included_deg) / 2) * bis,
             "inner_sharp": r_in / math.sin(math.radians(included_deg) / 2) * bis,
             "length": length, "included": included_deg}
    return mesh, truth


@pytest.mark.parametrize("t,included", [(2.0, 90.0), (1.0, 90.0), (3.0, 120.0)])
def test_thickness_mode(t, included):
    mesh, truth = _bent_sheet(t=t, included_deg=included)
    est = ts.estimate_thickness(mesh, n_samples=40_000)
    assert est.t_mm == pytest.approx(t, rel=0.02)
    assert est.confidence > 0.5
    assert est.exit_fraction > 0.95


def test_profile_loci_and_snap():
    mesh, tr = _bent_sheet(t=2.0, r_in=2.0, included_deg=90.0)
    zmid = tr["length"] / 2
    P, N = mesh.vertices, mesh.vertex_normals

    def line_at(p2):
        return np.array([p2[0], p2[1], 5.0]), np.array([p2[0], p2[1], tr["length"] - 5.0])

    s, e = line_at(tr["outer_apex"])
    prof = ts.bend_profile(P, N, s, e, tr["t"])
    assert prof.ok, prof.reason
    assert prof.included_angle_deg == pytest.approx(90.0, abs=2.0)
    got = prof.to3d(prof.outer_apex_2d)
    assert np.allclose(got[:2], tr["outer_apex"], atol=0.15)
    assert np.allclose(prof.to3d(prof.inner_apex_2d)[:2], tr["inner_apex"], atol=0.15)
    assert np.allclose(prof.to3d(prof.outer_sharp_2d)[:2], tr["outer_sharp"], atol=0.15)
    assert np.allclose(prof.to3d(prof.inner_sharp_2d)[:2], tr["inner_sharp"], atol=0.15)
    assert prof.local_t_mm == pytest.approx(2.0, abs=0.1)
    # the two virtual sharps are t / sin(theta/2) apart along the bisector
    sep = np.linalg.norm(np.subtract(prof.outer_sharp_2d, prof.inner_sharp_2d))
    assert sep == pytest.approx(2.0 / math.sin(math.radians(45)), abs=0.15)
    # a detection on the inner virtual sharp snaps onto the outer apex
    s2, e2 = line_at(tr["inner_sharp"])
    prof2 = ts.bend_profile(P, N, s2, e2, tr["t"])
    ns, ne, info = ts.snap_line_to_outer_apex(s2, e2, prof2, max_shift_mm=8.0)
    assert info["moved"] and info["locus"] == "outer_apex"
    assert np.allclose(0.5 * (ns + ne)[:2], tr["outer_apex"], atol=0.2)
    assert 0.5 * (ns + ne)[2] == pytest.approx(zmid, abs=1e-6)


def test_skin_sides_and_mid_surface():
    mesh, tr = _bent_sheet(t=2.0)
    lab = ts.label_skins(mesh, 2.0)
    summ = lab.summary()
    assert summ["two_sided_fraction"] > 0.8
    assert summ["pair_link_consistency"] > 0.95
    # the convex (outer) skin vertices all share one side label
    r = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
    arc_outer = (np.abs(r - 4.0) < 0.05) & (mesh.vertices[:, 0] > 0.5) & (mesh.vertices[:, 1] > 0.5)
    sides = set(lab.side[arc_outer].tolist()) - {-1}
    assert len(sides) == 1
    mp, _ = ts.mid_surface(mesh, lab, 2.0, keep_one_sided=False)
    flat = mp[mp[:, 0] < -5]
    assert np.allclose(flat[:, 1], 3.0, atol=0.05)          # mid-plane of arm 1 (y = r_in + t/2)


@pytest.mark.parametrize("bad", ["/x/HOLDOUT10/a.ply", "/x/ai_bend_lines_fresh_v1_SEALED/a.json",
                                 "/x/N015_scan.ply", "/x/16231000_03.ply", "/Users/u/rhino-preview/a.ply"])
def test_custody_refuses(bad):
    with pytest.raises(CustodyError):
        check_path(bad)


def test_custody_allows_plain_path(tmp_path):
    p = tmp_path / "49024000_A.ply"
    assert check_path(p) == p.resolve()
