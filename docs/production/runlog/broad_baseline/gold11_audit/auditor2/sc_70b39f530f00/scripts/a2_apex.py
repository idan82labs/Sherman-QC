"""Auditor-2: exact per-station locus of each gold line relative to its bend.

For each bend the two flanges' skin planes come from the planar region growing (a2_planes.py):
  M03 arm  -> web1 : outer (convex) skins R0 (arm top) + R6 (web1 top); inner R2 + R8
  M02 web1 -> web2 : outer skins R8 (web1 bottom) + R4/R10 (web2 bottom); inner R6 + R1
  M01 web2 -> frame: outer skins R4/R10 (web2 bottom) + R5 (frame +u); inner R1 + R3
Outer / inner virtual sharp lines = plane-pair intersections (exact 3D lines). The true bend axis
direction is their direction. Stations every 0.25 mm along that axis: the section plane is
perpendicular to the axis; the bisector ray (from the outer sharp toward the concave side) is
intersected with the exact mesh section polylines -> outer apex (first skin hit) and inner apex
(second hit). The gold line's pierce point is then measured in that frame:
  along  = signed distance from outer apex along the bisector (+ = toward concave side)
  across = signed distance across the bisector (in the section plane)
Also: signed distance gold -> mesh surface (+ = in free air, - = inside the material).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_mesh, section, unit  # noqa: E402

m = load_mesh()
gold = load_gold()
fn = np.asarray(m.face_normals)
fa = np.asarray(m.area_faces)
fc = np.asarray(m.triangles_center)
lab = np.load(OUT / "face_region_label.npy")


def plane_of(rids):
    sel = np.isin(lab, rids)
    P, W = fc[sel], fa[sel]
    c = (P * W[:, None]).sum(0) / W.sum()
    X = P - c
    ev, evec = np.linalg.eigh((X * W[:, None]).T @ X / W.sum())
    n = evec[:, 0]
    if n @ (fn[sel] * W[:, None]).sum(0) < 0:
        n = -n  # outward (mesh normal) orientation
    return c, n, float(np.sqrt(ev[0])), float(W.sum())


def plane_plane(c1, n1, c2, n2):
    a = unit(np.cross(n1, n2))
    # point on both planes closest to the midpoint of the centroids
    A = np.array([n1, n2, a])
    b = np.array([n1 @ c1, n2 @ c2, a @ ((c1 + c2) / 2)])
    return np.linalg.solve(A, b), a


BENDS = {
    "M03": {"outer": ([0], [6]), "inner": ([2], [8])},
    "M02": {"outer": ([8], [4, 10]), "inner": ([6], [1])},
    "M01": {"outer": ([4, 10], [5]), "inner": ([1], [3])},
}

pq = trimesh.proximity.ProximityQuery(m)
report = {}
for bid, spec in BENDS.items():
    (o1, o2), (i1, i2) = spec["outer"], spec["inner"]
    co1, no1, r_o1, A_o1 = plane_of(o1)
    co2, no2, r_o2, A_o2 = plane_of(o2)
    ci1, ni1, r_i1, A_i1 = plane_of(i1)
    ci2, ni2, r_i2, A_i2 = plane_of(i2)
    Xo, ax = plane_plane(co1, no1, co2, no2)
    Xi, ax_i = plane_plane(ci1, ni1, ci2, ni2)
    gs, ge = gold[bid]
    gd = unit(ge - gs)
    if ax @ gd < 0:
        ax = -ax
    if ax_i @ ax < 0:
        ax_i = -ax_i
    theta = np.degrees(np.arccos(np.clip(no1 @ no2, -1, 1)))  # bend angle (deviation from flat)
    # bisector toward the concave side = minus the mean outward normal of the outer skins
    sdir = unit(-(no1 + no2))
    sdir = unit(sdir - (sdir @ ax) * ax)
    # put Xi on the same station as Xo
    Xi_s = Xi + ((Xo - Xi) @ ax) * ax
    sharp_sep_along = (Xi_s - Xo) @ sdir
    sharp_sep_across = np.linalg.norm((Xi_s - Xo) - sharp_sep_along * sdir)
    cdir = unit(np.cross(ax, sdir))
    gold_axis_angle = np.degrees(np.arccos(np.clip(abs(gd @ ax), -1, 1)))
    # station range: gold extent +- 8 mm along the axis
    ts, te = sorted([(gs - Xo) @ ax, (ge - Xo) @ ax])
    stations = np.arange(np.floor(ts) - 8, np.ceil(te) + 8.001, 0.25)
    rows = []
    for t in stations:
        p0 = Xo + t * ax
        polys = section(m, p0, ax)
        # intersect the bisector line p0 + r*sdir with the section polylines (2D in (sdir, cdir))
        hits = []
        for pl in polys:
            q = np.c_[(pl - p0) @ sdir, (pl - p0) @ cdir]
            for k in range(len(q) - 1):
                a_, b_ = q[k], q[k + 1]
                if (a_[1] > 0) == (b_[1] > 0) or a_[1] == b_[1]:
                    continue
                lam = a_[1] / (a_[1] - b_[1])
                r = a_[0] + lam * (b_[0] - a_[0])
                if -6 < r < 12:
                    hits.append(r)
        hits = sorted(hits)
        # gold pierce point at this station (infinite line)
        lam = ((p0 - gs) @ ax) / (gd @ ax)
        G = gs + lam * gd
        g_along_from_Xo = (G - p0) @ sdir
        g_across = (G - p0) @ cdir
        inside_gold = (min(ts, te) - 1e-9) <= t <= (max(ts, te) + 1e-9)
        row = {"t": float(t), "hits": [float(h) for h in hits], "gold_inside": bool(inside_gold),
               "g_along_Xo": float(g_along_from_Xo), "g_across": float(g_across), "gold_xyz": G.tolist()}
        if len(hits) >= 2:
            oa, ia = hits[0], hits[1]
            row.update({"oa": float(oa), "ia": float(ia), "thk_on_bisector": float(ia - oa),
                        "oa_xyz": (p0 + oa * sdir).tolist(), "ia_xyz": (p0 + ia * sdir).tolist(),
                        "g_minus_oa": float(g_along_from_Xo - oa), "g_minus_ia": float(g_along_from_Xo - ia),
                        "g_minus_xi": float(g_along_from_Xo - sharp_sep_along)})
        rows.append(row)
    # signed distance of gold samples to the surface
    nS = 57
    samp = gs[None] + np.linspace(0, 1, nS)[:, None] * (ge - gs)[None]
    cp, dist, tri = trimesh.proximity.closest_point(m, samp)
    sgn = np.sign(np.einsum("ij,ij->i", samp - cp, fn[tri]))
    sdist = (dist * sgn).tolist()
    # radius estimate from apex-to-sharp distance: r_oa = R(1/cos(th/2) - 1)
    hc = np.cos(np.radians(theta / 2))
    oas = np.array([r["oa"] for r in rows if "oa" in r and r["gold_inside"]])
    ias = np.array([r["ia"] for r in rows if "ia" in r and r["gold_inside"]])
    R_o = float(np.median(oas) / (1 / hc - 1)) if len(oas) else None
    R_i = float((np.median(ias) - sharp_sep_along) / (1 / hc - 1)) if len(ias) else None
    report[bid] = {
        "planes_rms_mm": [r_o1, r_o2, r_i1, r_i2], "planes_area": [A_o1, A_o2, A_i1, A_i2],
        "bend_angle_deg": float(theta),
        "inner_angle_check_deg": float(np.degrees(np.arccos(np.clip(ni1 @ ni2, -1, 1)))),
        "axis": ax.tolist(), "sdir": sdir.tolist(), "cdir": cdir.tolist(), "axis_inner_vs_outer_deg": float(np.degrees(np.arccos(np.clip(ax @ ax_i, -1, 1)))),
        "gold_vs_axis_deg": float(gold_axis_angle),
        "Xo": Xo.tolist(), "Xi_sep_along_mm": float(sharp_sep_along), "Xi_sep_across_mm": float(sharp_sep_across),
        "gold_t_range_on_axis": [float(ts), float(te)],
        "R_outer_est": R_o, "R_inner_est": R_i,
        "gold_signed_dist_to_surface": sdist,
        "stations": rows,
    }
    print(f"\n=== {bid}: bend angle {theta:.2f} deg (inner skins {report[bid]['inner_angle_check_deg']:.2f}); "
          f"gold vs true axis {gold_axis_angle:.3f} deg; outer-inner sharp sep along bisector {sharp_sep_along:.3f} mm "
          f"(across {sharp_sep_across:.3f}); R_o~{R_o}, R_i~{R_i}")
    print(f"    plane rms {np.round([r_o1, r_o2, r_i1, r_i2], 3)}; gold t-range on axis [{ts:.2f}, {te:.2f}]")
    sd = np.array(sdist)
    print(f"    gold signed dist to surface (+ = free air, - = in material): min {sd.min():+.3f} med {np.median(sd):+.3f} max {sd.max():+.3f}")
    print(f"    {'t':>6} {'gold':>4} {'OA':>6} {'IA':>6} {'thk':>5} {'G-OA':>6} {'G-IA':>6} {'G-Xi':>6} {'across':>6}  hits")
    for r in rows:
        if abs((r["t"] * 4) % 4) > 1e-6 and not (abs(r["t"] - ts) < 0.3 or abs(r["t"] - te) < 0.3):
            continue  # print every 1 mm plus the gold ends
        if "oa" in r:
            print(f"    {r['t']:6.2f} {'IN' if r['gold_inside'] else '--':>4} {r['oa']:6.2f} {r['ia']:6.2f} {r['thk_on_bisector']:5.2f} "
                  f"{r['g_minus_oa']:6.2f} {r['g_minus_ia']:6.2f} {r['g_minus_xi']:6.2f} {r['g_across']:6.2f}  {np.round(r['hits'], 2).tolist()}")
        else:
            print(f"    {r['t']:6.2f} {'IN' if r['gold_inside'] else '--':>4}   no 2-skin hit; hits={np.round(r['hits'], 2).tolist()}  G along Xo {r['g_along_Xo']:.2f} across {r['g_across']:.2f}")

json.dump(report, open(OUT / "a2_apex_report.json", "w"), indent=1)
print("\nwrote", OUT / "a2_apex_report.json")
