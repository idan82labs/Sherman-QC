"""Auditor-2: light software z-buffer renders of the scan mesh (two-sided Lambert shading) with
gold lines (green), measured outer-apex lines (blue) and inner-apex lines (orange) overlaid,
lines drawn solid where visible and dotted where occluded. Also a 'leftover' map: faces that are
neither in a large flat region (>= 15 mm2) nor within 9 mm of a bend axis -> must be plate edges,
hole walls or an unlabeled formed feature."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_mesh, unit  # noqa: E402

m = load_mesh()
gold = load_gold()
L = json.loads((OUT / "a2_lines_report.json").read_text())["bends"]
V = np.asarray(m.vertices)
F = np.asarray(m.faces)
fn = np.asarray(m.face_normals)
fa = np.asarray(m.area_faces)
lab = np.load(OUT / "face_region_label.npy")
planes = json.loads((OUT / "a2_planes.json").read_text())
big = {r["rid"] for r in planes["regions"]}

# densely sample triangles (barycentric grid) for splatting
K = 4
bary = np.array([(i / K, j / K) for i in range(K + 1) for j in range(K + 1 - i)])
A_, B_, C_ = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
pts = (A_[:, None] * (1 - bary[:, 0] - bary[:, 1])[None, :, None] + B_[:, None] * bary[None, :, 0, None]
       + C_[:, None] * bary[None, :, 1, None]).reshape(-1, 3)
pfid = np.repeat(np.arange(len(F)), len(bary))


def view_matrix(az, el):
    az, el = np.radians(az), np.radians(el)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # camera looks along -fwd
    up0 = np.array([0, 0, 1.0]) if abs(fwd[2]) < 0.95 else np.array([0, 1.0, 0])
    right = unit(np.cross(up0, fwd))
    up = np.cross(fwd, right)
    return np.array([right, up, fwd])


def render(ax, R, centre, half, res=0.12, face_color=None, title="", lines=()):
    P = (pts - centre) @ R.T
    nx = int(2 * half / res)
    ix = ((P[:, 0] + half) / res).astype(int)
    iy = ((P[:, 1] + half) / res).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < nx)
    ix, iy, z, fid = ix[ok], iy[ok], P[ok, 2], pfid[ok]
    zb = np.full((nx, nx), -np.inf)
    order = np.argsort(z)
    zb[iy[order], ix[order]] = z[order]  # last write wins = largest z (nearest to camera)
    fb = np.full((nx, nx), -1)
    fb[iy[order], ix[order]] = fid[order]
    light = unit(np.array([0.3, 0.5, 1.0]))
    nrm = fn @ R.T
    shade = 0.25 + 0.75 * np.abs(nrm @ light)
    img = np.ones((nx, nx, 3))
    hit = fb >= 0
    if face_color is None:
        img[hit] = (shade[fb[hit]][:, None] * np.array([0.82, 0.82, 0.85]))
    else:
        img[hit] = shade[fb[hit]][:, None] * face_color[fb[hit]]
    ax.imshow(img, origin="lower", extent=[-half, half, -half, half], interpolation="nearest")
    for (p, q, col, lw, lab_) in lines:
        s = np.linspace(0, 1, 200)
        X = (p[None] + s[:, None] * (q - p)[None] - centre) @ R.T
        jx = ((X[:, 0] + half) / res).astype(int).clip(0, nx - 1)
        jy = ((X[:, 1] + half) / res).astype(int).clip(0, nx - 1)
        vis = X[:, 2] >= zb[jy, jx] - 0.35
        xs, ys = X[:, 0].copy(), X[:, 1].copy()
        xv, yv = xs.copy(), ys.copy()
        xv[~vis] = np.nan
        yv[~vis] = np.nan
        ax.plot(xv, yv, "-", color=col, lw=lw)
        ax.plot(xs, ys, ":", color=col, lw=max(1.0, lw - 1))
        if lab_:
            ax.text(xs[-1], ys[-1], lab_, color=col, fontsize=8, fontweight="bold")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


line_set = []
for bid in ("M01", "M02", "M03"):
    gs, ge = gold[bid]
    line_set.append((gs, ge, "#1a9c3a", 2.6, bid))
    oa = np.array(L[bid]["outer_apex_line"])
    ia = np.array(L[bid]["inner_apex_line"])
    line_set.append((oa[0], oa[1], "#1f77b4", 1.6, ""))
    line_set.append((ia[0], ia[1], "#ff7f0e", 1.2, ""))

centre_all = V.mean(0)
bendc = np.mean([np.mean(gold[b], axis=0) for b in gold], axis=0)
d = unit(gold["M01"][1] - gold["M01"][0])
fig, axs = plt.subplots(2, 3, figsize=(21, 14))
views = [(30, 35), (210, 35), (120, -30), (300, 60)]
for k, (az, el) in enumerate(views[:3]):
    render(axs[0, k], view_matrix(az, el), centre_all, 75, 0.2, title=f"whole part az {az} el {el}", lines=line_set)
# close-ups along the bend axis direction (looking down the bends) and two obliques
Rd = np.array([unit(np.cross(np.array([0, 0, 1.0]), d)), None, None], dtype=object)
fwd = -d
right = unit(np.cross(np.array([0, 0, 1.0]), fwd))
up = np.cross(fwd, right)
Rax = np.array([right, up, fwd])
render(axs[1, 0], Rax, bendc, 30, 0.06, title="close-up looking along the bend axes (gold green, outer apex blue, inner apex orange)", lines=line_set)
render(axs[1, 1], view_matrix(40, 25), bendc, 30, 0.07, title="close-up oblique (concave sides)", lines=line_set)
render(axs[1, 2], view_matrix(220, -25), bendc, 30, 0.07, title="close-up oblique (convex sides)", lines=line_set)
fig.suptitle("sc_70b39f530f00 auditor-2 shaded renders: gold (green, solid = visible, dotted = hidden), "
             "measured outer-apex lines (blue), inner-apex lines (orange)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "a2_render3d.png", dpi=72)
print("wrote", OUT / "a2_render3d.png")

# leftover map
fc = np.asarray(m.triangles_center)
near_bend = np.zeros(len(F), bool)
for bid in ("M01", "M02", "M03"):
    oa = np.array(L[bid]["outer_apex_line"])
    a, b = oa
    v = unit(b - a)
    tt = np.clip((fc - a) @ v, -2, np.linalg.norm(b - a) + 2)
    dist = np.linalg.norm(fc - (a + tt[:, None] * v), axis=1)
    near_bend |= dist < 9.0
in_big = np.isin(lab, list(big))
left = ~in_big & ~near_bend
print(f"leftover faces: {left.sum()} area {fa[left].sum():.0f} mm2 (in big flat regions {fa[in_big].sum():.0f}, near bends {fa[~in_big & near_bend].sum():.0f})")
# classify leftover by normal relative to its plate: edge walls have normals ~perpendicular to plate normals
col = np.tile(np.array([0.85, 0.85, 0.88]), (len(F), 1))
col[left] = np.array([0.85, 0.1, 0.1])
col[~in_big & near_bend] = np.array([0.1, 0.4, 0.9])
fig, axs = plt.subplots(1, 3, figsize=(21, 7.5))
for k, (az, el) in enumerate([(30, 35), (210, 35), (120, -30)]):
    render(axs[k], view_matrix(az, el), centre_all, 75, 0.2, face_color=col,
           title=f"red = not flat-region & >9 mm from any bend (edges/holes/other); blue = non-flat near bends; az {az} el {el}")
fig.tight_layout()
fig.savefig(OUT / "a2_leftover_faces.png", dpi=72)
print("wrote", OUT / "a2_leftover_faces.png")
