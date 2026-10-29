"""Cross-section look at a bend (DEV): slice the raw scan perpendicular to the zs line at 5 stations, fit one line
per flange (points 4-30 mm from the apex, split into two clusters by direction), report the included angle per
station and plot the middle station. Independent of zs, E3 and gold (only the zs line places the slice).
usage (repo root): .venv/bin/python -B .../section.py ANGLE_CHECK.json OUT_DIR scan:gold_id [scan:gold_id ...]
"""
import json, sys
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from research.zeroshot.common.gold_guard import require_runnable

R_IN, R_OUT, SLAB = 3.0, 20.0, 0.75


def frame(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    u = np.cross(d, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-3:
        u = np.cross(d, [0, 1.0, 0])
    u /= np.linalg.norm(u); v = np.cross(d, u)
    return a, b, d, u, v


def _ransac(uv, n_it=1500, tol=0.25, rng=np.random.default_rng(0), avoid=None):
    best = None
    for _ in range(n_it):
        i, j = rng.choice(len(uv), 2, replace=False)
        w = uv[j] - uv[i]; nw = np.linalg.norm(w)
        if nw < 1.0:
            continue
        w /= nw; n = np.array([-w[1], w[0]])
        if avoid is not None and abs(w @ avoid) > np.cos(np.radians(10)):
            continue
        if abs(uv[i] @ n) > 2.0:   # a flange line of the outer skin passes (near) the outer apex
            continue
        inl = np.abs((uv - uv[i]) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    return best


def section_angle(P, c, d, u, v, r_in=None, r_out=None):
    """Two dominant straight segments near the apex (sequential RANSAC), angle between rays leaving the apex."""
    r_in = R_IN if r_in is None else r_in; r_out = R_OUT if r_out is None else r_out
    q = P - c
    m = np.abs(q @ d) < SLAB
    uv_all = np.c_[q[m] @ u, q[m] @ v]
    r = np.linalg.norm(uv_all, axis=1)
    uv = uv_all[(r > r_in) & (r < r_out)]
    if len(uv) < 30:
        return None, uv_all, None
    rays, rest = [], uv
    for _ in range(2):
        if len(rest) < 15:
            return None, uv_all, None
        inl = _ransac(rest, avoid=rays[0] if rays else None)
        if inl is None or inl.sum() < 15:
            return None, uv_all, None
        g = rest[inl]
        w = np.linalg.svd(g - g.mean(0), full_matrices=False)[2][0]
        rays.append(w if w @ g.mean(0) > 0 else -w)
        # drop both skins of this flange: everything within 2 mm of this line
        n = np.array([-w[1], w[0]])
        rest = rest[np.abs((rest - g.mean(0)) @ n) > 2.0]
    ang = float(np.degrees(np.arccos(np.clip(rays[0] @ rays[1], -1, 1))))
    return ang, uv_all, (rays, None)


def main(js, outdir, keys):
    d = json.loads(Path(js).read_text())
    recs = {(r["scan"], r["gold_id"]): r for r in d["cli"]["records"]}
    rows = {r["scan"]: r for r in json.loads(Path(d["inputs"]["cohort"]).read_text())["rows"]}
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for k in keys:
        s, g = k.split(":")
        require_runnable(s, cad_path=True)
        r = recs[(s, g)]
        mesh = trimesh.load(rows[s]["scan_path"], process=False)
        P = np.asarray(mesh.vertices, float)
        a, b, dd, u, v = frame(*r["zs_line"])
        L = np.linalg.norm(b - a)
        st = []
        for f in (0.2, 0.35, 0.5, 0.65, 0.8):
            ang, uv, fit = section_angle(P, a + f * L * dd, dd, u, v)
            st.append(None if ang is None else round(ang, 2))
            if f == 0.5:
                mid = (uv, fit, ang)
        vals = [x for x in st if x is not None]
        res[k] = {"stations": st, "median": round(float(np.median(vals)), 2) if vals else None,
                  "zs": r["zs_incl"], "e3": r["e3_incl"], "gold": r["gold_incl"], "nominal": r["nominal"]}
        uv, fit, ang = mid
        fig, ax = plt.subplots(figsize=(5, 5))
        uv = uv[np.linalg.norm(uv, axis=1) < 40]
        ax.scatter(uv[:, 0], uv[:, 1], s=1, c="0.4")
        if fit:
            for ray in fit[0]:
                ax.plot([0, R_OUT * ray[0]], [0, R_OUT * ray[1]], "r-", lw=1)
        ax.plot(0, 0, "b+", ms=12)
        ax.set_aspect("equal")
        e3 = "-" if r["e3_incl"] is None else f"{r['e3_incl']:.1f}"
        ax.set_title(f"{s} {r.get('bend_label')} (gold {g})\nnom {r['nominal']}  zs {r['zs_incl']:.1f}  E3 {e3}  "
                     f"gold {r['gold_incl']}  section {'-' if ang is None else f'{ang:.1f}'}", fontsize=8)
        fig.savefig(out / f"{s}_{g}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        print(k, json.dumps(res[k]), flush=True)
    (out / "sections.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
