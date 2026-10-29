import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
P = np.load(W / "plateframe.npy")
c, e1, e2, e3 = np.load(W / "pca.npy"); R = np.load(W / "R_plate.npy"); B = np.vstack([e1, e2, e3])
def to_world(q): return (np.asarray(q) @ R) @ B + c
G = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.load(open(W / "gold_plateframe.json")).items()}
def fitl(a, b):
    A = np.c_[np.ones(len(a)), a]; cc, *_ = np.linalg.lstsq(A, b, rcond=None); return cc
out = {"joggles": {}, "lips": {}}
# ---------- joggles: sections along x at fixed y
for side, xin, xout, sgn in (("LEFT", (-158, -146), (-175, -168), -1), ("RIGHT", (146, 158), (168, 175), 1)):
    rows = []
    for y0 in np.arange(-155, 152, 5.0):
        m = (np.abs(P[:, 1] - y0) < 0.5) & (P[:, 2] < 2) & (P[:, 2] > -10)
        Q = P[m]
        if len(Q) < 50: continue
        # top skin = upper envelope: for plate points, separate skins by local median per x-bin
        def skins(lo, hi):
            mm = (Q[:, 0] > lo) & (Q[:, 0] < hi)
            z = Q[mm, 2]; x = Q[mm, 0]
            if len(z) < 10: return None
            zc = np.median(z)
            top = z > zc; bot = ~top
            if top.sum() < 4 or bot.sum() < 4: return None
            return fitl(x[top], z[top]), fitl(x[bot], z[bot])
        si = skins(min(xin), max(xin)); so = skins(min(xout), max(xout))
        if si is None or so is None: continue
        # step height on the top skin, evaluated at the transition centre
        # transition: top-skin points between the two flats; find x where the top skin is halfway
        xs = np.linspace(min(xin + xout), max(xin + xout), 400)
        # approximate top envelope by binning
        bins = np.arange(min(xin + xout), max(xin + xout) + 0.25, 0.25)
        idx = np.digitize(Q[:, 0], bins)
        topz = np.array([Q[idx == k, 2].max() if np.any(idx == k) else np.nan for k in range(1, len(bins))])
        botz = np.array([Q[idx == k, 2].min() if np.any(idx == k) else np.nan for k in range(1, len(bins))])
        xc = 0.5 * (bins[:-1] + bins[1:])
        zin_t = si[0][0] + si[0][1] * xc; zout_t = so[0][0] + so[0][1] * xc
        h = np.nanmedian(zout_t - zin_t)
        frac = (topz - zin_t) / (zout_t - zin_t)
        ok = np.isfinite(frac)
        # transition centre = x where frac crosses 0.5 (search between flats)
        between = ok & (frac > 0.1) & (frac < 0.9)
        if between.sum() < 2: continue
        xm = float(np.median(xc[between]))
        # slope of top skin in the transition (max |dz/dx| over bins with 0.25<frac<0.75)
        core = ok & (frac > 0.2) & (frac < 0.8)
        if core.sum() >= 2:
            cc = fitl(xc[core], topz[core]); ang = math.degrees(math.atan(abs(cc[1])))
        else:
            ang = float("nan")
        w10 = xc[ok & (frac > 0.1) & (frac < 0.9)]
        width = float(w10.max() - w10.min()) if len(w10) else float("nan")
        # gold crossing (M03/M04)
        rows.append({"y": float(y0), "x_mid": xm, "step_mm": float(h), "flank_angle_deg": ang, "transition_width_mm": width,
                     "thick_in": float(np.median(si[0][0] - si[1][0] + (si[0][1] - si[1][1]) * np.mean(xin))),
                     "edge_x": float(Q[:, 0].min() if side == "LEFT" else Q[:, 0].max())})
    xm = np.array([r["x_mid"] for r in rows]); hh = np.array([r["step_mm"] for r in rows]); aa = np.array([r["flank_angle_deg"] for r in rows]); ex = np.array([r["edge_x"] for r in rows])
    ys = np.array([r["y"] for r in rows])
    cf = fitl(ys, xm)
    print(f"{side} joggle: stations {len(rows)} y {ys.min():.0f}..{ys.max():.0f}; x_mid median {np.median(xm):.2f} (fit x = {cf[0]:.2f} + {cf[1]:.5f} y, resid max {np.abs(xm - (cf[0] + cf[1]*ys)).max():.2f}); step {np.median(hh):.2f} mm (min {hh.min():.2f} max {hh.max():.2f}); flank angle median {np.nanmedian(aa):.1f} deg (p10 {np.nanpercentile(aa,10):.1f} p90 {np.nanpercentile(aa,90):.1f}); transition width med {np.median([r['transition_width_mm'] for r in rows]):.2f}; free edge x median {np.median(ex):.2f}")
    out["joggles"][side] = {"rows": rows, "x_mid_fit": cf.tolist(), "step_mm_med": float(np.median(hh)), "flank_angle_med": float(np.nanmedian(aa))}
    # plate y-extent at the joggle (full plate width?)
    mj = (np.abs(P[:, 0] - np.median(xm)) < 1.0) & (P[:, 2] < 2)
    print(f"   plate y-extent at joggle x: {P[mj,1].min():.1f}..{P[mj,1].max():.1f}")
    # gold distance to joggle mid line
    for k in (("M03",) if side == "LEFT" else ("M04",)):
        a, b = G[k]
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            gp = a + t * (b - a)
            xj = cf[0] + cf[1] * gp[1]
            print(f"   {k} t={t:.2f} gold x={gp[0]:.2f} y={gp[1]:.1f} z={gp[2]:.2f} -> joggle mid x={xj:.2f} : dx(gold-joggle)={gp[0]-xj:+.2f} mm ; dist to free edge ~{(np.median(ex)-gp[0]) if side=='RIGHT' else (gp[0]-np.median(ex)):.2f} mm")
# ---------- lips: wall->lip folds, sections in (u,z) at fixed x
for side, yw, s in (("M01", 152.86, 1), ("M02", -158.66, -1)):
    rows = []
    for x0 in np.arange(-100, 101, 2.0):
        m = (np.abs(P[:, 0] - x0) < 0.5) & (np.abs(P[:, 1] - yw) < 20) & (P[:, 2] > 15)
        Q = P[m]
        if len(Q) < 30: continue
        u = s * (yw - Q[:, 1]); z = Q[:, 2]
        lipm = (u > 5) & (u < 14) & (z > 28)
        wallm = (u > -3) & (u < 3) & (z > 15) & (z < 26)
        if lipm.sum() < 8 or wallm.sum() < 8:
            rows.append({"x": float(x0), "lip": False}); continue
        zl = np.median(z[lipm]); ul = np.median(u[wallm])
        rows.append({"x": float(x0), "lip": True, "lip_z_mid": float(zl), "wall_u_mid": float(ul), "lip_len_u": float(np.percentile(u[(z > zl - 1.5) & (z < zl + 1.5) & (u > 2)], 98) if np.any((z > zl - 1.5) & (z < zl + 1.5) & (u > 2)) else np.nan)})
    lx = [r["x"] for r in rows if r["lip"]]
    zz = [r["lip_z_mid"] for r in rows if r["lip"]]; ll = [r["lip_len_u"] for r in rows if r["lip"]]
    # contiguous runs
    runs = []; cur = []
    for r in rows:
        if r["lip"]: cur.append(r["x"])
        else:
            if cur: runs.append((cur[0], cur[-1])); cur = []
    if cur: runs.append((cur[0], cur[-1]))
    print(f"{side} lip present at x runs {runs}; lip mid-surface z median {np.median(zz):.2f}; lip length (u, from wall) median {np.nanmedian(ll):.1f} mm")
    out["lips"][side] = {"runs": runs, "lip_z_med": float(np.median(zz)), "lip_len_med": float(np.nanmedian(ll))}
json.dump(out, open(W / "joggles_lips.json", "w"), indent=1)
