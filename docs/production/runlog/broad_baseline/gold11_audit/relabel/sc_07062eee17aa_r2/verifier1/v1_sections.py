"""Verifier1 independent cross-section check of sc_07062eee17aa r2 lines.
Loads the full scan (run under the shared lock). Writes only into this dir."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply"
LINES = os.path.join(HERE, "..", "lines.json")


def load_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while True:
            line = f.readline()
            hdr += line
            if line.strip() == b"end_header" or not line:
                break
        n = int([l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
        a = np.frombuffer(f.read(n * 24), dtype="<f4").reshape(n, 6).astype(np.float64)
    return a[:, :3], a[:, 3:]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def fit_line2(P):
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c)
    return c, vt[0]


def section(P, tree, L, t, slab, rad, plate_a):
    """Return dict for station t along label line L."""
    d, b0, c0 = L["d"], L["b"], L["c"]
    P0 = L["s"] + t * d
    idx = tree.query_ball_point(P0, rad)
    if len(idx) == 0:
        return None
    Q = P[idx] - P0
    m = np.abs(Q @ d) < slab
    Q = Q[m]
    q = np.c_[Q @ c0, Q @ b0]  # 2D: x=lateral (c), y=depth into material (b)
    out = {"t": t, "q": q, "ok": False}
    plates = []
    for sd in (L["sa"], L["sb"]):
        sa = np.array([sd @ c0, sd @ b0]); sa = unit(sa)
        na = np.array([-sa[1], sa[0]])
        if na[1] > 0:
            na = -na  # outward normal has -b component
        a = q @ sa; n = q @ na
        sel = (a > plate_a[0]) & (a < plate_a[1]) & (np.abs(n) < 4)
        if sel.sum() < 6:
            return out
        nn = n[sel]
        hi = np.percentile(nn, 98)
        keep = sel.copy(); keep[sel] = nn > hi - 0.5
        if keep.sum() < 5:
            return out
        cc, uu = fit_line2(q[keep])
        if uu @ sa < 0:
            uu = -uu
        plates.append((cc, uu, keep.sum()))
    (c1, u1, _), (c2, u2, _) = plates
    A = np.c_[u1, -u2]
    if abs(np.linalg.det(A)) < 1e-3:
        return out
    k = np.linalg.solve(A, c2 - c1)
    S = c1 + k[0] * u1
    bl = unit(u1 + u2)
    pl = np.array([-bl[1], bl[0]])
    r = q - S
    s = r @ bl; w = r @ pl
    strip = (np.abs(w) < 0.35) & (s > -0.5) & (s < 5)
    if strip.sum() < 3:
        out.update(S=S, bl=bl, gap=True)
        return out
    sv = s[strip]
    smin = sv.min()
    sa_ = np.median(sv[sv < smin + 0.2])
    Apex = S + sa_ * bl
    lab = np.zeros(2)
    off_b = (lab - Apex) @ bl  # +: label deeper into material than outer apex
    off_w = (lab - Apex) @ pl
    incl = np.degrees(np.arccos(np.clip(u1 @ u2, -1, 1)))
    out.update(ok=True, S=S, bl=bl, Apex=Apex, off_b=off_b, off_w=off_w,
               dist=np.hypot(off_b, off_w), incl=incl, sharp_to_apex=sa_,
               label_from_sharp=(lab - S) @ bl,
               apex3d=P0 + Apex[0] * c0 + Apex[1] * b0)
    return out


def main():
    import time, signal
    signal.alarm(int(os.environ.get("V1_ALARM", "400")))
    T0 = time.time()
    P, N = load_ply(SCAN)
    print("loaded", P.shape, time.time() - T0, flush=True)
    tree = cKDTree(P)
    print("tree", time.time() - T0, flush=True)
    J = json.load(open(LINES))
    res = {}
    for B in J["bends"]:
        bid = B["bend_id"]
        s = np.array(B["label_line"]["start"]); e = np.array(B["label_line"]["end"])
        d = unit(e - s); Ln = np.linalg.norm(e - s)
        b = np.array(B["bisector"]); b = unit(b - (b @ d) * d)
        c = np.cross(d, b)
        L = dict(s=s, d=d, b=b, c=c, sa=unit(B["side_dirs"]["plane_a"]), sb=unit(B["side_dirs"]["plane_b"]))
        ear = Ln < 20
        step = 0.25 if ear else 1.0
        slab = 0.25 if ear else 0.5
        plate_a = (2.5, 6.0) if ear else (3.5, 10.0)
        ext = 5.0 if ear else 25.0
        ts = np.arange(-ext, Ln + ext + 1e-9, step)
        st = [section(P, tree, L, t, slab, 14.0, plate_a) for t in ts]
        ang = B["included_angle_deg"]
        fold0 = np.array([x is not None and x["ok"] and abs(x["incl"] - ang) < 12 and x["dist"] < 3.0 for x in st])
        # strict: a real bend arc must be present (sharp-to-apex gap near the fold's typical value)
        med_sta = np.median([x["sharp_to_apex"] for x, f, t in zip(st, fold0, ts) if f and 0 <= t <= Ln])
        fold = np.array([f and (0.5 * med_sta < x["sharp_to_apex"] < 1.8 * med_sta) for x, f in zip(st, fold0)])
        rows = []
        for t, x, f in zip(ts, st, fold):
            if x is not None and x["ok"]:
                rows.append([t, f, x["off_b"], x["off_w"], x["dist"], x["incl"], x["sharp_to_apex"], x["label_from_sharp"]])
            else:
                rows.append([t, False] + [np.nan] * 6)
        R = np.array(rows, float)
        np.savetxt(os.path.join(HERE, f"stations_{bid}.csv"), R, delimiter=",", fmt="%.4f",
                   header="t_mm,fold,off_depth_mm(+into material),off_lateral_mm,dist_mm,incl_deg,sharp_to_apex_mm,label_from_sharp_mm")
        tf = ts[fold]
        fold_lo, fold_hi = (tf.min(), tf.max()) if len(tf) else (np.nan, np.nan)
        inlab = (ts >= 0) & (ts <= Ln)
        cov = np.clip(min(Ln, fold_hi) - max(0, fold_lo), 0, None) / (fold_hi - fold_lo)
        # gaps inside fold span
        span = (ts >= fold_lo) & (ts <= fold_hi)
        Fin = R[fold & inlab]
        if len(Fin) == 0:
            Fin = np.full((1, 8), np.nan)
        # best straight line through measured apex stations (3D)
        A3 = np.array([x["apex3d"] for x, f in zip(st, fold) if f])
        np.savez(os.path.join(HERE, f"apex3d_{bid}.npz"), t=ts[fold], apex=A3, bl3=np.array([x["bl"][0] * c + x["bl"][1] * b for x, f in zip(st, fold) if f]))
        if len(A3) < 3:
            A3 = np.array([s, s + d, s + 2 * d])
        cen = A3.mean(0); _, _, vt = np.linalg.svd(A3 - cen); dd = vt[0]
        def dist_to_line(p, c0, u):
            r = p - c0; return np.linalg.norm(r - np.outer(r @ u, u), axis=1)
        lab_pts = s + np.outer((A3 - s) @ d, d)
        lab_d = np.linalg.norm(A3 - lab_pts, axis=1)
        bf_d = dist_to_line(A3, cen, dd)
        # signed mean offset of label vs best-fit line along b (into material +)
        bf_at = cen + np.outer((lab_pts - cen) @ dd, dd)
        bias_b = np.median((lab_pts - bf_at) @ b); bias_c = np.median((lab_pts - bf_at) @ c)
        par = np.degrees(np.arccos(abs(dd @ d)))
        q = {}
        for fr in (0.25, 0.5, 0.75):
            t0 = fr * Ln
            i = np.argmin(np.abs(ts - t0))
            # nearest station with a valid fold within +-3 mm (notch handling)
            cand = [j for j in range(len(ts)) if abs(ts[j] - t0) <= 15 and fold[max(0, j - int(2 / step)):j + int(2 / step) + 1].all()]
            j = min(cand, key=lambda j: abs(ts[j] - t0)) if cand else i
            x = st[j]
            q[fr] = dict(t=float(ts[j]), ok=bool(fold[j]),
                         off_depth=float(x["off_b"]) if x and x["ok"] else None,
                         off_lat=float(x["off_w"]) if x and x["ok"] else None,
                         incl=float(x["incl"]) if x and x["ok"] else None)
            plot_section(bid, fr, x, os.path.join(HERE, f"sec_{bid}_{int(fr*100)}.png"), B)
        res[bid] = dict(length=float(Ln), median_sharp_to_apex=float(med_sta), fold_t=[float(fold_lo), float(fold_hi)],
                        fold_len=float(fold_hi - fold_lo), coverage_of_fold=float(cov),
                        fold_stations_in_span=int(fold[span].sum()), span_stations=int(span.sum()),
                        start_excess=float(0 - fold_lo), end_excess=float(Ln - fold_hi),
                        sections=q,
                        full_off_depth_median=float(np.nanmedian(Fin[:, 2])),
                        full_off_depth_p10_p90=[float(np.nanpercentile(Fin[:, 2], 10)), float(np.nanpercentile(Fin[:, 2], 90))],
                        full_off_lat_median=float(np.nanmedian(Fin[:, 3])),
                        full_dist_median=float(np.nanmedian(Fin[:, 4])), full_dist_p90=float(np.nanpercentile(Fin[:, 4], 90)),
                        full_dist_max=float(np.nanmax(Fin[:, 4])),
                        incl_median=float(np.nanmedian(Fin[:, 5])), labeler_incl=float(ang),
                        label_dist_to_apex_median=float(np.median(lab_d)), label_dist_p90=float(np.percentile(lab_d, 90)),
                        bestfit_dist_median=float(np.median(bf_d)), bestfit_dist_p90=float(np.percentile(bf_d, 90)),
                        label_vs_bestfit_bias_depth=float(bias_b), label_vs_bestfit_bias_lat=float(bias_c),
                        bestfit_vs_label_angle_deg=float(par))
        plot_profile(bid, ts, R, Ln, os.path.join(HERE, f"profile_{bid}.png"))
        print(bid, json.dumps(res[bid], indent=None), time.time() - T0, flush=True)
    json.dump(res, open(os.path.join(HERE, "v1_results.json"), "w"), indent=1)


def plot_section(bid, fr, x, path, B):
    fig, ax = plt.subplots(figsize=(6, 6))
    if x is None:
        plt.close(fig); return
    q = x["q"]
    ax.scatter(q[:, 0], q[:, 1], s=2, c="0.5", label="scan pts (slab)")
    ax.plot(0, 0, "o", c="tab:red", ms=8, label="label line")
    if x.get("ok"):
        ax.plot(*x["Apex"], "+", c="tab:green", ms=14, mew=2, label="my outer apex")
        ax.plot(*x["S"], "x", c="k", ms=10, mew=2, label="my virtual sharp")
        ax.set_title(f"{bid} @{int(fr*100)}% t={x['t']:.1f}  off depth {x['off_b']:+.2f} lat {x['off_w']:+.2f} mm\n(+depth = label inside material)  incl {x['incl']:.1f}")
        cx, cy = x["Apex"]
    else:
        ax.set_title(f"{bid} @{int(fr*100)}% t={x['t']:.1f} NO FOLD FOUND")
        cx, cy = 0, 0
    ax.set_xlim(cx - 6, cx + 6); ax.set_ylim(cy - 3, cy + 9)
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower left")
    ax.set_xlabel("lateral c (mm)"); ax.set_ylabel("b: into material (mm)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_profile(bid, ts, R, Ln, path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    f = R[:, 1] > 0.5
    ax[0].plot(ts[f], R[f, 2], ".", ms=3, label="depth offset (+ inside material)")
    ax[0].plot(ts[f], R[f, 3], ".", ms=3, label="lateral offset")
    ax[0].axhline(0, c="k", lw=.5); ax[0].set_ylabel("label - outer apex (mm)"); ax[0].legend(fontsize=7)
    ax[1].plot(ts, R[:, 1], "-", label="fold present")
    for a in ax:
        a.axvline(0, c="r", ls="--"); a.axvline(Ln, c="r", ls="--")
    ax[1].set_xlabel("t along label (mm); red = label ends"); ax[0].set_title(f"{bid} full-length check")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
