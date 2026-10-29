"""Verifier2 summary: fold extents, coverage, offsets along the whole fold, best straight line."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "v2_stations.json")))
cfg = {b["bend_id"]: b for b in json.load(open(os.path.join(HERE, "..", "lines.json")))["bends"]}
out = {}
lines_txt = []
for bid, r in R.items():
    L = r["L"]; rows = r["rows"]
    b = cfg[bid]
    Ls = np.array(b["label_line"]["start"]); Le = np.array(b["label_line"]["end"])
    d = (Le - Ls) / L
    ear = bid in ("B5", "B6")
    t = np.array([q["t"] for q in rows])
    pres = np.array([(q.get("leg_a_n", 0) >= 8) and (q.get("leg_b_n", 0) >= 8) and ("off_bis" in q) for q in rows])
    ob = np.array([q.get("off_bis", np.nan) for q in rows])
    ol = np.array([q.get("off_lat", np.nan) for q in rows])
    ang = np.array([q.get("angle_incl", np.nan) for q in rows])
    # sanity: apex must be a real skin point -> ignore stations where offsets are absurd
    pres &= np.abs(ob) < 3
    tp = t[pres]
    f0, f1 = tp.min(), tp.max()
    # gaps inside fold extent
    step = t[1] - t[0]
    gaps = []
    inside = (t >= f0) & (t <= f1)
    run = None
    for ti, pi in zip(t[inside], pres[inside]):
        if not pi and run is None:
            run = ti
        if pi and run is not None:
            gaps.append((run, ti - step)); run = None
    fold_len = f1 - f0
    ov = max(0.0, min(L, f1) - max(0.0, f0))
    cov = ov / fold_len
    inl = pres & (t >= 0) & (t <= L)
    # stations at 25/50/75 (nearest present station)
    def at(fr):
        k = np.argmin(np.where(inl, np.abs(t - fr * L), 1e9))
        return t[k], ob[k], ol[k]
    # best straight 3D line through measured apices (present stations within fold extent)
    A = np.array([rows[i]["apex_xyz"] for i in range(len(rows)) if pres[i]])
    c = A.mean(0); _, _, vt = np.linalg.svd(A - c); dv = vt[0]
    def dist_line(P, c0, d0):
        Q = P - c0
        return np.linalg.norm(Q - np.outer(Q @ d0, d0), axis=1)
    d_label = dist_line(A, Ls, d)
    d_best = dist_line(A, c, dv)
    par_deg = np.degrees(np.arccos(min(1, abs(dv @ d))))
    # bisector-direction mean offset of label vs best line
    u = np.array(b["bisector"]); u = u - (u @ d) * d; u /= np.linalg.norm(u)
    # signed offset of label line from best line, along u, at fold midpoint
    tm = 0.5 * (max(0, f0) + min(L, f1))
    Pl = Ls + tm * d
    Pb = c + ((Pl - c) @ dv) * dv
    sys_off_u = float((Pl - Pb) @ u)
    res = dict(L=L, fold_start=f0, fold_end=f1, fold_len=fold_len, gaps=gaps, coverage=cov,
               end_err_start=float(0 - f0), end_err_end=float(L - f1),
               off_bis_median=float(np.nanmedian(ob[inl])), off_bis_p10=float(np.nanpercentile(ob[inl], 10)),
               off_bis_p90=float(np.nanpercentile(ob[inl], 90)), off_bis_absmax=float(np.nanmax(np.abs(ob[inl]))),
               off_lat_median=float(np.nanmedian(ol[inl])), off_lat_absp90=float(np.nanpercentile(np.abs(ol[inl]), 90)),
               s25=at(0.25), s50=at(0.5), s75=at(0.75), angle_incl_median=float(np.nanmedian(ang[inl])),
               bend_angle_median=float(180 - np.nanmedian(ang[inl])), label_incl=r["label_incl"],
               n_present_in_label=int(inl.sum()),
               dist_apex_to_label_median=float(np.median(d_label)), dist_apex_to_label_p90=float(np.percentile(d_label, 90)),
               dist_apex_to_best_median=float(np.median(d_best)), dist_apex_to_best_p90=float(np.percentile(d_best, 90)),
               label_vs_best_parallel_deg=float(par_deg), label_minus_best_along_bisector_mm=sys_off_u)
    out[bid] = res
    # figure
    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax in axs:
        ax.axvspan(0, L, color="tab:red", alpha=0.07, label="label extent")
        for g0, g1 in gaps:
            ax.axvspan(g0, g1, color="0.6", alpha=0.3)
        ax.axvline(f0, color="tab:green", ls="--"); ax.axvline(f1, color="tab:green", ls="--")
        ax.grid(alpha=0.3)
    axs[0].plot(t[pres], ob[pres], ".", ms=4, label="off_bis (label - measured outer apex, + = into material)")
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_ylabel("mm"); axs[0].legend(fontsize=8)
    axs[1].plot(t[pres], ol[pres], ".", ms=4, color="tab:purple", label="lateral offset (across fold)")
    axs[1].axhline(0, color="k", lw=0.8); axs[1].set_ylabel("mm"); axs[1].legend(fontsize=8)
    axs[2].plot(t[pres], ang[pres], ".", ms=4, color="tab:orange", label="local included angle (deg)")
    axs[2].axhline(r["label_incl"], color="k", lw=0.8, ls=":", label="label included angle")
    axs[2].legend(fontsize=8); axs[2].set_xlabel("t along label line (mm); green dashed = measured fold ends, grey = gaps (notch)")
    fig.suptitle(f"verifier2 {bid}: whole-fold check, label L={L:.1f}, fold {f0:.1f}..{f1:.1f} (len {fold_len:.1f}), coverage {cov:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"v2_fold_{bid}.png"), dpi=85); plt.close(fig)
json.dump(out, open(os.path.join(HERE, "v2_summary.json"), "w"), indent=1, default=float)
for k, v in out.items():
    print(k, json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}, default=float))
