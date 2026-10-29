"""Verifier2: how close can ANY straight line get to the measured outer apex? (bowed-fold judgement)"""
import json, os
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "v2_stations.json")))
cfg = {b["bend_id"]: b for b in json.load(open(os.path.join(HERE, "..", "lines.json")))["bends"]}
# fold extents read from the sections (verifier2 visual + leg counts)
EXT = {"B1": (0, 306), "B2": (-1, 306), "B3": (-1, 78), "B4": (-1, 78), "B5": (-0.5, 12.0), "B6": (-0.5, 12.0)}
res = {}
fig, axs = plt.subplots(len(EXT), 2, figsize=(16, 3.2 * len(EXT)))
for i, (bid, (f0, f1)) in enumerate(EXT.items()):
    b = cfg[bid]
    Ls = np.array(b["label_line"]["start"]); Le = np.array(b["label_line"]["end"])
    L = np.linalg.norm(Le - Ls); d = (Le - Ls) / L
    u = np.array(b["bisector"]); u = u - (u @ d) * d; u /= np.linalg.norm(u); w = np.cross(d, u)
    rows = [q for q in R[bid]["rows"] if f0 <= q["t"] <= f1 and "apex_xyz" in q and abs(q.get("off_bis", 9)) < 3
            and q.get("leg_a_n", 0) >= 8 and q.get("leg_b_n", 0) >= 8]
    A = np.array([q["apex_xyz"] for q in rows]) - Ls
    t = A @ d; aw = A @ w; au = A @ u  # apex in label frame; label line is aw=au=0
    def dist(p):
        w0, w1, u0, u1 = p
        lw = w0 + (w1 - w0) * t / L; lu = u0 + (u1 - u0) * t / L
        return np.hypot(aw - lw, au - lu), (lu - au)
    dl, sl = dist(np.zeros(4))
    # LSQ
    M = np.c_[1 - t / L, t / L]
    cw = np.linalg.lstsq(M, aw, rcond=None)[0]; cu = np.linalg.lstsq(M, au, rcond=None)[0]
    p_lsq = np.array([cw[0], cw[1], cu[0], cu[1]])
    out = {"n": len(t), "label": dict(median=float(np.median(dl)), p90=float(np.percentile(dl, 90)), max=float(dl.max()),
                                     mean_signed_u=float(np.mean(sl)), median_signed_u=float(np.median(sl)))}
    for nm, obj in (("lsq", None), ("min_median", lambda p: np.median(dist(p)[0])), ("min_p90", lambda p: np.percentile(dist(p)[0], 90)),
                    ("minimax", lambda p: dist(p)[0].max())):
        if obj is None:
            p = p_lsq
        else:
            best = None
            for st in (p_lsq, np.zeros(4)):
                r = minimize(obj, st, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-5, maxiter=20000))
                if best is None or r.fun < best.fun:
                    best = r
            p = best.x
        dd, ss = dist(p)
        out[nm] = dict(median=float(np.median(dd)), p90=float(np.percentile(dd, 90)), max=float(dd.max()),
                       mean_signed_u=float(np.mean(ss)), median_signed_u=float(np.median(ss)), params_w0w1u0u1=[float(x) for x in p])
    res[bid] = out
    ax = axs[i, 0]
    ax.plot(t, au, ".", ms=3, label="measured outer apex u (label frame; + = into material)")
    ax.axhline(0, color="tab:red", label="label line")
    ax.plot(t, p_lsq[2] + (p_lsq[3] - p_lsq[2]) * t / L, "k--", lw=1, label="LSQ straight line")
    ax.set_ylabel("u mm"); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title(f"{bid}: apex depth vs label (u = label bisector)", fontsize=9)
    ax = axs[i, 1]
    ax.plot(t, aw, ".", ms=3, color="tab:purple", label="measured outer apex w (across fold)")
    ax.axhline(0, color="tab:red"); ax.plot(t, p_lsq[0] + (p_lsq[1] - p_lsq[0]) * t / L, "k--", lw=1)
    ax.set_ylabel("w mm"); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title(f"{bid}: apex lateral vs label", fontsize=9)
axs[-1, 0].set_xlabel("t along label (mm)"); axs[-1, 1].set_xlabel("t along label (mm)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "v2_bestline.png"), dpi=80)
json.dump(res, open(os.path.join(HERE, "v2_bestline.json"), "w"), indent=1)
for k, v in res.items():
    print(k, "n", v["n"])
    for m in ("label", "lsq", "min_median", "min_p90", "minimax"):
        print("   %-10s med %.3f p90 %.3f max %.3f  mean_signed_u %+.3f med_signed_u %+.3f" % (m, v[m]["median"], v[m]["p90"], v[m]["max"], v[m]["mean_signed_u"], v[m]["median_signed_u"]))
