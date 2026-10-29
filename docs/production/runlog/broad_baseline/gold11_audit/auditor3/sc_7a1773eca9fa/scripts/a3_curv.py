"""Auditor 3: outer-skin normal-angle vs arc-length profile through each fold (single arc vs compound)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
from a3_loci import analyse  # noqa: E402

OUT = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_7a1773eca9fa"
fig, axs = plt.subplots(2, 5, figsize=(26, 9))
report = {}
for k, bid in enumerate(("M01", "M02", "M03", "M04", "M05")):
    F, rows, (lo, hi) = analyse(bid)
    for fr, col in ((0.15, "tab:blue"), (0.5, "tab:green"), (0.85, "tab:red")):
        target = lo + fr * (hi - lo)
        r = min([q for q in rows if q["Ro"] is not None], key=lambda q: abs(q["s"] - target))
        m = np.abs(F.s - r["s"]) <= 0.5
        xy, nxy = F.xy[m], F.nxy[m]
        Xo, g, acr = r["Xo"], r["g"], r["acr"]
        half = np.radians(180 - r["bend"]) / 2
        C = Xo + g * r["Ro"] / np.sin(half)
        rel = xy - C
        rad = np.hypot(*rel.T)
        nout = -(nxy @ g)
        # outer skin: points at radius >= Ro-0.8 from the centre side, outward facing w.r.t. centre
        radial = (rel / rad[:, None])
        facing = np.einsum("ij,ij->i", radial, nxy / np.linalg.norm(nxy, axis=1, keepdims=True))
        sel = (facing > 0.5) & (np.abs(rad - r["Ro"]) < 1.2) | (
            (facing > 0.5) & (rad < r["Ro"] + 25) & (rad > r["Ro"] - 1.5) & ((rel @ g) < 0.5 * r["Ro"]))
        # keep the outer sheet: for each point, signed distance to the outer skin lines must be small
        pts = xy[sel]; nn = nxy[sel]
        # order by polar angle around C, measured from the bisector (-g direction = apex)
        ang = np.degrees(np.arctan2(rel[sel] @ acr, -(rel[sel] @ g)))
        o = np.argsort(ang)
        pts, nn, ang = pts[o], nn[o], ang[o]
        # restrict to the outer sheet near the arc: within 15 mm of apex along the curve
        nang = np.degrees(np.arctan2(nn @ acr, -(nn @ g)))  # normal direction angle, 0 at apex
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        # zero arc length at the apex (closest to angle 0)
        i0 = int(np.argmin(np.abs(ang)))
        seg = seg - seg[i0]
        keep = np.abs(seg) < 14
        axs[0, k].plot(seg[keep], nang[keep], ".", ms=2, color=col, label=f"{int(fr*100)}% s={r['s']:.0f} bend={r['bend']:.0f} Ro={r['Ro']:.1f}")
        # derivative (curvature proxy) in 0.5 mm bins
        bins = np.arange(-14, 14.01, 0.5)
        idx = np.digitize(seg[keep], bins)
        med = np.array([np.median(nang[keep][idx == i]) if (idx == i).sum() >= 2 else np.nan for i in range(1, len(bins))])
        cen = (bins[:-1] + bins[1:]) / 2
        dk = np.gradient(med, cen)
        axs[1, k].plot(cen, dk, "-", color=col)
        report.setdefault(bid, []).append(dict(frac=fr, s=r["s"], bend=r["bend"], Ro=r["Ro"],
                                              curv_peak_deg_per_mm=float(np.nanmax(np.abs(dk))),
                                              n_points=int(keep.sum())))
    axs[0, k].set_title(f"{bid}: outer-skin normal angle vs arc length (0 = apex)")
    axs[0, k].set_xlabel("arc length from outer apex (mm)"); axs[0, k].set_ylabel("normal angle (deg)")
    axs[0, k].legend(fontsize=7); axs[0, k].grid(alpha=0.3)
    axs[1, k].set_title(f"{bid}: d(angle)/ds (deg/mm); single arc = one plateau ~ 57.3/Ro")
    axs[1, k].set_xlabel("arc length (mm)"); axs[1, k].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/a3_curvature_profiles.png", dpi=75)
for bid, v in report.items():
    print(bid, v)
