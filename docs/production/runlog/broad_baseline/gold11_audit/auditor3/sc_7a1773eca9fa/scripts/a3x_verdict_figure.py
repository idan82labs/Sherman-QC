"""Auditor 3 (resumed run): one verdict figure, mid-span (and 25/75 %) sections of every gold line with the
concentric-arc fit from a3x_arcfit_check.json: outer apex, inner apex, inner tangent points, gold, 4 mm
matcher circle around the outer apex. Also prints M01 statistics restricted to stations where the inner
arc was actually scanned (direct, not inferred).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_7a1773eca9fa/scripts")
from a3_load import load  # noqa: E402
import a3x_arcfit_check as A  # noqa: E402

AUD = A.AUD
J = json.load(open(AUD + "/a3x_arcfit_check.json"))
v, n, f = load()
patches = {p["id"]: p for p in json.load(open(A.PATCHES))["patches"]}
gold = {b["bend_id"]: b for b in json.load(open(A.GOLD))["bends"]}

# M01: direct stats where the inner arc is scanned
rows = [r for r in J["detail"]["M01"]["rows"] if r.get("ok")]
lo, hi = J["detail"]["M01"]["gold_s"]
sc = [r for r in rows if lo <= r["s"] <= hi and r.get("inner_arc_scanned")]
nsc = [r for r in rows if lo <= r["s"] <= hi and not r.get("inner_arc_scanned")]
print(f"M01 stations in gold span: inner arc scanned {len(sc)}, not scanned {len(nsc)}")
for name, rr in (("scanned", sc), ("not-scanned", nsc)):
    if rr:
        dep = np.array([r["gold_depth_below_inner_skin"] for r in rr])
        ga = np.array([r["gold_to_outer_apex"] for r in rr])
        th = np.array([r["theta_gold"] for r in rr])
        ss = np.array([r["s"] for r in rr])
        print(f"  {name}: s in [{ss.min():.0f},{ss.max():.0f}]  depth below inner skin med {np.median(dep):.2f} (p10 {np.percentile(dep,10):.2f}, p90 {np.percentile(dep,90):.2f})"
              f"  gold->outer apex med {np.median(ga):.2f}  theta med {np.median(th):.1f} deg  Ri med {np.median([r['Ri'] for r in rr]):.2f}")

fig, axs = plt.subplots(3, 5, figsize=(30, 18))
for col, bid in enumerate(("M01", "M02", "M03", "M04", "M05")):
    det = J["detail"][bid]
    d = np.array(det["d"])
    pb, pp, pname = A.FOLDS[bid]
    nb = A.unit(patches[pb]["normal"])
    S, E = np.array(gold[bid]["line_start"]), np.array(gold[bid]["line_end"])
    mid = (S + E) / 2
    e1 = A.unit(nb - (nb @ d) * d)
    e2 = np.cross(d, e1)
    lo, hi = det["gold_s"]
    ok = [r for r in det["rows"] if r.get("ok") and lo <= r["s"] <= hi and not r.get("suspect_fit")]
    for rowi, fr in enumerate((0.25, 0.5, 0.75)):
        target = lo + fr * (hi - lo)
        r = min(ok, key=lambda q: abs(q["s"] - target))
        rel = v - mid
        s_all = rel @ d
        m = (np.abs(s_all - r["s"]) <= 0.5)
        X = np.c_[rel[m] @ e1, rel[m] @ e2]
        NX = np.c_[n[m] @ e1, n[m] @ e2]
        C = np.array(r["C"])
        keep = np.hypot(*(X - C).T) < 16
        X, NX = X[keep], NX[keep]
        radial = (X - C) / (np.hypot(*(X - C).T)[:, None] + 1e-12)
        facing = np.einsum("ij,ij->i", radial, NX / (np.linalg.norm(NX, axis=1, keepdims=True) + 1e-12))
        ax = axs[rowi, col]
        ax.scatter(*X[facing > 0].T, s=3, c="tab:blue", label="normal faces away from arc centre (outer side)")
        ax.scatter(*X[facing <= 0].T, s=3, c="tab:orange", label="normal faces arc centre (inner/concave side)")
        G = np.array(r["gold"])
        # recompute apex direction from the stored geometry: outer apex = C + Ro*n_apex
        # n_apex is the unit vector from C toward the outer apex: take it from the theta definition
        # (gold_minus_apex_* are stored; rebuild n_apex from C, Ro and the stored outer-apex distance)
        Ro, Ri = r["Ro"], r["Ri"]
        # n_apex: direction of the outer-facing arc points' mean around the bisector -> use the stored values
        # theta_gold = atan2(g.across, g.n_apex); |g| = r_gold  =>  solve for n_apex
        n_apex = np.array(r["n_apex"])  # stored by a3x_arcfit_check (bisector of the plates' outward normals)
        apex_o = C + Ro * n_apex
        apex_i = C + Ri * n_apex
        half = np.radians(r["beta"] / 2)
        for sgn in (+1, -1):
            rt = np.array([[np.cos(sgn * half), -np.sin(sgn * half)], [np.sin(sgn * half), np.cos(sgn * half)]])
            u = rt @ n_apex
            ax.plot(*np.c_[C + Ri * u, C + Ro * u], color="m", lw=2)
        tt = np.linspace(-half, half, 60)
        for R_, sty in ((Ri, "--"), (Ro, "--")):
            pts = np.array([C + R_ * (np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ n_apex) for a in tt])
            ax.plot(*pts.T, sty, color="0.3", lw=1)
        circ = np.array([apex_o + 4 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 100)])
        ax.plot(*circ.T, ":", color="green", lw=1.5, label="4 mm matcher radius around outer apex")
        ax.plot(*apex_o, "D", ms=10, mfc="cyan", mec="k", label="outer apex (owner/AI locus)")
        ax.plot(*apex_i, "s", ms=8, mfc="yellow", mec="k", label="inner apex")
        ax.plot(*C, "+", ms=12, color="brown", mew=2, label="arc centre")
        ax.plot(*G, "*", ms=18, mfc="red", mec="k", label="human gold")
        ax.set_aspect("equal")
        ax.set_xlim(C[0] - 11, C[0] + 11); ax.set_ylim(C[1] - 11, C[1] + 11)
        ax.grid(alpha=0.3)
        ax.set_title(f"{bid} ({pname}) @ {int(fr*100)}%  bend {r['beta']:.0f} deg  Ri {Ri:.1f} Ro {Ro:.1f}\n"
                     f"gold->outer apex {r['gold_to_outer_apex']:.2f} mm, ->inner apex {r['gold_to_inner_apex']:.2f}, "
                     f"nearest raw vtx {r['gold_nearest_scan']:.2f}\nzone {r['gold_zone']}"
                     + (f" past inner tangent {r['gold_past_inner_tangent']:.1f} mm" if "gold_past_inner_tangent" in r else
                        f" theta {r['theta_gold']:.0f}/{r['beta']/2:.0f} deg"), fontsize=10)
axs[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("sc_7a1773eca9fa (49001000) auditor 3: human gold vs fold geometry (magenta = inner/outer tangent lines; dashed = fitted arcs)", fontsize=15)
fig.tight_layout()
fig.savefig(AUD + "/a3x_verdict_sections.png", dpi=55)
print("wrote", AUD + "/a3x_verdict_sections.png")
