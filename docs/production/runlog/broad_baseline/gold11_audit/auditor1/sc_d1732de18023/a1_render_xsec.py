"""Auditor-1 perpendicular cross-section renders (own fit) for every gold line of sc_d1732de18023.
Rows: context (+-25 mm) and close-up (+-9 mm) at 10 / 50 / 90 % of the gold span.  The 4 mm circle is
centred on the fitted OUTER APEX (the locus the AI gold / bench 'lines' use), so a gold star outside
it cannot be matched by an apex-locus line under the frozen 4 mm transverse limit."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from a1_common import OUT, gold_lines, load_all, perp_basis, unit  # noqa: E402
from a1_xsec import RAD, fit_station  # noqa: E402

FRACS = (0.10, 0.50, 0.90)
SLAB = 0.6


def main():
    xyz, nrm, _ = load_all()
    for bid, (a, b) in gold_lines().items():
        d = unit(b - a)
        L = float(np.linalg.norm(b - a))
        u, w = perp_basis(d)
        rel = xyz - a
        t = rel @ d
        r2 = rel - np.outer(t, d)
        near = (np.linalg.norm(r2, axis=1) < 40) & (t > -2) & (t < L + 2)
        idx = np.where(near)[0]
        tt, Q = t[idx], np.stack([rel[idx] @ u, rel[idx] @ w], 1)
        M = np.stack([nrm[idx] @ u, nrm[idx] @ w], 1)
        Mn = np.linalg.norm(M, axis=1)
        fig, axs = plt.subplots(2, 3, figsize=(19, 13))
        for col, fr in enumerate(FRACS):
            s0 = fr * L
            sel = np.abs(tt - s0) < SLAB
            q = Q[sel] - (d @ d) * 0  # 2D coords, gold point = origin
            mm = M[sel]
            mn = Mn[sel]
            ok = mn > 0.5
            k = ok & (np.linalg.norm(q, axis=1) < RAD)
            r = fit_station(q[k], mm[k] / mn[k][:, None])
            if r is None or r["Ao"] is None:
                axs[0, col].set_title(f"{bid} @ {int(fr*100)}%: fit failed")
                continue
            Ao, Xo, Xi, Ai, b_in = r["Ao"], r["Xo"], r["Xi"], r["Ai"], r["b_in"]
            # rotate so that the concave bisector points up and the apex is the origin
            ang = np.arctan2(b_in[1], b_in[0]) - np.pi / 2
            R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
            T = lambda p: (np.atleast_2d(p) - Ao) @ R.T  # noqa: E731
            qq = T(q)
            g = T(np.zeros(2))[0]
            for row, half in ((0, 25.0), (1, 9.0)):
                ax = axs[row, col]
                ax.scatter(qq[:, 0], qq[:, 1], s=2 if row else 1, c="0.35", lw=0)
                # flange skin lines
                for fi in (0, 1):
                    n, tdir = r["flange_n"][fi], r["flange_t"][fi]
                    for si, sk in enumerate(r["skins"][fi]):
                        p0 = n * sk
                        pts = np.array([p0 - tdir * 60, p0 + tdir * 60])
                        pp = T(pts)
                        ax.plot(pp[:, 0], pp[:, 1], "--", lw=0.6, color="tab:blue" if si == 0 else "tab:purple")
                ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=":", color="tab:red", lw=1.2))
                ax.plot(*T(Xo)[0], "x", color="tab:blue", ms=10, mew=2, label="outer virtual sharp")
                ax.plot(*T(Xi)[0], "x", color="tab:purple", ms=10, mew=2, label="inner virtual sharp")
                ax.plot(0, 0, "D", color="cyan", mec="k", ms=8, label="outer apex (fitted)")
                if Ai is not None:
                    ax.plot(*T(Ai)[0], "s", color="violet", mec="k", ms=6, label="inner apex (fitted)")
                ax.plot(*g, "*", color="#1a9c3a", mec="k", ms=18, label=f"gold {bid}")
                ax.set_xlim(-half, half)
                ax.set_ylim(-half * 0.45, half * 1.55)
                ax.set_aspect("equal")
                ax.grid(alpha=0.3)
                dg = float(np.linalg.norm(g))
                if row == 0:
                    ax.set_title(f"{bid} @ {int(fr*100)}% (t={s0:.0f} mm)  bend {r['bend_angle']:.1f} deg  "
                                 f"t={np.mean(r['th']):.2f} mm\nflange lengths {r['flange_len'][0]:.1f}/"
                                 f"{r['flange_len'][1]:.1f} mm (fit window {RAD:.0f} mm)", fontsize=9)
                else:
                    ax.set_title(f"close-up: gold->outer apex {dg:.2f} mm (along concave bisector "
                                 f"{g[1]:+.2f}, across {g[0]:+.2f})\n gold->outer sharp "
                                 f"{np.linalg.norm(g - T(Xo)[0]):.2f}, ->inner sharp "
                                 f"{np.linalg.norm(g - T(Xi)[0]):.2f}"
                                 + ("" if Ai is None else f", ->inner apex {np.linalg.norm(g - T(Ai)[0]):.2f}")
                                 + " mm", fontsize=9)
                if row == 1 and col == 0:
                    ax.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"sc_d1732de18023 auditor1 own fit: gold {bid} (len {L:.1f} mm) perpendicular sections, "
                     f"slab +-{SLAB} mm; outer apex at origin, concave side up; red dotted = 4 mm around apex",
                     fontsize=11)
        fig.tight_layout()
        p = OUT / f"a1_xsec_{bid}.png"
        fig.savefig(p, dpi=85)
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
