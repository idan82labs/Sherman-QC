"""Auditor-1 (sc_71cf0b216f43): perpendicular cross-sections of every gold line at 5 stations.

Each panel: scan slab (+-0.75 mm) in the section plane of the fold the gold line belongs to,
points coloured by mesh normal (red = faces the convex/outer side, blue = faces the concave/inner side),
measured outer apex (magenta diamond) with the 4 mm matcher circle around it (dotted), outer virtual
sharp Xo (blue x), inner virtual sharp Xi (purple x), gold crossing (green star). Title: station,
gold->apex distance, along-bisector (+ = concave side) / across (+ = toward wall) components.
One PNG per line: a1_xsec_<Mxx>.png. Stations are at 3/25/50/75/97 % of the gold span; for M01/M02
two extra stations inside the measured lip fold are shown, because the gold overruns the fold.
"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_common import OUT, get, unit
from a1_xsec import FOLDS, section, analyse_station, gold_cross

P, VN, F, G, R, c = get()
gold = {bid: (s, e) for bid, s, e in G}
ap = json.load(open(OUT + "/a1_apex_lines.json"))["folds"]
sorted_by = {ax: (np.argsort(P[:, ax]), np.sort(P[:, ax])) for ax in (0, 1)}
for fid, f in FOLDS.items():
    gid = f["gold"]; ax_ = f["axis"]
    Gs, Ge = gold[gid]
    s_lo, s_hi = sorted([Gs[ax_], Ge[ax_]])
    fe = ap[fid]["fold_extent_axis_coord"]
    tlist = [0.03, 0.25, 0.5, 0.75, 0.97]
    stations = [Gs[ax_] + t * (Ge[ax_] - Gs[ax_]) for t in tlist]
    if fid in ("RT", "LT"):
        stations = [stations[0], fe[0] + 3.0, stations[2], fe[1] - 3.0, stations[4]]
    fig, axes = plt.subplots(1, 5, figsize=(26, 6.2), dpi=80)
    for axp, s0 in zip(axes, stations):
        q, nq, idx = section(P, VN, sorted_by[ax_], ax_, s0, f["corner"], rad=16.0)
        r = analyse_station(q, nq, f)
        g2, t = gold_cross(Gs, Ge, ax_, s0)
        c0 = np.asarray(f["corner"], float)
        if r.get("ok"):
            gin = np.array(r["g_in"])
            nn = nq / (np.linalg.norm(nq, axis=1)[:, None] + 1e-12)
            col = np.where(nn @ gin < 0, "#d62728", "#1f77b4")
            axp.scatter(q[:, 0], q[:, 1], s=3, c=col, linewidths=0)
            apex = np.array(r["apex"]); Xo = np.array(r["Xo"])
            axp.plot(*apex, "D", color="magenta", ms=7, zorder=5, label="outer apex (measured)")
            axp.add_patch(plt.Circle(apex, 4.0, fill=False, ls=":", color="magenta"))
            axp.plot(*Xo, "x", color="blue", ms=9, mew=2, label="outer virtual sharp")
            if r.get("Xi") is not None:
                axp.plot(*r["Xi"], "x", color="purple", ms=9, mew=2, label="inner virtual sharp")
            d = g2 - apex
            perp = np.array([-gin[1], gin[0]])
            if perp @ np.array(r["a"]) < 0:
                perp = -perp
            ttl = f"{gid} {('x' if ax_ == 0 else 'y')}={s0:.1f} (t={t:.2f})\n|gold-apex|={np.linalg.norm(d):.2f}  along={d @ gin:+.2f}  across={d @ perp:+.2f} mm"
        else:
            axp.scatter(q[:, 0], q[:, 1], s=3, c="gray", linewidths=0)
            ttl = f"{gid} {('x' if ax_ == 0 else 'y')}={s0:.1f} (t={t:.2f})\nno fold fit: {r.get('why')}"
        inside = fe[0] <= s0 <= fe[1]
        ttl += "" if inside else "\nOUTSIDE measured fold extent"
        axp.plot(*g2, "*", color="#00a000", ms=16, mec="k", zorder=6, label="gold crossing")
        cx, cy = 0.5 * (c0 + (g2 if np.linalg.norm(g2 - c0) < 20 else c0))
        half = max(8.0, 0.5 * np.linalg.norm(g2 - c0) + 6.0)
        axp.set_xlim(cx - half, cx + half); axp.set_ylim(cy - half, cy + half)
        axp.set_aspect("equal"); axp.grid(alpha=0.3); axp.set_title(ttl, fontsize=9)
    axes[0].legend(fontsize=7, loc="lower right")
    lab = "(y,z)" if ax_ == 0 else "(x,z)"
    fig.suptitle(f"sc_71cf0b216f43 {gid} -> fold {fid}: {f['desc']}; section coords {lab} part frame; "
                 f"fold extent {fe[0]:.1f}..{fe[1]:.1f}, gold span {s_lo:.1f}..{s_hi:.1f}; red = outer-facing skin, blue = inner-facing", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT + f"/a1_xsec_{gid}.png")
    plt.close(fig)
print("ok")
