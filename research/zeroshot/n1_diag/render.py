"""Close-up renders for the N1 47085000 diagnosis: per failure, a local 3-D close-up (scan vertices, gold lines,
engine lines, intermediate candidates) and Z3 cross-sections at chosen slices.
usage: .venv/bin/python -B -m research.zeroshot.n1_diag.render SCRATCH_DIAG_DIR OUT_DIR"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from research.zeroshot.n1_diag.common import gold, eval_lines, pm

GREEN, ORANGE, RED, BLUE, MAG, CYAN, GREY = "#1a9850", "#f28e2b", "#d62728", "#1f5fd6", "#b03fb0", "#17becf", "#9a9a9a"

CASES = {
    "B1": {"gold": ["B1"], "fam_chain": "B1", "sections": [30, 10], "rej": [], "z2": [0], "corners_fam": 1,
           "title": "B1 end flange: Z3 corner track only on s = 48-80 mm (mesh torn at the fold for s = 6-46); Z2 line full length but absorbed by the short Z3 candidate at pooling"},
    "B4": {"gold": ["B4"], "fam_chain": "B4", "sections": [24, 36], "rej": [25], "z2": [1], "corners_fam": 1,
           "title": "B4 lanced lip: Z3 track breaks where the saddle lip is < 2 mm tall (s = 28-41 mm); piece 2 rejected (p_zs 0.39); Z2 line absorbed"},
    "B14": {"gold": ["B14"], "fam_chain": "B14", "sections": [21], "rej": [26], "z2": [], "corners_fam": 0,
            "title": "B14 hook lip: Z3 candidate matches gold (p_z3 0.90) but the pooled GBM gives 0.215 (no Z2 agreement; Z2 never segmented the lip)"},
    "B25": {"gold": ["B25", "B26"], "fam_chain": "B25", "sections": [26, 22], "rej": [16, 14], "z2": [], "corners_fam": 1,
            "title": "B25/B26 3 mm edge joggle: both corners born by Z3 (cands 16/14, both would match) but scored 0.016 / 0.009; Z2 has no web flange"},
}


def frame(g, R, rej):
    a, b = g
    u = (b - a) / np.linalg.norm(b - a)
    # local 'up' from the nearest Z3 candidate's bisector when available, else a perpendicular
    best = None
    for c in R["z3"]["candidates"]:
        m = pm(g, np.array([c["start"], c["end"]]))
        if m["axis_angle_deg"] < 5 and m["transverse_offset_mm"] < 5:
            if best is None or m["transverse_offset_mm"] < best[0]:
                best = (m["transverse_offset_mm"], np.asarray(c["bis"], float))
    up = best[1] if best is not None else np.cross(u, [0, 0, 1.0])
    up = up - u * (up @ u); up /= np.linalg.norm(up)
    w = np.cross(up, u)
    return 0.5 * (a + b), u, w, up


def loc(P, o, u, w, up):
    Q = np.atleast_2d(P) - o
    return np.stack([Q @ u, Q @ w, Q @ up], 1)


def crossing(line, lev, U, e1, e2):
    a, b = np.asarray(line[0], float), np.asarray(line[1], float)
    den = (b - a) @ U
    if abs(den) < 1e-9:
        return None
    t = (lev - a @ U) / den
    if t < -0.05 or t > 1.05:
        return None
    p = a + t * (b - a)
    return np.array([p @ e1, p @ e2])


def render(case, cfg, R, V, outdir, chains):
    G = gold()
    ev = eval_lines("runs/v1/zs")
    g0 = G[cfg["gold"][0]]
    o, u, w, up = frame(g0, R, cfg["rej"])
    L = np.linalg.norm(g0[1] - g0[0])
    pad = 12.0
    box = (L / 2 + pad, 16.0, 16.0)
    Vl = loc(V, o, u, w, up)
    m = (np.abs(Vl[:, 0]) < box[0]) & (np.abs(Vl[:, 1]) < box[1]) & (np.abs(Vl[:, 2]) < box[2])
    pts = Vl[m]
    if len(pts) > 60000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]
    nsec = len(cfg["sections"])
    fig = plt.figure(figsize=(12, 6.2), dpi=100)
    ax = fig.add_axes([0.0, 0.30, 0.56, 0.62], projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.15, c=GREY, alpha=0.35, depthshade=False, rasterized=True)

    def seg3(line, **kw):
        q = loc(np.asarray(line, float), o, u, w, up)
        ax.plot(q[:, 0], q[:, 1], q[:, 2], **kw)

    for k, gid in enumerate(cfg["gold"]):
        seg3(G[gid], color=GREEN if k == 0 else ORANGE, lw=3.2, label=f"gold {gid}")
    near = [e for e in ev if pm(g0, e)["axis_angle_deg"] < 10 and pm(g0, e)["transverse_offset_mm"] < 16]
    for i, e in enumerate(near):
        seg3(e, color=RED, lw=2.0, label="engine line (v1 zs)" if i == 0 else None)
    for li in cfg["z2"]:
        z = R["z2_lines"][li]
        q = np.array([z["start"], z["end"]])
        q = q + 1.2 * up  # drawn 1.2 mm above for visibility
        seg3(q, color=MAG, lw=1.6, ls="--", label="Z2 line (absorbed at pooling), drawn +1.2 mm")
    for i, k in enumerate(cfg["rej"]):
        c = R["pool"][k]
        seg3([c["start_s"], c["end_s"]], color=CYAN, lw=2.0, ls=(0, (4, 2)),
             label=f"rejected Z3 cand {k} (p_zs {c['p_zs']:.2f}, p_z3 {c['p_z3']:.2f})")
    # Z3 per-slice corner apices of the family near the gold line(s)
    f = R["z3"]["families"][cfg["corners_fam"]]
    Cp = np.array([c["apex3"] for c in f["corners"]]) if f["corners"] else np.zeros((0, 3))
    if len(Cp):
        cl = loc(Cp, o, u, w, up)
        mm = (np.abs(cl[:, 0]) < box[0]) & (np.abs(cl[:, 1]) < 6) & (np.abs(cl[:, 2]) < 6)
        ax.scatter(cl[mm, 0], cl[mm, 1], cl[mm, 2], s=10, c=BLUE, depthshade=False, label="Z3 per-slice corners")
    for sk in cfg["sections"]:
        lv = next(l for l in chains["levels"] if l["k"] == sk)
        ax.text(lv["s"] - L / 2, -box[1], box[2] * 0.9, f"k{sk}", color="k", fontsize=8)
    ax.set_xlim(-box[0], box[0]); ax.set_ylim(-box[1], box[1]); ax.set_zlim(-box[2], box[2])
    ax.set_box_aspect((2 * box[0], 2 * box[1], 2 * box[2]))
    ax.view_init(elev=28, azim=-62)
    ax.set_xlabel("along gold (mm)", fontsize=8); ax.set_ylabel("", fontsize=8); ax.set_zlabel("", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85, bbox_to_anchor=(0.02, 1.02))
    # span bars along the gold axis (s = 0 at the gold start)
    bx = fig.add_axes([0.13, 0.07, 0.42, 0.2])
    a0 = g0[0]
    def sint(line):
        q = sorted([float((np.asarray(p, float) - a0) @ u) for p in line])
        return q
    rows = []
    for k, gid in enumerate(cfg["gold"]):
        rows.append((f"gold {gid}", [sint(G[gid])], GREEN if k == 0 else ORANGE))
    near_b = [e for e in near if (lambda q: q[1] > -pad and q[0] < L + pad)(sint(e))]
    if near_b:
        for e in near_b:
            rows.append((f"engine line (off {pm(g0, e)['transverse_offset_mm']:.1f} mm)", [sint(e)], RED))
    else:
        rows.append(("engine line: none", [], RED))
    for li in cfg["z2"]:
        z = R["z2_lines"][li]
        rows.append(("Z2 line", [sint([z["start"], z["end"]])], MAG))
    for k in cfg["rej"]:
        c = R["pool"][k]
        rows.append((f"rejected cand {k}", [sint([c["start_s"], c["end_s"]])], CYAN))
    for i, (name, ivs, col) in enumerate(rows):
        for q in ivs:
            bx.plot(q, [i, i], color=col, lw=6, solid_capstyle="butt")
    if len(Cp):
        dd_, ss_ = [], []
        for c in f["corners"]:
            p = np.asarray(c["apex3"]); s_ = float((p - a0) @ u)
            off = np.linalg.norm((p - a0) - s_ * u)
            if off < 4.0 and -pad < s_ < L + pad and any(np.linalg.norm((p - G[gid][0]) - ((p - G[gid][0]) @ u) * u) < 2.5 for gid in cfg["gold"]):
                ss_.append(s_)
        bx.plot(ss_, [len(rows)] * len(ss_), "o", color=BLUE, ms=4)
        rows.append(("Z3 slice corners", [], BLUE))
    bx.set_yticks(range(len(rows))); bx.set_yticklabels([r[0] for r in rows], fontsize=7)
    bx.set_xlim(-pad, L + pad); bx.set_ylim(len(rows) - 0.5, -0.5)
    bx.set_xlabel("position along the gold line (mm)", fontsize=7); bx.tick_params(axis="x", labelsize=7)
    bx.grid(axis="x", alpha=0.3)
    # cross-sections
    U, e1, e2 = chains["u"], chains["e1"], chains["e2"]
    for j, sk in enumerate(cfg["sections"]):
        lv = next(l for l in chains["levels"] if l["k"] == sk)
        h = 0.84 / nsec
        sx = fig.add_axes([0.62, 0.08 + (nsec - 1 - j) * h + 0.05, 0.36, h - 0.08])
        x2 = np.asarray(lv["x2"])
        span = 9.0 if case != "B25" else 6.0
        for ci, ch in enumerate(lv["chains"]):
            Q = np.asarray(ch["pts"])
            col = ["#444444", "#a0522d", "#6b8e23", "#708090", "#8b008b", "#2f4f4f"][ci % 6]
            sx.plot(Q[:, 0], Q[:, 1], color=col, lw=0.9)
            for r in ch["flats"]["runs"]:
                if "reject" not in r and r["len"] >= 2.0:
                    sx.plot(Q[r["i0"]:r["i1"] + 1, 0], Q[r["i0"]:r["i1"] + 1, 1], color=col, lw=2.6)
            for c in ch["corners"]:
                sx.plot(*c["apex"][:2], "x", color=BLUE, ms=8, mew=2)
        for k, gid in enumerate(cfg["gold"]):
            p = crossing(G[gid], lv["lev"], U, e1, e2)
            if p is not None:
                sx.plot(*p, "*", color=GREEN if k == 0 else ORANGE, ms=13, mec="k", mew=0.5)
        for e in near:
            p = crossing(e, lv["lev"], U, e1, e2)
            if p is not None:
                sx.plot(*p, "o", color=RED, ms=7, mfc="none", mew=2)
        for k in cfg["rej"]:
            c = R["pool"][k]
            p = crossing([c["start_s"], c["end_s"]], lv["lev"], U, e1, e2)
            if p is not None:
                sx.plot(*p, "s", color=CYAN, ms=7, mfc="none", mew=2)
        for li in cfg["z2"]:
            z = R["z2_lines"][li]
            p = crossing([z["start"], z["end"]], lv["lev"], U, e1, e2)
            if p is not None:
                sx.plot(*p, "D", color=MAG, ms=6, mfc="none", mew=1.5)
        sx.set_xlim(x2[0] - span, x2[0] + span); sx.set_ylim(x2[1] - span, x2[1] + span)
        sx.set_aspect("equal"); sx.tick_params(labelsize=6)
        ncorn = sum(len(ch["corners"]) for ch in lv["chains"] if any(np.linalg.norm(np.asarray(c["apex"][:2]) - x2) < 5 for c in ch["corners"]))
        sx.set_title(f"Z3 slice k{sk} at s = {lv['s']:.1f} mm: {len([c for c in lv['chains'] if c['dmin'] < 3])} section chain(s) near the fold, "
                     f"{'corner found' if ncorn else 'NO corner'}", fontsize=8)
    fig.text(0.60, 0.03, "one colour per connected section chain; thick = accepted flat (>= 2 mm); x = Z3 corner;\n"
             "* gold apex; o engine line; square = rejected cand; diamond = Z2 line (axes in mm)", fontsize=7)
    fig.suptitle(cfg["title"], fontsize=9.5)
    out = Path(outdir) / f"{case}_closeup.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def main(dd, outdir):
    dd = Path(dd)
    R = pickle.loads((dd / "run/diag.pkl").read_bytes())
    V = np.load(dd / "run/mesh.npz")["V"]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for case, cfg in CASES.items():
        chains = pickle.loads((dd / f"chains_{cfg['fam_chain']}.pkl").read_bytes())
        print(render(case, cfg, R, V, outdir, chains))


if __name__ == "__main__":
    main(*sys.argv[1:])
