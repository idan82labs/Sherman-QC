"""Read-only visual triage for sc_2714fb7e0c0b (Gold11 C06, 38172000_01_V).

Loads the scan, downsamples, and renders overlays / cross-sections of gold vs
predicted lines. Writes only under docs/production/runlog/broad_baseline/triage/.
"""
import json, sys
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
OUT = ROOT / "docs/production/runlog/broad_baseline/triage/sc_2714fb7e0c0b"
OUT.mkdir(parents=True, exist_ok=True)
SCAN = ROOT / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_V.ply"
RUNS = ROOT / "docs/production/runlog/broad_baseline/runs"
GOLD = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"


def load_points():
    npz = OUT / "points_ds.npz"
    if npz.exists():
        return np.load(npz)["p"]
    pc = o3d.io.read_point_cloud(str(SCAN))
    if len(pc.points) == 0:
        m = o3d.io.read_triangle_mesh(str(SCAN))
        pc = o3d.geometry.PointCloud(m.vertices)
    p = np.asarray(pc.voxel_down_sample(0.7).points)
    np.savez_compressed(npz, p=p)
    return p


def gold_lines():
    d = json.load(open(GOLD))
    def find(o):
        if isinstance(o, dict):
            if o.get("case_id") == "shaked_xv/C06":
                return o
            for v in o.values():
                r = find(v)
                if r: return r
        if isinstance(o, list):
            for v in o:
                r = find(v)
                if r: return r
    c = find(d)
    return {b["bend_id"]: (np.array(b["line_start"]), np.array(b["line_end"])) for b in c["bends"]}


def pred_lines(rel):
    d = json.load(open(RUNS / rel))
    return {r["bend_id"]: (np.array(r["line_start"]), np.array(r["line_end"])) for r in d["owned_bend_regions"]}


def main():
    p = load_points()
    G = gold_lines()
    B = pred_lines("best_f1/sc_2714fb7e0c0b/predictions_best_f1.json")
    P = pred_lines("runtime/sc_2714fb7e0c0b/predictions_runtime_with_placeholders.json")
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    R = vt  # rows = PC axes
    q = (p - c) @ R.T
    tf = lambda x: (np.asarray(x) - c) @ R.T
    print("points", len(p), "extent PC", q.min(0).round(1), q.max(0).round(1))
    info = {}
    for name, L in [("gold", G), ("best", B), ("ph", P)]:
        for k, (a, b) in L.items():
            A, Bb = tf(a), tf(b)
            d = (Bb - A); ln = np.linalg.norm(d)
            info[f"{name}:{k}"] = dict(a=A.round(1).tolist(), b=Bb.round(1).tolist(), len=round(float(ln), 1),
                                       dir=(d / ln).round(3).tolist())
            print(name, k, A.round(1), Bb.round(1), "len", round(ln, 1), "dir", (d / ln).round(3))
    json.dump(info, open(OUT / "lines_pc_frame.json", "w"), indent=1)

    # ---- 3 orthographic views with labels
    views = [(0, 1, "PC1-PC2 top"), (0, 2, "PC1-PC3 front"), (1, 2, "PC2-PC3 side")]
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax, (i, j, t) in zip(axs, views):
        sub = q[:: max(1, len(q) // 120000)]
        ax.scatter(sub[:, i], sub[:, j], s=0.2, c="0.7")
        for name, L, col, ls in [("gold", G, "green", "-"), ("best", B, "red", "--"), ("ph", P, "blue", ":")]:
            for k, (a, b) in L.items():
                A, Bb = tf(a), tf(b)
                ax.plot([A[i], Bb[i]], [A[j], Bb[j]], ls, color=col, lw=2.5 if name == "gold" else 1.6)
                m = (A + Bb) / 2
                ax.text(m[i], m[j], k.replace("CADF_", "").replace("F1C_", "F")[:8], color=col, fontsize=8)
        ax.set_aspect("equal"); ax.set_title(t)
    fig.suptitle("sc_2714fb7e0c0b: green=gold, red dashed=best_f1, blue dotted=runtime placeholders")
    fig.savefig(OUT / "labelled_views.png", dpi=80, bbox_inches="tight"); plt.close(fig)

    # ---- cross-sections perpendicular to selected lines at several stations
    def xsec(tag, a, b, stations=(0.2, 0.5, 0.8), half=30.0, slab=1.5, extra=()):
        a, b = np.asarray(a), np.asarray(b)
        d = (b - a) / np.linalg.norm(b - a)
        fig, axs = plt.subplots(1, len(stations), figsize=(6 * len(stations), 6))
        for ax, s in zip(np.atleast_1d(axs), stations):
            o = a + s * (b - a)
            rel = p - o
            along = rel @ d
            perp = rel - np.outer(along, d)
            m = (np.abs(along) < slab) & (np.linalg.norm(perp, axis=1) < half)
            # 2D basis in the perpendicular plane: use PC3 projected + cross
            u = R[2] - (R[2] @ d) * d; u /= np.linalg.norm(u)
            w = np.cross(d, u)
            pts = perp[m]
            ax.scatter(pts @ w, pts @ u, s=2, c="k")
            ax.plot(0, 0, "g+", ms=18, mew=2)
            for lab, (ea, eb), col in extra:
                ea, eb = np.asarray(ea), np.asarray(eb)
                ed = eb - ea
                # intersect other line with slab plane
                den = ed @ d
                if abs(den) > 1e-6:
                    t = ((o - ea) @ d) / den
                    if -0.05 <= t <= 1.05:
                        x = ea + t * ed - o
                        ax.plot(x @ w, x @ u, "x", color=col, ms=12, mew=2); ax.text(x @ w, x @ u, lab, color=col)
                else:
                    x = (ea + eb) / 2 - o
                    if np.linalg.norm(x - (x @ d) * d) < half:
                        ax.plot(x @ w, x @ u, "s", color=col, mfc="none", ms=12); ax.text(x @ w, x @ u, lab + "(par)", color=col)
            ax.set_aspect("equal"); ax.set_xlim(-half, half); ax.set_ylim(-half, half)
            ax.set_title(f"{tag} @ {s:.1f}  n={m.sum()}")
            ax.grid(alpha=0.3)
        fig.savefig(OUT / f"xsec_{tag}.png", dpi=70, bbox_inches="tight"); plt.close(fig)

    others = [(f"g{k}", v, "green") for k, v in G.items()] + [(k.replace("CADF_", "").replace("F1C_", "F")[:9], v, "red") for k, v in B.items()] + \
             [(f"ph{k}", v, "blue") for k, v in P.items()]
    for k in ["B07", "B09", "B10", "B01", "B03", "B05"]:
        xsec("gold_" + k, *G[k], extra=others)
    for k in ["CADF_CAD_B5", "CADF_CAD_B7"]:
        xsec("best_" + k, *B[k], extra=others)
    for k in ["CAD_B3", "CAD_B8", "CAD_B10", "CAD_B11"]:
        a, b = P[k]
        xsec("ph_" + k, a, b, extra=others)


if __name__ == "__main__":
    main()
