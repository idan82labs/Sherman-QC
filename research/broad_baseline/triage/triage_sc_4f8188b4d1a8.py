#!/usr/bin/env python3
"""Read-only visual/geometric triage for sc_4f8188b4d1a8 (Gold11 CG1E0BBA2933, family 38172000, X scan).

For every gold line and every prediction (runtime_with_placeholders, best_f1), measure in scan_raw:
  * point support along the line (fraction of samples with a scan vertex within 3 mm)
  * local two-face dihedral from the scan's stored vertex normals in a 10 mm tube around the line
    (2-means on normals; a real bend gives two well-populated normal clusters at ~the bend angle)
  * nearest gold line relation (angle / transverse offset / span overlap)
and write close-up renders (cross-section across the line + along-line top view).
Outputs go to docs/production/runlog/broad_baseline/overlays/sc_4f8188b4d1a8/triage/.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
CASE = "sc_4f8188b4d1a8"
SCAN = ROOT / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_X.ply"
GOLD = ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/manual_adjudication/labels/C05/bend_instance_labels.json"
RUNS = ROOT / "docs/production/runlog/broad_baseline/runs"
OUT = ROOT / "docs/production/runlog/broad_baseline/overlays" / CASE / "triage"


def load_ply(path: Path):
    with open(path, "rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("no end_header")
            header += line
            if line.strip() == b"end_header":
                break
        n = int([l for l in header.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
        v = np.frombuffer(fh.read(n * 24), dtype="<f4").reshape(n, 6).astype(np.float64)
    return v[:, :3], v[:, 3:]


def unit(v):
    return v / np.linalg.norm(v)


def line_metrics(a, b, pts, nrm, tree):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    s = np.linspace(0, 1, 101)[:, None] * d + a
    dist, _ = tree.query(s)
    support = float((dist < 3.0).mean())
    # tube
    mid = (a + b) / 2
    idx = tree.query_ball_point(mid, L / 2 + 10.0)
    P, N = pts[idx], nrm[idx]
    rel = P - a
    t = rel @ u
    perp = rel - np.outer(t, u)
    r = np.linalg.norm(perp, axis=1)
    m = (t > 0.05 * L) & (t < 0.95 * L) & (r < 10.0) & (r > 1.5)
    P, N = P[m], N[m]
    res = {"length_mm": round(L, 2), "support_3mm": round(support, 3), "median_nn_mm": round(float(np.median(dist)), 3),
           "tube_points": int(m.sum())}
    if m.sum() < 30:
        res["dihedral"] = None
        return res
    # 2-means on sign-agnostic normals (flip to a common hemisphere first)
    ref = N[np.argmax(np.abs(N @ np.cross(u, [0.3, 0.5, 0.8])))]
    Nf = N * np.sign(N @ ref + 1e-9)[:, None]
    c = np.stack([Nf[0], Nf[np.argmin(Nf @ Nf[0])]])
    for _ in range(25):
        lab = np.argmax(Nf @ c.T, axis=1)
        c = np.stack([unit(Nf[lab == k].mean(0)) if (lab == k).any() else c[k] for k in range(2)])
    frac = [float((lab == k).mean()) for k in range(2)]
    ang = float(np.degrees(np.arccos(np.clip(abs(c[0] @ c[1]), -1, 1))))
    # how parallel each face normal is to the line axis (a real bend: both normals ⟂ axis)
    axial = [float(abs(ck @ u)) for ck in c]
    res.update({"dihedral": {"normal_angle_deg": round(ang, 1), "cluster_frac": [round(f, 2) for f in frac],
                             "normal_dot_axis": [round(x, 2) for x in axial]}})
    return res


def rel_to(g, p):
    ga, gb, pa, pb = (np.asarray(x, float) for x in (g[0], g[1], p[0], p[1]))
    gu, pu = unit(gb - ga), unit(pb - pa)
    ang = float(np.degrees(np.arccos(min(1.0, abs(gu @ pu)))))
    pm = (pa + pb) / 2
    off = float(np.linalg.norm((pm - ga) - ((pm - ga) @ gu) * gu))
    gl = np.linalg.norm(gb - ga)
    t = sorted([(pa - ga) @ gu, (pb - ga) @ gu])
    ov = max(0.0, min(gl, t[1]) - max(0.0, t[0])) / gl
    return round(ang, 2), round(off, 2), round(ov, 3)


def main():
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    pts, nrm = load_ply(SCAN)
    tree = cKDTree(pts)
    gold = {b["bend_id"]: (b["line_start"], b["line_end"]) for b in json.load(open(GOLD))["manual_bends"]}
    lanes = {}
    for lane, f in (("runtime_with_placeholders", RUNS / "runtime" / CASE / "predictions_runtime_with_placeholders.json"),
                    ("best_f1", RUNS / "best_f1" / CASE / "predictions_best_f1.json"),
                    ("cad_raw_ungated", RUNS / "best_f1" / CASE / "predictions_cad_raw_ungated.json"),
                    ("a0", RUNS / "best_f1" / CASE / "predictions_a0.json")):
        if f.is_file():
            lanes[lane] = {r["bend_id"]: (r["line_start"], r["line_end"]) for r in json.load(open(f))["owned_bend_regions"]
                           if r.get("line_start") is not None}
    report = {"gold": {}, "pred": {}}
    for gid, (a, b) in gold.items():
        report["gold"][gid] = line_metrics(a, b, pts, nrm, tree)
    for lane, lines in lanes.items():
        for pid, (a, b) in lines.items():
            m = line_metrics(a, b, pts, nrm, tree)
            rels = {gid: rel_to(g, (a, b)) for gid, g in gold.items()}
            best = min(rels.items(), key=lambda kv: (kv[1][0] > 5) * 1e3 + kv[1][1])
            m["nearest_gold"] = {"id": best[0], "angle_deg": best[1][0], "offset_mm": best[1][1], "gold_span_overlap": best[1][2]}
            report["pred"][f"{lane}:{pid}"] = m
    (OUT / "triage_metrics.json").write_text(json.dumps(report, indent=1))

    # close-ups: for each item of interest, cross-section slab perpendicular to the line at its midpoint,
    # plus a local "top" view of a box around the line.
    rng = np.random.default_rng(0)
    focus = [("gold", g, gold[g]) for g in ("B09", "B10", "B05", "B02", "B07", "B08")]
    for lane in ("best_f1", "runtime_with_placeholders"):
        for pid in lanes.get(lane, {}):
            if pid.startswith("CAD"):
                focus.append((lane, pid, lanes[lane][pid]))
    for kind, name, (a, b) in focus:
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = unit(b - a)
        L = np.linalg.norm(b - a)
        mid = (a + b) / 2
        idx = np.array(tree.query_ball_point(mid, L / 2 + 40.0))
        P = pts[idx]
        # local frame: u along line, w = dominant local normal, v = u x w
        loc = tree.query_ball_point(mid, 12.0)
        w = unit(np.linalg.svd(pts[loc] - pts[loc].mean(0))[2][2]) if len(loc) > 10 else unit(np.cross(u, [0, 0, 1]))
        w = unit(w - (w @ u) * u)
        v = np.cross(u, w)
        R = P - mid
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=80)
        # (1) cross-section at mid (|t|<2 mm)
        t = R @ u
        sl = np.abs(t) < 2.0
        ax[0].scatter(R[sl] @ v, R[sl] @ w, s=2, c="0.3")
        ax[0].plot([0], [0], "r+", ms=14, mew=2)
        ax[0].set_aspect("equal"); ax[0].set_title(f"{kind}:{name} cross-section |t|<2mm (axes v,w)")
        # (2) cross-sections at 20%/80%
        for frac, col in ((0.2, "tab:blue"), (0.8, "tab:orange")):
            c0 = a + frac * (b - a) - mid
            sl2 = np.abs((R - c0) @ u) < 2.0
            ax[1].scatter((R[sl2] - c0) @ v, (R[sl2] - c0) @ w, s=2, c=col, label=f"t={frac}")
        ax[1].plot([0], [0], "r+", ms=14, mew=2); ax[1].legend(); ax[1].set_aspect("equal")
        ax[1].set_title("cross-sections at 20% / 80% of span")
        # (3) along-line plan view coloured by height w
        sub = R if len(R) < 60000 else R[rng.choice(len(R), 60000, replace=False)]
        sc_ = ax[2].scatter(sub @ u, sub @ v, s=1, c=np.clip(sub @ w, -20, 20), cmap="coolwarm")
        ax[2].plot([-L / 2, L / 2], [0, 0], "k-", lw=2)
        for gid, (ga, gb) in gold.items():
            ga2, gb2 = np.asarray(ga) - mid, np.asarray(gb) - mid
            ax[2].plot([ga2 @ u, gb2 @ u], [ga2 @ v, gb2 @ v], "g--", lw=1)
            gm = (ga2 + gb2) / 2
            if abs(gm @ u) < L / 2 + 40 and abs(gm @ v) < 60:
                ax[2].text(gm @ u, gm @ v, gid, color="g")
        ax[2].set_aspect("equal"); ax[2].set_title("plan view along line (colour = height w), black = item, green = gold")
        plt.colorbar(sc_, ax=ax[2], fraction=0.03)
        fig.tight_layout()
        fig.savefig(OUT / f"closeup_{kind}_{name}.png")
        plt.close(fig)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
