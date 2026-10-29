"""Gold-free and gold-referenced sanity check of Z3 it7 selections on 5 random DEV scans (open-gold only).
For each line: median distance line->mesh, and the median two-cluster dihedral of vertex normals in thin slabs
across the line (a real bend shows a dihedral close to the predicted bend angle; flat surface ~0)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

REPO = Path("/Users/idantal/ShermanQC/FullProject")
for p in (REPO / "research/broad_baseline", REPO, REPO / "backend"):
    sys.path.insert(0, str(p))
import score as S
from research.zeroshot.z3_slicing import slicer
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RL = REPO / "docs/production/runlog/zeroshot_2026-10-25"
OUT = RL / "audit_z3"
sp = json.loads((RL / "SPLIT_FROZEN.json").read_text())
DEV = sp["dev"]["scans"]
INV = {r["scan_id"]: r for r in json.loads((RL / "INVENTORY.json").read_text())["scans"] if r["scan_id"] in set(DEV)}
OPEN = [s for s in DEV if "sealed" not in INV[s]["gold"]["path"].lower() and not INV[s]["gold"].get("no_bends_present")]
rng = np.random.default_rng(424242)
pick = list(rng.choice(sorted(OPEN), 5, replace=False))


def dihedral(V, N, tree, a, b, t):
    d = b - a; L = np.linalg.norm(d); u = d / L
    r = 3 * t + 6
    idx = np.array(tree.query_ball_point((a + b) / 2, L / 2 + r))
    if len(idx) == 0:
        return np.nan, np.nan
    P = V[idx] - a; s = P @ u; perp = P - np.outer(s, u); rad = np.linalg.norm(perp, axis=1)
    out = []
    for sk in np.arange(0.1 * L, 0.9 * L + 1e-9, max(2.0, L / 25)):
        m = (np.abs(s - sk) < 1.0) & (rad < r)
        if m.sum() < 10:
            continue
        n = N[idx[m]]; n = n - np.outer(n @ u, u); n /= np.maximum(np.linalg.norm(n, axis=1), 1e-9)[:, None]
        e1 = np.cross(u, [1, 0, 0] if abs(u[0]) < .9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
        th = 2 * np.arctan2(n @ e2, n @ e1)          # doubled angle: two skins (n, -n) are the same orientation
        n = np.column_stack([np.cos(th), np.sin(th), np.zeros(len(th))])
        # 2-means on the circle
        c1, c2 = n[0], n[np.argmin(n @ n[0])]
        for _ in range(10):
            g = (n @ c1) >= (n @ c2)
            if g.all() or (~g).all():
                break
            c1 = n[g].mean(0); c1 /= np.linalg.norm(c1); c2 = n[~g].mean(0); c2 /= np.linalg.norm(c2)
        out.append(0.5 * np.degrees(np.arccos(np.clip(c1 @ c2, -1, 1))))   # axial angle in [0, 90]
    near = tree.query(a + np.outer(np.linspace(0.1, 0.9, 9), d))[0]
    return (float(np.median(out)) if out else np.nan), float(np.median(near))


res = {}
for s in pick:
    ev = json.loads((RL / "z3/runs/it7" / s / "eval_lines.json").read_text())
    mesh = slicer.load_mesh(str(Path(INV[s]["path"]).expanduser()))
    t = float(json.loads((RL / "z3/runs/it7" / s / "run_record.json").read_text())["t_mm"])
    tree = cKDTree(mesh.V)
    gold = S.load_gold({"kind": "ai_studio", "path": INV[s]["gold"]["path"]}, frame_records=None, locus="apex")["lines"]
    rows = []
    for kind, lines in (("pred", ev["owned_bend_regions"]), ("gold", gold)):
        for r in lines:
            a, b = np.array(r["line_start"], float), np.array(r["line_end"], float)
            dh, dist = dihedral(mesh.V, mesh.vn, tree, a, b, t)
            rows.append({"kind": kind, "id": r.get("region_id") or r.get("bend_id") or r.get("id"), "len": round(float(np.linalg.norm(b - a)), 1),
                         "pred_bend_axial": (round(min(180 - r["incl"], r["incl"]), 1) if r.get("incl") is not None else None),
                         "axial_dihedral_med": round(dh, 1), "dist_to_mesh_med": round(dist, 2)})
    # control: random chords between surface points 20-60 mm apart
    ctl = []
    for _ in range(15):
        i = rng.integers(len(mesh.V)); a = mesh.V[i]
        j = tree.query_ball_point(a, 60); j = [k for k in j if np.linalg.norm(mesh.V[k] - a) > 20]
        if not j:
            continue
        b = mesh.V[rng.choice(j)]
        ctl.append(dihedral(mesh.V, mesh.vn, tree, a, b, t)[0])
    # score
    case = {"case_id": s, "family_id": INV[s]["family"], "tier": "x", "gold": {"kind": "ai_studio", "path": INV[s]["gold"]["path"]},
            "label_complete": True, "predictions": str(RL / "z3/runs/it7" / s / "eval_lines.json")}
    sc = S.score_case(case, frame_records=None)
    res[s] = {"family": INV[s]["family"], "t_mm": round(t, 2), "tp_fp_fn": [sc["true_positive"], sc["false_positive"], sc["false_negative"]],
              "lines": rows, "random_chord_dihedral_med": round(float(np.nanmedian(ctl)), 1) if ctl else None}
    # render
    V = mesh.V[rng.choice(len(mesh.V), min(len(mesh.V), 40000), replace=False)]
    fig = plt.figure(figsize=(14, 7))
    for k, (el, az) in enumerate(((25, -60), (60, 30))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=0.3, c="0.6", alpha=0.25)
        for g in gold:
            p = np.array([g["line_start"], g["line_end"]]); ax.plot(p[:, 0], p[:, 1], p[:, 2], "g-", lw=5, alpha=0.6)
        for r in ev["owned_bend_regions"]:
            p = np.array([r["line_start"], r["line_end"]]); ax.plot(p[:, 0], p[:, 1], p[:, 2], "r-", lw=1.6)
        ax.view_init(el, az); ax.set_box_aspect(np.ptp(mesh.V, axis=0))
    fig.suptitle(f"{s} {INV[s]['family']}  red=Z3 it7 OOF selection, green=gold  TP/FP/FN={res[s]['tp_fp_fn']}")
    fig.savefig(OUT / f"sanity_{s}.png", dpi=90); plt.close(fig)
(OUT / "sanity_5scans.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
