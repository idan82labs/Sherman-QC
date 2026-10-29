"""Read-only visual triage for sc_6eba02723d4b (family 23553000).

Renders (a) along-line profiles for the long merged predictions and
(b) foot cross-sections showing gold vs runtime/best_f1 loci.
Outputs go to docs/production/runlog/broad_baseline/overlays/sc_6eba02723d4b/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "docs/production/runlog"
OUT = RUN / "broad_baseline/overlays/sc_6eba02723d4b"
PTS = np.load(RUN / "benchmark_v2/label_prep/sc_6eba02723d4b/points_ds.npz")
P, N = PTS["points"], PTS["normals"]
GOLD = {
    b["bend_id"]: (np.array(b["line_start"]), np.array(b["line_end"]))
    for b in json.load(open(ROOT / "data/bend_corpus/annotations/ai_bend_lines_v1/71f5e686304eac1244183772.json"))["bends"]
}
BB = RUN / "broad_baseline/runs"


def load(p, tag):
    d = json.load(open(p))
    return {f"{tag}:{r['bend_id']}": (np.array(r["line_start"]), np.array(r["line_end"])) for r in d["owned_bend_regions"]}


PRED = {}
PRED.update(load(BB / "runtime/sc_6eba02723d4b/predictions_runtime.json", "rt"))
PRED.update(load(BB / "runtime/sc_6eba02723d4b/predictions_runtime_gated.json", "rtg"))
PRED.update(load(BB / "best_f1/sc_6eba02723d4b/predictions_best_f1.json", "bf"))


def frame(a, b, up_hint):
    u = (b - a) / np.linalg.norm(b - a)
    w = up_hint - (up_hint @ u) * u
    w /= np.linalg.norm(w)
    v = np.cross(w, u)
    return u, v, w


def along_profile(key, ax, radius=25.0):
    a, b = PRED[key]
    u = (b - a) / np.linalg.norm(b - a)
    # plate normal near the line: median normal of nearby points with |n.u| small
    rel = P - a
    t = rel @ u
    perp = rel - np.outer(t, u)
    dist = np.linalg.norm(perp, axis=1)
    near = (dist < radius) & (t > -20) & (t < np.linalg.norm(b - a) + 20)
    nn = N[near]
    # dominant normal (plate) via SVD of normals
    _, _, vt = np.linalg.svd(nn - 0, full_matrices=False)
    w = vt[0] - (vt[0] @ u) * u
    w /= np.linalg.norm(w)
    v = np.cross(w, u)
    x = t[near]
    y = (perp[near] @ w)
    z = (perp[near] @ v)
    ax.scatter(x, y, s=0.3, c=np.abs(nn @ w), cmap="viridis", vmin=0, vmax=1)
    ax.plot([0, np.linalg.norm(b - a)], [0, 0], "r-", lw=1.5, label=key)
    for gid, (ga, gb) in GOLD.items():
        ta, tb = (ga - a) @ u, (gb - a) @ u
        da = np.linalg.norm((ga - a) - ta * u)
        db = np.linalg.norm((gb - a) - tb * u)
        if max(da, db) < 8 and abs(((gb - ga) / np.linalg.norm(gb - ga)) @ u) > 0.95:
            ax.plot([ta, tb], [((ga - a) @ w), ((gb - a) @ w)], "g-", lw=4, alpha=0.7)
            ax.text((ta + tb) / 2, 3, gid, color="g")
    ax.set_title(f"{key}: points within {radius:.0f} mm, x=along line, y=plate-normal offset (colour=|n.plate_n|)")
    ax.set_xlabel("along line (mm)")
    ax.set_ylabel("plate-normal offset (mm)")
    ax.legend(loc="lower right")


def xsec(gid, keys, ax, half=6.0, box=30.0):
    a, b = GOLD[gid]
    u = (b - a) / np.linalg.norm(b - a)
    m = (a + b) / 2
    rel = P - m
    t = rel @ u
    sl = np.abs(t) < half
    perp = rel - np.outer(t, u)
    # local 2D basis
    e1 = np.cross(u, [0, 0, 1.0])
    if np.linalg.norm(e1) < 1e-3:
        e1 = np.cross(u, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    X, Y = perp @ e1, perp @ e2
    sel = sl & (np.abs(X) < box) & (np.abs(Y) < box)
    ax.scatter(X[sel], Y[sel], s=1.0, c="0.5")
    for g2, (ga, gb) in GOLD.items():
        gm = (ga + gb) / 2 - m
        ax_t = gm @ u
        if abs(abs(((gb - ga) / np.linalg.norm(gb - ga)) @ u) - 1) < 0.05:
            q = gm - ax_t * u
            if abs(q @ e1) < box and abs(q @ e2) < box:
                ax.plot(q @ e1, q @ e2, "o", ms=10, mfc="none", mec="g", mew=2)
                ax.text(q @ e1 + 0.8, q @ e2 + 0.8, g2, color="g")
    cols = {"rt": "b", "rtg": "c", "bf": "r"}
    for k in keys:
        pa, pb = PRED[k]
        pu = (pb - pa) / np.linalg.norm(pb - pa)
        # point on pred line closest to section plane centre
        s = ((m - pa) @ pu)
        q = pa + s * pu - m
        ax.plot(q @ e1, q @ e2, "x", ms=10, mew=2, color=cols[k.split(":")[0]])
        ax.text(q @ e1 + 0.8, q @ e2 - 1.8, k, color=cols[k.split(":")[0]], fontsize=7)
    ax.set_aspect("equal")
    ax.set_title(f"cross-section at gold {gid} (slab +/-{half} mm along axis)")


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    along_profile("bf:F1C_6c50503a93bcd449", axs[0])
    along_profile("bf:F1C_c7d8614bb6a86c7c", axs[1])
    fig.tight_layout()
    fig.savefig(OUT / "triage_long_lines_profile.png", dpi=110)

    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    xsec("B10", ["rt:CAD_B4", "rtg:CAD_B4", "bf:CADF_CAD_B4"], axs[0, 0])
    xsec("B11", ["rt:CAD_B4", "rtg:CAD_B4", "bf:CADF_CAD_B4"], axs[0, 1])
    xsec("B9", ["rt:CAD_B3", "rtg:CAD_B3", "bf:CADF_CAD_B3"], axs[0, 2])
    xsec("B12", ["rt:CAD_B3", "rtg:CAD_B3", "bf:CADF_CAD_B3"], axs[1, 0])
    xsec("B2", ["bf:F1C_6c50503a93bcd449"], axs[1, 1])
    xsec("B8", ["rt:CAD_B2", "rtg:CAD_B2", "bf:F1C_e2779d82112acf45"], axs[1, 2])
    fig.tight_layout()
    fig.savefig(OUT / "triage_xsections.png", dpi=110)

    # extents of predictions along gold axes (for span reasoning)
    for k, (a, b) in PRED.items():
        print(k, "len", round(float(np.linalg.norm(b - a)), 1))
