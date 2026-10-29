"""Read-only visual triage for sc_2233372b8df7 (family 42288005, AI gold).

Inputs (all read-only): label_prep points_ds.npz + patches.json, AI gold ai_bend_lines_v1,
broad_baseline prediction files. Outputs PNG + JSON to
docs/production/runlog/broad_baseline/triage/sc_2233372b8df7/.
Run from FullProject/ with .venv/bin/python.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC = "sc_2233372b8df7"
OUT = f"docs/production/runlog/broad_baseline/triage/{SC}/"
PREP = f"docs/production/runlog/benchmark_v2/label_prep/{SC}/"
RUNS = f"docs/production/runlog/broad_baseline/runs/"

D = np.load(PREP + "points_ds.npz")
P, LAB = D["points"], D["labels"]
patches = {p["id"]: p for p in json.load(open(PREP + "patches.json"))["patches"]}
gold = json.load(open("data/bend_corpus/annotations/ai_bend_lines_v1/ae525e4954effa70d87ba8eb.json"))["bends"]
G = {b["bend_id"]: (np.array(b["line_start"]), np.array(b["line_end"])) for b in gold}


def preds(path):
    return {b["bend_id"]: (np.array(b["line_start"]), np.array(b["line_end"]))
            for b in json.load(open(path))["owned_bend_regions"]}


LANES = {
    "best_f1": preds(RUNS + f"best_f1/{SC}/predictions_best_f1.json"),
    "rt_gated": preds(RUNS + f"runtime/{SC}/predictions_runtime_gated.json"),
    "rt_ph": preds(RUNS + f"runtime/{SC}/predictions_runtime_with_placeholders.json"),
}

# part frame: e1 along the long bends (gold B1), e3 = panel normal (patch 1), e2 = e3 x e1
a, b = G["B1"]
e1 = (b - a) / np.linalg.norm(b - a)
e3 = np.array(patches[1]["normal"]); e3 -= (e3 @ e1) * e1; e3 /= np.linalg.norm(e3)
e2 = np.cross(e3, e1)
R = np.stack([e1, e2, e3]); C = P.mean(0)


def L(X):
    return (np.atleast_2d(X) - C) @ R.T


q = L(P)


def seg_dist(p, A, B):
    d = B - A; t = np.clip(((p - A) @ d) / (d @ d), 0, 1)
    return np.linalg.norm(p - (A + np.outer(t, d)), axis=1)


report = {"frame": {"e1": e1.tolist(), "e2": e2.tolist(), "e3": e3.tolist(), "origin": C.tolist()}, "lines": {}}
for tag, dct in [("gold", G)] + [(k, v) for k, v in LANES.items()]:
    for k, (A, B) in dct.items():
        la, lb = L(A)[0], L(B)[0]
        dirv = (B - A) / np.linalg.norm(B - A)
        samples = A + np.linspace(0, 1, 41)[:, None] * (B - A)
        # nearest scan point distance along the line (support)
        dn = np.array([np.min(np.linalg.norm(P - s, axis=1)) for s in samples])
        report["lines"][f"{tag}:{k}"] = {
            "e1_range": [round(float(min(la[0], lb[0])), 1), round(float(max(la[0], lb[0])), 1)],
            "start_e123": la.round(1).tolist(), "end_e123": lb.round(1).tolist(),
            "angle_to_e1_deg": round(float(np.degrees(np.arccos(min(1, abs(dirv @ e1))))), 2),
            "len_mm": round(float(np.linalg.norm(B - A)), 1),
            "support_frac_within_2mm": round(float(np.mean(dn < 2.0)), 3),
            "median_nn_mm": round(float(np.median(dn)), 2),
        }

# shallow-fold check: plane-intersection lines of the 165 deg patch pairs (3|13, 13|8 / 4|12, 12|9)
def plane(pid):
    p = patches[pid]; return np.array(p["normal"]), np.array(p["centroid"])


def inter(p1, p2):
    n1, c1 = plane(p1); n2, c2 = plane(p2)
    d = np.cross(n1, n2); d /= np.linalg.norm(d)
    A = np.array([n1, n2, d]); bvec = np.array([n1 @ c1, n2 @ c2, d @ ((c1 + c2) / 2)])
    x = np.linalg.solve(A, bvec)
    m = np.isin(LAB, [p1, p2]); t = (P[m] - x) @ d
    return x + d * t.min(), x + d * t.max(), float(np.degrees(np.arccos(abs(n1 @ n2))))


report["shallow_folds"] = {}
for p1, p2 in [(3, 13), (13, 8), (4, 12), (12, 9)]:
    A, B, ang = inter(p1, p2)
    rec = {"bend_deg": round(ang, 2), "start_e123": L(A)[0].round(1).tolist(), "end_e123": L(B)[0].round(1).tolist()}
    for lane, nm in [("best_f1", "CADF_CAD_B7"), ("rt_gated", "CAD_B7")]:
        PA, PB = LANES[lane][nm]
        s = PA + np.linspace(0, 1, 21)[:, None] * (PB - PA)
        rec[f"dist_{lane}_{nm}_mm"] = round(float(np.median(seg_dist(s, A, B))), 2)
    report["shallow_folds"][f"{p1}|{p2}"] = rec
# offset of patch 8 plane vs wall 3 plane (joggle depth)
n3, c3 = plane(3)
report["joggle_depth_mm_patch8_vs_wall3"] = round(float(n3 @ (np.array(patches[8]["centroid"]) - c3)), 2)
report["joggle_depth_mm_patch9_vs_wall4"] = round(float(plane(4)[0] @ (np.array(patches[9]["centroid"]) - plane(4)[1])), 2)

json.dump(report, open(OUT + "triage_geometry.json", "w"), indent=1)

COL = {"best_f1": "r", "rt_gated": "b", "rt_ph": "m"}


def draw(ax, i, j, mask, lanes=("best_f1",), lim=None, label=True, xr=None):
    qq = q[mask]
    ax.scatter(qq[:, i], qq[:, j], s=0.6, c="0.55", alpha=0.5, rasterized=True)
    items = [("gold", G, "g")] + [(ln, LANES[ln], COL[ln]) for ln in lanes]
    for tag, dct, col in items:
        for k, (A, B) in dct.items():
            la, lb = L(A)[0], L(B)[0]
            if xr is not None:  # section view: plot the line's crossing of the e1 window, if any
                lo, hi = sorted([la[0], lb[0]])
                xm = 0.5 * (xr[0] + xr[1])
                if hi < xr[0] - 2 or lo > xr[1] + 2:
                    continue
                if abs(lb[0] - la[0]) > 1e-6:
                    t = np.clip((xm - la[0]) / (lb[0] - la[0]), 0, 1)
                else:
                    t = 0.5
                pt = la + t * (lb - la)
                ax.plot(pt[i], pt[j], "o" if tag == "gold" else "x", color=col, ms=11 if tag == "gold" else 9,
                        mfc="none" if tag == "gold" else None, mew=2)
                if label:
                    ax.annotate(k if tag == "gold" else k[-8:], (pt[i], pt[j]), color=col, fontsize=8,
                                xytext=(4, 4 if tag == "gold" else -10), textcoords="offset points")
            else:
                ax.plot([la[i], lb[i]], [la[j], lb[j]], color=col, lw=3.5 if tag == "gold" else 1.5,
                        ls="-" if tag == "gold" else "--", alpha=0.8)
                if label:
                    m = (la + lb) / 2
                    ax.text(m[i], m[j], k if tag == "gold" else k[-8:], color=col, fontsize=7)
    ax.set_aspect("equal")
    if lim:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])


# 1) cross sections at several e1 stations
e1min, e1max = q[:, 0].min(), q[:, 0].max()
stations = [e1min + 6, e1min + 25, (e1min + e1max) / 2, e1max - 40, e1max - 8]
fig, axs = plt.subplots(2, len(stations), figsize=(5 * len(stations), 11))
for c, s in enumerate(stations):
    xr = (s - 1.5, s + 1.5)
    m = (q[:, 0] > xr[0]) & (q[:, 0] < xr[1])
    draw(axs[0, c], 1, 2, m, lanes=("best_f1",), xr=xr)
    axs[0, c].set_title(f"section e1={s:.0f}+-1.5  gold(o) best_f1(x)", fontsize=9)
    draw(axs[1, c], 1, 2, m, lanes=("rt_gated", "rt_ph"), xr=xr)
    axs[1, c].set_title(f"section e1={s:.0f}  gold(o) rt_gated(b) rt_ph(m)", fontsize=9)
plt.tight_layout(); plt.savefig(OUT + "sections_e1.png", dpi=80); plt.close()

# 2) zoom on the channel (B2/B3/B4/B5) cross-sections
gB2 = L(G["B2"][0])[0]
lim = ((gB2[1] - 25, gB2[1] + 12), (gB2[2] - 30, gB2[2] + 10))
fig, axs = plt.subplots(1, 4, figsize=(22, 7))
for c, s in enumerate([e1min + 8, (e1min + e1max) / 2, e1max - 60, e1max - 8]):
    xr = (s - 1.0, s + 1.0); m = (q[:, 0] > xr[0]) & (q[:, 0] < xr[1])
    draw(axs[c], 1, 2, m, lanes=("best_f1", "rt_ph"), xr=xr, lim=lim)
    axs[c].set_title(f"channel zoom e1={s:.0f}+-1: gold(o) best_f1 red x, rt_ph mag x", fontsize=9)
    axs[c].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT + "channel_zoom_sections.png", dpi=85); plt.close()

# 3) wall face (flange 3/4) seen face-on: e1 vs height, with patch colours -> shows joggle 8/9/12/13 and tab 10/11
n_w = np.array(patches[3]["normal"]); w2 = np.cross(n_w, e1); w2 /= np.linalg.norm(w2)
fig, axs = plt.subplots(2, 1, figsize=(18, 13))
for ax, lanes in zip(axs, [("best_f1",), ("rt_gated", "rt_ph")]):
    mw = np.isin(LAB, [3, 4, 8, 9, 10, 11, 12, 13, 5, 17, 19])
    cmap = {3: "0.7", 4: "0.7", 8: "orange", 9: "orange", 12: "purple", 13: "purple", 10: "c", 11: "c", 5: "0.4", 17: "y", 19: "k"}
    for pid, col in cmap.items():
        mm = LAB == pid
        ax.scatter((P[mm] - C) @ e1, (P[mm] - C) @ w2, s=0.6, c=col, rasterized=True, label=f"patch {pid}")
    for tag, dct, col in [("gold", G, "g")] + [(ln, LANES[ln], COL[ln]) for ln in lanes]:
        for k, (A, B) in dct.items():
            xa, xb = (A - C) @ e1, (B - C) @ e1; ya, yb = (A - C) @ w2, (B - C) @ w2
            ax.plot([xa, xb], [ya, yb], color=col, lw=3.5 if tag == "gold" else 1.6, ls="-" if tag == "gold" else "--")
            ax.text((xa + xb) / 2, (ya + yb) / 2, k if tag == "gold" else k[-8:], color=col, fontsize=8)
    ax.set_aspect("equal"); ax.legend(markerscale=10, fontsize=7, ncol=6, loc="lower right")
    ax.set_title(f"flange 3/4 face-on (e1 vs in-wall height); purple=165deg strip 12/13, orange=joggle land 8/9; lanes {lanes}")
plt.tight_layout(); plt.savefig(OUT + "wall_faceon.png", dpi=80); plt.close()

# 4) joggle close-up section at left end through patches 12/13
c12 = (np.array(patches[12]["centroid"]) - C) @ e1
xr = (c12 - 1.5, c12 + 1.5); m = (q[:, 0] > xr[0]) & (q[:, 0] < xr[1])
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
draw(axs[0], 1, 2, m, lanes=("best_f1", "rt_gated"), xr=xr)
axs[0].set_title(f"section at e1={c12:.0f} (through 165deg strip 12/13)")
mm = m & np.isin(LAB, [3, 4, 8, 9, 12, 13])
for pid, col in {3: "0.6", 4: "0.6", 8: "orange", 9: "orange", 12: "purple", 13: "purple"}.items():
    k = mm & (LAB == pid); axs[1].scatter(q[k, 1], q[k, 2], s=4, c=col, label=str(pid))
draw(axs[1], 1, 2, np.zeros(len(q), bool), lanes=("best_f1", "rt_gated"), xr=xr)
qq = q[mm]
if len(qq):
    axs[1].set_xlim(qq[:, 1].min() - 5, qq[:, 1].max() + 5); axs[1].set_ylim(qq[:, 2].min() - 5, qq[:, 2].max() + 5)
axs[1].legend(); axs[1].set_title("zoom: wall 3/4 (grey), strip 12/13 (purple), land 8/9 (orange)")
plt.tight_layout(); plt.savefig(OUT + "joggle_section.png", dpi=85); plt.close()
print(json.dumps(report, indent=1))
