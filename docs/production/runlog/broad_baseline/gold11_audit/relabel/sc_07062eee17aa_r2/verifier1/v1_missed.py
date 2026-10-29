"""Verifier1 missed-bend sweep over the whole scan (loads full scan; run under the shared lock)."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import v1_sections as V
H = os.path.dirname(os.path.abspath(__file__))
P, N = V.load_ply(V.SCAN)
N = N / np.linalg.norm(N, axis=1, keepdims=True)
J = json.load(open(os.path.join(H, "../lines.json")))
nw = np.array(J["bends"][0]["plane_a"]["normal"]); nw /= np.linalg.norm(nw)  # web normal (B1 plane_a = web)
# frame: e3 = web normal, e1 = fold direction of B1 (long axis), e2 = e3 x e1
e1 = np.array(J["bends"][0]["direction"]); e1 -= (e1 @ nw) * nw; e1 /= np.linalg.norm(e1); e2 = np.cross(nw, e1)
X = np.c_[P @ e1, P @ e2, P @ nw]
# distance to any labeled line (bend zones)
def seg_dist(Pp, a, b):
    d = b - a; t = np.clip(((Pp - a) @ d) / (d @ d), 0, 1); return np.linalg.norm(Pp - (a + np.outer(t, d)), axis=1)
dz = np.min([seg_dist(P, np.array(B["label_line"]["start"]), np.array(B["label_line"]["end"])) for B in J["bends"]], axis=0)
# 1) normal-orientation clusters (sign-folded), greedy mode finding
rng = np.random.default_rng(0); idx = rng.choice(len(N), 120000, replace=False); M = N[idx]
rem = np.ones(len(M), bool); clusters = []
for _ in range(14):
    if rem.sum() < 200: break
    S = M[rem]; samp = S[rng.choice(len(S), min(3000, len(S)), replace=False)]
    cnt = (np.abs(samp @ S.T) > np.cos(np.radians(4))).sum(1)
    c = samp[cnt.argmax()]; mem = rem & (np.abs(M @ c) > np.cos(np.radians(4)))
    c = np.linalg.svd(M[mem] * np.sign(M[mem] @ c)[:, None], full_matrices=False)[2][0]
    mem = rem & (np.abs(M @ c) > np.cos(np.radians(4)))
    frac = mem.sum() / len(M)
    if frac < 0.002: break
    clusters.append(dict(dir=[round(float(v), 4) for v in c], frac=round(float(frac), 4),
                         ang_to_web_deg=round(float(np.degrees(np.arccos(min(1, abs(c @ nw))))), 2),
                         median_dist_to_labeled_line_mm=round(float(np.median(dz[idx][mem])), 2)))
    rem &= ~mem
unassigned = rem.sum() / len(M)
# 2) web-facing points: angle from web normal, away from labeled bend zones
ang = np.degrees(np.arccos(np.clip(np.abs(N @ nw), 0, 1)))
webf = ang < 45
# local-plane smoothing of the angle: median over 1.5 mm neighbourhood (on a 4x subsample) to suppress noise
sub = np.arange(0, len(P), 3)
tr = cKDTree(P)
nb = tr.query(P[sub], k=24)[1]
ang_s = np.median(ang[nb], axis=1)
flag = (ang_s > 8) & (ang_s < 45) & (dz[sub] > 4.0)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
w = webf[sub]
sc = ax[0].scatter(X[sub][w, 0], X[sub][w, 1], c=ang_s[w], s=0.3, cmap="viridis", vmin=0, vmax=20)
plt.colorbar(sc, ax=ax[0], label="normal angle from web normal (deg, 1.5mm median)")
for B in J["bends"]:
    a = np.array(B["label_line"]["start"]); b = np.array(B["label_line"]["end"])
    ax[0].plot([a @ e1, b @ e1], [a @ e2, b @ e2], "r-", lw=1); ax[0].text(a @ e1, a @ e2, B["bend_id"], color="r")
ax[0].set_aspect("equal"); ax[0].set_title("web-facing points (top view): any formed bend in the web would show as a band >= 8 deg")
nf = ~webf[sub]
# side-facing points (flanges, tabs, ears, rims): plot front view (e1 vs e3) colored by angle of normal to e2-ish face normal
nfl = np.array(J["bends"][0]["plane_b"]["normal"]); nfl /= np.linalg.norm(nfl)
angf = np.degrees(np.arccos(np.clip(np.abs(N[sub] @ nfl), 0, 1)))
sc2 = ax[1].scatter(X[sub][nf, 0], X[sub][nf, 2] + np.where(X[sub][nf, 1] > np.median(X[:, 1]), 30, 0), c=angf[nf], s=0.3, cmap="viridis", vmin=0, vmax=40)
plt.colorbar(sc2, ax=ax[1], label="normal angle from flange normal (deg)")
ax[1].set_title("non-web-facing points, front view (+Y side shifted up 30 mm)"); ax[1].set_aspect("equal")
fig.tight_layout(); fig.savefig(os.path.join(H, "missed_bend_maps.png"), dpi=110); plt.close(fig)
fl = sub[flag]
out = dict(web_normal=nw.tolist(), clusters=clusters, unassigned_frac=round(float(unassigned), 4),
           web_facing_flagged_points_away_from_labels=int(flag.sum()), web_facing_points=int(w.sum()))
if flag.sum():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[sub][w, 0], X[sub][w, 1], s=0.1, c="0.8"); ax.scatter(X[fl, 0], X[fl, 1], s=2, c="r")
    ax.set_aspect("equal"); ax.set_title(f"flagged web-facing points (>8 deg, >4 mm from any label): {flag.sum()}")
    fig.savefig(os.path.join(H, "missed_bend_flagged.png"), dpi=110); plt.close(fig)
json.dump(out, open(os.path.join(H, "missed_bend.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
