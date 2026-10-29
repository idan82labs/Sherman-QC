"""Web height profiles (deviation from web plane) to tell a formed crease from gradual warp (loads full scan; run under lock)."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import v1_sections as V
H = os.path.dirname(os.path.abspath(__file__))
P, N = V.load_ply(V.SCAN); N /= np.linalg.norm(N, axis=1, keepdims=True)
J = json.load(open(os.path.join(H, "../lines.json")))
B1 = J["bends"][0]; nw = np.array(B1["plane_a"]["normal"]); nw /= np.linalg.norm(nw); p0 = np.array(B1["plane_a"]["point"])
e1 = np.array(B1["direction"]); e1 -= (e1 @ nw) * nw; e1 /= np.linalg.norm(e1); e2 = np.cross(nw, e1)
X = np.c_[P @ e1, P @ e2, (P - p0) @ nw]
top = (N @ nw) > 0.9  # web top-skin-facing points
fig, ax = plt.subplots(3, 1, figsize=(11, 10))
# A: profiles across the part (vs e2) at several e1 positions
for x0 in (-480, -400, -350, -300, -230):
    m = top & (np.abs(X[:, 0] - x0) < 0.5)
    o = np.argsort(X[m, 1]); ax[0].plot(X[m, 1][o], X[m, 2][o], ".", ms=1, label=f"e1={x0}")
ax[0].set_xlabel("e2 (mm, across part; -Y flange at right)"); ax[0].set_ylabel("height above web plane"); ax[0].legend(fontsize=7, markerscale=8)
ax[0].set_title("web top skin height across the part (a formed crease = kink; warp = smooth)")
ax[0].set_ylim(-3, 3)
# B: profile along e1 near the faint mark at e1~-258, e2 130..170
for y0 in (133, 150, 168):
    m = top & (np.abs(X[:, 1] - y0) < 0.5) & (np.abs(X[:, 0] + 258) < 15)
    o = np.argsort(X[m, 0]); ax[1].plot(X[m, 0][o], X[m, 2][o], ".", ms=2, label=f"e2={y0}")
ax[1].set_xlabel("e1 (mm)"); ax[1].set_ylabel("height (mm)"); ax[1].legend(fontsize=7, markerscale=5); ax[1].set_title("web height along e1 across the faint mark at e1 ~ -258")
# C: strip between tab-B window and -Y flange: height vs e2 at e1=-350,-330 zoom
for x0 in (-440, -350, -330, -250):
    m = top & (np.abs(X[:, 0] - x0) < 0.5) & (X[:, 1] > 200)
    o = np.argsort(X[m, 1]); ax[2].plot(X[m, 1][o], X[m, 2][o], ".", ms=2, label=f"e1={x0}")
ax[2].set_xlabel("e2 (mm)"); ax[2].set_ylabel("height (mm)"); ax[2].legend(fontsize=7, markerscale=5); ax[2].set_title("-Y web edge strip (e2>200)")
fig.tight_layout(); fig.savefig(os.path.join(H, "web_profiles.png"), dpi=100)
print("ok")
