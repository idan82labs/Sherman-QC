"""Tab ends (30 deg ears?) via horizontal slices; X-Z elevation; end/corner close-ups."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from a3x_common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = load_points()
F = json.load(open(OUT / "a3x_frame.json"))
O, X, Y, Z = (np.array(F[k]) for k in ("O", "X", "Y", "Z"))
L = np.c_[(p - O) @ X, (p - O) @ Y, (p - O) @ Z]
gold = load_gold()
GL = {bid: (np.array([(a - O) @ X, (a - O) @ Y, (a - O) @ Z]), np.array([(b - O) @ X, (b - O) @ Y, (b - O) @ Z])) for bid, a, b in gold}

fig, axs = plt.subplots(2, 3, figsize=(24, 14))
info = {}
for row, (name, ylo, yhi, bid) in enumerate([("tab_A", 66, 92, "M01"), ("tab_B", -106, -80, "M04")]):
    m = (L[:, 0] > -25) & (L[:, 0] < 90) & (L[:, 1] > ylo) & (L[:, 1] < yhi) & (L[:, 2] > 2.5)
    T = L[m]
    ax = axs[row, 0]
    s = ax.scatter(T[:, 0], T[:, 1], c=T[:, 2], s=0.5, cmap="viridis")
    plt.colorbar(s, ax=ax, fraction=0.03, label="Z")
    a, b = GL[bid]
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#1a9c3a", lw=1.5, label=f"gold {bid}")
    ax.set_xlim(-25, 90); ax.set_ylim(ylo, yhi); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"{name}: points with Z>2.5 (tab body) top view, gold {bid} in green")
    # zoom on the end x<10, horizontal slices
    ax = axs[row, 1]
    res = []
    for zc, col in [(4.0, "C0"), (7.0, "C1"), (10.0, "C2"), (13.0, "C3")]:
        mm = (L[:, 0] > -20) & (L[:, 0] < 25) & (L[:, 1] > ylo) & (L[:, 1] < yhi) & (np.abs(L[:, 2] - zc) < 0.3)
        S = L[mm]
        ax.scatter(S[:, 0], S[:, 1], s=1, c=col, label=f"Z={zc}+-0.3")
        # fit straight part (x in 10..25) and end part (x < x_knee)
        st = S[(S[:, 0] > 8) & (S[:, 0] < 25)]
        en = S[(S[:, 0] < -2)]
        if len(st) > 10 and len(en) > 10:
            k1 = np.polyfit(st[:, 0], st[:, 1], 1)
            k2 = np.polyfit(en[:, 0], en[:, 1], 1)
            ang = math.degrees(abs(math.atan(k2[0]) - math.atan(k1[0])))
            xk = (k2[1] - k1[1]) / (k1[0] - k2[0])
            res.append(dict(z=zc, ear_angle_deg=ang, knee_x=xk, knee_y=float(np.polyval(k1, xk)), end_xmin=float(S[:, 0].min()),
                            n_end=int(len(en))))
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_xlim(-20, 25)
    ax.set_title(f"{name} end: horizontal slices (top view)")
    info[name] = res
    # vertical slice through the ear: points of the end part, show Z-extent of ear vs x
    ax = axs[row, 2]
    mm = (L[:, 0] > -20) & (L[:, 0] < 12) & (L[:, 1] > ylo) & (L[:, 1] < yhi) & (L[:, 2] > -3)
    S = L[mm]
    ax.scatter(S[:, 0], S[:, 2], s=0.5, c=S[:, 1], cmap="plasma")
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"{name} end: X-Z elevation (colour = Y), web at Z~0")
fig.tight_layout()
fig.savefig(OUT / "a3x_tab_ends.png", dpi=70)
print(json.dumps(info, indent=1))

# X-Z elevation of full part and end close-ups
fig, axs = plt.subplots(3, 1, figsize=(22, 16))
rng = np.random.default_rng(1)
sub = L[rng.choice(len(L), 300000, replace=False)]
ax = axs[0]
ax.scatter(sub[:, 0], sub[:, 2], s=0.2, c=sub[:, 1], cmap="coolwarm")
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_title("X-Z elevation, all Y (colour=Y). Web flat? any end bends?")
ax.set_ylim(-20, 20)
for k, (x0, x1) in enumerate([(-166, -136), (136, 166)]):
    ax = axs[1 + k]
    m = (L[:, 0] > x0) & (L[:, 0] < x1)
    S = L[m]
    s = ax.scatter(S[:, 0], S[:, 1], c=np.clip(S[:, 2], -16, 16), s=0.4, cmap="coolwarm", vmin=-16, vmax=16)
    for bid, (a, b) in GL.items():
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#1a9c3a", lw=1)
    ax.set_xlim(x0, x1); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"end region X in [{x0},{x1}] top view, colour = Z")
fig.tight_layout()
fig.savefig(OUT / "a3x_ends.png", dpi=70)
