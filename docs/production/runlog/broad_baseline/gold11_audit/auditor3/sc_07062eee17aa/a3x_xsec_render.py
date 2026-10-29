"""Perpendicular cross-section panels (Y-Z, slab +-0.6 mm in X) at chosen stations, with gold points,
4 mm matcher circles, fitted outer apex / outer & inner virtual sharp / flange tip from a3x_stations.json."""
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
order = np.argsort(L[:, 0]); xs_sorted = L[order, 0]
gold = load_gold()
GL = {bid: (np.array([(a - O) @ X, (a - O) @ Y, (a - O) @ Z]), np.array([(b - O) @ X, (b - O) @ Y, (b - O) @ Z])) for bid, a, b in gold}
ST = json.load(open(OUT / "a3x_stations.json"))


def gold_at(bid, x):
    a, b = GL[bid]
    t = (x - a[0]) / (b[0] - a[0])
    return a + t * (b - a), t


def nearest_station(bid, x):
    rows = ST[bid]
    r = min(rows, key=lambda r: abs(r["x"] - x))
    return r if abs(r["x"] - x) <= 1.01 else None


def panel(ax, x, bids, center, half, title):
    i0, i1 = np.searchsorted(xs_sorted, [x - 0.6, x + 0.6])
    S = L[order[i0:i1]]
    m = (np.abs(S[:, 1] - center[0]) < half * 1.3) & (np.abs(S[:, 2] - center[1]) < half * 1.3)
    ax.scatter(S[m, 1], S[m, 2], s=2, c="0.35")
    for bid, col in bids:
        g, t = gold_at(bid, x)
        inside = 0 <= t <= 1
        ax.plot(g[1], g[2], "*", ms=16 if inside else 10, mec="k", mfc=col if inside else "white", zorder=5)
        ax.add_patch(plt.Circle((g[1], g[2]), 4.0, fill=False, ls=":", color=col, lw=1))
        ax.text(g[1] + 0.6, g[2] + 0.6, bid + ("" if inside else " (beyond end)"), color=col, fontsize=9, weight="bold")
        r = nearest_station(bid, x)
        if r and r.get("fold") and r.get("apex"):
            ax.plot(*r["apex"], "D", mfc="cyan", mec="k", ms=7, zorder=6)
            ax.plot(*r["Xo"], "x", color="tab:blue", ms=9, mew=2, zorder=6)
            if r.get("Xi"):
                ax.plot(*r["Xi"], "x", color="tab:purple", ms=9, mew=2, zorder=6)
            if r.get("tip"):
                ax.plot(*r["tip"], "s", mfc="orange", mec="k", ms=7, zorder=6)
    ax.set_xlim(center[0] - half, center[0] + half); ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Y (across) mm"); ax.set_ylabel("Z (off web) mm")


def figure(fname, suptitle, bids, center_fn, xs_list, half=11):
    n = len(xs_list)
    cols = 4
    rows = int(math.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax, x in zip(axs, xs_list):
        lines = []
        for bid, _ in bids:
            r = nearest_station(bid, x)
            if r is None:
                continue
            s = f"{bid}: "
            if r.get("fold") and r.get("d_apex") is not None:
                s += f"d_apex {r['d_apex']:.2f}"
                if r.get("d_tip") is not None and bid in ("M03", "M06"):
                    s += f", d_tip {r['d_tip']:.2f} (beyond tip {r['gold_minus_tip_along_flange']:+.2f})"
            else:
                s += f"no fold; d_scan {r.get('d_gold_nearest_scan', float('nan')):.2f}"
            lines.append(s)
        panel(ax, x, bids, center_fn(x), half, f"X = {x:.0f} mm\n" + "\n".join(lines))
    for ax in axs[n:]:
        ax.axis("off")
    fig.suptitle(suptitle + "\nstar = gold point on this section (dotted = 4 mm matcher circle); cyan diamond = outer apex; blue x = outer virtual sharp; purple x = inner virtual sharp; orange square = flange/tab free-edge tip", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=72)


def flange_center(side):
    def f(x):
        return (109.0, -6.5) if side > 0 else (-113.0, -6.5)
    return f


def tab_center(bid):
    def f(x):
        g, _ = gold_at(bid, x)
        return (g[1], 5.0)
    return f


figure("a3x_xsec_M02_M03.png", "sc_07062eee17aa +Y side flange: gold M02 (green) and M03 (red)",
       [("M02", "#1a9c3a"), ("M03", "#d62728")], flange_center(+1), [-146, -138, -120, -80, -56, -20, 30, 100, 144, 150, 156, 160], half=11)
figure("a3x_xsec_M05_M06.png", "sc_07062eee17aa -Y side flange: gold M05 (green) and M06 (red)",
       [("M05", "#1a9c3a"), ("M06", "#d62728")], flange_center(-1), [-146, -138, -120, -80, -56, -20, 30, 100, 144, 150, 156, 160], half=11)
figure("a3x_xsec_M01.png", "sc_07062eee17aa tab A: gold M01 (green) - stations inside and outside the tab",
       [("M01", "#1a9c3a")], tab_center("M01"), [-141, -101, -61, -21, -7, -1, 9, 39, 69, 75, 81, 121], half=11)
figure("a3x_xsec_M04.png", "sc_07062eee17aa tab B: gold M04 (green) - stations inside and outside the tab",
       [("M04", "#1a9c3a")], tab_center("M04"), [-140, -100, -60, -20, -8, -2, 8, 38, 68, 76, 82, 120], half=11)
print("ok")
