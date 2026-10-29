"""Worst-case PNGs for E3_STABILITY.md. usage: plots.py OUTDIR"""
import sys, json, pickle
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject"]
from domains.production.engine.inspect.scan_io import read_scan
R = Path(__file__).resolve().parents[1]; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C_NAT, C_SUB, C_GREY = "#1f5fa8", "#d9640f", "#b8b8b8"

def section(probe_nat, probe_sub, regscan, bend, title, fname, half=4.0):
    A = pickle.load(open(R / f"probe/{probe_nat}/probe.pkl", "rb"))["bends"][bend]
    B = pickle.load(open(R / f"probe/{probe_sub}/probe.pkl", "rb"))["bends"][bend]
    # bring the subsample run's points and planes into the native run's registered frame (its pose differs by ~0.1 deg / 0.17 mm)
    Tn = np.array(json.load(open(R / f"probe/{probe_nat}/engine/result.json"))["inspection_validity"]["registration"]["raw_to_frame"])
    Ts = np.array(json.load(open(R / f"probe/{probe_sub}/engine/result.json"))["inspection_validity"]["registration"]["raw_to_frame"])
    M = Tn @ np.linalg.inv(Ts); Rm, tv = M[:3, :3], M[:3, 3]
    B = dict(B)
    for k in ("parent_pts", "child_pts"): B[k] = (np.asarray(B[k], float) @ Rm.T + tv)
    for k in ("plane_parent", "plane_child", "region_parent"):
        n = Rm @ np.array(B[k][0]); B[k] = [n, float(B[k][1] + n @ tv)]
    B["line_point"] = (Rm @ np.array(B["line_point"]) + tv).tolist()
    a = np.array(A["line_dir"]); lp = np.array(A["line_point"]); zc = 0.5 * (A["visible"][0] + A["visible"][1])
    o = lp + a * zc
    e1 = np.array(A["plane_parent"][0]); e1 = e1 - (e1 @ a) * a; e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    def proj(p):
        r = p - o; m = np.abs(r @ a) < half; return np.c_[r[m] @ e2, r[m] @ e1]
    allp = read_scan(str(regscan)).points
    q = proj(allp); keep = (np.abs(q[:, 0]) < 28) & (np.abs(q[:, 1]) < 28)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100, gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.scatter(q[keep, 0], q[keep, 1], s=1, c=C_GREY, lw=0, label="native scan points (registered, ±%.0f mm axial)" % half, rasterized=True)
    pn = proj(A["parent_pts"]); ps = proj(B["parent_pts"])
    ax.scatter(pn[:, 0], pn[:, 1], s=3, c=C_NAT, lw=0, alpha=.6, label=f"short panel, native: {len(A['parent_pts'])} pts (both skins)", rasterized=True)
    ax.scatter(ps[:, 0], ps[:, 1] + 0.0, s=4, c=C_SUB, lw=0, alpha=.9, zorder=3, label=f"short panel, 95% subsample: {len(B['parent_pts'])} pts (one skin)", rasterized=True)
    for X, col, nm in ((A, C_NAT, "native"), (B, C_SUB, "95% subsample")):
        for key, ls in (("plane_parent", "-"), ("plane_child", "--")):
            n, d = np.array(X[key][0]), X[key][1]
            # trace of plane n.x=d in the section: points x = o + s*e2 + h*e1 with n.(o+s e2+h e1)=d
            s = np.linspace(-28, 28, 400); ne1, ne2 = n @ e1, n @ e2; base = d - n @ o
            if abs(ne1) > abs(ne2): h = (base - s * ne2) / ne1; xs, ys = s, h
            else: h = np.linspace(-28, 28, 400); xs, ys = (base - h * ne1) / ne2, h
            ax.plot(xs, ys, ls, c=col, lw=1.3, label=f"{nm} OML plane fit ({'short panel' if key == 'plane_parent' else 'other flange'})")
        sh = np.array(X["sharp"]); v = np.array(X["valid"]); sp = proj(sh[v]) if v.any() else np.zeros((0, 2))
        lpX = np.array(X["line_point"]) + a * (zc - (np.array(X["line_point"]) - o) @ a * 0)
        qq = np.array([(lpX - o) @ e2, (lpX - o) @ e1])
        ax.plot(*qq, marker="x", ms=10, mew=2, c=col)
    ax.set_xlim(-26, 26); ax.set_ylim(-20, 20); ax.set_aspect("equal")
    ax.set_xlabel("in-section mm"); ax.set_ylabel("mm along native short-panel normal")
    ax.set_title(title + "\n(subsample run mapped into the native frame)", fontsize=11, loc="left"); ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax = axes[1]
    n, d = np.array(A["region_parent"][0]), A["region_parent"][1]
    bins = np.arange(-2.0, 1.0, 0.05)
    ax.hist(A["parent_pts"] @ n - d, bins=bins, color=C_NAT, alpha=.7, label="native: panel owns both skins")
    ax.hist(B["parent_pts"] @ n - d, bins=bins, color=C_SUB, alpha=.7, label="95% subsample: panel owns one skin")
    nb, db = np.array(B["region_parent"][0]), B["region_parent"][1]
    off = float(np.median(B["parent_pts"] @ nb - db) - np.median(B["parent_pts"] @ n - d))
    ax.axvline(0, c=C_NAT, lw=1.5); ax.axvline(-A["t"], c=C_NAT, lw=1, ls=":")
    ax.axvline(-off, c=C_SUB, lw=1.5)
    ax.text(0.02, -0.12, "", transform=ax.transAxes)
    ax.set_xlabel("distance to native OML plane of the short panel (mm)"); ax.set_ylabel("points")
    ax.set_title(f"Skin assignment: subsample OML plane sits {abs(off):.2f} mm inside, on the inner skin\n(t = {A['t']:.2f} mm; dotted = native inner skin)", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=100); plt.close(fig)

section("B_nat", "B_s9501", R / "runs/75611b1d377fec21/native/engine/registered_scan.ply", "B5",
        "23553000 B full scan (sc_6eba02723d4b), section through B5\nnative vs 95% subsample seed 9501", "e3stab_B5_section_full_scan.png")

# value modes figure
A = json.load(open(R / "analysis.json")); by = {j["key"]: j for j in A["jobs"]}
VARS = ["native"] + [f"s95_{s}" for s in (9501, 9502, 9503, 9504, 9505)] + ["thin040"]
def modes(key, picks, title, fname):
    j = by[key]; cs = [c for c in j["chars"] if (c["label"], c["kind"], c["side"]) in picks]
    cs.sort(key=lambda c: picks.index((c["label"], c["kind"], c["side"])))
    fig, axes = plt.subplots(1, len(cs), figsize=(12, 4.6), dpi=100)
    col = {"PASS": "#2e8b57", "FAIL": "#c0392b", "REVIEW": "#e0a400", "UNKNOWN": "#777777"}
    for ax, c in zip(axes, cs):
        nom, tp, tm = c["expected"], c["tp"], c["tm"]
        ax.axhspan(nom - tm, nom + tp, color="#2e8b57", alpha=.10, lw=0)
        ax.axhline(nom, c="#2e8b57", lw=.8, ls=":")
        for i, v in enumerate(VARS):
            if v == "native": d, x, u = c["native"][1], c["native"][2], c["u"]
            else:
                y = c["var"].get(v)
                if not y or y[2] is None: continue
                d, x, u = y[1], y[2], y[3]
            ax.errorbar(i, x, yerr=u, fmt="o", ms=6, c=col.get(d, "k"), capsize=3, mec="k" if v == "native" else col.get(d, "k"))
        ax.set_xticks(range(len(VARS))); ax.set_xticklabels(["N", "1", "2", "3", "4", "5", "T"], fontsize=8)
        k = {"FLANGE_LENGTH": "flange", "INCLUDED_ANGLE": "angle", "BEND_LINE_POSITION": "line pos."}[c["kind"]]
        ax.set_title(f"{c['label']} {k}{' ' + c['side'] if c['side'] else ''}", fontsize=10)
        ax.set_ylabel("deg" if c["kind"] == "INCLUDED_ANGLE" else "mm", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", c=v, label=k) for k, v in col.items() if k != "UNKNOWN"]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#2e8b57", alpha=.15, label="tolerance band"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9)); fig.savefig(OUT / fname, dpi=100); plt.close(fig)
modes("75611b1d377fec21", [("B5", "FLANGE_LENGTH", "B"), ("B6", "FLANGE_LENGTH", "A"), ("B6", "FLANGE_LENGTH", "B"), ("B7", "FLANGE_LENGTH", "A"), ("B5", "INCLUDED_ANGLE", None), ("B5", "BEND_LINE_POSITION", None)],
      "23553000 B full scan: each characteristic over the 7 realizations (bars = reported U; N native, 1-5 seeds of 95%, T 0.40 mm thinning).\nTwo value modes; the native run is in one, 5 of 6 perturbations in the other.",
      "e3stab_B_full_scan_modes.png")
modes("f50be239406b9dce", [("B5", "FLANGE_LENGTH", "A"), ("B5", "FLANGE_LENGTH", "B"), ("B6", "FLANGE_LENGTH", "B"), ("B5", "INCLUDED_ANGLE", None), ("B5", "BEND_LINE_POSITION", None), ("B6", "BEND_LINE_POSITION", None)],
      "23553000 B scan 2 (pc_186463746daa), 7 realizations (bars = reported U; N native, 1-5 seeds of 95%, T 0.40 mm thinning).\nModes alternate seed by seed (PASS/REVIEW/FAIL on B5 flange B).",
      "e3stab_B_scan2_modes.png")

# station skin choice figure for scan 2
def stations(pn, ps, bend, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=100)
    for ax, p, nm in ((axes[0], pn, "native"), (axes[1], ps, "95% subsample seed 9501")):
        X = pickle.load(open(R / f"probe/{p}/probe.pkl", "rb"))["bends"][bend]
        fits = X["fits"]; par = fits[0::2]; chi = fits[1::2]
        z = np.array(X["z"])
        for k, (fs, lab, yy) in enumerate(((par, "short panel (parent)", 1), (chi, "other flange (child)", 0))):
            for i, f in enumerate(fs):
                ax.scatter(z[i], yy, s=40 + f["support"] / 2, c="#8e44ad" if f["inner"] else "#16a085", edgecolor="k", lw=.5)
                ax.text(z[i], yy + 0.18, str(f["support"]), ha="center", fontsize=7)
        ang = np.array(X["angles"])
        ax2 = ax.twinx(); ax2.plot(z, ang, "-o", c="#444", ms=3, lw=1); ax2.set_ylabel("station included angle (deg)", fontsize=8)
        ax2.set_ylim(58, 68)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["other flange", "short panel"]); ax.set_ylim(-0.6, 1.8)
        ax.set_xlabel("station axial position (mm)")
        fs = X["flat_shift"]
        ax.set_title(f"{nm}\nflat-start shift {fs[0]:.2f} / {fs[1]:.2f} mm; reseat {'yes' if any(r['plane_out'] is not None for r in X['reseat']) else 'no'}; inner-skin share {X['inner'][0]:.1f} / {X['inner'][1]:.1f}", fontsize=9.5, loc="left")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    h = [plt.Line2D([], [], marker="o", ls="", c="#16a085", mec="k", label="station fit on OML (outer) skin"),
         plt.Line2D([], [], marker="o", ls="", c="#8e44ad", mec="k", label="station fit on inner skin + t"),
         plt.Line2D([], [], marker="o", c="#444", label="station angle")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title + "  (marker size and number = station support points)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94)); fig.savefig(OUT / fname, dpi=100); plt.close(fig)
stations("B2_nat", "B2_s9501", "B5", "23553000 B scan 2, bend B5 station fits", "e3stab_B_scan2_B5_stations.png")
print("ok")
