"""Auditor-3 final artefacts for sc_f2475286312b:
  a3_corrected_apex_lines.json  apex-locus lines (raw scan frame) for every formed fold, extents from
                                flange occupancy, plus frozen-matcher metrics of every gold line vs
                                every fold line (so a reviewer can see what would match).
  a3_gold_vs_apex_xsec.png      one perpendicular mesh section per gold line at the middle of its target
                                fold: gold point vs outer apex / outer sharp Xo / inner sharp Xi /
                                inner apex / arc centre C, with the 4 mm matcher circle around the apex.
  a3_topview_verdicts.png       top view: gold lines (coloured by verdict) vs fold apex lines (black).
Read-only on the scan and the gold."""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3_foldlib import (FOLDS, fit_station, load_gold, unit, OUT, R, o, AX, V, F, VEC, TC, slice_mesh,
                        pair_metrics, Thresholds, is_match, fit_line)

G = load_gold()
EXT = json.loads((OUT / "a3_extents.json").read_text())
FD = json.loads((OUT / "a3_folds.json").read_text())["folds"]
NAME = {"LIP_+b (M01 side)": "LIP_+b", "LIP_-b (M02 side)": "LIP_-b", "TAB_+b (window b 64..82)": "TAB_+b",
        "TAB_-b (window b -90..-73)": "TAB_-b", "WALL a~113 (M07 side)": "WALL", "SHORT_LIP a~-140": "SHORT_LIP",
        "EAR_+b (wall end)": "EAR_+b", "EAR_-b (wall end)": "EAR_-b"}
# physical extents (occupancy runs; stray boxes from neighbouring features dropped)
SPAN = {"LIP_+b": (-149, 71), "LIP_-b": (-148, 71), "TAB_+b": (-67, 12.5), "TAB_-b": (-66, 12.5),
        "WALL": (-110, 112), "SHORT_LIP": (-101, 103), "EAR_+b": (-46, 0), "EAR_-b": (-47, -1)}
lines = {}
for long, short in NAME.items():
    st = [s for s in FD[long]["stations"] if s["ok"] and 60 <= s["bend_deg"] <= 120]
    fd = FOLDS[long]
    P = []
    for s in st:
        v = np.zeros(3)
        v[AX[fd["axis"]]] = s["v"]
        v[AX[fd["x"]]], v[AX[fd["y"]]] = s["apex"]
        P.append(o + v @ R)
    P = np.array(P)
    c, d, _ = fit_line(P)
    d = d * np.sign(d @ VEC[fd["axis"]])
    # parametrise by the frame axis coordinate
    ax = AX[fd["axis"]]
    c_ax = ((c - o) @ R.T)[ax]
    d_ax = (d @ R.T)[ax]
    lo, hi = SPAN[short]
    p0 = c + d * (lo - c_ax) / d_ax
    p1 = c + d * (hi - c_ax) / d_ax
    lines[short] = {"line_start": p0.tolist(), "line_end": p1.tolist(), "length_mm": float(np.linalg.norm(p1 - p0)),
                    "bend_deg_median": FD[long]["bend_deg_median"], "Ro_median": FD[long]["Ro_median"],
                    "t_median": FD[long]["t_median"], "n_apex_stations": len(P),
                    "apex_fit_resid_rms_mm": float(np.sqrt(np.mean(np.linalg.norm((P - c) - np.outer((P - c) @ d, d), axis=1) ** 2))),
                    "extent_axis": fd["axis"], "extent_mm_in_plate_frame": [lo, hi]}
th = Thresholds()
cross = {}
for gid, (s, e) in G.items():
    row = {}
    for k, L in lines.items():
        pm = pair_metrics(np.array([s, e]), np.array([L["line_start"], L["line_end"]]))
        if pm["axis_angle_deg"] <= 10 and pm["transverse_offset_mm"] <= 25:
            row[k] = {kk: round(vv, 3) for kk, vv in pm.items()} | {"match": bool(is_match(pm, th))}
    cross[gid] = row
doc = {"scan_id": "sc_f2475286312b", "scan_sha256": "f2475286312b7e11be773166afae5d7eaacd826f8b59cee84289d05b4a222b63",
       "frame": "scan_raw (== reference_mesh_raw, identity)", "locus": "outer-skin apex (arc midpoint)",
       "note": "Auditor-3 reference lines, NOT a label file. TAB lines are the straight part only; each tab's last ~9 mm "
               "turns 45 deg about the plate normal (vertical kink at a~12.5).",
       "fold_lines": lines, "gold_vs_fold_lines_frozen_matcher": cross}
(OUT / "a3_corrected_apex_lines.json").write_text(json.dumps(doc, indent=1))
for gid, row in cross.items():
    print(gid, {k: (v["transverse_offset_mm"], v["gold_span_overlap"], v["prediction_span_overlap"], v["match"]) for k, v in row.items()})
for k, L in lines.items():
    print(k, "len %.1f" % L["length_mm"], "bend %.1f" % L["bend_deg_median"], "resid %.2f" % L["apex_fit_resid_rms_mm"])

# ---------------- cross-section figure: one panel per gold line
TARGET = {"M01": "LIP_+b (M01 side)", "M02": "LIP_-b (M02 side)", "M03": "TAB_+b (window b 64..82)",
          "M04": "TAB_+b (window b 64..82)", "M05": "TAB_-b (window b -90..-73)", "M06": "TAB_-b (window b -90..-73)",
          "M07": "WALL a~113 (M07 side)", "M08": "WALL a~113 (M07 side)"}
STATION = {"M01": -20.0, "M02": -20.0, "M03": -25.0, "M04": -25.0, "M05": -25.0, "M06": -25.0, "M07": 0.0, "M08": 0.0}
HALF = {"M01": 9, "M02": 9, "M03": 12, "M04": 14, "M05": 20, "M06": 22, "M07": 9, "M08": 36}
fig, axes = plt.subplots(2, 4, figsize=(24, 12.5), dpi=90)
for ax_, gid in zip(axes.ravel(), G):
    long = TARGET[gid]
    fd = FOLDS[long]
    v = STATION[gid]
    r = fit_station(fd, v)
    xi, yi = AX[fd["x"]], AX[fd["y"]]
    c3 = np.zeros(3)
    c3[AX[fd["axis"]]] = v
    c3[xi], c3[yi] = fd["c0"]
    segs = slice_mesh(V, F, o + c3 @ R, VEC[fd["axis"]], tri_centroid=TC, radius=90)
    S = ((segs.reshape(-1, 3) - o) @ R.T).reshape(-1, 2, 3)
    s2, e2 = (G[gid][0] - o) @ R.T, (G[gid][1] - o) @ R.T
    t = (v - s2[AX[fd["axis"]]]) / (e2[AX[fd["axis"]]] - s2[AX[fd["axis"]]])
    gp = s2 + t * (e2 - s2)
    gp2 = np.array([gp[xi], gp[yi]])
    apex = r["apex"]
    cen = 0.5 * (apex + gp2)
    h = HALF[gid]
    for sg in S:
        if abs(sg[0, xi] - cen[0]) < h + 5 and abs(sg[0, yi] - cen[1]) < h + 5:
            ax_.plot(sg[:, xi], sg[:, yi], color="k", lw=0.8)
    g = r["g"]
    half = math.radians(r["included_deg"]) / 2
    loci = {"outer apex": (apex, "D", "#17becf"), "outer sharp Xo": (r["Xo"], "x", "#1f77b4"),
            "inner sharp Xi": (r["Xo"] + (r["t"] / math.sin(half)) * g, "x", "#9467bd"),
            "inner apex": (apex + r["t"] * g, "s", "#bcbd22"),
            "arc centre C": (r["Xo"] + (r["Ro"] / math.sin(half)) * g, "+", "#8c564b")}
    for nm, (p, mk, col) in loci.items():
        ax_.plot(p[0], p[1], mk, color=col, ms=8, mew=2, mec="k" if mk in "Ds" else col, label=nm)
    ax_.add_patch(plt.Circle(tuple(apex), 4.0, fill=False, ls=":", color="#d62728", lw=1.2))
    ax_.plot(*gp2, "*", color="#1a9c3a", ms=16, mec="k", mew=0.6, label=f"gold {gid}")
    d = np.linalg.norm(gp2 - apex)
    al = (gp2 - apex) @ g
    ax_.set_title(f"{gid} at {fd['axis']}={v:.0f} on {NAME[long]} ({r['bend_deg']:.0f} deg, Ro {r['Ro']:.2f}, t {r['t']:.2f})\n"
                  f"gold-apex {d:.2f} mm, along bisector {al:+.2f} (+ = concave), red dotted = 4 mm around apex", fontsize=8.5)
    ax_.set_xlim(cen[0] - h, cen[0] + h)
    ax_.set_ylim(cen[1] - h, cen[1] + h)
    ax_.set_aspect("equal")
    ax_.grid(alpha=0.3)
    ax_.set_xlabel(fd["x"] + " (mm, plate frame)", fontsize=7)
    ax_.set_ylabel(fd["y"] + " (mm)", fontsize=7)
    ax_.tick_params(labelsize=7)
    if gid == "M01":
        ax_.legend(fontsize=7, loc="lower left")
fig.suptitle("sc_f2475286312b (38172000 scan 2) - auditor 3: exact mesh sections perpendicular to each gold line's target fold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "a3_gold_vs_apex_xsec.png")
plt.close(fig)

# ---------------- top view with verdict colours
VERD = json.loads((OUT / "a3_verdicts.json").read_text()) if (OUT / "a3_verdicts.json").is_file() else {}
VC = {"OK": "#1a9c3a", "CONVENTION_OFFSET": "#ff7f0e", "REAL_ERROR": "#d62728", "NOT_A_BEND": "#9467bd", "UNCLEAR": "#7f7f7f"}
X = (V - o) @ R.T
rng = np.random.default_rng(0)
sel = rng.choice(len(X), 200000, replace=False)
fig, ax = plt.subplots(figsize=(13, 11), dpi=95)
ax.scatter(X[sel, 0], X[sel, 1], c=np.clip(X[sel, 2], -20, 15), s=0.25, cmap="Greys", vmin=-25, vmax=20, linewidths=0)
for k, L in lines.items():
    p0, p1 = (np.array(L["line_start"]) - o) @ R.T, (np.array(L["line_end"]) - o) @ R.T
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="k", lw=3.2, alpha=0.55)
    ax.text(p1[0], p1[1] - 4, k, fontsize=7, color="k")
for gid, (s, e) in G.items():
    s2, e2 = (s - o) @ R.T, (e - o) @ R.T
    col = VC.get(VERD.get(gid, ""), "#1a9c3a")
    ax.plot([s2[0], e2[0]], [s2[1], e2[1]], color=col, lw=1.4)
    ax.text(s2[0] - 14, s2[1] + (2 if gid in ("M03", "M05") else -4), f"{gid} {VERD.get(gid, '')}", fontsize=7, color=col)
ax.set_aspect("equal")
ax.set_xlabel("a (mm)")
ax.set_ylabel("b (mm)")
ax.grid(alpha=0.3)
ax.set_title("sc_f2475286312b top view: black = auditor-3 fold apex lines (8 formed folds), colour = gold by verdict", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "a3_topview_verdicts.png")
plt.close(fig)
print("ok")
