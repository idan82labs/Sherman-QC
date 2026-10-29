"""Auditor-1 (sc_71cf0b216f43): per-line offset profiles + verdict summary JSON.

Reads a1_xsec_measurements.json, a1_apex_lines.json, a1_joggles.json (all produced by the
scripts in this folder from the raw scan, read-only) and writes:
  a1_offset_profiles.png   gold->outer-apex distance along every gold line (fold extent shaded,
                           4 mm matcher limit dashed)
  a1_verdicts.json         per-line numbers + verdict, missing bends, relabel proposal (apex lines,
                           raw scan frame, clipped to the measured fold extents)
"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_common import OUT, get

P, VN, F, G, R, c = get()
gold = {bid: (s, e) for bid, s, e in G}
meas = json.load(open(OUT + "/a1_xsec_measurements.json"))["folds"]
ap = json.load(open(OUT + "/a1_apex_lines.json"))["folds"]
jg = json.load(open(OUT + "/a1_joggles.json"))["joggles"]

VERDICT = {
    "M01": "REAL_ERROR", "M02": "REAL_ERROR", "M03": "OK", "M04": "OK",
    "M05": "REAL_ERROR", "M06": "REAL_ERROR", "M07": "REAL_ERROR", "M08": "REAL_ERROR",
}
rows = {}
fig, axes = plt.subplots(4, 2, figsize=(18, 16), dpi=80)
for axp, (fid, fd) in zip(axes.ravel(), meas.items()):
    gid = fd["def"]["gold"]
    a = ap[fid]
    f0, f1 = a["fold_extent_axis_coord"]
    st = [s for s in fd["stations"] if s.get("gold") and f0 <= s["s"] <= f1]
    s_ = np.array([s["s"] for s in st]); d_ = np.array([s["gold"]["d_apex"] for s in st])
    al = np.array([s["gold"]["along_bisector_concave"] for s in st]); ac = np.array([s["gold"]["across_toward_A"] for s in st])
    nn = np.array([s.get("gold_d_nearest_scan_3d", np.nan) for s in st])
    gs, ge = a["gold_span_axis_coord"]
    within4 = float(np.mean(d_ <= 4.0))
    rows[gid] = {
        "fold": fid, "fold_desc": a["desc"], "verdict": VERDICT[gid],
        "fold_extent_axis_mm": [f0, f1], "fold_length_mm": a["fold_length_mm"],
        "gold_span_axis_mm": [gs, ge], "gold_length_mm": float(np.linalg.norm(gold[gid][1] - gold[gid][0])),
        "coverage_of_fold_pct": round(100 * a["gold_coverage_of_fold"], 1),
        "gold_outside_fold_mm": a["gold_beyond_fold_mm"],
        "d_outer_apex_mm_min_med_max": [float(d_.min()), float(np.median(d_)), float(d_.max())],
        "along_bisector_concave_mm_min_max": [float(al.min()), float(al.max())],
        "across_toward_wall_mm_min_max": [float(ac.min()), float(ac.max())],
        "gold_nearest_scan_mm_median": float(np.nanmedian(nn)),
        "frac_of_fold_stations_within_4mm_of_apex": round(within4, 3),
        "angle_to_apex_line_deg": a["gold_angle_to_apex_deg"],
        "matcher_vs_measured_apex_line": a["matcher_vs_apex_line"], "matcher_pass": a["matcher_pass_vs_apex_line"],
        "n_stations": len(st),
    }
    axp.axvspan(f0, f1, color="#e8f4e8", label="measured fold extent")
    axp.axhline(4.0, color="r", ls="--", lw=1, label="4 mm matcher limit")
    stall = [s for s in fd["stations"] if s.get("gold")]
    axp.plot([s["s"] for s in stall], [s["gold"]["d_apex"] for s in stall], ".", color="#999999", ms=3, label="outside fold (no fold there)")
    axp.plot(s_, d_, "-o", color="#00a000", ms=3, label="gold -> outer apex")
    axp.axvline(gs, color="k", lw=0.8); axp.axvline(ge, color="k", lw=0.8)
    axp.set_title(f"{gid} ({fid}: {a['desc']}) -> {VERDICT[gid]}; cov {100 * a['gold_coverage_of_fold']:.0f}%, "
                  f"d_apex {d_.min():.1f}-{d_.max():.1f} mm", fontsize=9)
    axp.set_xlabel(("x" if fd["def"]["axis"] == 0 else "y") + " along fold (part frame, mm); black = gold ends")
    axp.set_ylabel("mm"); axp.set_ylim(0, max(6, d_.max() + 1)); axp.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7)
fig.suptitle("sc_71cf0b216f43 (10839000) auditor-1: distance of each human gold line to the measured outer apex of its fold", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT + "/a1_offset_profiles.png")

# relabel proposal: 8 wall-fold apex lines (clipped to fold extent) + 4 joggle folds
prop = []
for fid, a in ap.items():
    prop.append({"fold": fid, "replaces_gold": a["gold_id"], "locus": "outer_apex", "desc": a["desc"],
                 "line_start": a["apex_line_raw"][0], "line_end": a["apex_line_raw"][1],
                 "length_mm": a["fold_length_mm"], "apex_fit_resid_med_mm": a["apex_fit_resid_med_mm"]})
for name, v in jg.items():
    for fold, ln in v["fold_lines_topskin_virtual_sharp"].items():
        e = np.array(ln["part"], float)
        # clip to the flange extent (end faces at x = -160.3 / 145.7); beyond that the step is the short-wall top cut edge
        d = e[1] - e[0]
        t0 = (-160.3 - e[0][0]) / d[0]; t1 = (145.7 - e[0][0]) / d[0]
        e0, e1 = e[0] + t0 * d, e[0] + t1 * d
        prop.append({"fold": f"{name}_{fold}", "replaces_gold": None, "locus": "top-skin virtual sharp of the joggle (~apex for J_upper; "
                     "J_lower apex is ~1.4 mm lower on the bottom skin)", "desc": f"joggle fold in {name} "
                     f"(drop {v['summary_median_p5_p95']['drop'][0]:.2f} mm, web {v['summary_median_p5_p95']['web_slope_deg'][0]:.0f} deg)",
                     "line_start": (e0 @ R + c).tolist(), "line_end": (e1 @ R + c).tolist(), "length_mm": float(np.linalg.norm(e1 - e0))})
out = {"scan_id": "sc_71cf0b216f43", "part": "10839000", "gold11_id": "CG84C8FA12D2", "auditor": "auditor1",
       "frame_note": "all *_axis_mm values are in the auditor part frame (x long, y short, z flange normal; a1_frame.py); line_start/line_end are raw scan coordinates",
       "sheet_thickness_mm": 1.42, "outer_radius_mm_typ": 2.8,
       "lines": rows, "relabel_proposal_apex_lines_raw": prop,
       "missing_bends": [p["fold"] for p in prop if p["replaces_gold"] is None],
       "label_job_note": "output/bend_label_01_10839000/bend_inspection_meta.json says 'manual bend truth (8 expected)': the joggles were not in the expected count"}
json.dump(out, open(OUT + "/a1_verdicts.json", "w"), indent=1)
for gid, r in rows.items():
    print(gid, r["verdict"], "cov", r["coverage_of_fold_pct"], "outside", [round(v, 1) for v in r["gold_outside_fold_mm"]],
          "d_apex", [round(v, 2) for v in r["d_outer_apex_mm_min_med_max"]], "within4", r["frac_of_fold_stations_within_4mm_of_apex"],
          "ang", round(r["angle_to_apex_line_deg"], 2), "nn", round(r["gold_nearest_scan_mm_median"], 2), "match", r["matcher_pass"])
for p in prop:
    print(p["fold"], np.round(p["line_start"], 2), np.round(p["line_end"], 2), round(p["length_mm"], 1))
