"""Extent / locus probe for the sc_08686dc1d5d4 AI relabel (read-only; downsampled prep cloud only).

For every ok bend of a lines.json run:
- fold support (bench_labeling.geometry.fold_support, same params as the tool) over a wide t window,
  so the unsupported gaps show where the formed fold really ends;
- t-range of prep points within 0.8 mm of the label (apex) line;
- per-flange t-range of points in a band 3..10 mm from the line inside each flange plane;
- distance/angle of each auditor's measured line of the same fold to the label line.
Usage: python -I extent_probe.py <lines.json>
"""
import json
import sys

sys.dont_write_bytecode = True
import numpy as np

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from research.bench_labeling.geometry import fold_support  # noqa: E402

ROOT = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/"
PREP = ROOT + "prep/sc_08686dc1d5d4/"
lines_path = sys.argv[1]
L = json.load(open(lines_path))
man = json.load(open(PREP + "prep_manifest.json"))
ds = np.load(PREP + "points_ds.npz")
pts = ds["points"]
voxel = float(man["voxel_mm"])
t_mm, ri = float(L["thickness_mm"]), float(L["ri_mm"])

A3 = json.load(open(ROOT + "auditor3/sc_08686dc1d5d4/formed_bends_apex.json"))
A1 = json.load(open(ROOT + "auditor1/sc_08686dc1d5d4/a1_missing_bends.json"))
aud = {k: (v["start"], v["end"]) for k, v in A3.items()}
aud.update({"a1_" + k: tuple(v["line_raw"]) for k, v in A1.items()})


def perp(p, o, d):
    r = np.asarray(p, float) - o
    return float(np.linalg.norm(r - (r @ d) * d))


out = {}
for b in L["bends"]:
    if b["status"] != "ok":
        continue
    o = np.asarray(b["line_origin"], float)
    d = np.asarray(b["direction"], float)
    bis = np.asarray(b["bisector"], float)
    shift = b["apex_offset_mm"] if b["label_locus"] == "apex" else 0.0
    origin = o + shift * bis
    ua = np.asarray(b["side_dirs"]["plane_a"], float)
    ub = np.asarray(b["side_dirs"]["plane_b"], float)
    wide = fold_support(pts, origin, d, ua, ub, -180.0, 180.0, reach_mm=ri + t_mm + 1.0, min_points=3,
                        half_window_mm=max(0.5, 0.75 * voxel), arc_band_mm=max(0.5, 0.5 * voxel))
    # supported runs from the gaps
    gaps = wide["gaps"]
    runs, cur = [], -180.0
    for g0, g1 in gaps:
        if g0 > cur:
            runs.append([round(cur, 1), round(g0, 1)])
        cur = g1
    if cur < 180.0:
        runs.append([round(cur, 1), 180.0])
    rel = pts - origin
    s = rel @ d
    w = rel - np.outer(s, d)
    r = np.linalg.norm(w, axis=1)
    near = r <= 0.8
    tn = s[near]
    # flange bands: points 3..10 mm from the line, close to the flange direction (within 1.5 mm of the plane ray)
    band = {}
    for key, u in (("a", ua), ("b", ub)):
        along = w @ u
        off = np.linalg.norm(w - np.outer(along, u), axis=1)
        sel = (along >= 3.0) & (along <= 10.0) & (off <= 1.5 * t_mm + 0.5)
        ts = s[sel]
        band[key] = [round(float(np.percentile(ts, 0.5)), 1), round(float(np.percentile(ts, 99.5)), 1)] if len(ts) else None
    lab = b["label_line"]
    S, E = np.asarray(lab["start"]), np.asarray(lab["end"])
    close = []
    for k, (p0, p1) in aud.items():
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        dv = (p1 - p0) / np.linalg.norm(p1 - p0)
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(dv @ d))))))
        mid = (p0 + p1) / 2
        dm = perp(mid, origin, d)
        if ang < 5 and dm < 15:
            ta = sorted(float((q - origin) @ d) for q in (p0, p1))
            close.append({"auditor_line": k, "angle_deg": round(ang, 2), "mid_dist_mm": round(dm, 2),
                          "end_dists_mm": [round(perp(p0, origin, d), 2), round(perp(p1, origin, d), 2)],
                          "t_span": [round(v, 1) for v in ta], "len": round(ta[1] - ta[0], 1)})
    close.sort(key=lambda c: c["mid_dist_mm"])
    out[b["bend_id"]] = {
        "patches": b["patches"], "t_range": [round(v, 1) for v in b["t_range"]],
        "t_range_auto": b["t_range_auto"] and [round(v, 1) for v in b["t_range_auto"]],
        "length": round(b["length_mm"], 1),
        "wide_supported_runs_t": runs,
        "apex_cyl_0p8mm_t": [round(float(tn.min()), 1), round(float(tn.max()), 1)] if len(tn) else None,
        "apex_cyl_0p8mm_t_p0.5_p99.5": [round(float(np.percentile(tn, 0.5)), 1),
                                       round(float(np.percentile(tn, 99.5)), 1)] if len(tn) else None,
        "flange_band_t": band, "auditor_lines_same_fold": close[:4]}
for k, v in out.items():
    print(k, json.dumps(v))
