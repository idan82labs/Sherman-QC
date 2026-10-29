"""Fold-support probe beyond each bend's current extent (prep downsampled cloud, read-only).

For every ok bend of a lines.json, runs research.bench_labeling.geometry.fold_support on
[t0 - pad, t1 + pad] around the label line (same reach/window/band as the tool) and prints the
supported intervals, so the real fold ends can be read off. Also prints, per bend, how far the
auditors' measured apex endpoints lie from the label line (perp) and their t positions."""
import json, sys
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from research.bench_labeling.geometry import fold_support

lines_path, prep_dir = sys.argv[1:3]
pad = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
L = json.load(open(lines_path))
M = json.load(open(prep_dir + "/prep_manifest.json"))
voxel = float(M["voxel_mm"])
pts = np.load(prep_dir + "/points_ds.npz")["points"]
ri, t = L["ri_mm"], L["thickness_mm"]
AUD = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/"
a1 = json.load(open(AUD + "auditor1/sc_152f1b1aacb2/a1_measured_apex_lines.json"))
a2 = json.load(open(AUD + "auditor2/sc_152f1b1aacb2/a2_audit.json"))
aud = {f"a1:{k}": (v["line_start_raw"], v["line_end_raw"]) for k, v in a1.items()}
for f in a2["formed_folds_scan_measured"]:
    aud[f"a2:{f['truth_id']}"] = (f["line_start"], f["line_end"])


def supported_intervals(sup, t0, t1):
    n = sup["n_stations"]; step = (t1 - t0) / n
    gaps = sup["gaps"]
    # build supported intervals as complement of gaps
    out, cur = [], t0
    for g in gaps:
        if g[0] > cur + 1e-6:
            out.append([round(cur, 2), round(g[0], 2)])
        cur = g[1]
    if cur < t1 - 1e-6:
        out.append([round(cur, 2), round(t1, 2)])
    return out


for b in L["bends"]:
    if b.get("status") != "ok":
        continue
    o = np.array(b["line_origin"]); d = np.array(b["direction"]); bis = np.array(b["bisector"])
    shift = b["apex_offset_mm"] if b["label_locus"] == "apex" else 0.0
    origin = o + shift * bis
    t0, t1 = b["t_range"]
    sup = fold_support(pts, origin, d, b["side_dirs"]["plane_a"], b["side_dirs"]["plane_b"],
                       t0 - pad, t1 + pad, reach_mm=ri + t + 1.0, min_points=3,
                       half_window_mm=max(0.5, 0.75 * voxel), arc_band_mm=max(0.5, 0.5 * voxel))
    print(f"{b['bend_id']} patches {b['patches']} t_range {np.round(b['t_range'], 2).tolist()} "
          f"auto {None if b['t_range_auto'] is None else np.round(b['t_range_auto'], 2).tolist()} "
          f"apex_off {b['apex_offset_mm']:.3f} skin {b['skin']} conf {b['confidence']}")
    print(f"   supported in [{t0 - pad:.1f},{t1 + pad:.1f}]: {supported_intervals(sup, t0 - pad, t1 + pad)}")
    best = []
    for name, (p, q) in aud.items():
        P, Q = np.array(p), np.array(q)
        ang = np.degrees(np.arccos(min(1.0, abs(float((Q - P) @ d) / np.linalg.norm(Q - P)))))
        if ang > 5:
            continue
        dist = []
        for X in (P, Q):
            r = X - origin
            dist.append(float(np.linalg.norm(r - (r @ d) * d)))
        if max(dist) > 6:
            continue
        ts = sorted(float((X - o) @ d) for X in (P, Q))
        best.append(f"{name}: t {np.round(ts, 2).tolist()} perp {np.round(dist, 2).tolist()} ang {ang:.2f}")
    for line in best:
        print("   ", line)
