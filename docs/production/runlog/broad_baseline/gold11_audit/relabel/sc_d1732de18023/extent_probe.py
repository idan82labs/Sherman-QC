import json, sys
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from research.bench_labeling.geometry import fold_support

ROOT = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/"
ds = np.load(ROOT + "prep/sc_d1732de18023/points_ds.npz")
pts = ds["points"]
L = json.load(open(ROOT + "auditor2/sc_d1732de18023/bl_lines_apex/lines.json"))
A1 = json.load(open(ROOT + "auditor1/sc_d1732de18023/a1_extent_match.json"))
H = json.load(open("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/05df0a708054946a1d77f7ee.json"))
hum = {b["bend_id"]: b for b in H["bends"]}
gmap = {"B1": "M01", "B2": "M02", "B3": "M03", "B4": "M04"}
voxel = 1.0368
for b in L["bends"]:
    o = np.array(b["line_origin"]); d = np.array(b["direction"]); bis = np.array(b["bisector"])
    origin = o + b["apex_offset_mm"] * bis
    sup = fold_support(pts, origin, d, b["side_dirs"]["plane_a"], b["side_dirs"]["plane_b"], -160, 160,
                       reach_mm=1.35 + 1.42 + 1.0, min_points=3, half_window_mm=max(0.5, 0.75 * voxel),
                       arc_band_mm=max(0.5, 0.5 * voxel))
    g = gmap[b["bend_id"]]
    a1 = A1[g]["apex_line_xyz"]
    ta1 = sorted(float((np.array(q) - o) @ d) for q in a1)
    hb = hum[g]
    th = sorted(float((np.array(hb[k]) - o) @ d) for k in ("line_start", "line_end"))
    # perpendicular distance of auditor-1 apex endpoints from this bench apex line
    dist_a1 = []
    for q in a1:
        r = np.array(q) - origin
        dist_a1.append(float(np.linalg.norm(r - (r @ d) * d)))
    print(b["bend_id"], g, "auto t", np.round(b["t_range"], 1), "len", round(b["length_mm"], 1))
    print("   fold_support(-160..160) frac", round(sup["frac"], 3), "gaps", sup["gaps"])
    print("   auditor1 apex t", np.round(ta1, 1), "len", round(ta1[1] - ta1[0], 1), "perp dist of a1 ends to bench apex line", np.round(dist_a1, 3))
    print("   human t", np.round(th, 1), "len", round(th[1] - th[0], 1))
