import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics, Thresholds, is_match
st = json.load(open(W / "fold_stations.json"))
c, e1, e2, e3 = np.load(W / "pca.npy"); R = np.load(W / "R_plate.npy")
B = np.vstack([e1, e2, e3])            # world->local: (p-c) @ B.T ; local->plate: @ R.T
def plate_to_world(q):
    return (np.asarray(q) @ R) @ B + c
gold_world = {b["bend_id"]: (np.array(b["line_start"]), np.array(b["line_end"])) for b in json.load(open("/Users/idantal/ShermanQC/FullProject/output/bend_label_02_10839000/bend_instance_labels.json"))["manual_bends"]}
for side in ("M01", "M02"):
    rows = st[side]
    print(side, "failed stations:", [(r["x"], r.get("why"), r.get("n_plate"), r.get("n_wall")) for r in rows if not r["ok"]])
apex_lines = {}
for side in ("M01", "M02"):
    ok = [r for r in st[side] if r["ok"]]
    A = np.array([r["apex_xyz_plate"] for r in ok])
    # fit 3D line (PCA) through apex points in plate frame
    m = A.mean(0); u, s, vt = np.linalg.svd(A - m); d = vt[0]
    if d[0] < 0: d = -d
    resid = np.linalg.norm((A - m) - np.outer((A - m) @ d, d), axis=1)
    print(side, "apex line resid rms %.3f max %.3f; dir (plate) %s" % (math.sqrt((resid ** 2).mean()), resid.max(), d.round(5)))
    apex_lines[side] = (m, d)
# fold extents in plate x (from wall-point extents measured earlier): M01 wall x -99.97..101.0, M02 wall x -99.4..100.8
ext = {"M01": (-99.97, 101.0), "M02": (-99.4, 100.8)}
res = {}
for side in ("M01", "M02"):
    m, d = apex_lines[side]
    x0, x1 = ext[side]
    p0 = m + d * ((x0 - m[0]) / d[0]); p1 = m + d * ((x1 - m[0]) / d[0])
    w0, w1 = plate_to_world(p0), plate_to_world(p1)
    g = np.array(gold_world[side])
    met = pair_metrics(g, np.array([w0, w1]))
    print(side, "TRUE apex line (fold extent) world", w0.round(2), w1.round(2), "len %.1f" % np.linalg.norm(w1 - w0))
    print("   frozen-matcher metrics gold vs true apex line:", {k: round(v, 3) for k, v in met.items()}, "MATCH" if is_match(met, Thresholds()) else "NO MATCH")
    res[side] = {"apex_line_world": [w0.tolist(), w1.tolist()], "metrics_vs_gold": met, "match": bool(is_match(met, Thresholds()))}
    # same but gold clipped to the fold extent (what an extent-corrected human line would score)
    ga, gb = g
    # gold in plate frame
    def world_to_plate(p): return ((np.asarray(p) - c) @ B.T) @ R.T
    pa, pb = world_to_plate(ga), world_to_plate(gb)
    def at(x): return pa + (pb - pa) * ((x - pa[0]) / (pb[0] - pa[0]))
    gc0, gc1 = at(max(x0, min(pa[0], pb[0]))), at(min(x1, max(pa[0], pb[0])))
    met2 = pair_metrics(np.array([plate_to_world(gc0), plate_to_world(gc1)]), np.array([w0, w1]))
    print("   gold CLIPPED to fold extent vs true apex:", {k: round(v, 3) for k, v in met2.items()}, "MATCH" if is_match(met2, Thresholds()) else "NO MATCH")
    # angle between gold and apex line
    gd = (gb - ga) / np.linalg.norm(gb - ga); ad = (w1 - w0) / np.linalg.norm(w1 - w0)
    print("   gold vs apex axis angle %.3f deg" % math.degrees(math.acos(abs(gd @ ad))))
    # perpendicular distance from gold endpoints/midpoint to the apex line (3D)
    for nm, p in (("gold start", ga), ("gold mid", (ga + gb) / 2), ("gold end", gb)):
        v = p - w0; dist = np.linalg.norm(v - (v @ ad) * ad); along = v @ ad
        print("   %s: perp dist to apex line %.2f mm, along-from-fold-start %.1f mm (fold len %.1f)" % (nm, dist, along, np.linalg.norm(w1 - w0)))
    # coverage
    gx0, gx1 = sorted([pa[0], pb[0]])
    ov = max(0, min(gx1, x1) - max(gx0, x0))
    print("   coverage of fold by gold span: %.1f%% ; gold length beyond fold: %.1f mm (%.1f%% of gold)" % (100 * ov / (x1 - x0), (gx1 - gx0) - ov, 100 * ((gx1 - gx0) - ov) / (gx1 - gx0)))
# engine lines vs true apex for reference
eng = {"CADF_CAD_B1": [[-80.36354444735174, -147.94615528768375, 224.2560882889684], [-279.9344741650192, 64.16806069095682, 376.31476471950134]],
       "F1C_90211c283156a5d8": [[-472.09327682761, -185.26452238749684, 334.22352462088776], [-353.23621143152354, -309.3575022868147, 243.4557761087437]]}
for k, v in eng.items():
    for side in ("M01", "M02"):
        met = pair_metrics(np.array(res[side]["apex_line_world"]), np.array(v))
        if met["axis_angle_deg"] < 5 and met["transverse_offset_mm"] < 20:
            print("engine", k, "vs TRUE apex", side, {kk: round(vv, 3) for kk, vv in met.items()})
        metg = pair_metrics(np.array(gold_world[side]), np.array(v))
        if metg["axis_angle_deg"] < 5 and metg["transverse_offset_mm"] < 20:
            print("engine", k, "vs GOLD", side, {kk: round(vv, 3) for kk, vv in metg.items()}, "MATCH" if is_match(metg, Thresholds()) else "NO MATCH")
json.dump(res, open(W / "apex_lines_match.json", "w"), indent=1, default=float)
