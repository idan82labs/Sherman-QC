"""Fine fold-end search (0.5 mm) for tabs/flanges, 3D apex-line fits of formed folds (raw frame),
ear bends from horizontal slices; matcher metrics of each gold line vs the fitted apex line."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
src = open(__import__('pathlib').Path(__file__).parent / "a3x_stations.py").read()
exec(src.split("res = {}")[0])          # reuse station() etc. without re-running the sweep
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics, Thresholds

T = Thresholds()


def to_raw(v):
    v = np.asarray(v, float)
    return O + v[0] * X + v[1] * Y + v[2] * Z


def fine_ends(bid, lo, hi, step=0.5):
    xs = np.arange(lo, hi + 1e-9, step)
    return [(float(x), bool(station(bid, float(x)).get("fold"))) for x in xs]


out = {"frame": F, "folds": {}, "gold_vs_fold": {}}
# fine ends
ends = {
    "tab_A": ("M01", [(-6, 2), (72, 80)]),
    "tab_B": ("M04", [(-6, 2), (72, 80)]),
    "flange_+Y": ("M02", [(-152, -144), (-108, -100), (-64, -56), (154, 162)]),
    "flange_-Y": ("M05", [(-152, -144), (-106, -98), (-64, -56), (154, 162)]),
}
for fold, (bid, wins) in ends.items():
    rep = []
    for lo, hi in wins:
        r = fine_ends(bid, lo, hi)
        on = [x for x, f in r if f]
        rep.append(dict(window=[lo, hi], fold_x=[min(on), max(on)] if on else None, pattern="".join("#" if f else "." for _, f in r)))
    out["folds"].setdefault(fold, {})["fine_ends"] = rep
    print(fold, rep)

# 3D apex line fits from the 2 mm sweep
ST = json.load(open(OUT / "a3x_stations.json"))
fold_src = {"tab_A": "M01", "tab_B": "M04", "flange_+Y": "M02", "flange_-Y": "M05"}
fold_xrange = {"tab_A": (-2.0, 77.0), "tab_B": (-2.0, 77.0), "flange_+Y": (-148.0, 158.0), "flange_-Y": (-146.0, 158.0)}
for fold, bid in fold_src.items():
    rows = [r for r in ST[bid] if r.get("fold") and r.get("apex") is not None]
    x0, x1 = fold_xrange[fold]
    rows = [r for r in rows if x0 - 0.5 <= r["x"] <= x1 + 0.5]
    P = np.array([[r["x"], r["apex"][0], r["apex"][1]] for r in rows])
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    d = vt[0] if vt[0][0] > 0 else -vt[0]
    a = c + d * ((x0 - c[0]) / d[0]); b = c + d * ((x1 - c[0]) / d[0])
    resid = np.linalg.norm((P - c) - np.outer((P - c) @ d, d), axis=1)
    out["folds"][fold].update(dict(apex_line_part=[a.tolist(), b.tolist()], apex_line_raw=[to_raw(a).tolist(), to_raw(b).tolist()],
                                   length_mm=float(np.linalg.norm(b - a)), n_stations=len(rows), apex_resid_med=float(np.median(resid)),
                                   apex_resid_max=float(resid.max()),
                                   bend_angle_med=float(np.median([r["bend_angle_deg"] for r in rows]))))
    print(fold, "apex line part", np.round(a, 2), np.round(b, 2), "resid med/max %.2f/%.2f" % (np.median(resid), resid.max()))

# ears from horizontal slices: knee (mid-surface) per Z, fit vertical-ish line
p_all = p
Lp = np.c_[(p_all - O) @ X, (p_all - O) @ Y, (p_all - O) @ Z]
for ear, (ylo, yhi) in {"ear_A": (66, 92), "ear_B": (-106, -80)}.items():
    knees = []
    angs = []
    for zc in np.arange(3.0, 14.01, 1.0):
        mm = (Lp[:, 0] > -20) & (Lp[:, 0] < 25) & (Lp[:, 1] > ylo) & (Lp[:, 1] < yhi) & (np.abs(Lp[:, 2] - zc) < 0.3)
        S = Lp[mm]
        st = S[(S[:, 0] > 6) & (S[:, 0] < 25)]
        en = S[(S[:, 0] < -3.5) & (S[:, 0] > -8)]
        if len(st) < 10 or len(en) < 10:
            continue
        k1 = np.polyfit(st[:, 0], st[:, 1], 1); k2 = np.polyfit(en[:, 0], en[:, 1], 1)
        xk = (k2[1] - k1[1]) / (k1[0] - k2[0])
        knees.append([xk, float(np.polyval(k1, xk)), zc])
        angs.append(math.degrees(abs(math.atan(k2[0]) - math.atan(k1[0]))))
    K = np.array(knees)
    zlo = float(Lp[(np.abs(Lp[:, 0] - K[:, 0].mean()) < 1.5) & (np.abs(Lp[:, 1] - K[:, 1].mean()) < 3) & (Lp[:, 2] > 1.2), 2].min())
    zhi = float(Lp[(np.abs(Lp[:, 0] - K[:, 0].mean()) < 1.5) & (np.abs(Lp[:, 1] - K[:, 1].mean()) < 3), 2].max())
    a = np.array([K[:, 0].mean(), K[:, 1].mean(), max(zlo, 2.0)]); b = np.array([K[:, 0].mean(), K[:, 1].mean(), zhi])
    out["folds"][ear] = dict(kind="30deg_ear_vertical_axis", angle_med_deg=float(np.median(angs)), angle_range=[float(min(angs)), float(max(angs))],
                             knee_x_med=float(np.median(K[:, 0])), knee_y_med=float(np.median(K[:, 1])), knee_x_spread=float(np.ptp(K[:, 0])),
                             axis_line_part_mid_surface=[a.tolist(), b.tolist()], axis_line_raw_mid_surface=[to_raw(a).tolist(), to_raw(b).tolist()],
                             span_mm=float(zhi - max(zlo, 2.0)))
    print(ear, "angle med %.1f range %.1f-%.1f, knee X %.2f Y %.2f (spread %.2f), Z %.1f..%.1f" % (np.median(angs), min(angs), max(angs), np.median(K[:, 0]), np.median(K[:, 1]), np.ptp(K[:, 0]), max(zlo, 2.0), zhi))

# matcher metrics: each gold line vs fitted apex lines (raw frame) -- frozen thresholds
gold = load_gold()
for bid, ga, gb in gold:
    best = None
    for fold, f in out["folds"].items():
        line = f.get("apex_line_raw") or f.get("axis_line_raw_mid_surface")
        m = pair_metrics(np.array([ga, gb]), np.array(line))
        ok = (m["axis_angle_deg"] <= T.angle_deg and m["transverse_offset_mm"] <= T.transverse_offset_mm and
              m["gold_span_overlap"] >= T.gold_span_overlap and m["prediction_span_overlap"] >= T.prediction_span_overlap)
        m["would_match"] = bool(ok)
        if best is None or (m["axis_angle_deg"] < 10 and m["transverse_offset_mm"] < best[1]["transverse_offset_mm"]):
            best = (fold, m)
    out["gold_vs_fold"][bid] = dict(nearest_fold=best[0], **{k: (float(v) if not isinstance(v, bool) else v) for k, v in best[1].items()})
    m = best[1]
    print(bid, "vs", best[0], "angle %.2f trans %.2f gold_ov %.2f pred_ov %.2f -> match %s" % (m["axis_angle_deg"], m["transverse_offset_mm"], m["gold_span_overlap"], m["prediction_span_overlap"], m["would_match"]))
json.dump(out, open(OUT / "a3x_apexlines.json", "w"), indent=1)
