"""Fit straight OUTER-APEX lines (outer-skin arc midline) per bend from the per-station corner
fits (a3_stations.json) and the joggle fits (a3_joggle.json); score every gold line against its
own bend's apex line with the frozen pair_metrics (5 deg / 4 mm / 60 %), and write
a3_apex_lines.json (scan_raw coordinates) + a3_gold_vs_apex.json."""
import sys
sys.dont_write_bytecode = True
import json
import math
import numpy as np
from a3_common import *
from a3_stations import SIDES, GOLD_SIDE
sys.path.insert(0, str(ROOT))
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds  # frozen matcher (read-only import)

C0 = np.load(SCR / "center.npy")
st = json.loads((OUT / "a3_stations.json").read_text())
jg = json.loads((OUT / "a3_joggle.json").read_text())


def uh_to_raw(side, a, u, h):
    sd = SIDES[side]
    perp = sd["ref"] + sd["sgn"] * u
    LSH = np.zeros(3)
    LSH[sd["ax"]] = a; LSH[sd["pe"]] = perp; LSH[2] = h
    return C0 + LSH[0] * L0 + LSH[1] * S0 + LSH[2] * H0


def fit_line3(P, trim=2.5):
    P = np.asarray(P)
    m = np.ones(len(P), bool)
    for _ in range(3):
        c = P[m].mean(0)
        _u, _s, vt = np.linalg.svd(P[m] - c)
        d = vt[0]
        r = np.linalg.norm((P - c) - np.outer((P - c) @ d, d), axis=1)
        s = 1.4826 * np.median(r[m]) + 0.05
        m = r < trim * s
    c = P[m].mean(0)
    _u, _s, vt = np.linalg.svd(P[m] - c)
    d = vt[0]
    t = (P[m] - c) @ d
    r = np.linalg.norm((P[m] - c) - np.outer(t, d), axis=1)
    return c, d, float(t.min()), float(t.max()), float(np.sqrt((r ** 2).mean())), int(m.sum())


def main():
    lines = {}
    for key, rows in st["bends"].items():
        side, corner = key.split("_")
        pts = [uh_to_raw(side, r["a"], *r["apex"]) for r in rows if r.get("ok") and r.get("apex")]
        c, d, t0, t1, rms, n = fit_line3(pts)
        lines[key] = {"start": (c + t0 * d).tolist(), "end": (c + t1 * d).tolist(), "length_mm": t1 - t0,
                      "rms_mm": rms, "n_stations": n, "kind": "wall_bend_90",
                      "bend_deg_median": float(np.median([r["bend_deg"] for r in rows if r.get("ok")])),
                      "R_outer_median": float(np.median([r["R_outer"] for r in rows if r.get("ok")]))}
    # joggle bends: J1 convex = lower skin at lower-fit ua ; J2 convex = upper skin at upper-fit ub
    for side, rows in jg.items():
        for nm, skin, kk, hk in (("J1", "lower", "ua", "h1"), ("J2", "upper", "ub", "h2")):
            pts = [uh_to_raw(side, r["a"], r[skin][kk], r[skin][hk]) for r in rows
                   if skin in r and abs(r[skin]["step"]) > 1.0]
            c, d, t0, t1, rms, n = fit_line3(pts)
            lines[f"{side}_joggle_{nm}"] = {"start": (c + t0 * d).tolist(), "end": (c + t1 * d).tolist(),
                                           "length_mm": t1 - t0, "rms_mm": rms, "n_stations": n,
                                           "kind": "joggle_bend", "convex_skin": skin,
                                           "incline_deg_median": float(np.median([r[skin]["incline_deg"] for r in rows if skin in r])),
                                           "step_mm_median": float(np.median([r[skin]["step"] for r in rows if skin in r]))}
    (OUT / "a3_apex_lines.json").write_text(json.dumps({"frame": "scan_raw (= reference_mesh_raw, identity)",
                                                         "locus": "outer apex (outer-skin arc midline); joggles: breakpoint on convex skin",
                                                         "lines": lines}, indent=1))
    gold = {b: np.array([a, e]) for b, a, e in load_gold()}
    th = Thresholds()
    res = {}
    for bid, (side, corner) in GOLD_SIDE.items():
        key = f"{side}_{corner}"
        ap = np.array([lines[key]["start"], lines[key]["end"]])
        m = pair_metrics(gold[bid], ap)
        m["match"] = bool(is_match(m, th))
        # also best match against ANY apex line (incl joggles)
        allm = {k: pair_metrics(gold[bid], np.array([v["start"], v["end"]])) for k, v in lines.items()}
        best = min(allm, key=lambda k: allm[k]["transverse_offset_mm"] + 10 * (allm[k]["axis_angle_deg"] > 5))
        res[bid] = {"own_bend": key, **{k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()},
                    "nearest_apex_line": best, "nearest_transverse_mm": round(allm[best]["transverse_offset_mm"], 3)}
        print(bid, key, "angle %.2f  transverse %.2f  gold_overlap %.2f  pred_overlap %.2f  match %s | nearest %s %.2f" % (
            m["axis_angle_deg"], m["transverse_offset_mm"], m["gold_span_overlap"], m["prediction_span_overlap"], m["match"], best,
            allm[best]["transverse_offset_mm"]))
    (OUT / "a3_gold_vs_apex.json").write_text(json.dumps(res, indent=1))
    for k, v in lines.items():
        print(k, "len %.1f rms %.2f n %d" % (v["length_mm"], v["rms_mm"], v["n_stations"]), {kk: round(vv, 2) for kk, vv in v.items() if kk.endswith("median")})


if __name__ == "__main__":
    main()
