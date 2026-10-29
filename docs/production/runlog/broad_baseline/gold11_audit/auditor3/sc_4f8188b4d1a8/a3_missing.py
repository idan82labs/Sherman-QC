"""Auditor-3: measure the formed folds that have NO gold line (upper-section side flanges, tab-end kinks).

Seeds are rough lines in the auditor's part frame (from the slice renders); each seed is measured
with the same perpendicular-slab outer-apex machinery as the gold audit (a3_xsec.station), giving the
fold extent, bend angle and an outer-apex reference line in RAW scan coordinates. The matcher is then
used to confirm that no gold line matches the measured apex line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_common as c  # noqa: E402
import a3_xsec as X  # noqa: E402

sys.path.insert(0, str(c.ROOT))
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402

# seeds in part frame (X', Y', Z'): start, end
SEEDS = {
    "M1_upper_side_flange_Ypos": ([-125.0, 112.3, 30.6], [-62.0, 112.3, 30.6]),
    "M2_upper_side_flange_Yneg": ([-125.0, -111.8, 30.2], [-62.0, -111.8, 30.2]),
    "M3_tab09_end_kink": ([-13.6, 92.6, -10.0], [-13.6, 92.6, -29.0]),
    "M4_tab10_end_kink": ([-13.6, -81.2, -10.0], [-13.6, -81.2, -29.0]),
}


def measure(p, n, s0, e0, step=1.0, win=None, slab=None):
    L = float(np.linalg.norm(e0 - s0))
    d, e1, e2 = c.basis(e0 - s0)
    (nA, nB), _, peaks = X.leg_normals(p, n, s0, d, e1, e2, L)
    rel = p - s0
    t = rel @ d
    u, v = rel @ e1, rel @ e2
    W = win or X.WIN
    sl = slab or X.SLAB
    pre = (t > -5) & (t < L + 5) & (u * u + v * v < (W * 1.5) ** 2)
    idx = np.nonzero(pre)[0]
    tt, uu, vv = t[idx], u[idx], v[idx]
    nn2 = np.stack([n[idx] @ e1, n[idx] @ e2], 1)
    rows = []
    for sk in np.arange(0, L + 1e-9, step):
        m = (np.abs(tt - sk) < sl) & (np.abs(uu) < W) & (np.abs(vv) < W)
        r = X.station(np.stack([uu[m], vv[m]], 1), nn2[m], nA, nB)
        ok = bool(r.get("ok")) and min(r.get("leg_len", [0, 0])) > 3.0 and 0.05 < r.get("apex_from_outer_sharp", 0) < 3.0
        row = {"s": float(sk), "ok": ok, "why": r.get("why")}
        if r.get("ok"):
            A2 = np.array(r["A"])
            row.update({"apex": (s0 + d * sk + e1 * A2[0] + e2 * A2[1]).tolist(), "bend_deg": r["bend_deg"],
                        "R_o": r.get("R_o"), "skins": r.get("skins_measured"), "leg_len": r.get("leg_len")})
        rows.append(row)
    return rows, L, d, peaks


def longest_run(rows, gap=4.0):
    runs, cur = [], None
    for r in rows:
        if r["ok"]:
            if cur is None or r["s"] - cur[-1]["s"] > gap:
                cur = [r]
                runs.append(cur)
            else:
                cur.append(r)
    return max(runs, key=lambda x: x[-1]["s"] - x[0]["s"]) if runs else []


def main():
    p, n, _ = c.load_scan()
    R = np.load(c.CACHE / "part_R.npy")
    o = np.load(c.CACHE / "part_o.npy")
    gold = c.load_gold()[0]
    th = Thresholds()
    out = {}
    for name, (a, b) in SEEDS.items():
        s0 = o + R.T @ np.array(a)
        e0 = o + R.T @ np.array(b)
        rows, L, d, peaks = measure(p, n, s0, e0)
        run = longest_run(rows)
        if not run:
            print(name, "NO FOLD FOUND")
            out[name] = {"found": False}
            continue
        apex = np.array([r["apex"] for r in run])
        cen = apex.mean(0)
        _, _, vt = np.linalg.svd(apex - cen)
        ax = vt[0] if vt[0] @ d > 0 else -vt[0]
        lo, hi = run[0]["s"], run[-1]["s"]
        a_lo = cen + ax * ((s0 + d * lo - cen) @ ax)
        a_hi = cen + ax * ((s0 + d * hi - cen) @ ax)
        resid = np.linalg.norm((apex - cen) - np.outer((apex - cen) @ ax, ax), axis=1)
        bends = [r["bend_deg"] for r in run]
        best = None
        for gl in gold:
            met = pair_metrics(np.array([gl["s"], gl["e"]]), np.array([a_lo, a_hi]))
            score = met["axis_angle_deg"] / 5 + met["transverse_offset_mm"] / 4
            if best is None or score < best[0]:
                best = (score, gl["id"], met, is_match(met, th))
        pf = lambda x: ((np.asarray(x) - o) @ R.T).round(1).tolist()  # noqa: E731
        out[name] = {
            "found": True,
            "apex_line_raw": [a_lo.tolist(), a_hi.tolist()],
            "apex_line_partframe": [pf(a_lo), pf(a_hi)],
            "fold_length_mm": float(np.linalg.norm(a_hi - a_lo)),
            "n_apex_stations": len(run),
            "apex_fit_rms_mm": float(np.sqrt(np.mean(resid ** 2))),
            "bend_deg_median": float(np.median(bends)),
            "bend_deg_range": [float(np.min(bends)), float(np.max(bends))],
            "R_o_median": float(np.nanmedian([r["R_o"] for r in run])),
            "unfit_stations_inside": [r["s"] for r in rows if lo <= r["s"] <= hi and not r["ok"]],
            "nearest_gold": best[1],
            "nearest_gold_metrics": best[2],
            "nearest_gold_matches": bool(best[3]),
        }
        oo = out[name]
        print(f"{name}: fold {oo['fold_length_mm']:.1f} mm  pf {oo['apex_line_partframe']}  bend {oo['bend_deg_median']:.1f} "
              f"[{oo['bend_deg_range'][0]:.1f},{oo['bend_deg_range'][1]:.1f}]  R_o {oo['R_o_median']:.2f}  rms {oo['apex_fit_rms_mm']:.2f}  "
              f"nearest gold {best[1]} ang {best[2]['axis_angle_deg']:.1f} off {best[2]['transverse_offset_mm']:.1f} -> {'MATCH' if best[3] else 'no match'}"
              f"  unfit {oo['unfit_stations_inside'][:10]}")
    (c.OUT / "a3_missing_bends.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
