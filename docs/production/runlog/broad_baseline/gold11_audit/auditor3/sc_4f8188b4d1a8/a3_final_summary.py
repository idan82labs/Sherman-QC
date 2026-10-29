"""Auditor-3 final per-line summary for sc_4f8188b4d1a8 (Gold11 CG1E0BBA2933, shaked_xv/C05).

Consolidates the already-computed evidence (no re-run of the station fits):
  a3_xsec_stations.json  - per-2 mm perpendicular-slab fits (outer apex, sharps, zones)
  a3_extent_match.json   - fold extents (1 mm scan + 0.25 mm refine) + frozen-matcher vs apex line
  a3_missing_bends.json  - measured folds without gold (M1-M4)
and applies the one correction found by the fine X' slab renders (a3_tab09/10_foldstart_x.png):
the B09/B10 90-deg tab folds start where the tab re-attaches to the plate (X' ~ -11.7 / -11.6),
not at the kink relief, so the B09 start is moved from s=-3.5 to s=+3.3 (B10 kept at +3.75).
The apex reference line for B09/B10 is refit over the corrected extent and re-matched with the frozen
matcher (scripts.evaluate_f1_against_manual_gold, as used by research/broad_baseline/score.py).
B07/B08: distance from gold samples to the nearest scan vertex (KD-tree).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_common as c  # noqa: E402

sys.path.insert(0, str(c.ROOT))
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402

FOLD_START_FIX = {"B09": 3.3}   # from a3_tab09_foldstart_x.png (attached from X' -11.7)


def main():
    st = json.loads((c.OUT / "a3_xsec_stations.json").read_text())
    ex = json.loads((c.OUT / "a3_extent_match.json").read_text())
    p, _, _ = c.load_scan()
    tree = cKDTree(p)
    gold = {g["id"]: g for g in c.load_gold()[0]}
    th = Thresholds()
    out = {}
    for bid, g in gold.items():
        L = float(np.linalg.norm(g["e"] - g["s"]))
        d = (g["e"] - g["s"]) / L
        samples = g["s"][None] + np.linspace(0, L, 200)[:, None] * d[None]
        dscan, _ = tree.query(samples)
        row = {"gold_length_mm": L, "gold_to_nearest_scan_mm_median": float(np.median(dscan)),
               "gold_frac_within_1mm_of_scan": float(np.mean(dscan < 1.0))}
        if bid in ex:
            lo, hi = ex[bid]["fold_extent_s_mm"]
            lo = FOLD_START_FIX.get(bid, lo)
            ok = [s for s in st[bid]["stations"] if s.get("ok") and lo <= s["s"] <= hi]
            ok_in = [s for s in ok if 0 <= s["s"] <= L]
            dist = np.array([s["gold_to_apex_mm"] for s in ok_in])
            ab = np.array([s["gold_minus_apex_along_bisector"] for s in ok_in])
            ac = np.array([s["gold_minus_apex_across"] for s in ok_in])
            gis = np.array([s["gold_to_inner_sharp_mm"] for s in ok_in])
            gos = np.array([s["gold_to_outer_sharp_mm"] for s in ok_in])
            zones = {}
            for s in ok_in:
                zones[s["gold_zone"]] = zones.get(s["gold_zone"], 0) + 1
            apex = np.array([s["apex_xyz"] for s in ok])
            cen = apex.mean(0)
            _, _, vt = np.linalg.svd(apex - cen)
            ax = vt[0] if vt[0] @ d > 0 else -vt[0]
            a_lo = cen + ax * ((g["s"] + d * lo - cen) @ ax)
            a_hi = cen + ax * ((g["s"] + d * hi - cen) @ ax)
            met = pair_metrics(np.array([g["s"], g["e"]]), np.array([a_lo, a_hi]))
            overlap = max(0.0, min(hi, L) - max(lo, 0.0))
            row.update({
                "fold_extent_s_mm": [float(lo), float(hi)], "fold_length_mm": float(hi - lo),
                "coverage_of_fold_pct": float(100 * overlap / (hi - lo)),
                "gold_outside_fold_mm": float(L - overlap),
                "gold_start_vs_fold_mm": float(-lo), "gold_end_vs_fold_mm": float(L - hi),
                "n_stations": len(ok_in),
                "gold_to_outer_apex_mm": {"median": float(np.median(dist)), "p5": float(np.percentile(dist, 5)),
                                          "p95": float(np.percentile(dist, 95)), "max": float(dist.max())},
                "along_bisector_mm_median_p5_p95": [float(np.median(ab)), float(np.percentile(ab, 5)), float(np.percentile(ab, 95))],
                "across_mm_median_p5_p95": [float(np.median(ac)), float(np.percentile(ac, 5)), float(np.percentile(ac, 95))],
                "gold_to_inner_sharp_mm_median": float(np.median(gis)),
                "gold_to_outer_sharp_mm_median": float(np.median(gos)),
                "zones": zones,
                "bend_deg_median": float(np.median([s["bend_deg"] for s in ok_in])),
                "R_o_median": float(np.nanmedian([s.get("R_o", np.nan) for s in ok_in])),
                "apex_ref_line_raw": [a_lo.tolist(), a_hi.tolist()],
                "matcher_vs_apex_ref": met, "matcher_match": bool(is_match(met, th)),
            })
        out[bid] = row
        r = row
        if "fold_extent_s_mm" in r:
            m = r["matcher_vs_apex_ref"]
            print(f"{bid}: L {L:.1f} fold [{r['fold_extent_s_mm'][0]:.2f},{r['fold_extent_s_mm'][1]:.2f}] cover {r['coverage_of_fold_pct']:.1f}% "
                  f"outside {r['gold_outside_fold_mm']:.1f} | apex dist med {r['gold_to_outer_apex_mm']['median']:.2f} "
                  f"p95 {r['gold_to_outer_apex_mm']['p95']:.2f} max {r['gold_to_outer_apex_mm']['max']:.2f} | along {r['along_bisector_mm_median_p5_p95'][0]:+.2f} "
                  f"across {r['across_mm_median_p5_p95'][0]:+.2f} | inner sharp {r['gold_to_inner_sharp_mm_median']:.2f} | zones {r['zones']} | "
                  f"matcher ang {m['axis_angle_deg']:.2f} off {m['transverse_offset_mm']:.2f} gov {m['gold_span_overlap']:.2f} pov {m['prediction_span_overlap']:.2f} "
                  f"-> {'MATCH' if r['matcher_match'] else 'NO'} | scan dist med {r['gold_to_nearest_scan_mm_median']:.2f}")
        else:
            print(f"{bid}: L {L:.1f} (no fold) gold->nearest scan median {r['gold_to_nearest_scan_mm_median']:.2f} "
                  f"frac<1mm {r['gold_frac_within_1mm_of_scan']:.2f}")
    (c.OUT / "a3_final_summary.json").write_text(json.dumps(out, indent=1))
    print(c.OUT / "a3_final_summary.json")


if __name__ == "__main__":
    main()
