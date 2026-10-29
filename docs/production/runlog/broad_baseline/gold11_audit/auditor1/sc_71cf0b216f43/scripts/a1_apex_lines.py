"""Auditor-1 (sc_71cf0b216f43): apex lines, fold extents, gold coverage, matcher metrics.

Uses a1_xsec_measurements.json (per-station apex in section coords) to fit one straight 3D apex
line per fold (robust, interior stations only), then scans the whole fold at 0.5 mm pitch for
outer-arc evidence at that apex line (points within 0.7 mm of the apex locus whose mesh normal is
within 30 deg of the outward bisector) to get the physical fold extent.
Gold coverage = overlap of the gold span (projected on the apex line) with the fold extent.
Matcher metrics (frozen score.py rules: 5 deg / 4 mm midpoint transverse / 60 % span both ways) are
reported against this apex line clipped to the fold extent.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_common import OUT, get, unit  # noqa: E402

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402


def main():
    P, VN, F, G, R, c = get()
    gold = {bid: (s, e) for bid, s, e in G}
    meas = json.load(open(OUT + "/a1_xsec_measurements.json"))
    res = {"note": __doc__, "folds": {}}
    for fid, fd in meas["folds"].items():
        f = fd["def"]
        ax = f["axis"]
        cols = [1, 2] if ax == 0 else [0, 2]
        st = [s for s in fd["stations"] if s["ok"] and s.get("apex_perp_spread", 9) < 0.5]
        s_all = np.array([s["s"] for s in st])
        lo, hi = np.percentile(s_all, [6, 94])
        st_in = [s for s in st if lo <= s["s"] <= hi]
        A3 = []
        for s in st_in:
            p = np.zeros(3)
            p[cols] = s["apex"]
            p[ax] = s["s"]
            A3.append(p)
        A3 = np.array(A3)
        # robust line fit
        keep = np.ones(len(A3), bool)
        for _ in range(4):
            cen = A3[keep].mean(0)
            u, sv, vt = np.linalg.svd(A3[keep] - cen)
            d = vt[0] * np.sign(vt[0][ax])
            r = A3 - cen - np.outer((A3 - cen) @ d, d)
            rn = np.linalg.norm(r, axis=1)
            thr = max(0.35, 3 * np.median(rn[keep]))
            keep = rn < thr
        resid = rn[keep]
        g_in_med = unit(np.median(np.array([s["g_in"] for s in st_in]), axis=0))
        g3 = np.zeros(3)
        g3[cols] = g_in_med  # inward bisector (3D, no component along the fold)
        # fold extent: scan stations at 0.5 mm along the fold axis
        order = np.argsort(P[:, ax])
        keys = P[order, ax]
        ext_s = np.arange(-172.0, 158.0, 0.5) if ax == 0 else np.arange(-106.0, 114.0, 0.5)
        present = []
        for s0 in ext_s:
            a, b = np.searchsorted(keys, [s0 - 0.5, s0 + 0.5])
            idx = order[a:b]
            if len(idx) == 0:
                present.append(0)
                continue
            t = (s0 - cen[ax]) / d[ax]
            apx = cen + t * d
            q = P[idx]
            rel = q - apx
            rel[:, ax] = 0
            near = np.linalg.norm(rel, axis=1) < 0.7
            nrm = VN[idx]
            ok = near & ((nrm @ (-g3)) > math.cos(math.radians(30)))
            present.append(int(ok.sum()))
        present = np.array(present)
        on = present > 0
        # largest run allowing gaps <= 6 mm (holes / scan gaps); report gaps
        runs = []
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                runs.append((ext_s[i], ext_s[j]))
                i = j + 1
            else:
                i += 1
        merged = []
        for r0, r1 in runs:
            if merged and r0 - merged[-1][1] <= 6.0:
                merged[-1] = (merged[-1][0], r1)
            else:
                merged.append((r0, r1))
        main_run = max(merged, key=lambda x: x[1] - x[0])
        gaps = []
        for (a0, a1), (b0, b1) in zip(runs[:-1], runs[1:]):
            if b0 - a1 > 1.0 and a1 >= main_run[0] and b0 <= main_run[1]:
                gaps.append([float(a1), float(b0)])
        f0, f1 = main_run
        fold_len = (f1 - f0) / abs(d[ax])
        # apex line endpoints at the fold extent
        e0 = cen + ((f0 - cen[ax]) / d[ax]) * d
        e1 = cen + ((f1 - cen[ax]) / d[ax]) * d
        Gs, Ge = gold[f["gold"]]
        gs_t = (Gs - cen) @ d
        ge_t = (Ge - cen) @ d
        f0_t, f1_t = (e0 - cen) @ d, (e1 - cen) @ d
        glo, ghi = sorted([gs_t, ge_t])
        ov = max(0.0, min(ghi, f1_t) - max(glo, f0_t))
        coverage = ov / (f1_t - f0_t)
        gold_beyond = [float(max(0.0, f0_t - glo)), float(max(0.0, ghi - f1_t))]
        # gold endpoints offsets to the apex line
        def off(p):
            v = p - cen
            return float(np.linalg.norm(v - (v @ d) * d))
        ang = math.degrees(math.acos(min(1.0, abs(unit(Ge - Gs) @ d))))
        m = pair_metrics(np.array([Gs, Ge]), np.array([e0, e1]))
        res["folds"][fid] = {
            "gold_id": f["gold"], "desc": f["desc"],
            "apex_line_part": [e0.tolist(), e1.tolist()],
            "apex_line_raw": [(e0 @ R + c).tolist(), (e1 @ R + c).tolist()],
            "apex_fit_resid_med_mm": float(np.median(resid)), "apex_fit_resid_max_mm": float(resid.max()),
            "apex_fit_n": int(keep.sum()),
            "fold_extent_axis_coord": [float(f0), float(f1)], "fold_length_mm": float(fold_len), "fold_gaps": gaps,
            "gold_span_axis_coord": [float(min(Gs[ax], Ge[ax])), float(max(Gs[ax], Ge[ax]))],
            "gold_coverage_of_fold": float(coverage), "gold_beyond_fold_mm": gold_beyond,
            "gold_angle_to_apex_deg": ang,
            "gold_start_offset_mm": off(Gs), "gold_end_offset_mm": off(Ge),
            "gold_mid_offset_mm": off(0.5 * (Gs + Ge)),
            "matcher_vs_apex_line": {k: float(v) for k, v in m.items()},
            "matcher_pass_vs_apex_line": bool(is_match(m, Thresholds())),
        }
        r = res["folds"][fid]
        print(f"{fid} {f['gold']}: fold {f0:.1f}..{f1:.1f} len {fold_len:.1f} gaps {gaps} | gold {r['gold_span_axis_coord'][0]:.1f}..{r['gold_span_axis_coord'][1]:.1f} "
              f"cov {coverage*100:.1f}% beyond {gold_beyond[0]:.1f}/{gold_beyond[1]:.1f} | ang {ang:.2f} off s/m/e {r['gold_start_offset_mm']:.2f}/{r['gold_mid_offset_mm']:.2f}/{r['gold_end_offset_mm']:.2f} "
              f"| apex resid med {r['apex_fit_resid_med_mm']:.2f} max {r['apex_fit_resid_max_mm']:.2f} | matcher {m['axis_angle_deg']:.2f}deg {m['transverse_offset_mm']:.2f}mm "
              f"ov {m['gold_span_overlap']:.2f}/{m['prediction_span_overlap']:.2f} pass={r['matcher_pass_vs_apex_line']}")
    with open(OUT + "/a1_apex_lines.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
