"""Auditor-3: fold extents (fine end scans), coverage, and frozen-matcher metrics of each gold line
against an independent OUTER-APEX reference line (PCA fit through the measured apex points of
the fold, spanning the measured fold extent)."""
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

LINES = ["B01", "B02", "B03", "B04", "B05", "B06", "B09", "B10"]


def fold_ok(r):
    return bool(r.get("ok")) and min(r.get("leg_len", [0, 0])) > 3.0 and 0.2 < r.get("apex_from_outer_sharp", 0) < 3.0


def main():
    p, n, _ = c.load_scan()
    gold = {g["id"]: g for g in c.load_gold()[0]}
    th = Thresholds()
    out = {}
    for bid in LINES:
        g = gold[bid]
        s0, e0 = g["s"], g["e"]
        L = float(np.linalg.norm(e0 - s0))
        d, e1, e2 = c.basis(e0 - s0)
        (nA, nB), _, _ = X.leg_normals(p, n, s0, d, e1, e2, L)
        rel = p - s0
        t = rel @ d
        u, v = rel @ e1, rel @ e2
        pre = (t > -40) & (t < L + 40) & (u * u + v * v < 17 ** 2)
        idx = np.nonzero(pre)[0]
        tt, uu, vv = t[idx], u[idx], v[idx]
        nn2 = np.stack([n[idx] @ e1, n[idx] @ e2], 1)

        def st(sk, slab=X.SLAB):
            m = (np.abs(tt - sk) < slab) & (np.abs(uu) < X.WIN) & (np.abs(vv) < X.WIN)
            return X.station(np.stack([uu[m], vv[m]], 1), nn2[m], nA, nB)

        # coarse scan every 1 mm, then the longest contiguous fold run (gaps <= 6 mm bridged = reliefs/holes)
        ss = np.arange(-35.0, L + 35.0 + 1e-9, 1.0)
        okv = np.array([fold_ok(st(s)) for s in ss])
        runs, cur = [], None
        for s, ok in zip(ss, okv):
            if ok:
                if cur is None:
                    cur = [s, s]
                elif s - cur[1] <= 6.0:
                    cur[1] = s
                else:
                    runs.append(cur)
                    cur = [s, s]
        if cur:
            runs.append(cur)
        # pick the run overlapping the gold most
        def ov(r):
            return max(0.0, min(r[1], L) - max(r[0], 0.0))
        run = max(runs, key=ov)
        # refine ends with 0.25 mm steps and thinner slab
        lo = run[0]
        for s in np.arange(run[0] - 1.0, run[0] + 1.01, 0.25):
            if fold_ok(st(s, 0.3)):
                lo = s
                break
        hi = run[1]
        for s in np.arange(run[1] + 1.0, run[1] - 1.01, -0.25):
            if fold_ok(st(s, 0.3)):
                hi = s
                break
        gaps = [(float(a), float(b)) for a, b in zip(ss[:-1], ss[1:]) if False]
        miss_inside = [float(s) for s, ok in zip(ss, okv) if lo <= s <= hi and not ok]
        # apex reference line through apex points inside [lo, hi]
        apex = []
        for s in np.arange(lo, hi + 1e-9, 1.0):
            r = st(s)
            if fold_ok(r):
                A2 = np.array(r["A"])
                apex.append(s0 + d * s + e1 * A2[0] + e2 * A2[1])
        apex = np.array(apex)
        cen = apex.mean(0)
        _, _, vt = np.linalg.svd(apex - cen)
        ax = vt[0] if vt[0] @ d > 0 else -vt[0]
        res = np.linalg.norm((apex - cen) - np.outer((apex - cen) @ ax, ax), axis=1)
        a_lo = cen + ax * ((s0 + d * lo - cen) @ ax)
        a_hi = cen + ax * ((s0 + d * hi - cen) @ ax)
        ref = np.array([a_lo, a_hi])
        met = pair_metrics(np.array([s0, e0]), ref)
        fold_len = hi - lo
        overlap = max(0.0, min(hi, L) - max(lo, 0.0))
        out[bid] = {
            "gold_length_mm": L,
            "fold_extent_s_mm": [float(lo), float(hi)],
            "fold_length_mm": float(fold_len),
            "fold_internal_unfit_stations_s": miss_inside,
            "gold_overlap_with_fold_mm": float(overlap),
            "coverage_of_fold_pct": float(100 * overlap / fold_len),
            "gold_outside_fold_mm": float(L - overlap),
            "gold_overshoot_start_mm": float(max(0.0, lo - 0.0)) if lo > 0 else 0.0,
            "gold_short_at_start_mm": float(max(0.0, -lo)) if lo < 0 else 0.0,
            "gold_overshoot_end_mm": float(max(0.0, L - hi)),
            "gold_short_at_end_mm": float(max(0.0, hi - L)),
            "apex_ref_line": [a_lo.tolist(), a_hi.tolist()],
            "apex_ref_fit_rms_mm": float(np.sqrt(np.mean(res ** 2))),
            "apex_ref_n": int(len(apex)),
            "matcher_vs_apex_ref": met,
            "matcher_match": bool(is_match(met, th)),
            "angle_gold_vs_apex_deg": float(np.degrees(np.arccos(min(1.0, abs(ax @ d))))),
        }
        o = out[bid]
        print(f"{bid}: gold L {L:.1f}  fold s[{lo:.2f},{hi:.2f}] len {fold_len:.1f}  cover {o['coverage_of_fold_pct']:.1f}%"
              f"  gold outside fold {o['gold_outside_fold_mm']:.1f} mm  (start ovs {o['gold_overshoot_start_mm']:.1f} short {o['gold_short_at_start_mm']:.1f};"
              f" end ovs {o['gold_overshoot_end_mm']:.1f} short {o['gold_short_at_end_mm']:.1f})  apexfit rms {o['apex_ref_fit_rms_mm']:.2f}"
              f"  matcher: ang {met['axis_angle_deg']:.2f} off {met['transverse_offset_mm']:.2f} gov {met['gold_span_overlap']:.2f}"
              f" pov {met['prediction_span_overlap']:.2f} -> {'MATCH' if o['matcher_match'] else 'NO'}  unfit-in-fold {miss_inside[:12]}")
    (c.OUT / "a3_extent_match.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
