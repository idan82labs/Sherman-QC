"""Auditor-2 per-line audit for sc_4f8188b4d1a8: fitted outer-apex line, fold extent, gold offsets/coverage,
plus independent fits of candidate bends that have no gold line.  Writes a2_audit.json."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import math

import numpy as np

from a2_common import OUT, get_scan, gold_lines, unit
from a2_fit import extent_from_presence, fit_bend, fold_presence
from a2_frame import from_part, to_part

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics  # frozen matcher metric (read-only use)

xyz, nrm, faces = get_scan()
G = gold_lines()

SEEDS = {
    # name: part-frame seed endpoints (L, W, N)
    "X_endwall_Lmin": ((-4.2, 2.0, 0.8), (-4.2, 43.0, 0.8)),
    "X_endwall_Lmax": ((219.9, 2.0, 0.8), (219.9, 43.0, 0.8)),
    "X_tab30_B10end": ((26.0, 104.5, 45.5), (26.0, 104.5, 57.0)),
    "X_tab30_B09end": ((200.0, 104.5, 45.5), (200.0, 104.5, 57.0)),
}

only = sys.argv[1:]
targets = []
for bid in sorted(G):
    targets.append((bid, G[bid][0], G[bid][1], True))
for name, (pa, pb) in SEEDS.items():
    targets.append((name, from_part(np.array(pa)), from_part(np.array(pb)), False))
if only:
    targets = [t for t in targets if t[0] in only]

report = {}
for name, a, b, is_gold in targets:
    Lg = float(np.linalg.norm(b - a))
    short = Lg < 30
    fb = fit_bend(xyz, nrm, a, b, ext=(6.0 if short else 14.0), step=(0.5 if short else 1.0),
                  rmax=(10.0 if short else 13.0))
    rec = {"name": name, "is_gold": is_gold, "seed_len": Lg, "fit_ok": fb.get("ok")}
    if not fb.get("ok"):
        rec["why"] = fb.get("why")
        report[name] = rec
        print(name, "FIT FAILED", fb.get("why"))
        continue
    c, d = fb["c"], fb["d"]
    sa, sb = (a - c) @ d, (b - c) @ d
    lo, hi = min(sa, sb), max(sa, sb)
    pres = fold_presence(xyz, nrm, c, d, fb["bis"], fb["tA"], fb["tB"], lo - 25, hi + 25,
                         step=0.5, leg_lo=3.0, leg_hi=(5.0 if short else 6.0))
    best, runs = extent_from_presence(pres, min_leg=3)
    rec.update({k: fb[k] for k in ("n_apex", "n_keep", "apex_resid_mm_med", "bend_angle_med", "thick_med",
                                    "apex_to_Xo_med", "apex_to_Xi_med", "apex_to_inner_apex_med")})
    rec["apex_line_point"] = c.tolist()
    rec["apex_line_dir"] = d.tolist()
    rec["presence_runs"] = [(round(float(r0), 2), round(float(r1), 2)) for r0, r1 in runs]
    if best is not None:
        f0, f1 = best
        rec["fold_extent_s"] = (float(f0), float(f1))
        rec["fold_len"] = float(f1 - f0)
        fa, fbp = c + f0 * d, c + f1 * d
        rec["fold_apex_start"] = fa.tolist()
        rec["fold_apex_end"] = fbp.tolist()
        rec["fold_apex_start_part"] = to_part(fa).round(2).tolist()
        rec["fold_apex_end_part"] = to_part(fbp).round(2).tolist()
    if is_gold:
        # gold offsets relative to fitted apex line, decomposed: along bisector (+ = concave side) and lateral
        perpb = np.cross(d, fb["bis"])
        offs = {}
        for tag, f in (("start", 0.0), ("q25", 0.25), ("mid", 0.5), ("q75", 0.75), ("end", 1.0)):
            p = a + f * (b - a)
            r = p - c
            r = r - d * (r @ d)
            offs[tag] = {"dist": float(np.linalg.norm(r)), "bis": float(r @ fb["bis"]), "lat": float(r @ perpb)}
        rec["gold_offsets_from_apex"] = offs
        rec["gold_s_range"] = (float(lo), float(hi))
        ang = math.degrees(math.acos(min(1.0, abs(unit(b - a) @ d))))
        rec["gold_vs_apex_angle_deg"] = ang
        if best is not None:
            ov = max(0.0, min(hi, f1) - max(lo, f0))
            rec["coverage_of_fold"] = ov / (f1 - f0)
            rec["gold_outside_fold_mm"] = (lo - f0 if lo < f0 else 0.0, hi - f1 if hi > f1 else 0.0)
            rec["gold_frac_on_fold"] = ov / (hi - lo)
            gm = np.array([a, b])
            pm = np.array([fa, fbp])
            rec["matcher_vs_fitted_apex"] = pair_metrics(gm, pm)
            # also vs outer virtual sharp line (apex shifted by -apex_to_Xo along bisector)
            if fb.get("apex_to_Xo_med") is not None:
                sh = -fb["apex_to_Xo_med"] * fb["bis"]
                rec["matcher_vs_fitted_outer_sharp"] = pair_metrics(gm, pm + sh)
    report[name] = rec
    o = rec.get("gold_offsets_from_apex")
    print(f"== {name}: angle {rec['bend_angle_med']:.1f}  t {rec['thick_med']}  apexN {rec['n_apex']}/{rec['n_keep']} "
          f"resid {rec['apex_resid_mm_med']:.2f}  apex->Xo {rec['apex_to_Xo_med']}  apex->Xi {rec['apex_to_Xi_med']}")
    print("   presence runs", rec["presence_runs"])
    if best is not None:
        print(f"   fold extent s [{f0:.1f},{f1:.1f}] len {f1 - f0:.1f}  part {rec['fold_apex_start_part']} -> {rec['fold_apex_end_part']}")
    if o:
        print("   gold s-range [%.1f, %.1f] len %.1f angle %.2f deg" % (lo, hi, hi - lo, rec["gold_vs_apex_angle_deg"]))
        print("   offsets (dist/bis/lat):", {k: (round(v['dist'], 2), round(v['bis'], 2), round(v['lat'], 2)) for k, v in o.items()})
        if "coverage_of_fold" in rec:
            print(f"   coverage {rec['coverage_of_fold']*100:.1f}%  gold outside fold {rec['gold_outside_fold_mm']}  gold frac on fold {rec['gold_frac_on_fold']*100:.1f}%")
            mm = rec["matcher_vs_fitted_apex"]
            print("   matcher vs apex: ang %.2f off %.2f govl %.2f povl %.2f" % (mm["axis_angle_deg"], mm["transverse_offset_mm"], mm["gold_span_overlap"], mm["prediction_span_overlap"]))
            if "matcher_vs_fitted_outer_sharp" in rec:
                mm = rec["matcher_vs_fitted_outer_sharp"]
                print("   matcher vs outer sharp: ang %.2f off %.2f govl %.2f povl %.2f" % (mm["axis_angle_deg"], mm["transverse_offset_mm"], mm["gold_span_overlap"], mm["prediction_span_overlap"]))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


outp = OUT / ("a2_audit.json" if not only else f"a2_audit_{'_'.join(only)}.json")
outp.write_text(json.dumps(_clean(report), indent=1))
print(outp)
