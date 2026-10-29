#!/usr/bin/env python3
"""Aggregate the per-scan D1 records: gold locus, error classes, thickness accuracy, A/B line-F1.

Light (JSON only). Every match decision uses the frozen matcher through
research/broad_baseline/score.score_case (5 deg / 4 mm / 0.6 both spans, Hungarian, apex locus,
raw frame) with preloaded predictions; geometric diagnostics use the frozen pair_metrics.

usage: analyze.py --scans d1/scans --out d1
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import copy
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/broad_baseline"))
import score as frozen  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import pair_metrics  # noqa: E402

RL = REPO / "docs/production/runlog"
SCORED_LIST = RL / "gold_set_v1/rescore/case_list_gold_v1__scored.json"
FRAME_RECORDS = RL / "broad_baseline/broad_frame_records.json"
LOCI = ("outer_apex_2d", "outer_sharp_2d", "mid_sharp_2d", "inner_apex_2d", "inner_sharp_2d")


def _pm(a, b):
    return pair_metrics(np.asarray([a["start"], a["end"]]), np.asarray([b["start"], b["end"]]))


def to2d(prof, p3):
    o, e1, e2 = (np.asarray(prof[k]) for k in ("origin", "e1", "e2"))
    d = np.asarray(p3) - o
    return np.array([d @ e1, d @ e2])


# ------------------------------------------------------------------ gold locus

def gold_locus(recs):
    rows = []
    for r in recs:
        for g in (r.get("gold") or {}).get("lines", []):
            p = g["profile"]
            if not p["ok"]:
                rows.append({"case": r["case_id"], "bend": g["bend_id"], "ok": False, "reason": p["reason"]})
                continue
            d = {k: (None if p[k] is None else float(np.hypot(*p[k]))) for k in LOCI}
            b = np.asarray(p["bisector_2d"])
            ao = p["outer_apex_2d"]
            row = {"case": r["case_id"], "bend": g["bend_id"], "ok": True, "theta": p["included_angle_deg"],
                   "t": r["thickness"]["t_mm"], **{f"d_{k[:-3]}": v for k, v in d.items()}}
            # signed position of the gold along the bisector relative to the outer apex (+ = outside the part)
            if ao is not None:
                row["bisector_from_outer_apex_mm"] = float(-(np.asarray(ao) @ b))
            row["bisector_from_outer_sharp_mm"] = float(-(np.asarray(p["outer_sharp_2d"]) @ b))
            avail = {k: v for k, v in d.items() if v is not None}
            row["nearest"] = min(avail, key=avail.get)[:-3]
            rows.append(row)
    ok = [x for x in rows if x["ok"]]
    summ = {"gold_lines": len(rows), "profiled": len(ok), "nearest_locus": dict(Counter(x["nearest"] for x in ok))}
    for k in LOCI:
        v = [x[f"d_{k[:-3]}"] for x in ok if x[f"d_{k[:-3]}"] is not None]
        if v:
            summ[f"median_dist_{k[:-3]}_mm"] = round(statistics.median(v), 3)
            summ[f"frac_within_1mm_{k[:-3]}"] = round(float(np.mean(np.array(v) < 1.0)), 3)
    v = [x["bisector_from_outer_apex_mm"] for x in ok if "bisector_from_outer_apex_mm" in x]
    if v:
        summ["bisector_offset_from_outer_apex_mm"] = {"median": round(statistics.median(v), 3),
                                                     "p10": round(float(np.percentile(v, 10)), 3),
                                                     "p90": round(float(np.percentile(v, 90)), 3)}
    return summ, rows


# ------------------------------------------------------------- error classes

def _match(gold, preds):
    thresholds, match_lines = frozen._matcher()
    g = [{"bend_id": x["bend_id"], "line_start": x["start"], "line_end": x["end"]} for x in gold]
    p = [{"bend_id": x["bend_id"], "line_start": x["start"], "line_end": x["end"]} for x in preds]
    return match_lines(g, p, thresholds)


def _inner_tol(t):
    return max(1.5, 0.5 * t)


def _skin_relation(gline, pline, t):
    """Where the predicted line sits in the gold line's cross-section (needs the gold profile)."""
    prof = gline["profile"]
    if not prof["ok"]:
        return None
    mid = 0.5 * (np.asarray(pline["start"]) + np.asarray(pline["end"]))
    q = to2d(prof, mid)
    b = np.asarray(prof["bisector_2d"])
    perp = np.array([-b[1], b[0]])
    rel = {"along_bisector_mm": float(q @ b), "lateral_mm": float(q @ perp),
           "theta": prof["included_angle_deg"]}
    s = math.sin(math.radians(prof["included_angle_deg"]) / 2)
    rel["t_over_sin_half"] = t / max(s, 1e-6)
    dists = {k[:-3]: float(np.hypot(*(q - np.asarray(prof[k])))) for k in LOCI if prof[k] is not None}
    rel["dist_to_loci_mm"] = dists
    inner = {k: v for k, v in dists.items() if k.startswith(("inner", "mid"))}
    rel["nearest_inner_locus"] = min(inner, key=inner.get) if inner else None
    rel["inner_locus_dist_mm"] = min(inner.values()) if inner else None
    # pure-geometry test of the hypothesis: inside the part, on the bisector, about t..t/sin(theta/2) deep
    rel["on_inner_side"] = bool(q @ b < -0.4 * t and abs(q @ perp) < max(2.0, 0.6 * t)
                                and -(q @ b) < rel["t_over_sin_half"] + _inner_tol(t))
    return rel


def classify_case(r, lane):
    t = r["thickness"]["t_mm"]
    gold = (r.get("gold") or {}).get("lines") or []
    preds = ((r.get("lanes") or {}).get(lane) or {}).get("lines") or []
    m = _match(gold, preds)
    matched = {x["gold_id"]: x["prediction_id"] for x in m["matches"]}
    gl = {g["bend_id"]: g for g in gold}
    pl = {p["bend_id"]: p for p in preds}
    unp = set(m["unmatched_prediction_ids"])
    fn_rows, fp_rows = [], []
    fp_explained = {}
    for gid in m["unmatched_gold_ids"]:
        g = gl[gid]
        best = None
        for pid in unp:
            pm = _pm(g, pl[pid])
            if pm["axis_angle_deg"] <= 15 and pm["transverse_offset_mm"] <= 15:
                key = pm["axis_angle_deg"] / 5 + pm["transverse_offset_mm"] / 4 + (1 - min(pm["gold_span_overlap"], pm["prediction_span_overlap"]))
                if best is None or key < best[0]:
                    best = (key, pid, pm)
        if best is None:
            fn_rows.append({"gold_id": gid, "class": "MISSED_ENTIRELY"})
            continue
        _, pid, pm = best
        row = {"gold_id": gid, "nearest_prediction": pid, **{k: round(pm[k], 3) for k in
               ("axis_angle_deg", "transverse_offset_mm", "gold_span_overlap", "prediction_span_overlap")}}
        if pm["transverse_offset_mm"] <= 4.0:
            row["class"] = "OTHER"
            row["subclass"] = "OFFSET_OK_SPAN_OR_ANGLE_FAIL"
        else:
            rel = _skin_relation(g, pl[pid], t)
            row["skin_relation"] = rel
            parallel = pm["axis_angle_deg"] <= 5
            if rel is not None and parallel and (rel["on_inner_side"] or (rel["inner_locus_dist_mm"] is not None and rel["inner_locus_dist_mm"] < _inner_tol(t))):
                row["class"] = "INNER_SKIN_OFFSET"
                fp_explained[pid] = "INNER_SKIN_OFFSET"
            else:
                row["class"] = "OTHER"
                row["subclass"] = ("OFFSET_GT_4_NOT_SKIN" if parallel else "ANGLE_GT_5_AND_OFFSET")
                if rel is None:
                    row["subclass"] += "_NO_GOLD_PROFILE"
        fn_rows.append(row)
    for pid in sorted(unp):
        p = pl[pid]
        row = {"prediction_id": pid}
        if pid in fp_explained:
            row["class"] = fp_explained[pid]
            fp_rows.append(row)
            continue
        dup = None
        for gid, qid in matched.items():
            pm = _pm(gl[gid], p)
            if pm["axis_angle_deg"] > 5 or max(pm["gold_span_overlap"], pm["prediction_span_overlap"]) < 0.6:
                continue
            rel = _skin_relation(gl[gid], p, t)
            pq = _pm(pl[qid], p)
            sep = pq["transverse_offset_mm"]
            s = None if rel is None else rel["t_over_sin_half"]
            two_skin_sep = (s is not None and 0.5 * t <= sep <= s + _inner_tol(t))
            if (rel is not None and (rel["on_inner_side"] or (rel["inner_locus_dist_mm"] or 99) < _inner_tol(t))) or two_skin_sep:
                dup = {"gold_id": gid, "tp_prediction": qid, "separation_from_tp_mm": round(sep, 3),
                       "t_over_sin_half": None if s is None else round(s, 3)}
                break
        if dup:
            row.update({"class": "DUPLICATE_SKIN_LINE", **dup})
        else:
            near = None
            for gid in gl:
                pm = _pm(gl[gid], p)
                if pm["axis_angle_deg"] <= 15 and pm["transverse_offset_mm"] <= 15:
                    near = gid
            row["class"] = "OTHER"
            row["subclass"] = "NEAR_A_GOLD_LINE" if near else "SPURIOUS"
        fp_rows.append(row)
    return {"case_id": r["case_id"], "tp": len(m["matches"]), "fn": fn_rows, "fp": fp_rows,
            "matches": m["matches"]}


# ------------------------------------------------------------------- variants

def _variant_lines(lines, mode, t, max_shift=8.0):
    out = []
    for x in lines:
        y = {"bend_id": x["bend_id"], "line_start": x["start"], "line_end": x["end"], "origin": x.get("origin")}
        p = x["profile"]
        if mode != "baseline" and p["ok"]:
            tgt = p["outer_apex_2d"] if (mode.startswith("apex") and p["outer_apex_2d"] is not None) else p["outer_sharp_2d"]
            if mode.startswith("apex") and p["outer_apex_2d"] is None:
                tgt = p["outer_sharp_2d"]
            cap = max_shift
            if "gated" in mode:
                # D1 only explains a skin offset: never move a line further than the two-skin separation
                cap = t / max(math.sin(math.radians(p["included_angle_deg"]) / 2), 0.05) + 1.0
            if tgt is not None and float(np.hypot(*tgt)) <= cap:
                d3 = tgt[0] * np.asarray(p["e1"]) + tgt[1] * np.asarray(p["e2"])
                y["line_start"] = (np.asarray(x["start"]) + d3).tolist()
                y["line_end"] = (np.asarray(x["end"]) + d3).tolist()
                y["d1_shift_mm"] = round(float(np.hypot(*tgt)), 4)
        out.append(y)
    if mode.endswith("+dedupe"):
        tol = max(1.5, 0.5 * t)
        keep = [True] * len(out)
        L = [float(np.linalg.norm(np.subtract(y["line_end"], y["line_start"]))) for y in out]
        order = np.argsort(L)[::-1]
        for ii, i in enumerate(order):
            if not keep[i]:
                continue
            for j in order[ii + 1:]:
                if not keep[j]:
                    continue
                pm = pair_metrics(np.asarray([out[i]["line_start"], out[i]["line_end"]]),
                                  np.asarray([out[j]["line_start"], out[j]["line_end"]]))
                if pm["axis_angle_deg"] <= 5 and pm["transverse_offset_mm"] <= tol and pm["shorter_span_overlap"] >= 0.6:
                    keep[j] = False
        out = [y for y, k in zip(out, keep) if k]
    return out


def ab(recs, lane, modes):
    scored = {c["case_id"]: c for c in json.loads(SCORED_LIST.read_text())["cases"]}
    frames = json.loads(FRAME_RECORDS.read_text())
    res = {}
    for mode in modes:
        rows = []
        for r in recs:
            pid = r.get("prediction_case_id")
            if pid is None or pid not in scored or lane not in (r.get("lanes") or {}):
                continue
            c = scored[pid]
            if c.get("zero_bend_control"):
                pass
            lanep = r["lanes"][lane]
            payload = json.loads((REPO / lanep["path"]).read_text())
            payload = copy.deepcopy(payload)
            payload["owned_bend_regions"] = _variant_lines(lanep["lines"], mode, r["thickness"]["t_mm"] or 1.0)
            row = frozen.score_case({"case_id": pid, "tier": c["tier"], "family_id": c["family_id"],
                                     "gold": c["gold"], "label_complete": c.get("label_complete", True)},
                                    frame_records=frames, preloaded_predictions=payload)
            row["in_gold11"] = c.get("in_gold11")
            rows.append(row)
        complete = [x for x in rows if x.get("disposition") == "SCORED_COMPLETE_LABEL"]
        tp = sum(x["true_positive"] for x in complete)
        fp = sum(x["false_positive"] for x in complete)
        fn = sum(x["false_negative"] for x in complete)
        zero = [x for x in rows if x.get("disposition") == "ZERO_BEND_CONTROL"]
        per = {x["case_id"]: x["f1"] for x in complete}
        fam = defaultdict(list)
        for x in complete:
            fam[x["family_id"]].append(x)
        fam_f1 = []
        for xs in fam.values():
            a, b_, c_ = (sum(x[k] for x in xs) for k in ("true_positive", "false_positive", "false_negative"))
            fam_f1.append(frozen.prf(a, b_, c_)["f1"])
        g11 = [x for x in complete if x.get("in_gold11")]
        offs = [m_["transverse_offset_mm"] for x in complete for m_ in x.get("matches") or []]
        res[mode] = {"scans": len(complete),
                     "matched_offset_mm_median": round(statistics.median(offs), 4) if offs else None,
                     "matched_offset_mm_p90": round(float(np.percentile(offs, 90)), 4) if offs else None, **{k: v for k, v in frozen.prf(tp, fp, fn).items() if not k.endswith("_6")},
                     "family_macro_f1": round(float(np.mean(fam_f1)), 6) if fam_f1 else None,
                     "gold11": frozen.prf(*(sum(x[k] for x in g11) for k in ("true_positive", "false_positive", "false_negative")))["f1"],
                     "zero_bend_fp_lines": sum(x["prediction_count"] for x in zero),
                     "per_scan_f1": per}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    recs = [json.loads(p.read_text()) for p in sorted(args.scans.glob("sc_*.json"))]
    recs = [r for r in recs if r.get("status") == "OK"]
    out = {"schema": "sherman.zeroshot.d1.analysis.v1", "scans_loaded": len(recs)}
    # thickness
    th = []
    for r in recs:
        c = r.get("cad_thickness") or {}
        row = {"case": r["case_id"], "family": r["family_id"], "t_scan": r["thickness"]["t_mm"],
               "conf": r["thickness"]["confidence"], "two_sided": r["thickness"]["two_sided_fraction"],
               "t_cad": c.get("t_mm"), "cad_planar_cyl_agree": c.get("planar_cylinder_agree")}
        if row["t_cad"] and row["t_scan"]:
            row["err_mm"] = row["t_scan"] - row["t_cad"]
            row["rel_err"] = row["err_mm"] / row["t_cad"]
        th.append(row)
    e = [x for x in th if "err_mm" in x]
    out["thickness"] = {"scans": len(th), "with_cad": len(e),
                        "abs_err_mm_median": round(statistics.median(abs(x["err_mm"]) for x in e), 4) if e else None,
                        "abs_err_mm_p90": round(float(np.percentile([abs(x["err_mm"]) for x in e], 90)), 4) if e else None,
                        "abs_err_mm_max": round(max(abs(x["err_mm"]) for x in e), 4) if e else None,
                        "signed_err_mm_median": round(statistics.median(x["err_mm"] for x in e), 4) if e else None,
                        "rel_err_median": round(statistics.median(x["rel_err"] for x in e), 4) if e else None,
                        "within_0.2mm": sum(abs(x["err_mm"]) <= 0.2 for x in e),
                        "within_10pct": sum(abs(x["rel_err"]) <= 0.10 for x in e),
                        "per_scan": th}
    gsum, grows = gold_locus(recs)
    out["gold_locus"] = {**gsum, "per_line": grows}
    for lane in ("a0", "best_f1"):
        cases = [classify_case(r, lane) for r in recs if lane in (r.get("lanes") or {}) and (r.get("gold") or {}).get("lines") is not None]
        fnc = Counter(x["class"] for c in cases for x in c["fn"])
        fns = Counter(x.get("subclass", "") for c in cases for x in c["fn"] if x["class"] == "OTHER")
        fpc = Counter(x["class"] for c in cases for x in c["fp"])
        fps = Counter(x.get("subclass", "") for c in cases for x in c["fp"] if x["class"] == "OTHER")
        out[f"errors_{lane}"] = {"scans": len(cases), "tp": sum(c["tp"] for c in cases),
                                 "fn_classes": dict(fnc), "fn_other_subclasses": dict(fns),
                                 "fp_classes": dict(fpc), "fp_other_subclasses": dict(fps), "per_case": cases}
        out[f"ab_{lane}"] = ab(recs, lane, ["baseline", "apex", "sharp", "apex+dedupe", "apex_gated", "apex_gated+dedupe"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "d1_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    brief = {k: v for k, v in out.items() if k not in ("thickness", "gold_locus") and not k.startswith(("errors_", "ab_"))}
    brief["thickness"] = {k: v for k, v in out["thickness"].items() if k != "per_scan"}
    brief["gold_locus"] = gsum
    for lane in ("a0", "best_f1"):
        brief[f"errors_{lane}"] = {k: v for k, v in out[f"errors_{lane}"].items() if k != "per_case"}
        brief[f"ab_{lane}"] = {m: {k: v for k, v in x.items() if k != "per_scan_f1"} for m, x in out[f"ab_{lane}"].items()}
    print(json.dumps(brief, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
