"""Build docs/production/runlog/broad_baseline/SUMMARY.json for the 2026-10-17 broad baseline.

Read-only over the broad-baseline outputs; no engine code is imported or run.

Inputs (all under docs/production/runlog/broad_baseline/):
  scores_best_f1.json, scores_runtime.json   frozen-matcher scores on case_list_broad_v3
  case_list_broad_v3.json                    the scored case list
  gold11_repro.json                          Gold11 (a)/(b)/(c) reproduction record
  cad_crosscheck.json                        STEP bend-count cross-check of AI gold
  triage/triage_results_24.json              the 24 visual-triage verdict sets
Writes SUMMARY.json and prints the markdown tables used in
docs/production/BROAD_BASELINE_2026-10-17.md.

Usage: .venv/bin/python -B research/broad_baseline/report/build_summary.py [--print-tables]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import triage_map  # noqa: E402
import length_bins  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "docs/production/runlog/broad_baseline"
GOLD11_FAMILIES = {"10839000", "49001000", "38172000", "37361005", "49024000", "40500000"}
DEMO_SOURCE_MESHES = {"sc_9b9fd762f361", "sc_d45186653eed"}

# Scans whose gold the triage found wrong in a way that changes the score (single-agent verdicts).
LABEL_DEFECT_SCANS = {
    "sc_71cf0b216f43": {"tier": "GOLD11", "gold11_id": "CG84C8FA12D2", "wrong": ["M01", "M02", "M05", "M06", "M07", "M08"], "gold": 8,
                         "omitted_real_bends": 0, "note": "6/8 lines off the bend (4 on the flat wall face, slanted, 1.6-16.4 mm off; 2 on the short flange 3.5-5 mm out). Two 35 deg joggles also unlabelled (policy)."},
    "sc_f2475286312b": {"tier": "GOLD11", "gold11_id": "CGEBA9A869AC", "wrong": ["M01", "M03", "M04", "M05", "M06", "M07", "M08"], "gold": 8,
                         "omitted_real_bends": 3, "note": "M01/M03/M07 4.7-5.7 mm off the corner; M05 on the wrong side of a cutout (14.6 mm); M04/M06/M08 are free edges; +PC1 lip and two ear bends missing. M02 matched and is not counted wrong."},
    "sc_cec08dfca9ab": {"tier": "GOLD11", "gold11_id": "CG08506AE97D", "wrong": ["HB1", "HB2", "HB3", "HB4", "HB5", "HB6"], "gold": 6,
                         "omitted_real_bends": 0, "note": "All 6 lines in free air 2.6-3.6 mm (median) from the scan, 4-6 mm inside the concave side; no single transform repairs them."},
    "sc_152f1b1aacb2": {"tier": "GOLD11", "gold11_id": "CG64C412CC73", "wrong": ["M03", "M04", "M05", "M06"], "gold": 7,
                         "omitted_real_bends": 4, "note": "M03-M06 are wall free ends / straight wall, not bends; M01/M02/M07 over-long (engine-seed extents); two 20 mm lips and two 14 mm tabs missing."},
    "sc_2714fb7e0c0b": {"tier": "GOLD11", "gold11_id": "CGF41F49066A", "wrong": ["B06", "B08"], "gold": 10,
                         "omitted_real_bends": 2, "note": "Codex multiview gold. B06/B08 on seam corners (90 deg off); the two 44 mm end-wall bends are missing."},
    "sc_4f8188b4d1a8": {"tier": "GOLD11", "gold11_id": "CG1E0BBA2933", "wrong": ["B07", "B08"], "gold": 10,
                         "omitted_real_bends": 2, "note": "Codex multiview gold. B07/B08 on butt seams; the two end-flange hinges (STEP C1/C9) are missing. The 2 best_f1 TPs on B07/B08 are false lines matching the mislabel."},
    "sc_36e5d6998b9b": {"tier": "GOLD_LINES_HUMAN_NON_GOLD11", "gold11_id": None, "wrong": ["M02", "M04", "M08", "M09", "M10", "M11"], "gold": 12,
                         "omitted_real_bends": 4, "note": "M08-M11 are free edges/flat feet; M02/M04 merge opposite-direction bends ~5 mm off every crease; 4 ear-tab bends missing. M01-M11 are bit-identical to the engine's own zero-shot decomposition output (circular)."},
    "sc_a0305a6cf787": {"tier": "GOLD_LINES_HUMAN_NON_GOLD11", "gold11_id": None, "wrong": ["HB3", "HB4", "HB5"], "gold": 5,
                         "omitted_real_bends": 1, "note": "HB3 4.6 mm, HB4 ~6 mm, HB5 8 mm off the fold; a sixth formed fold (STEP has 6) is unlabelled."},
}

# Per-scan best_f1 counts under corrected gold, as estimated (not rescored) by the triage agents.
# (historical TP, FP, FN) -> (estimated TP, FP, FN)
GOLD11_CORRECTED_ESTIMATE = {
    "CG08506AE97D": {"orig": (0, 4, 6), "est": (3, 1, 3), "src": "triage sc_cec08dfca9ab: HB1/HB3/HB4 moved onto the apex -> 3/1/3 (F1 0.60)"},
    "CG84C8FA12D2": {"orig": (2, 9, 6), "est": (8, 3, 0), "src": "triage sc_71cf0b216f43: corrected positions -> ~8/3/0 (F1 ~0.84)"},
    "CGEBA9A869AC": {"orig": (1, 5, 7), "est": (6, 0, 2), "src": "triage sc_f2475286312b: against 8 corner-fitted bends -> ~6/0/2 (F1 ~0.86)"},
    "CG64C412CC73": {"orig": (3, 3, 4), "est": (3, 3, 4), "src": "triage sc_152f1b1aacb2: M03-M06 removed and the 4 omitted short bends added (none predicted) -> 3/3/4; with removal only it would be 3/3/0"},
    "CGF41F49066A": {"orig": (7, 2, 3), "est": (5, 4, 5), "src": "triage sc_2714fb7e0c0b: corrected gold -> 5/4/5 (F1 0.526)"},
    "CG1E0BBA2933": {"orig": (8, 3, 2), "est": (7, 4, 3), "src": "triage sc_4f8188b4d1a8: corrected gold -> 7/4/3 (F1 0.667)"},
}
RUNTIME_GOLD11_CORRECTED_ESTIMATE = {
    "CG84C8FA12D2": {"orig": (0, 5, 8), "est": (4, 1, 4), "src": "triage sc_71cf0b216f43: CAD_B1/B2/B3/B6 are on the real corners -> 4/1/4"},
}
# A perfect engine (one apex line per real bend, true extent) scored against the CURRENT Gold11 labels.
PERFECT_ENGINE_ON_CURRENT_GOLD11 = {
    "CG84C8FA12D2": {"tp_fp_fn": (2, 6, 6), "why": "only M03/M04 are on bends; 8 real bends (joggles ignored)"},
    "CGEBA9A869AC": {"tp_fp_fn": (1, 7, 7), "why": "only M02 survives; 8 real bends"},
    "CG08506AE97D": {"tp_fp_fn": (0, 6, 6), "why": "every gold line >4 mm from the apex"},
    "CG64C412CC73": {"tp_fp_fn": (2, 5, 5), "why": "7 real bends; M01/M02 match despite over-long gold (overlap 0.70/0.74), M07 fails span (48 vs 113 mm), M03-M06 unmatchable"},
    "CGF41F49066A": {"tp_fp_fn": (8, 2, 2), "why": "B06/B08 unmatchable; 10 real bends"},
    "CG1E0BBA2933": {"tp_fp_fn": (8, 2, 2), "why": "B07/B08 unmatchable; 10 real bends"},
    "UNTRIAGED_5": {"tp_fp_fn": (22, 0, 0), "why": "CG191DEC7F81, CG4BDD1371B6, CG4FAF273093, CG65CD575B1C, CGE4B0549E41 assumed correct (not triaged; optimistic)"},
}

# Mechanism tags per triaged scan (report author's reading of the 24 triage summaries).
MECHANISMS = {
    "BEND_MISS": "a real formed bend gets no line in the lane at all (mostly short: tabs, lips, feet, return flanges; see recall_by_gold_length)",
    "OVER_EXTENSION": "line on the right fold but stretched along a free edge or past the part (fails the 60% span rule; usually 1 FN + 1 FP)",
    "COLLINEAR_MERGE": "one line across two collinear bends separated by a gap, relief, window or slot",
    "SEAM_OR_EDGE_FALSE_LINE": "line on a box-corner butt/open seam, a cut free edge or flat plate (no bend)",
    "CAD_TARGET_EXTRACTION_WRONG": "the lane's own CAD bend list is wrong (phantom rolled/corner bends, collapsed collinear bends, missing targets)",
    "DETECTION_DISCARDED_NO_CAD_SLOT": "the runtime measured/detected a real bend but emitted nothing (unmatched detection or placeholder-only geometry)",
    "REGISTRATION_FAILURE": "runtime worker refused/crashed on alignment, or the shared alignment slid so nothing matched",
    "SELECTION_DROPPED_GOOD_CANDIDATE": "a correct candidate existed (decomposition/pool/CAD-gated) but the typed union or fusion dropped it",
    "NO_CAD_RUNTIME_ABSTAINS": "runtime returned RUNTIME_REQUIRES_CAD (no STEP)",
    "LABEL_ERROR": "gold line(s) not on a physical bend or real bends missing from gold",
}
SCAN_MECHANISMS = {
    "sc_6eba02723d4b": ["BEND_MISS", "COLLINEAR_MERGE", "OVER_EXTENSION", "CAD_TARGET_EXTRACTION_WRONG", "DETECTION_DISCARDED_NO_CAD_SLOT"],
    "sc_71cf0b216f43": ["LABEL_ERROR", "SEAM_OR_EDGE_FALSE_LINE", "DETECTION_DISCARDED_NO_CAD_SLOT"],
    "sc_d45186653eed": ["COLLINEAR_MERGE", "OVER_EXTENSION", "SEAM_OR_EDGE_FALSE_LINE", "CAD_TARGET_EXTRACTION_WRONG"],
    "sc_eaa89c6d1f1f": ["SEAM_OR_EDGE_FALSE_LINE", "CAD_TARGET_EXTRACTION_WRONG", "DETECTION_DISCARDED_NO_CAD_SLOT", "OVER_EXTENSION", "COLLINEAR_MERGE", "BEND_MISS"],
    "sc_36e5d6998b9b": ["LABEL_ERROR", "SELECTION_DROPPED_GOOD_CANDIDATE", "BEND_MISS", "NO_CAD_RUNTIME_ABSTAINS"],
    "sc_9bd69bc6535c": ["BEND_MISS", "OVER_EXTENSION", "NO_CAD_RUNTIME_ABSTAINS"],
    "sc_1059a9192687": ["BEND_MISS", "OVER_EXTENSION", "NO_CAD_RUNTIME_ABSTAINS"],
    "sc_f2475286312b": ["LABEL_ERROR", "BEND_MISS", "OVER_EXTENSION", "DETECTION_DISCARDED_NO_CAD_SLOT"],
    "sc_e080ae094f35": ["CAD_TARGET_EXTRACTION_WRONG", "COLLINEAR_MERGE", "OVER_EXTENSION", "BEND_MISS"],
    "sc_752e2bd54aad": ["CAD_TARGET_EXTRACTION_WRONG", "DETECTION_DISCARDED_NO_CAD_SLOT", "BEND_MISS", "OVER_EXTENSION", "SEAM_OR_EDGE_FALSE_LINE"],
    "sc_777c64c88967": ["BEND_MISS", "OVER_EXTENSION", "CAD_TARGET_EXTRACTION_WRONG", "SEAM_OR_EDGE_FALSE_LINE"],
    "sc_186463746daa": ["COLLINEAR_MERGE", "BEND_MISS", "REGISTRATION_FAILURE"],
    "sc_cec08dfca9ab": ["LABEL_ERROR", "BEND_MISS", "SEAM_OR_EDGE_FALSE_LINE", "REGISTRATION_FAILURE"],
    "sc_2714fb7e0c0b": ["LABEL_ERROR", "BEND_MISS", "SEAM_OR_EDGE_FALSE_LINE", "CAD_TARGET_EXTRACTION_WRONG", "DETECTION_DISCARDED_NO_CAD_SLOT"],
    "sc_4f8188b4d1a8": ["LABEL_ERROR", "BEND_MISS", "SEAM_OR_EDGE_FALSE_LINE", "DETECTION_DISCARDED_NO_CAD_SLOT"],
    "sc_1268c72dcc30": ["SEAM_OR_EDGE_FALSE_LINE", "CAD_TARGET_EXTRACTION_WRONG", "DETECTION_DISCARDED_NO_CAD_SLOT", "SELECTION_DROPPED_GOOD_CANDIDATE", "BEND_MISS"],
    "sc_152f1b1aacb2": ["LABEL_ERROR", "OVER_EXTENSION", "COLLINEAR_MERGE", "REGISTRATION_FAILURE"],
    "sc_c623d2e45bff": ["OVER_EXTENSION", "SEAM_OR_EDGE_FALSE_LINE", "BEND_MISS"],
    "sc_8bb74bda6756": ["OVER_EXTENSION", "COLLINEAR_MERGE", "SELECTION_DROPPED_GOOD_CANDIDATE", "BEND_MISS", "NO_CAD_RUNTIME_ABSTAINS"],
    "sc_8935b8a8e8aa": ["COLLINEAR_MERGE", "SELECTION_DROPPED_GOOD_CANDIDATE", "BEND_MISS", "NO_CAD_RUNTIME_ABSTAINS"],
    "sc_6e5d5a2f60f5": ["SELECTION_DROPPED_GOOD_CANDIDATE", "SEAM_OR_EDGE_FALSE_LINE", "CAD_TARGET_EXTRACTION_WRONG", "REGISTRATION_FAILURE"],
    "sc_8379cebef9e3": ["COLLINEAR_MERGE", "SEAM_OR_EDGE_FALSE_LINE", "SELECTION_DROPPED_GOOD_CANDIDATE", "BEND_MISS", "REGISTRATION_FAILURE"],
    "sc_2233372b8df7": ["BEND_MISS", "COLLINEAR_MERGE", "SEAM_OR_EDGE_FALSE_LINE", "SELECTION_DROPPED_GOOD_CANDIDATE", "DETECTION_DISCARDED_NO_CAD_SLOT"],
    "sc_a0305a6cf787": ["LABEL_ERROR", "BEND_MISS", "COLLINEAR_MERGE", "OVER_EXTENSION", "REGISTRATION_FAILURE", "DETECTION_DISCARDED_NO_CAD_SLOT"],
}

EXAMPLE_IMAGES = {
    "LABEL_ERROR (gold on the wall face, Gold11)": "docs/production/runlog/broad_baseline/overlays/sc_71cf0b216f43/triage_xsections.png",
    "LABEL_ERROR (gold in free air, Gold11 partial)": "docs/production/runlog/broad_baseline/overlays/sc_cec08dfca9ab/triage_zoom_midspan.png",
    "LABEL_ERROR (gold on wall free ends, Gold11 partial)": "docs/production/runlog/broad_baseline/triage/sc_152f1b1aacb2/closeups_M03_M06.png",
    "LABEL_ERROR (gold on butt seam vs real hinge, Gold11 Codex)": "docs/production/runlog/broad_baseline/overlays/sc_4f8188b4d1a8/triage/zoom_endpiece_bend_vs_seam.png",
    "SEAM_OR_EDGE_FALSE_LINE (box corner seams)": "docs/production/runlog/broad_baseline/overlays/sc_1268c72dcc30/triage_corner_closeups.png",
    "SEAM_OR_EDGE_FALSE_LINE (open corner seams, 10839000)": "docs/production/runlog/broad_baseline/overlays/sc_71cf0b216f43/triage_corners.png",
    "COLLINEAR_MERGE + OVER_EXTENSION (395-404 mm lines over leg roots)": "docs/production/runlog/broad_baseline/overlays/sc_6eba02723d4b/triage_long_lines_profile.png",
    "COLLINEAR_MERGE (three tabs merged across relief slots)": "docs/production/runlog/broad_baseline/triage/sc_8935b8a8e8aa/slot_and_endflange_closeups.png",
    "COLLINEAR_MERGE (B7+B8 across an 11 mm relief)": "docs/production/runlog/broad_baseline/overlays/sc_d45186653eed/triage/b7_b8_gap.png",
    "OVER_EXTENSION (plate-to-wall lines 175-191 mm vs ~98 mm bend)": "docs/production/runlog/broad_baseline/triage/sc_c623d2e45bff/plate_wall_bend_zone_extent.png",
    "BEND_MISS (tab/foot bends, 50757000)": "docs/production/runlog/broad_baseline/overlays/sc_777c64c88967/triage/zoom_B8_B9.png",
    "CAD_TARGET_EXTRACTION_WRONG (6 of 12 bends, merged ears)": "docs/production/runlog/broad_baseline/overlays/sc_e080ae094f35/triage_sections.png",
    "BEND_MISS (slot tabs) + seam false lines, Gold11": "docs/production/runlog/broad_baseline/overlays/sc_4f8188b4d1a8/triage/slices_slot_tabs.png",
    "AI gold missing a 15 deg joggle (42288005)": "docs/production/runlog/broad_baseline/triage/sc_2233372b8df7/joggle_section.png",
    "Overlay, 4 views (any scan)": "docs/production/runlog/broad_baseline/overlays/<scan_id>/overlay_4views.png",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def scored(cases):
    return [c for c in cases if c["disposition"] == "SCORED_COMPLETE_LABEL"]


def cohort(cases, pred=lambda c: True) -> dict:
    cs = [c for c in scored(cases) if pred(c)]
    tp = sum(c["true_positive"] for c in cs)
    fp = sum(c["false_positive"] for c in cs)
    fn = sum(c["false_negative"] for c in cs)
    out = prf(tp, fp, fn)
    out.update({"scans": len(cs), "gold_lines": tp + fn,
                "macro_f1": round(float(np.mean([c["f1"] or 0.0 for c in cs])), 4) if cs else None})
    return out


def pct(vals, q):
    return round(float(np.percentile(np.asarray(vals, dtype=float), q)), 3) if vals else None


def resources(cases, pred=lambda c: True) -> dict:
    cs = [c for c in cases if pred(c)]
    wall = [c["run"]["lane_wall_s"] for c in cs if c.get("run") and c["run"].get("lane_wall_s") is not None]
    rss = [c["run"]["peak_rss_gib"] for c in cs if c.get("run") and c["run"].get("peak_rss_gib") is not None]
    return {"runs": len(cs), "wall_s_p50": pct(wall, 50), "wall_s_p95": pct(wall, 95), "wall_s_max": round(max(wall), 3) if wall else None,
            "wall_s_total": round(sum(wall), 1), "peak_rss_gib_p50": pct(rss, 50), "peak_rss_gib_p95": pct(rss, 95),
            "peak_rss_gib_max": round(max(rss), 3) if rss else None, "runs_over_4_gib": sorted(c["case_id"] for c in cs if (c.get("run") or {}).get("peak_rss_gib", 0) > 4.0),
            "runs_over_45_s": sum(1 for w in wall if w > 45.0)}


def runtime_structure(rt_cases) -> dict:
    buckets = defaultdict(lambda: {"scans": 0, "gold_lines_missed": 0, "true_positive": 0, "false_positive": 0, "case_ids": []})
    for c in scored(rt_cases):
        f = c.get("failure") or {}
        reason = f.get("reason") if isinstance(f, dict) else f
        if reason == "RUNTIME_REQUIRES_CAD":
            key = "NO_CAD_RUNTIME_REQUIRES_CAD"
        elif reason == "RUNTIME_WORKER_FAILED":
            we = f.get("worker_error") or ""
            key = "WORKER_EXCEPTION_INVALID_TRANSFORM" if "Invalid raw scan" in we else "WORKER_EXCEPTION_ALIGNMENT_FITNESS"
        elif c["prediction_count"] == 0:
            key = "RAN_ZERO_LINES"
        else:
            key = "RAN_EMITTED_LINES"
        b = buckets[key]
        b["scans"] += 1
        b["gold_lines_missed"] += c["false_negative"]
        b["true_positive"] += c["true_positive"]
        b["false_positive"] += c["false_positive"]
        b["case_ids"].append(c["case_id"])
    return dict(buckets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-tables", action="store_true")
    args = ap.parse_args()

    S = {lane: json.loads((OUT / f"scores_{lane}.json").read_text()) for lane in ("best_f1", "runtime")}
    C = {lane: S[lane]["cases"] for lane in S}
    case_list = json.loads((OUT / "case_list_broad_v3.json").read_text())
    repro = json.loads((OUT / "gold11_repro.json").read_text())
    cadx = json.loads((OUT / "cad_crosscheck.json").read_text())
    tri = json.loads((OUT / "triage/triage_results_24.json").read_text())

    inputs = {p: sha256(OUT / p) for p in ("scores_best_f1.json", "scores_runtime.json", "case_list_broad_v3.json", "gold11_repro.json",
                                           "cad_crosscheck.json", "triage/triage_results_24.json", "BROAD_BASELINE_SUMMARY.json")}

    # ---------- cohorts ----------
    preds = {
        "ALL": lambda c: True,
        "GOLD11": lambda c: c["tier"] == "GOLD11",
        "NON_GOLD11": lambda c: c["tier"] != "GOLD11",
        "NEW_FAMILIES_NOT_IN_GOLD11": lambda c: c["family_id"] not in GOLD11_FAMILIES,
        "AI_GOLD": lambda c: c["tier"] == "AI_GOLD_ACCEPTED",
        "HUMAN_GOLD_NON_GOLD11": lambda c: c["tier"] == "GOLD_LINES_HUMAN_NON_GOLD11",
        "FULL_SCANS": lambda c: not c["staged"],
        "STAGED_SCANS": lambda c: c["staged"],
        "CAD_AVAILABLE": lambda c: c["has_cad"],
        "NO_CAD": lambda c: not c["has_cad"],
        "AI_GOLD_CAD_AVAILABLE": lambda c: c["tier"] == "AI_GOLD_ACCEPTED" and c["has_cad"],
    }
    cohorts = {lane: {k: cohort(C[lane], p) for k, p in preds.items()} for lane in C}
    for lane in C:  # carry the scorer's own aggregates for ALL
        a = S[lane]["cohorts"]["ALL"]
        cohorts[lane]["ALL"].update({k: a[k] for k in ("macro_precision", "macro_recall", "macro_f1", "family_macro_f1", "exact_scans",
                                                          "scans_with_f1_ge_0_8", "scans_with_zero_tp", "scans_with_no_predictions",
                                                          "angle_mae_deg_matched", "position_error_mean_mm_matched",
                                                          "position_error_median_mm_matched", "fn_cause_counts_diagnostic",
                                                          "fp_cause_counts_diagnostic", "zero_bend_controls")})
        for k in ("GOLD11", "NON_GOLD11", "FULL_SCANS", "STAGED_SCANS", "AI_GOLD", "CAD_AVAILABLE", "NO_CAD"):
            a = S[lane]["cohorts"].get(k if k != "AI_GOLD" else "AI_GOLD")
            if a:
                cohorts[lane][k]["angle_mae_deg_matched"] = a.get("angle_mae_deg_matched")
                cohorts[lane][k]["position_error_median_mm_matched"] = a.get("position_error_median_mm_matched")
    # consistency check against the scorer
    for lane in C:
        a = S[lane]["cohorts"]["ALL"]
        assert (cohorts[lane]["ALL"]["true_positive"], cohorts[lane]["ALL"]["false_positive"], cohorts[lane]["ALL"]["false_negative"]) == \
               (a["true_positive"], a["false_positive"], a["false_negative"]), lane

    # ---------- scan-level bootstrap CIs (10k resamples, fixed seed) ----------
    def boot(pred, reps=10000, seed=20261017):
        rng = np.random.default_rng(seed)
        ids = [c["case_id"] for c in scored(C["best_f1"]) if pred(c)]
        rows = {lane: {c["case_id"]: (c["true_positive"], c["false_positive"], c["false_negative"]) for c in scored(C[lane])} for lane in C}
        arr = {lane: np.array([rows[lane][i] for i in ids], dtype=float) for lane in C}
        out = {}
        idx = rng.integers(0, len(ids), size=(reps, len(ids)))
        f = {}
        for lane in C:
            t = arr[lane][idx].sum(axis=1)
            den = 2 * t[:, 0] + t[:, 1] + t[:, 2]
            f[lane] = np.where(den > 0, 2 * t[:, 0] / np.maximum(den, 1e-9), 0.0)
            out[lane] = {"f1_ci95": [round(float(np.percentile(f[lane], 2.5)), 3), round(float(np.percentile(f[lane], 97.5)), 3)]}
        d = f["best_f1"] - f["runtime"]
        out["best_minus_runtime_ci95"] = [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]
        out["scans"] = len(ids)
        return out
    ci = {k: boot(preds[k]) for k in ("ALL", "GOLD11", "NON_GOLD11", "AI_GOLD", "CAD_AVAILABLE", "FULL_SCANS", "STAGED_SCANS")}

    # ---------- sensitivities ----------
    stage_fix = {"sc_eaa89c6d1f1f": False, "sc_927a2fa13067": True}  # CAD cross-check: real stage
    def staged_fixed(c):
        return stage_fix.get(c["case_id"], c["staged"])
    sens = {}
    for lane in C:
        sens[lane] = {
            "excl_demo_source_meshes": cohort(C[lane], lambda c: c["case_id"] not in DEMO_SOURCE_MESHES),
            "excl_8_label_defect_scans": cohort(C[lane], lambda c: c["case_id"] not in LABEL_DEFECT_SCANS),
            "gold11_excl_6_label_defect_scans": cohort(C[lane], lambda c: c["tier"] == "GOLD11" and c["case_id"] not in LABEL_DEFECT_SCANS),
            "full_scans_stage_tags_fixed": cohort(C[lane], lambda c: not staged_fixed(c)),
            "staged_scans_stage_tags_fixed": cohort(C[lane], lambda c: staged_fixed(c)),
        }

    # ---------- label-corrected estimates ----------
    g11_rows = {r["case_id"]: r for r in repro["per_scan_table"]}
    def est_sum(table, lane_key):
        tp = fp = fn = 0
        for cid, r in g11_rows.items():
            t = table.get(cid)
            if t:
                a, b, d = t["est"]
            else:
                h = r[lane_key]
                a, b, d = h["tp"], h["fp"], h["fn"]
            tp, fp, fn = tp + a, fp + b, fn + d
        return prf(tp, fp, fn)
    est_best = est_sum(GOLD11_CORRECTED_ESTIMATE, "a_historical")
    est_rt = est_sum(RUNTIME_GOLD11_CORRECTED_ESTIMATE, "c_active_runtime")
    pe = [v["tp_fp_fn"] for v in PERFECT_ENGINE_ON_CURRENT_GOLD11.values()]
    perfect = prf(sum(x[0] for x in pe), sum(x[1] for x in pe), sum(x[2] for x in pe))

    # ---------- family table ----------
    fam = {}
    for f in sorted(S["best_f1"]["by_family"]):
        b, r = S["best_f1"]["by_family"][f], S["runtime"]["by_family"][f]
        any_case = next(c for c in C["best_f1"] if c["family_id"] == f)
        fam[f] = {"scans": b["scans"], "gold_lines": b["gold_lines"], "in_gold11_families": f in GOLD11_FAMILIES,
                  "has_cad": any(c["has_cad"] for c in C["best_f1"] if c["family_id"] == f),
                  "best_f1": {k: b[k] for k in ("true_positive", "false_positive", "false_negative", "micro_f1")},
                  "runtime": {k: r[k] for k in ("true_positive", "false_positive", "false_negative", "micro_f1")}}

    # ---------- resources ----------
    res = {"best_f1": {"all_60_runs": resources(C["best_f1"]),
                       "with_cad": resources(C["best_f1"], lambda c: c["has_cad"]),
                       "no_cad_a0_only": resources(C["best_f1"], lambda c: not c["has_cad"]),
                       "timeouts": S["best_f1"]["resources"]["timeouts"]},
           "runtime": {"all_60_runs": resources(C["runtime"]),
                       "worker_executed_cad_scans": resources(C["runtime"], lambda c: c["has_cad"]),
                       "note": "the 21 no-CAD runs return RUNTIME_REQUIRES_CAD in <0.11 s; use worker_executed_cad_scans for speed"}}

    # ---------- runtime structure ----------
    rstruct = runtime_structure(C["runtime"])
    bstruct = Counter(c["lane_status"] for c in scored(C["best_f1"]))

    # ---------- triage taxonomy ----------
    mapped, unmapped = triage_map.map_all()
    tier_of = {c["case_id"]: c["tier"] for c in C["best_f1"]}
    tax = {}
    for lane in ("best_f1", "runtime"):
        recs = [m for m in mapped if m["lane"] == lane]
        byk = {k: dict(Counter(m["cause"] for m in recs if m["kind"] == k)) for k in ("FN", "FP")}
        bytier = defaultdict(lambda: Counter())
        for m in recs:
            bytier[tier_of[m["scan_id"]]][m["kind"] + ":" + m["cause"]] += 1
        tax[lane] = {"errors": len(recs), "fn": sum(byk["FN"].values()), "fp": sum(byk["FP"].values()),
                     "fn_by_cause": byk["FN"], "fp_by_cause": byk["FP"],
                     "by_tier": {t: dict(v) for t, v in bytier.items()}}
    # runtime triage causes split by the scan's structural outcome
    bucket_of = {cid: k for k, v in rstruct.items() for cid in v["case_ids"]}
    rt_by_bucket = defaultdict(Counter)
    for m in mapped:
        if m["lane"] == "runtime":
            rt_by_bucket[bucket_of.get(m["scan_id"], "?")][m["kind"] + ":" + m["cause"]] += 1
    tax["runtime"]["by_structural_outcome"] = {k: dict(v) for k, v in rt_by_bucket.items()}
    tax["runtime"]["triaged_scans_by_structural_outcome"] = dict(Counter(bucket_of.get(s) for s in {m["scan_id"] for m in mapped}))
    mech_counts = Counter(m for v in SCAN_MECHANISMS.values() for m in v)
    triaged = [r["scan_id"] for r in tri["results"]]
    assert set(triaged) == set(SCAN_MECHANISMS), set(triaged) ^ set(SCAN_MECHANISMS)
    triage_cov = {}
    for lane in C:
        cs = {c["case_id"]: c for c in scored(C[lane])}
        fn_all = sum(c["false_negative"] for c in cs.values())
        fp_all = sum(c["false_positive"] for c in cs.values())
        fn_t = sum(cs[s]["false_negative"] for s in triaged if s in cs)
        fp_t = sum(cs[s]["false_positive"] for s in triaged if s in cs)
        triage_cov[lane] = {"fn_triaged": fn_t, "fn_total": fn_all, "fp_triaged": fp_t, "fp_total": fp_all}
    tier_triaged = Counter(tier_of[s] for s in triaged)
    gold_lines_triaged = Counter()
    for c in scored(C["best_f1"]):
        if c["case_id"] in triaged:
            gold_lines_triaged[c["tier"]] += c["gold_count"]

    # ---------- recall by gold length, best-lane source split ----------
    glens = length_bins.gold_lengths()
    rbl = length_bins.recall_by_length(glens)
    rbl["best_f1_missed_line_diagnostic_by_length"] = length_bins.fn_causes_by_length(glens, "best_f1")
    rbl["note"] = "diagnostic causes are the scorer's nearest-counterpart classification, not the metric"
    src = {"A0_scan_only": {"tp": 0, "fp": 0}, "CADF_cad_guided_rescue": {"tp": 0, "fp": 0}}
    for c in scored(C["best_f1"]):
        for m in c.get("matches") or []:
            src["CADF_cad_guided_rescue" if m["prediction_id"].startswith("CADF") else "A0_scan_only"]["tp"] += 1
        for p in c.get("unmatched_prediction_ids") or []:
            src["CADF_cad_guided_rescue" if p.startswith("CADF") else "A0_scan_only"]["fp"] += 1
    for v in src.values():
        v["precision"] = round(v["tp"] / (v["tp"] + v["fp"]), 3)
    a0 = src["A0_scan_only"]
    src["A0_lines_alone_f1"] = round(2 * a0["tp"] / (2 * a0["tp"] + a0["fp"] + (cohorts["best_f1"]["ALL"]["gold_lines"] - a0["tp"])), 4)

    # ---------- runtime diagnostic variants rescored on v3 (report_rescore_v3/) ----------
    variants = {}
    for name in ("runtime", "runtime_gated", "runtime_with_placeholders"):
        p = OUT / "report_rescore_v3" / f"scores_runtime__{name}.json"
        if p.is_file():
            a = json.loads(p.read_text())["cohorts"]["ALL"]
            variants[name] = {"tp_fp_fn": [a["true_positive"], a["false_positive"], a["false_negative"]],
                              "precision": a["micro_precision"], "recall": a["micro_recall"], "f1": round(a["micro_f1"], 4),
                              "file": str(p.relative_to(ROOT)), "sha256": sha256(p)}
    variants["note"] = ("runtime_gated = shipped runtime lines cropped by the research-side raw-scan support gate (not shipped); "
                        "runtime_with_placeholders = runtime lines plus nominal CAD axes (not scan detections). Rescored on v3 by "
                        "research/broad_baseline/report/rescore_runtime_variants_v3.py; the older scores_runtime__*.json files are on the v2 list.")

    # ---------- Gold11 per-scan table ----------
    triage_by_scan = {r["scan_id"]: r for r in tri["results"]}
    g11_table = []
    rt_by_id = {c["case_id"]: c for c in C["runtime"]}
    for c in sorted((c for c in C["best_f1"] if c["tier"] == "GOLD11"), key=lambda c: c["gold11_id"]):
        rc = rt_by_id[c["case_id"]]
        f = rc.get("failure") or {}
        g11_table.append({"gold11_id": c["gold11_id"], "case_id": c["case_id"], "family_id": c["family_id"], "scan": c["scan_name"],
                          "stage_hints": c["stage_hints"], "gold": c["gold_count"],
                          "best_f1": [c["true_positive"], c["false_positive"], c["false_negative"]], "best_f1_f1": round(c["f1"], 3),
                          "runtime": [rc["true_positive"], rc["false_positive"], rc["false_negative"]], "runtime_f1": round(rc["f1"], 3),
                          "runtime_status": rc["lane_status"], "runtime_error": f.get("worker_error") if isinstance(f, dict) else None,
                          "triaged": c["case_id"] in triage_by_scan,
                          "label_verdict": (LABEL_DEFECT_SCANS.get(c["case_id"]) or {}).get("note") if c["case_id"] in triage_by_scan else "not triaged"})

    # ---------- CAD cross-check ----------
    cs_sum = cadx["summary"]
    cad_out = {"parts_with_step": cs_sum.get("parts_with_step"), "ai_accepted_full_scans_on_step_parts": cs_sum.get("ai_accepted_full_scans"),
               "full_scans_flagged_ai_ne_v10": cs_sum.get("full_scans_flagged_ai_ne_v10"),
               "verdicts": {
                   "sc_2233372b8df7": "AI ERROR: 42288005 FULL misses two formed 15 deg joggle bends (scan 14.1/15.1 deg); 6 labelled vs 8 CAD (counts only if joggles are gold)",
                   "sc_927a2fa13067": "STAGE TAG WRONG: 15651004 REPEAT_3RD_SCAN is staged (2nd flange unbent, 0.2 deg); AI count 2 correct",
                   "sc_38091ac91d9d": "V10 WRONG: rejected a real 27 deg bend; AI 2 correct",
                   "sc_d45186653eed": "convention: 3 collinear CAD segments = 1 AI line; AI correct",
                   "sc_777c64c88967": "convention: collinear CAD groups = 1 AI line each; AI correct",
                   "sc_752e2bd54aad": "convention as above; AI correct; DEFECTIVE deviations visible",
                   "sc_eaa89c6d1f1f": "STAGE TAG WRONG: 40500000 PARTIAL_SCAN_1 has all 14 CAD segments formed (really FULL); AI 9 = 9 CAD bend lines",
                   "sc_3bde9996f14d": "INCONCLUSIVE zero-bend label (28870000 SCAN_1): rolled R30 next to the end tab; both lanes emit 3 lines here",
               },
               "matching_full_scans": 17,
               "v10_undercounts": {"28870000": "1 vs 2", "49001000": "3 vs 5", "37126000": "0 vs 2"},
               "open_policy_question": cadx.get("open_convention_question")}

    summary = {
        "schema": "sherman.broad_baseline.report_summary.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "report": "docs/production/BROAD_BASELINE_2026-10-17.md",
        "ledger": "CONSUMED DEVELOPMENT scores (Gold11 + BENCHMARK_V2 CONSUMED with AI gold). Not unseen-family accuracy, not promotion evidence.",
        "inputs_sha256": inputs,
        "case_list": {"path": "docs/production/runlog/broad_baseline/case_list_broad_v3.json", "case_list_sha256": case_list["case_list_sha256"],
                      "cases": case_list["summary"]["cases"], "scored_line_cases": len(scored(C["best_f1"])),
                      "zero_bend_controls": case_list["summary"]["zero_bend_controls"], "gold_lines": case_list["summary"]["gold_lines_total"],
                      "by_tier": case_list["summary"]["by_tier"], "benchmark_v2_sha256": case_list["benchmark"]["sha256"]},
        "matcher": S["best_f1"]["matcher"],
        "gate_g2a": {"line_f1_min": 0.8, "precision_min": 0.9, "source": "docs/production/PRODUCTION_LINE_PLAN.md section 5 G2(a)"},
        "verdict": {
            "gold11_green": False,
            "gold11_best_lane_f1": 0.4580152672, "gold11_best_lane_precision": 0.5,
            "gold11_runtime_f1": 0.0740740741, "gold11_runtime_precision": 0.3,
            "broad_best_lane_f1": cohorts["best_f1"]["ALL"]["f1"], "broad_runtime_f1": cohorts["runtime"]["ALL"]["f1"],
            "one_line": "Not green. Best lane 0.458 on Gold11 (reproduced exactly, fresh), 0.445 on the 57-scan broad set; the shipped runtime 0.074 / 0.114. Gate is F1>=0.80 and P>=0.90.",
        },
        "gold11_repro": {
            "a_historical_rescored": {"tp_fp_fn": [30, 30, 41], "precision": 0.5, "recall": 0.4225, "f1": 0.4580152672, "reproduced_exactly": True},
            "b_fresh_best_lane": {"tp_fp_fn": [30, 30, 41], "f1": 0.4580152672, "per_scan_identical": "11/11",
                                  "component_difference": "C001 (CG84C8FA12D2) A0: 4 lines/0 TP fresh vs 5/1 historical; historical A0 came from a probe pool that broke the 200k-point contract; union unchanged because CAD B5 covers M04",
                                  "fresh_a0_alone": {"tp_fp_fn": [18, 14, 53], "f1": 0.3495}, "historical_a0_alone": {"tp_fp_fn": [19, 14, 52], "f1": 0.3654},
                                  "cad_gated_alone": {"tp_fp_fn": [20, 24, 51], "f1": 0.3478},
                                  "compute_s_per_scan": [74, 460], "peak_rss_gib": 1.64},
            "c_runtime": {"tp_fp_fn": [3, 7, 68], "precision": 0.3, "recall": 0.0423, "f1": 0.0740740741,
                          "with_placeholders_f1": 0.154, "repeat_pass_identical": "11/11", "peak_rss_gib": 10.58,
                          "causes": {"registration_refused_scans": 3, "registration_refused_gold_lines": 16,
                                     "no_scan_fitted_lines_scans": 4, "no_scan_fitted_lines_gold_lines": 35,
                                     "emitted_10_lines_3_correct_on_4_scans": True,
                                     "strict_spatial_matching_probe": "shared-registration path forces strict_spatial_matching + position_mode=enabled (40 mm midpoint gate); partial views put midpoints 68-73 mm apart: 38172000_V keeps 1/8 detections (8 under defaults), 38172000_X 1/5, 40500000 0/1"}},
            "gap_to_gate": {"f1": round(0.8 - 0.4580152672, 3), "precision": 0.4},
            "record": "docs/production/runlog/broad_baseline/gold11_repro.json",
        },
        "broad": {
            "cohorts": cohorts,
            "sensitivity": sens,
            "bootstrap_ci95_scan_resampling": ci,
            "best_f1_lane_status_counts": dict(bstruct),
            "runtime_outcome_structure": rstruct,
            "family_table": fam,
            "resources": res,
            "worst_15_by_best_f1": [{"case_id": w["case_id"], "family_id": w["family_id"], "tier": w["tier"], "staged": w["staged"], "has_cad": w["has_cad"],
                                     "gold": w["gold_count"], "best_f1": w["f1"]} for w in S["best_f1"]["worst_15"]],
            "zero_bend_controls": {"best_f1": S["best_f1"]["cohorts"]["ALL"]["zero_bend_controls"], "runtime": S["runtime"]["cohorts"]["ALL"]["zero_bend_controls"]},
            "recall_by_gold_length": rbl,
            "runtime_variants_v3": variants,
            "gold11_per_scan": g11_table,
            "best_f1_prediction_source_split": src,
            "stale_files_not_used": ["scores_runtime__runtime_gated.json", "scores_runtime__runtime_with_placeholders.json (67-case v2 list)"],
        },
        "cad_crosscheck": cad_out,
        "triage": {
            "results_file": "docs/production/runlog/broad_baseline/triage/triage_results_24.json",
            "scans": len(triaged), "scans_by_tier": dict(tier_triaged), "gold_lines_on_triaged_scans_by_tier": dict(gold_lines_triaged),
            "selection_bias": "the 24 scans with the most FN+FP across both lanes; not a random sample",
            "coverage_of_scored_errors": triage_cov,
            "verdict_quality": "one triage agent per scan, cross-sections from raw points; no second blind verifier",
            "taxonomy_mapped_to_scored_errors": tax,
            "unmapped_scored_errors": len(unmapped),
            "mechanisms": MECHANISMS,
            "mechanism_scan_counts": dict(mech_counts),
            "scan_mechanisms": SCAN_MECHANISMS,
            "example_images": EXAMPLE_IMAGES,
        },
        "label_quality": {
            "label_defect_scans": LABEL_DEFECT_SCANS,
            "gold11": {"triaged": 6, "with_outcome_changing_label_errors": 6,
                       "lines_judged_wrong": sum(len(v["wrong"]) for v in LABEL_DEFECT_SCANS.values() if v["tier"] == "GOLD11"),
                       "lines_on_those_scans": sum(v["gold"] for v in LABEL_DEFECT_SCANS.values() if v["tier"] == "GOLD11"),
                       "real_bends_missing_from_gold": sum(v["omitted_real_bends"] for v in LABEL_DEFECT_SCANS.values() if v["tier"] == "GOLD11"),
                       "untriaged": ["CG191DEC7F81", "CG4BDD1371B6", "CG4FAF273093", "CG65CD575B1C", "CGE4B0549E41"]},
            "human_non_gold11": {"triaged": 2, "with_outcome_changing_label_errors": 2,
                                 "untriaged": ["sc_fee685eee4db", "sc_8dfdab44605d"]},
            "ai_gold": {"triaged": 16, "gold_lines_on_triaged": gold_lines_triaged.get("AI_GOLD_ACCEPTED"), "with_outcome_changing_label_errors": 0,
                        "minor_defects": {
                            "sc_2233372b8df7": "two 15 deg joggle folds (~43-45 mm) unlabelled; no prediction there, so no count changes; label if joggles are gold (CAD cross-check counts it as an AI error)",
                            "sc_752e2bd54aad": "B2 truncated at 284 mm, real ~394 mm (score unchanged: overlap 0.72)",
                            "sc_c623d2e45bff": "B2 ~17 mm short (score unchanged)",
                            "sc_186463746daa": "labeler notes swap names of B1/B2 and B3/B4; coordinates right"},
                        "caveat": "AI labelers, AI verifiers and AI triage may share blind spots; the STEP cross-check is the independent check (17/23 FULL scans match by count, every AI line matched a CAD bend; 1 real AI miss)"},
            "policy_decisions_needed": [
                "Joggles: are 15 deg (42288005) and 35 deg micro-joggles (10839000) gold lines?",
                "Shallow lanced tabs (47251002 B12, 15 deg): gold or not?",
                "Collinear pieces split by a full-width window/relief (11979003 B2/B3, B4/B5): one line (ENGINE_COMPILE 'Windows' rule) or two (gold, V10)?",
                "Locus: gold is the visible outer apex; runtime CAD placeholders sit at the virtual sharp (~0.41*(Ri+t) off)",
            ],
        },
        "estimates_not_rescored": {
            "gold11_best_lane_with_triage_corrected_labels": {"per_scan": GOLD11_CORRECTED_ESTIMATE, "aggregate": est_best},
            "gold11_runtime_with_triage_corrected_labels": {"per_scan": RUNTIME_GOLD11_CORRECTED_ESTIMATE, "aggregate": est_rt},
            "perfect_engine_on_current_gold11_labels": {"per_scan": PERFECT_ENGINE_ON_CURRENT_GOLD11, "aggregate": perfect,
                                                        "meaning": "even an engine that draws every real bend exactly would score about this against today's Gold11 references, if the triage verdicts hold"},
        },
        "bar_for_recipe_guided_engine": {
            "same_case_list_sha256": case_list["case_list_sha256"],
            "matcher": "5 deg / 4 mm / 0.60 / 0.60 one-to-one Hungarian, raw-scan frame",
            "gold11": {"best_lane": cohorts["best_f1"]["GOLD11"], "runtime": cohorts["runtime"]["GOLD11"], "gate": "F1>=0.80 and P>=0.90 (G2a)"},
            "cad_available_36": {"best_lane": cohorts["best_f1"]["CAD_AVAILABLE"], "runtime": cohorts["runtime"]["CAD_AVAILABLE"]},
            "ai_gold_cad_available": {"best_lane": cohorts["best_f1"]["AI_GOLD_CAD_AVAILABLE"], "runtime": cohorts["runtime"]["AI_GOLD_CAD_AVAILABLE"]},
            "staged_13": {"best_lane": cohorts["best_f1"]["STAGED_SCANS"], "runtime": cohorts["runtime"]["STAGED_SCANS"]},
            "exact_scans": {"best_lane": cohorts["best_f1"]["ALL"]["exact_scans"], "runtime": cohorts["runtime"]["ALL"]["exact_scans"], "of": len(scored(C["best_f1"]))},
            "zero_bend_controls_false_lines": {"best_lane": 5, "runtime": 3, "of_scans": 3},
            "runtime_speed_legacy_worker_cad_scans": res["runtime"]["worker_executed_cad_scans"],
            "crash_rate_legacy_worker": f"{sum(v['scans'] for k, v in rstruct.items() if k.startswith('WORKER_EXCEPTION'))}/{sum(1 for c in scored(C['runtime']) if c['has_cad'])} scored CAD scans (+2 of 3 zero-bend controls) exited with a Python exception",
            "not_like_for_like": "a recipe-guided engine needs CAD; the 21 no-CAD scans (97 lines) are outside its scope and must be reported as such, not dropped silently",
        },
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT / "SUMMARY.json")
    if args.print_tables:
        print_tables(summary)


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def print_tables(s):
    co = s["broad"]["cohorts"]
    print("\n## cohorts")
    for k in co["best_f1"]:
        b, r = co["best_f1"][k], co["runtime"][k]
        print(f"| {k} | {b['scans']} | {b['gold_lines']} | {b['true_positive']}/{b['false_positive']}/{b['false_negative']} | {f3(b['precision'])} | {f3(b['recall'])} | **{f3(b['f1'])}** | {f3(b['macro_f1'])} | {r['true_positive']}/{r['false_positive']}/{r['false_negative']} | {f3(r['precision'])} | {f3(r['recall'])} | **{f3(r['f1'])}** | {f3(r['macro_f1'])} |")
    print("\n## bootstrap")
    print(json.dumps(s["broad"]["bootstrap_ci95_scan_resampling"], indent=1))
    print("\n## sensitivity")
    for k in s["broad"]["sensitivity"]["best_f1"]:
        b, r = s["broad"]["sensitivity"]["best_f1"][k], s["broad"]["sensitivity"]["runtime"][k]
        print(f"| {k} | {b['scans']} | {b['true_positive']}/{b['false_positive']}/{b['false_negative']} | {f3(b['f1'])} | {r['true_positive']}/{r['false_positive']}/{r['false_negative']} | {f3(r['f1'])} |")
    print("\n## families")
    for f, v in s["broad"]["family_table"].items():
        b, r = v["best_f1"], v["runtime"]
        print(f"| {f} | {'G11' if v['in_gold11_families'] else ''} | {'yes' if v['has_cad'] else 'no'} | {v['scans']} | {v['gold_lines']} | {b['true_positive']}/{b['false_positive']}/{b['false_negative']} | {f3(b['micro_f1'])} | {r['true_positive']}/{r['false_positive']}/{r['false_negative']} | {f3(r['micro_f1'])} |")
    print("\n## resources")
    print(json.dumps(s["broad"]["resources"], indent=1))
    print("\n## runtime structure")
    for k, v in s["broad"]["runtime_outcome_structure"].items():
        print(k, v["scans"], v["gold_lines_missed"], v["true_positive"], v["false_positive"])
    print("\n## taxonomy")
    print(json.dumps(s["triage"]["taxonomy_mapped_to_scored_errors"], indent=1))
    print(json.dumps(s["triage"]["mechanism_scan_counts"], indent=1))
    print(json.dumps(s["triage"]["coverage_of_scored_errors"], indent=1))
    print("\n## runtime by structural outcome")
    print(json.dumps(s["triage"]["taxonomy_mapped_to_scored_errors"]["runtime"]["by_structural_outcome"], indent=1))
    print(json.dumps(s["triage"]["taxonomy_mapped_to_scored_errors"]["runtime"]["triaged_scans_by_structural_outcome"], indent=1))
    print(json.dumps(s["broad"]["runtime_variants_v3"], indent=1))
    for r in s["broad"]["gold11_per_scan"]:
        print(r["gold11_id"], r["case_id"], r["family_id"], r["scan"], r["gold"], r["best_f1"], r["best_f1_f1"], r["runtime"], r["runtime_f1"], r["runtime_status"], (r["runtime_error"] or "")[:30], r["triaged"])
    print("\n## length bins")
    print(json.dumps(s["broad"]["recall_by_gold_length"], indent=1))
    print(json.dumps(s["broad"]["best_f1_prediction_source_split"], indent=1))
    print("\n## estimates")
    e = s["estimates_not_rescored"]
    print("best corrected", e["gold11_best_lane_with_triage_corrected_labels"]["aggregate"])
    print("runtime corrected", e["gold11_runtime_with_triage_corrected_labels"]["aggregate"])
    print("perfect engine", e["perfect_engine_on_current_gold11_labels"]["aggregate"])
    print(s["label_quality"]["gold11"])


if __name__ == "__main__":
    main()
