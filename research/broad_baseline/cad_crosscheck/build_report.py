"""Assemble docs/production/runlog/broad_baseline/cad_crosscheck.json from the
automatic outputs plus the adjudication recorded below (renders + scan-angle
measurements). Read-only on all inputs."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
O = ROOT / "docs/production/runlog/broad_baseline"
B = ROOT / "docs/production/runlog/benchmark_v2"
auto = json.loads((O / "cad_crosscheck_auto.json").read_text())
sp = json.loads((O / "cad_crosscheck_match/_same_plates_merge.json").read_text())
resid = json.loads((O / "cad_crosscheck_match/_surface_residual.json").read_text())
runs = {r["scan_id"]: r for r in json.loads((O / "cad_crosscheck_match/_run_summary.json").read_text())}
steps = {r["part_id"]: r for r in json.loads((O / "cad_crosscheck_raw/_run_summary.json").read_text())}

# CAD bend-line count under the AI/human convention (one line per fold between
# the same two plates). Lenient extras adjudicated by geometry:
CAD_LINES_OVERRIDE = {
    "28870000": (2, "V10 found only the 25 deg bend; lenient census adds a 27 deg discrete bend (V10 rejected: ambiguous_pair_missing_two_full_tangent_flange_rails) and an R30 154 deg rolled curl (not a press-brake line)."),
    "49001000": (5, "V10 found 3; lenient census finds 5 (all 90 deg R3); Gold11 human gold on the PRELIM scan also has 5."),
    "37126000": (2, "V10 found 0; lenient census finds 2 tiny 45 deg bends (7.2 mm span)."),
    "47479000": (0, "Lenient 178 deg R74/R75 coaxial pair is a rolled half-shell (414 mm), not a press-brake bend line."),
    "47266000": (1, "Lenient second candidate is a 3.2 mm-tall large-radius cut edge (306/235 deg sweep), not a bend."),
}

ADJ = {
 "sc_2233372b8df7": {"flag": True, "verdict": "AI_LABEL_ERROR_MISSED_BENDS",
   "detail": "CAD has a 15 deg + 15 deg joggle (C5/C6, 45 mm long, 11.7 mm apart) in the back wall. After CAD-to-scan registration (92% scan inliers at 2 mm) the scan shows 14.1 deg and 15.1 deg folds there; flat-wall control lines at 4 offsets measure 0.3-1.1 deg. The AI label has 6 lines (all six 90 deg bends, every one matched to CAD within 1.4 mm) and no line on either joggle bend; the verifier still reported 0 missed bends. Caveat: Gold11 human gold on 10839000 leaves out a 3 mm micro-joggle, so the rule for joggles must be set explicitly. This one is a visible 45 mm offset, not a micro-step.",
   "corrected_expected_count": 8},
 "sc_927a2fa13067": {"flag": True, "verdict": "AI_CORRECT_SCAN_IS_STAGED (stage metadata wrong)",
   "detail": "In this '3RD_SCAN' the second 155 mm flange (CAD C2) is not formed: the scan measures 0.2 deg at C2 with full point support, and the web runs flat about 10 mm past the CAD bend line. The 2 AI lines match C1 and C3. The scan is a staged part, so 2 bends is correct; the REPEAT_3RD_SCAN hint is misleading. The owner-verified gold on 15651004_01_SECOND_SCAN has 1 line, which also suggests the repeat scans are staged.",
   "corrected_expected_count": 2},
 "sc_38091ac91d9d": {"flag": True, "verdict": "AI_CORRECT_V10_UNDERCOUNT",
   "detail": "V10 reports 1 bend. The CAD has 2 discrete bends (25 deg, plus a 27 deg bend that V10 rejected) and an R30 154 deg rolled curl. The two AI lines match both discrete bends: B1 is 24.2 deg against 25 deg CAD, and B2 is 17.6 deg against 27 deg CAD; the scan measures 17.9 deg at that bend. So the count agrees. The 27-to-18 deg gap is real part geometry or springback, not a label error.",
   "corrected_expected_count": 2},
 "sc_d45186653eed": {"flag": True, "verdict": "CONSISTENT_COLLINEAR_SEGMENTS_MERGED",
   "detail": "V10 reports 11 segments. C7 (148 mm), C8 (48 mm) and C9 (12 mm) fold the same two plates across reliefs and form one 287 mm AI line, B1. The other 8 CAD bends match 1:1. With same-plate merging, CAD has 9 bend lines, equal to the AI count. All 11 segments are formed in the scan.",
   "corrected_expected_count": 9},
 "sc_777c64c88967": {"flag": True, "verdict": "CONSISTENT_COLLINEAR_SEGMENTS_MERGED",
   "detail": "V10 reports 12 segments. C3 and C11 (194 mm each) merge into AI B1 (420 mm), and C4, C5 and C6 (136/38/38 mm) merge into AI B4 (290 mm). Same-plate merging gives 9 CAD bend lines, equal to the AI count. All segments are formed.",
   "corrected_expected_count": 9},
 "sc_752e2bd54aad": {"flag": True, "verdict": "CONSISTENT_COLLINEAR_SEGMENTS_MERGED",
   "detail": "Same merges as the V scan, so 9 CAD bend lines equal the AI count of 9. This defective X part measures C1 at 76.9 deg, C2 at 99 deg and C3/C11 at about 85 deg against 90 deg nominal, which fits its DEFECTIVE label.",
   "corrected_expected_count": 9},
 "sc_eaa89c6d1f1f": {"flag": False, "verdict": "CONSISTENT_COLLINEAR_SEGMENTS_MERGED; STAGE_HINT_WRONG",
   "detail": "The scan is tagged PARTIAL_SCAN_1, yet all 14 CAD segments are formed (98% scan inliers, 98% CAD coverage). Same-plate merging gives 9 bend lines, equal to the AI count of 9. Treat this as a full scan."},
 "sc_2d3e4a623a31": {"flag": False, "verdict": "CONSISTENT",
   "detail": "Count and angle/length multiset agree: AI 117/357, 91/244 and 116/109 against CAD 117/359, 90/244.5 and 115.5/109. Registration is only partial (70% inliers), so the automatic line assignment swapped B2 and B3. That is a registration artefact, not a label problem."},
 "sc_b4719db3dd69": {"flag": False, "verdict": "CONSISTENT",
   "detail": "Rolled half-shell (R74/R75, 178 deg, 414 mm) with no press-brake bend line. The AI zero-bend label agrees with the V10 count of 0."},
 "sc_3bde9996f14d": {"flag": False, "verdict": "STAGED_INCONCLUSIVE",
   "detail": "Staged scan with a zero-bend AI label. The local fold measured at CAD C1 is 13.6 deg, but the adjacent R30 roll makes local fold measurement unreliable, because a control profile varies 18-43 deg. A human should look at the end tab."},
}


def sha(p):
    try: return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception: return None


rows = []
for r in auto["scans"]:
    pid = r["part_id"]
    cad_lines = CAD_LINES_OVERRIDE.get(pid, (sp[pid]["bend_lines_same_plates_merged"], None))[0]
    is_full = not r["stage_class"].startswith("STAGED")
    acc = r["ai_status"] == "AI_GOLD_ACCEPTED"
    row = {k: r.get(k) for k in ("part_id", "scan_id", "file", "stage_hints", "stage_class", "primary_truth", "ai_status", "ai_count", "cad_v10_count", "cad_lenient_count")}
    row["cad_bend_lines_same_plate_merged"] = cad_lines
    row["count_rule_flag_full_scan_ai_ne_v10"] = bool(acc and is_full and r["ai_count"] != r["cad_v10_count"])
    row["full_scan_ai_ne_cad_bend_lines"] = bool(acc and is_full and r["ai_count"] != cad_lines)
    if r.get("registration"):
        row["registration"] = r["registration"]; row["surface_residual_mm"] = resid.get(r["scan_id"])
        row["auto"] = r["auto"]; row["cad_bends"] = r["cad_bends"]; row["unmatched_ai_lines"] = r["unmatched_ai_lines"]
        row["lines_json"] = r.get("lines_json"); row["lines_json_sha256"] = sha(r.get("lines_json"))
        row["match_render"] = str(O / "cad_crosscheck_renders" / f"{r['scan_id']}_match.png")
        rr = runs.get(r["scan_id"], {}); row["match_run"] = {"wall_s": rr.get("wall_s"), "peak_rss_bytes": rr.get("peak_rss_bytes")}
    if r["gold_counts_other_tiers"]: row["gold_counts_other_tiers"] = r["gold_counts_other_tiers"]
    if r["scan_id"] in ADJ: row["adjudication"] = ADJ[r["scan_id"]]
    elif acc and is_full:
        row["adjudication"] = {"flag": False, "verdict": "CONSISTENT", "detail": "AI count equals the CAD count, and every AI line matches a CAD bend after registration."}
    elif acc:
        row["adjudication"] = {"flag": False, "verdict": "STAGED_FEWER_OR_EQUAL_OK" if (r["ai_count"] or 0) <= cad_lines else "STAGED_MORE_THAN_CAD",
                               "detail": "Staged scan; fewer bends than CAD is allowed."}
    rows.append(row)

parts = []
for p in auto["parts"]:
    pid = p["part_id"]; o = CAD_LINES_OVERRIDE.get(pid)
    parts.append({**p, "step_sha256": steps[pid].get("step_sha256"), "extractor_wall_s": steps[pid].get("wall_s"),
                  "extractor_peak_rss_bytes": steps[pid].get("peak_rss_bytes"),
                  "same_plate_merged_groups": sp[pid]["merged_groups"],
                  "cad_bend_lines_same_plate_merged": o[0] if o else sp[pid]["bend_lines_same_plates_merged"],
                  "cad_bend_lines_note": o[1] if o else None,
                  "cad_render": str(O / "cad_crosscheck_renders" / f"{pid}.png") if (O / "cad_crosscheck_renders" / f"{pid}.png").exists() else None})

full_acc = [r for r in rows if r["ai_status"] == "AI_GOLD_ACCEPTED" and not r["stage_class"].startswith("STAGED")]
report = {
 "schema": "sherman.broad_baseline.cad_crosscheck.v1",
 "generated_at": datetime.now(timezone.utc).isoformat(),
 "question": "Cross-check AI-gold bend counts against STEP CAD bend counts for every BENCHMARK_V2 part with a STEP file.",
 "custody": "Inputs: BENCHMARK_V2.json, CONSUMED parts' STEP files listed in it (source-assets/02_CAD_QC_3D, ~/Downloads, incoming_email_2026-10-17), AI label lines.json and renders (label_lines*/), label_prep downsampled points, Gold11 gold catalog (counts and line lengths only) and the Gold11 10839000 full scan for calibration. Not read: incoming_codex_fetch_2026-10-17, FRESH, HOLDOUT10, sealed truth, N001-N016, EHO4, authorization or held-out artifacts. No engine run, nothing edited outside research/broad_baseline/** and this runlog directory.",
 "method": [
  "1. Ran scripts/extract_step_topology_bend_axes_v10.extract_step_bend_axes on 25 STEP files, serially, by import because the script has no CLI. Each call took 0.5-1.2 s wall time at about 0.32 GiB peak RSS. A lenient coaxial inner/outer cylinder census was recorded too, because V10 is strict and can under-count.",
  "2. Built a CAD bend-line count. Collinear V10 segments that fold the same two plates (sharing at least 3 adjacent native plane faces, meaning one fold interrupted by reliefs or holes) count as one line, which is the convention both the AI and the Gold11 humans use. Separate tabs on a shared line stay separate. Lenient-only candidates were adjudicated by geometry.",
  "3. Registered each AI-labelled scan to its CAD. The CAD was tessellated and surface-sampled, then fitted by PCA with 48 axis-permutation/sign initialisations and a 3-stage point-to-point ICP; the fit was scored by the fraction of scan points within 2 mm. Each CAD bend was then mapped into the raw scan frame.",
  "4. For each CAD bend: measured scan support (points within 3 mm of the line) and the scan's own fold angle across the bend (two-sided PCA in the cross-section 3-18 mm from the line). Matched CAD bends to AI lines one-to-one with Hungarian assignment (gates: axis angle < 12 deg, perpendicular offset < 10 mm, overlap > 30%). Recorded unmatched CAD bends that a longer collinear AI line covers.",
  "5. Adjudicated each flagged FULL scan from the AI renders, CAD renders, match overlays and scan fold angles, with flat-region control lines where needed."],
 "stage_rule": "FULL = FULL, GOOD_V, DEFECTIVE_X, LIDAR_DEMO_CAPTURE or no stage hint (presumed full). REPEAT_* counts as full for flagging. PARTIAL_SCAN_*, PRELIM and STAGE0_BLANK count as staged, where fewer bends than CAD are allowed.",
 "summary": {
  "parts_with_step": len(parts),
  "scans_on_step_parts": len(rows),
  "ai_accepted_scans_on_step_parts": sum(r["ai_status"] == "AI_GOLD_ACCEPTED" for r in rows),
  "ai_accepted_full_scans": len(full_acc),
  "full_scans_flagged_ai_ne_v10": [r["scan_id"] for r in full_acc if r["count_rule_flag_full_scan_ai_ne_v10"]],
  "full_scans_ai_ne_cad_bend_lines": [r["scan_id"] for r in full_acc if r["full_scan_ai_ne_cad_bend_lines"]],
  "confirmed_ai_label_errors": [r["scan_id"] for r in rows if (r.get("adjudication") or {}).get("verdict", "").startswith("AI_LABEL_ERROR")],
  "stage_metadata_wrong": ["sc_927a2fa13067 (REPEAT_3RD_SCAN is a staged part)", "sc_eaa89c6d1f1f (PARTIAL_SCAN_1 has every bend formed)"],
  "v10_undercounts": {"28870000": "1 vs 2 discrete bends", "49001000": "3 vs 5 (human gold 5)", "37126000": "0 vs 2 tiny 45 deg bends"},
  "headline": "23 accepted AI labels are on FULL scans of parts with a STEP file (8 of them are 47266000 scans with 1 bend each). 6 of them differ from the raw V10 count. One is a real AI label error: 42288005 FULL is missing two formed 15 deg joggle bends (6 labelled vs 8). Of the other 5, three are a counting convention (collinear CAD segments folding the same plates drawn as one line: 11979003 STL, 50757000 V and X). One is a staged scan with a wrong stage tag (15651004 3RD_SCAN, where one flange is unformed). One is a V10 under-count (28870000 FULL). The other 17 FULL accepted labels match CAD by count and line by line. Staged scans had no AI count above the CAD count; 40500000 PARTIAL_SCAN_1 has every bend formed and should be retagged as full."},
 "calibration_gold11_human": {
  "37361005": "human 4 = CAD 4",
  "49024000": "human 3 = CAD 3",
  "49001000_PRELIM": "human 5 = lenient CAD 5 (V10 3)",
  "40500000_SCAN2": "human lines of 185 and 174 mm exceed every single CAD segment except one, so humans also merge same-plate collinear segments",
  "10839000_FULL": "human 8 against V10 12. The 4 extra CAD bends are two micro-joggles (35 deg + 35 deg, 3.1 mm apart, along 307 mm). In the Gold11 full scan the measured step offset is 1.6 and 2.4 mm with a net angle of 2.3-2.6 deg, so they are formed. Humans did not label them. Convention: micro-joggles are not gold lines."},
 "open_convention_question": "Joggles: Gold11 humans skip a 3 mm micro-joggle (10839000). Should the 42288005 joggle (two 15 deg bends 11.7 mm apart, 45 mm long) be gold lines? On the AI gate's own definition ('zero visible bends missed') it should; an owner decision is needed before scoring.",
 "parts": parts,
 "scans": rows,
 "artifacts": {"code": str(ROOT / "research/broad_baseline/cad_crosscheck"),
               "raw_v10": str(O / "cad_crosscheck_raw"), "matches": str(O / "cad_crosscheck_match"),
               "renders": str(O / "cad_crosscheck_renders"), "calibration": str(O / "cad_crosscheck_calib"),
               "auto_summary": str(O / "cad_crosscheck_auto.json")},
}
(O / "cad_crosscheck.json").write_text(json.dumps(report, indent=1))
for r in rows:
    if r["ai_status"]:
        a = r.get("adjudication", {})
        print(f"{r['part_id']:9} {r['scan_id']} {r['stage_class'][:14]:14} {r['ai_status'][:6]:6} ai={r['ai_count']} v10={r['cad_v10_count']} lines={r['cad_bend_lines_same_plate_merged']} flag={r['count_rule_flag_full_scan_ai_ne_v10']} {a.get('verdict','')}")
print(json.dumps(report["summary"], indent=1))
