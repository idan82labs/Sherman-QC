"""Final held-out evaluation 2026-10-28, batch FE1: adjudication (PREREG section 4.4-4.6), modelled on
docs/production/runlog/zeroshot_2026-10-25/tools/adjudicate_n1.py.

  main                A/B agreement with the frozen matcher (research/broad_baseline/score.py score_case, 5 deg / 4 mm /
                      60 % both spans; gold = marker A, prediction = marker B), canary QC against its GOLD_SET_V2 gold of
                      record, final lines per item (source A, B or the adjudicator re-run in adjudication/<gid>/lines),
                      adjudication/<gid>/adjudication.json and adjudication/ADJUDICATION_FE1.partial.json.
  labels-and-freeze   write-label of the 27 scored items into the new root
                      data/bend_corpus/annotations/ai_bend_lines_final_eval_2026_10_28_v1 (asset = the original scan when
                      the unchanged bench_labeling custody guard admits it, else a byte-identical guard-clean copy staged
                      through research.zeroshot.common.run_allowlist.stage, PREREG 4.6), provenance sidecars, FROZEN appends
                      via tools/gold_freeze_fe1.py, the gold-root manifest and adjudication/ADJUDICATION_FE1.json.
  ledger-note         the freeze-ledger NOTE (domains.production.engine.eval.freeze.append_entry, kind NOTE) binding the
                      gold root manifest, gold-ledger sha256, counts, canary scores, A/B agreement and new conventions.
No engine, lane, pool, E3, CAD or baseline output is read; none exists for FE1. Verdicts were taken by the adjudicator from
the marker renders/close-ups/sweeps and its own raw-scan sections (adjudication/evidence/), see DECISIONS below.
usage (repo root): .venv/bin/python -B docs/production/runlog/final_eval_2026-10-28_heldout/tools/adjudicate_fe1.py [cmd]
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).resolve().parents[1]
REPO = D.parents[3]
sys.path[:0] = [str(REPO), str(REPO / "research/broad_baseline")]
import bb_common as bb  # noqa: E402
import score as S  # noqa: E402

PK = D / "packets"
FE1 = PK / "FE1"
ADJ = D / "adjudication"
STAGE = D.parents[0] / "fe1_stage_2026-10-28/scans"
ROOT = REPO / "data/bend_corpus/annotations/ai_bend_lines_final_eval_2026_10_28_v1"
ASSET_COPIES = REPO / "output/research/fe1_gold_assets_2026-10-28"
GOLD_LEDGER = REPO / "docs/production/runlog/zeroshot_2026-10-25/gold/GOLD_LEDGER.jsonl"
FREEZE_LEDGER = REPO / "docs/production/runlog/freeze/LEDGER.json"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
BY = "FE1/adjudicator"
PROTOCOL = ("PREREG.md section 4 (sha c50f58e0...); frozen matcher research/broad_baseline/score.py score_case (5 deg / 4 mm "
            "/ 60 % both spans, one-to-one); accept threshold 1.5 mm / 2 deg; gold rule GOLD_SET_V2 (MARKER_INSTRUCTIONS.md)")

R_V2 = "GOLD_SET_V2 gold rule (docs/production/GOLD_SET_V2.md; packets/MARKER_INSTRUCTIONS.md)"
R_D0 = "D0 gb_44b168 hem ruling (zeroshot gold/D0/ADJUDICATION_D0.json: closed 180 deg hem counts; tangent-band construction)"
R_TABS = ("separate tabs = separate lines: T1 (canary-confirmed), T4 gb_efbb1e, T5 gb_0f8d41/leg-3 feet, GOLD_SET_V2 "
          "sc_b76452ea6fa7 ('two separate collinear folds (different tabs) = 2 bends')")
R_CORNER = "T1 box-corner rulings (formed fold only when the section shows a concentric arc; sharp corner/slit = not a fold)"
R_WIN = "GOLD_SET_V2 windows/reliefs rule; T2 gb_d2aa20, T3 gb_630bfb, T6 gb_9412f6 (one fold between the same two plates)"
R_LOUVRE = "T3 gb_c61b1b frozen gold (lanced louvres excluded as emboss-type features); GOLD_SET_V2 'embosses are not bends'"

# final line source per item: A = marker A lines.json, B = marker B lines.json, ADJ = adjudicator re-run
# (adjudication/<gid>/lines: marker A ok pairs reproduced bit-identically + the hem lines)
SOURCE = {g: "A" for g in ("gb_04738b", "gb_138090", "gb_1f3fe0", "gb_42dd99", "gb_44ddfb", "gb_47b48f", "gb_61b6da",
                           "gb_8ccb40", "gb_8df897", "gb_9528f8", "gb_a0538b", "gb_a2dd83", "gb_a64e51", "gb_b21bad",
                           "gb_b3ae40", "gb_b97c8e", "gb_bbc3a8", "gb_c6e707", "gb_cdecd9", "gb_f63c89")}
SOURCE.update({"gb_437879": "B", "gb_d1093e": "B"})
SOURCE.update({g: "ADJ" for g in ("gb_1baa22", "gb_d11e33", "gb_231669", "gb_ac2239", "gb_68cd06", "gb_959f4a")})

PART_CLASS = {"gb_42dd99": "mixed", "gb_61b6da": "rolled", "gb_cdecd9": "rolled"}

HEM = ("180 deg hem: both skins of one sheet run round a ~180 deg arc to a folded-back leg (adjudicator raw sections, "
       "adjudication/evidence/sections/*_v2.png). A formed fold >= 10 deg counts; the tool refuses parallel planes, so the "
       "line is built with the D0 tangent-band construction (two plane_from_points bands at +-45 deg on the outer arc, "
       "extent = measured crown extent); no manual line. Check: label line vs measured crown in evidence/hemcheck.json.")

DECISIONS = {
    "gb_138090": [
        {"lines_A": ["B3", "B4"], "doubt": "short ears (~6 mm folds) at one end of each long flange", "verdict": "ACCEPT_A",
         "rule": f"{R_V2}: formed fold >= 10 deg between two flat plates; short folds counted in T5 (6.5 mm feet) and N1 gb_4dd5ba (5 mm hook lips)",
         "reason": "A sweep ear_end_B1/B2 and closeups: the flange overhang turns ~90 deg into an ~11 mm ear with a visible "
                   "concentric radius. Adjudicator station check on the raw scan (evidence/st_out.json): outer envelope "
                   "1.83-1.95 mm from the plane sharp along the whole line vs expected outer apex 2.0 mm, i.e. a formed arc. "
                   "B classed the ear strips as end faces without a section. Lines cover the full flange height."}],
    "gb_1f3fe0": [
        {"lines_A": ["B3", "B4"], "doubt": "short ears (~5 mm folds)", "verdict": "ACCEPT_A",
         "rule": f"{R_V2}; T5 / N1 gb_4dd5ba short-fold precedent",
         "reason": "Same design as gb_138090. A closeup_B3 section shows a formed ~100 deg arc with the dot on the outer "
                   "apex. Station check: outer envelope 2.65-2.9 mm (B4, expected 2.48) and 3.1-4.6 mm (B3, expected 2.83; "
                   "the ear is slightly twisted so the straight line drifts <= 1.8 mm off the arc, inside tolerance). Formed "
                   "folds; tool confidence uncertain kept."}],
    "gb_231669": [
        {"lines_A": ["B10", "B11", "B12"], "doubt": "lanced tabs in the end wall (A only; B did not mention them)",
         "verdict": "ACCEPT_A", "rule": f"{R_TABS}; T1 'tab 16 bent up in the wall window is its own fold'",
         "reason": "A sweep g7/g8 and adjudicator sections (evidence/sections/231669A_B1*.png): tabs 28 and 30 (~14 mm) and "
                   "33 (~38 mm, in a 40 x 8 mm slot) stand ~90 deg to the end wall, each hinged along the edge of its own "
                   "opening with a rounded inner corner. Each tab is a separate plate, so one line each. The tab undersides "
                   "(outer arc) are barely scanned: the tool apex sits near the plane sharp, ~1.4 mm from the expected outer "
                   "arc (inside the 4 mm tolerance); lines keep confidence uncertain."},
        {"doubt": "box wall-to-wall corners (3-5, 3-7, 3-9, 9-11, base-11)", "verdict": "NOT_A_BEND", "rule": R_CORNER,
         "reason": "Both markers rejected them. B measured apex offsets at the corners on the plane sharp (no arc) vs ~1.6 mm "
                   "for every formed fold here: welded/seamed corners. A's 12 vs B's 9 is the three lanced tabs, not corners."},
        {"lines_final": ["B13", "B14"], "doubt": "two 180 deg hems on the tongue edges (both markers flagged, neither in lines)",
         "verdict": "MISSING_ADD", "rule": R_D0,
         "reason": HEM + " Open hems, outer stack 5.4-5.5 mm (2t + ~1.5 mm gap), crown 105-106 mm; label lines 0.02-0.07 mm "
                   "from the measured crown apex along the outward direction, <= 0.62 mm end to end."}],
    "gb_ac2239": [
        {"lines_A": ["B10", "B11", "B12"], "doubt": "lanced tabs in the end wall (A only)", "verdict": "ACCEPT_A",
         "rule": f"{R_TABS}; T1 tab 16",
         "reason": "Same part design as gb_231669; tabs 30, 31 (~15 mm) and 32 (~40 mm slot) at ~90 deg to the end wall "
                   "(adjudicator sections evidence/sections/ac2239A_B1*.png). Separate plates, one line each; tool apex "
                   "~1.3-1.6 mm from the expected outer arc (underside unscanned), confidence uncertain."},
        {"doubt": "box wall-to-wall corners", "verdict": "NOT_A_BEND", "rule": R_CORNER,
         "reason": "Both markers rejected them; B apex offsets 0.18-0.87 mm vs ~1.6 formed (3-5 at 0.87 is the closest; no "
                   "concentric arc)."},
        {"lines_final": ["B13", "B14"], "doubt": "two 180 deg tongue hems (both flagged)", "verdict": "MISSING_ADD",
         "rule": R_D0, "reason": HEM + " Open hems, stack 5.4-5.5 mm, crown 106 mm; lines within 0.31 mm of the crown."}],
    "gb_68cd06": [
        {"doubt": "14 lanced tabs separated by flat bridges: one fold or many", "verdict": "KEEP_SEPARATE (both markers)",
         "rule": R_TABS,
         "reason": "Each tab is bent ~88 deg out of its own window and is joined to the strip only along its own hinge; the "
                   "flat ~6 mm bridges between windows do not fold. Separate plates, so separate lines (the windows rule "
                   "merges one plate interrupted by windows, not disconnected tabs). B's single-line trial had the apex "
                   "collapse onto the sharp over the bridges, consistent with this."},
        {"lines_A": ["B18"], "doubt": "15th short tab (~5 mm) in the narrow end window", "verdict": "ACCEPT_A",
         "rule": f"{R_TABS}; {R_V2}",
         "reason": "B flagged the same tab (patches 39/40) as a real bent tab but excluded it only because its pairings came "
                   "out on the wrong skin. A closeup_B18: tab bent ~90 deg down at the window edge, line covers ~85 % of "
                   "the window width; apex ~1 mm off the arc (tool flags uncertain), inside tolerance."},
        {"lines_final": ["B19", "B20"], "doubt": "end hems (A: 'possible' hem at the patch-10 end; B: 'end plate fragments')",
         "verdict": "MISSING_ADD", "rule": R_D0,
         "reason": HEM + " Adjudicator sections at both strip ends (patch 10 end and patch 9 end): the plate runs to a "
                   "rounded crown and folds back ~8 mm under itself, outer stack 1.9-2.0 mm = 2t (closed hem); crown 24/25 mm "
                   "across the strip. Neither marker drew the patch-9 hem (A thought it 'unclear'); the section is "
                   "unambiguous. Lines within 0.24 mm of the measured crown."}],
    "gb_959f4a": [
        {"doubt": "14 lanced tabs separated by bridges", "verdict": "KEEP_SEPARATE (both markers)", "rule": R_TABS,
         "reason": "Same design as gb_68cd06; each tab is its own plate."},
        {"lines_final": ["B18", "B19"], "doubt": "end hems (A flagged both with manual lines; B: 'end fragments')",
         "verdict": "MISSING_ADD", "rule": R_D0,
         "reason": HEM + " Closed hems at both ends, outer stack 2.1-2.4 mm (t 1.0), crown 26 mm; lines 0.41-0.80 mm from "
                   "the measured crown (dot on the crown in closeup_B18)."}],
    "gb_1baa22": [
        {"lines_final": ["B3"], "doubt": "180 deg open hem at the short end (both markers flagged)", "verdict": "MISSING_ADD",
         "rule": R_D0, "reason": HEM + " Open hem, outer stack 5.0 mm, crown 19 mm; line 0.14-0.16 mm from the crown."}],
    "gb_d11e33": [
        {"lines_final": ["B3"], "doubt": "180 deg open hem at the short end (both flagged)", "verdict": "MISSING_ADD",
         "rule": R_D0, "reason": HEM + " Open hem, outer stack 4.9 mm, crown 19 mm; line 0.44-0.48 mm from the crown."}],
    "gb_a2dd83": [
        {"lines_A": ["B12", "B13"], "doubt": "short (~7 mm tall) inset end-flange pieces of strip 5/6 (A 21 vs B 19)",
         "verdict": "ACCEPT_A", "rule": f"{R_V2} (formed fold >= 10 deg); non-collinear separate flanges = separate lines (T1/T5)",
         "reason": "Adjudicator station check on the raw scan: B12 outer envelope 1.21-1.39 mm from the plane sharp over "
                   "0-19 mm of its 23 mm (expected formed apex 1.23), B13 0.64-1.53 mm over 0-17 mm; A closeups B12/B13 "
                   "show both skins on concentric arcs. They are ~7 mm tall, ~23 mm wide flanges folded 90 deg from strip 6, "
                   "inset ~10 mm from the main end flange, so separate folds. B's 0.4 mm figure is the tool envelope apex, "
                   "pulled down by the last ~3 mm of the line where it meets the corner; the fold itself is formed."}],
    "gb_d1093e": [
        {"lines_A": ["B21", "B22"], "doubt": "small (~8 mm) inset pieces beside the relief on strip 6 (A 22 vs B 20)",
         "verdict": "ACCEPT_B (not a bend)", "rule": f"{R_V2} (only formed folds); {R_CORNER}",
         "reason": "Station check: outer envelope 0.13-0.48 mm (B21) and 0.10-0.23 mm (B22) from the plane sharp at every "
                   "station, vs 1.25 expected for the formed folds of this part and >= 0.61 mm for any 90 deg fold of "
                   "t 1.48 even at zero inner radius. A closeup_B21: the plate's inner skin ends square with no arc and "
                   "the outer skins meet at the sharp. A sharp/abutting corner, not a formed fold. The sister scan "
                   "gb_a2dd83's inset pieces are formed (above); each scan is judged on its own geometry. Final = B."}],
    "gb_437879": [
        {"lines_B": ["B1", "B2"], "doubt": "A: flat/warped strip, 0 bends; B: shallow joggle, 2 kinks ~12 deg",
         "verdict": "ACCEPT_B", "rule": f"{R_V2}: joggle = 2 lines; formed folds >= 10 deg; a 10-15 deg bend must be a "
                                         "distinct formed fold between two flat plates",
         "reason": "Adjudicator profile of the raw mesh (evidence/k437.log, both skins, 0.5 mm stations, t ~0.55): three "
                   "straight segments with constant slope (top-skin heights linear to +-0.01 mm within each): normal tilt "
                   "-5.0 / +9.0 / -5.3 deg (top) and +5.2 / -9.1 / +5.4 deg (bottom), with ~1-1.5 mm transitions at "
                   "u ~ -2.4 and +1.9 mm. The end segments are parallel and offset (Z shape). Two distinct folds of "
                   "14.0-14.5 deg between flat plates (tool plane fits 11.9 / 12.3 deg); both >= 10 deg. A's own numbers "
                   "(-4/+8/-5 deg, ~13 deg each) agree; a smooth warp would not give piecewise-flat skins. B's 2 lines "
                   "(apex on the convex skin at each kink) are the gold; confidence uncertain kept."}],
    "gb_b3ae40": [
        {"doubt": "windowed ladder web: one line per side vs per rung; apex offset", "verdict": "KEEP_1_LINE_PER_SIDE",
         "rule": R_WIN,
         "reason": "The web is one connected frame plate (rungs joined through both rails; the windows are closed openings), "
                   "folded from the same rail plate along one straight fold: the windows rule gives 1 line per side. This "
                   "differs from T4/T5, where slit tabs were joined only along their own fold. Apex: station check shows "
                   "the formed arc at the rungs at 1.17-1.25 mm from the plane sharp and the rail-edge corners between "
                   "rungs at ~0.2 mm; the tool envelope apex (0.35 / 0.39 mm) follows the outermost material between rungs, "
                   "so the line is ~0.85 mm outside the rung arc apex. Kept (both markers identical, well inside the 4 mm "
                   "tolerance), confidence uncertain; see new-convention note NC2."}],
    "gb_b97c8e": [
        {"doubt": "windowed ladder web (mirror of gb_b3ae40)", "verdict": "KEEP_1_LINE_PER_SIDE", "rule": R_WIN,
         "reason": "As gb_b3ae40. Rung arc apex 1.14-1.20 mm; tool apex -0.08 / 0.26 mm, i.e. the lines are 0.93-1.25 mm "
                   "outside the rung arc apex (inside tolerance); kept, NC2."}],
    "gb_bbc3a8": [
        {"doubt": "~30 louvres in the top plate (both markers rejected)", "verdict": "NOT_A_BEND", "rule": R_LOUVRE,
         "reason": "Lanced and formed ~24-25 deg hoods, 35 x 12 mm, attached at both ends: stamped features of the emboss "
                   "kind, exactly the T3 gb_c61b1b case (frozen gold 2 lines, louvres excluded). Confirmed."}],
    "gb_04738b": [
        {"doubt": "stepped end tabs vs continuation of the side flanges", "verdict": "KEEP_SEPARATE (both markers)",
         "rule": f"{R_TABS}; windows rule needs the same two plates on one fold line",
         "reason": "Both markers drew them separately; the end-tab fold lines are offset ~7 mm from the side-flange lines "
                   "(different, non-collinear folds)."}],
    "gb_b21bad": [
        {"doubt": "stepped end tabs", "verdict": "KEEP_SEPARATE (both markers)", "rule": R_TABS,
         "reason": "As gb_04738b (same design family); offset ~7 mm, not collinear."}],
    "gb_42dd99": [
        {"doubt": "rolled trough facets", "verdict": "NOT_A_BEND", "rule": f"{R_V2}: rolled kinks are not bends",
         "reason": "Both markers: smooth roll (circle fit R ~122 mm, facet steps < 10 deg); only the formed lips count."}],
    "gb_61b6da": [{"doubt": "none (rolled plate)", "verdict": "ZERO_BEND", "rule": f"{R_V2}: rolled shell = 0 bends",
                   "reason": "Both markers: normal angle linear across the width, no step."}],
    "gb_cdecd9": [{"doubt": "none (rolled plate)", "verdict": "ZERO_BEND", "rule": f"{R_V2}: rolled shell = 0 bends",
                   "reason": "Both markers: constant curvature, no step."}],
}

NEW_CONVENTIONS = [
    {"id": "NC1", "items": ["gb_1baa22", "gb_d11e33", "gb_231669", "gb_ac2239"],
     "decision": "Open (gapped) 180 deg hems count as 1 line on the outer crown, built with the D0 tangent-band "
                 "construction, exactly like the closed hem of D0 gb_44b168.",
     "why_new": "D0 ruled on a closed hem only. Physically an open hem is the same formed 180 deg fold (>= 10 deg) with a "
                "gap between the legs; the GOLD_SET_V2 rule counts it. Effect: +6 lines over 4 items (1, 1, 2, 2)."},
    {"id": "NC2", "items": ["gb_b3ae40", "gb_b97c8e"],
     "decision": "On a fold interrupted by windows so large that the fold exists only at short rungs, the gold keeps the "
                 "tool's envelope apex (the outermost material, mostly rail-edge corners between rungs) instead of the "
                 "rung arc apex. Measured offset from the rung arc apex: 0.85 mm (b3ae40), 0.93-1.25 mm (b97c8e).",
     "why_new": "T4/T6 preferred a station-checked outer-skin apex when a better line existed; here no tool construction "
                "gives a full-length line on the rung apex. The offset is inside the 4 mm tolerance; per-rung lines were "
                "not used because the windows rule decides the count."},
    {"id": "NC3", "items": ["gb_d1093e", "gb_a2dd83"],
     "decision": "Formed-fold test for short or ambiguous pieces: if the outer material at the bisector sits closer to "
                 "the plane sharp than any formed fold of that thickness allows ((sqrt2 - 1) * t for 90 deg, i.e. Ri = 0), "
                 "at every station, the junction is a sharp/abutting corner and not a bend; if it sits at the expected "
                 "formed-apex offset over >= 80 % of the line, it is a formed fold.",
     "why_new": "It puts the T1 'concentric arc vs sharp corner' ruling into a number. It rejects d1093e B21/B22 and "
                "accepts a2dd83 B12/B13."},
    {"id": "NC4", "items": ["gb_68cd06"],
     "decision": "The adjudicator added the patch-9 end hem of gb_68cd06 (MISSING_ADD) on its own raw-scan sections, "
                 "with no third marking. Neither marker drew it; marker A noted the overlap but called it unclear.",
     "why_new": "Earlier MISSING_ADDs (T1, T4) came from a FRESH third marking. Here the adjudicator's evidence is the "
                "section (2t closed stack with a rounded crown). The owner may want a third marking to confirm."},
]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ok_lines(doc):
    return [b for b in doc["bends"] if b.get("status") == "ok" and b.get("label_line")]


def score(goldspec, pred_bends):
    regs = [bb.as_line(b["bend_id"], b["label_line"]["start"], b["label_line"]["end"]) for b in pred_bends]
    case = {"case_id": "x", "family_id": "x", "tier": "FE1_ADJ", "gold": goldspec, "label_complete": True}
    return S.score_case(case, frame_records=None,
                        preloaded_predictions={"owned_bend_regions": regs, "status": "OK", "scan": {}, "lane": "m"})


def mrow(m):
    return [m["gold_id"], m["prediction_id"], round(m["transverse_offset_mm"], 3), round(m["axis_angle_deg"], 3),
            round(m["gold_span_overlap"], 3), round(m["prediction_span_overlap"], 3)]


def key():
    return json.loads((D / "PRIVATE_KEY.json").read_text())["key"]


def registered():
    out = {}
    for line in GOLD_LEDGER.read_text().splitlines():
        e = json.loads(line)
        if e.get("batch_id") == "FE1" and e.get("event") == "REGISTER":
            out[e["gid"]] = e
    return out


def main():
    K = key()
    reg = registered()
    items = [i["gid"] for i in json.loads((PK / "batches.json").read_text())["items"]]
    summary = {"schema": "sherman.final_eval.gold_batch_adjudication.v1", "batch": "FE1", "adjudicated_at": NOW,
               "adjudicator": f"{BY} (claude-vision-ai-v1)", "protocol": PROTOCOL,
               "not_seen": "no engine, lane, pool, E3, CAD or baseline output (none exists for FE1)",
               "per_item": {}, "new_conventions": NEW_CONVENTIONS}
    tot = {"tp": 0, "fp": 0, "fn": 0, "nA": 0, "nB": 0, "accepted": 0, "outside": 0}
    tot_scored = {"tp": 0, "fp": 0, "fn": 0}
    tot27 = {"tp": 0, "fp": 0, "fn": 0}
    tot27_lines = {"tp": 0, "fp": 0, "fn": 0}
    zero = {"agree": [], "disagree": []}
    for gid in items:
        d = FE1 / gid
        pa, pb = d / "marker_A/lines/lines.json", d / "marker_B/lines/lines.json"
        A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
        la, lb = ok_lines(A), ok_lines(B)
        r = score({"kind": "bench_lines", "path": str(pa)}, lb)
        zb = r["disposition"] == "ZERO_BEND_CONTROL"
        matches = [] if zb else r["matches"]
        accepted = [m for m in matches if m["transverse_offset_mm"] <= 1.5 and m["axis_angle_deg"] <= 2.0]
        if zb:
            f1 = 1.0 if not lb else 0.0
            (zero["agree"] if not lb else zero["disagree"]).append(gid)
            tp, fp, fn = 0, len(lb), 0
        else:
            f1, tp, fp, fn = r["f1"], r["true_positive"], r["false_positive"], r["false_negative"]
        ab = {"nA": len(la), "nB": len(lb), "tp": tp, "fp_B_only": fp, "fn_A_only": fn, "f1": f1,
              "disposition": r["disposition"], "accepted_within_1p5mm_2deg": len(accepted),
              "matched_not_accepted": [[m["gold_id"], m["prediction_id"]] for m in matches if m not in accepted],
              "A_only": [] if zb else r["unmatched_gold_ids"], "B_only": [] if zb else r["unmatched_prediction_ids"],
              "matches": [mrow(m) for m in matches]}
        if zb and lb:
            ab["B_only"] = [b["bend_id"] for b in lb]
        for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("nA", len(la)), ("nB", len(lb)), ("accepted", len(accepted)),
                     ("outside", len(matches) - len(accepted))):
            tot[k] += v
        if not zb:
            for k, v in (("tp", tp), ("fp", fp), ("fn", fn)):
                tot_scored[k] += v
        if K[gid]["role"] == "SCORED":
            for k, v in (("tp", tp), ("fp", fp), ("fn", fn)):
                tot27[k] += v
                if not zb:
                    tot27_lines[k] += v
        # final lines
        src = SOURCE[gid]
        sp = {"A": pa, "B": pb, "ADJ": ADJ / gid / "lines/lines.json"}[src]
        fin = copy.deepcopy(json.loads(sp.read_text()))
        fl = ok_lines(fin)
        if src == "ADJ":
            assert all(b.get("status") == "ok" for b in fin["bends"]), gid
        n = len(fl)
        fin["notes"] = (f"FE1 final held-out gold (adjudicated, PREREG 4.5): blind markers A and B, line source {src}, "
                        f"{n} lines{' (no_bends_present)' if n == 0 else ''}; A/B F1 {f1:.4f}; record "
                        f"adjudication/{gid}/adjudication.json.")
        fin["adjudicated_at"] = NOW
        fin["marker_lines_sha256"] = {"A": sha(pa), "B": sha(pb)}
        fin["final_source"] = src
        if src == "ADJ":
            fin["adjudicator_lines_sha256"] = sha(sp)
            fin["adjudicator_pairs_sha256"] = sha(ADJ / gid / "pairs.json")
        fdir = ADJ / gid / "final/lines"
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / "lines.json").write_text(json.dumps(fin, indent=1) + "\n")
        # per-line provenance
        a_by_b = {m["prediction_id"]: m for m in matches}
        a_by_a = {m["gold_id"]: m for m in matches}
        dec = DECISIONS.get(gid, [])
        lines = []
        a_ids = [b["bend_id"] for b in la]
        for i, b in enumerate(fl):
            row = {"final_id": b["bend_id"], "bend_angle_deg": round(b["bend_angle_deg"], 2),
                   "length_mm": round(b["length_mm"], 1), "confidence": b.get("confidence")}
            if src == "A" or (src == "ADJ" and i < len(la)):
                aid = b["bend_id"] if src == "A" else a_ids[i]
                m = a_by_a.get(aid)
                row.update({"marker_A_id": aid, "marker_B_id": m["prediction_id"] if m else None,
                            "verdict": "AGREED" if m else "ACCEPT_A (A only)", "from": "A"})
                if m:
                    row.update({"ab_offset_mm": round(m["transverse_offset_mm"], 3),
                                "ab_angle_deg": round(m["axis_angle_deg"], 3)})
            elif src == "B":
                m = a_by_b.get(b["bend_id"])
                row.update({"marker_B_id": b["bend_id"], "marker_A_id": m["gold_id"] if m else None,
                            "verdict": "AGREED" if m else "ACCEPT_B (B only)", "from": "B"})
                if m:
                    row.update({"ab_offset_mm": round(m["transverse_offset_mm"], 3),
                                "ab_angle_deg": round(m["axis_angle_deg"], 3)})
            else:
                row.update({"marker_A_id": None, "marker_B_id": None, "verdict": "MISSING_ADD (hem, adjudicator line)",
                            "from": "ADJ"})
            lines.append(row)
        rejected = []
        if src == "B" and not zb:
            rejected = [{"marker_A_id": x, "verdict": "REJECTED (not a formed fold)"} for x in r["unmatched_gold_ids"]]
        is_canary = K[gid]["role"] == "CANARY"
        pc = PART_CLASS.get(gid, "bent")
        adj = {"schema": "sherman.final_eval.gold_adjudication.v1", "batch": "FE1", "gid": gid,
               "scan_id": K[gid]["scan_id"], "role": K[gid]["role"], "subset": K[gid]["subset"],
               "part_id": None if is_canary else reg[gid]["part_id"],
               "original_path": K[gid]["original_path"], "scan_sha256": K[gid]["sha256"],
               "adjudicator": f"{BY} (claude-vision-ai-v1)", "adjudicated_at": NOW, "protocol": PROTOCOL,
               "inputs_seen": ["marker_A and marker_B lines/pairs/notes/renders/sweeps", "shared prep renders",
                               "adjudicator raw-scan sections and station checks (adjudication/evidence/)",
                               "canary GOLD_SET_V2 gold (QC only)" if is_canary else "private key (gid -> original)"],
               "not_seen": "no engine, lane, pool, E3, CAD or baseline output",
               "marker_A_lines_sha256": sha(pa), "marker_B_lines_sha256": sha(pb),
               "final_source": src, "final_lines_sha256": sha(fdir / "lines.json"), "final_line_count": n,
               "no_bends_present": n == 0, "label_complete": True, "part_class": pc, "zero_bend": n == 0,
               "partial_scan": False, "partial_note": "complete scan (both markers; all views)",
               "excluded": False, "ab_match": ab, "lines": lines, "rejected_marker_lines": rejected,
               "doubts_adjudicated": dec}
        if is_canary:
            gp = REPO / K[gid]["gold_path"]
            q = {m: score({"kind": "ai_studio", "path": str(gp)}, L) for m, L in (("A", la), ("B", lb), ("final", fl))}
            adj["canary_qc"] = {"gold_path": K[gid]["gold_path"], "gold_n_lines": K[gid]["gold_n_lines"],
                                "marker_A_f1": q["A"]["f1"], "marker_B_f1": q["B"]["f1"],
                                "adjudicated_f1": q["final"]["f1"],
                                "marker_A_matches": [mrow(m) for m in q["A"]["matches"]],
                                "marker_B_matches": [mrow(m) for m in q["B"]["matches"]],
                                "rule": ">= 0.95 per marker (PREREG 4.4)",
                                "pass": min(q["A"]["f1"], q["B"]["f1"]) >= 0.95, "label_written": False}
            if not adj["canary_qc"]["pass"]:
                (ADJ / gid / "adjudication.json").write_text(json.dumps(adj, indent=1) + "\n")
                raise SystemExit("CANARY FAILED: redo the failing marker's batch")
            summary["canary"] = {"gid": gid, "scan_id": K[gid]["scan_id"],
                                 **{k: adj["canary_qc"][k] for k in ("marker_A_f1", "marker_B_f1", "adjudicated_f1",
                                                                     "pass", "label_written")}}
        (ADJ / gid / "adjudication.json").write_text(json.dumps(adj, indent=1) + "\n")
        summary["per_item"][gid] = {"role": K[gid]["role"], "subset": K[gid]["subset"], "final_lines": n,
                                    "final_source": src, "nA": len(la), "nB": len(lb), "ab_f1": f1,
                                    "ab_disposition": r["disposition"], "part_class": pc,
                                    "verdicts": [x["verdict"] for x in dec]}
    def f1c(t):
        p = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else 0.0
        rr = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else 0.0
        return round(2 * p * rr / (p + rr), 4) if p + rr else 0.0
    sc = {k: v for k, v in summary["per_item"].items() if v["role"] == "SCORED"}
    summary["ab_agreement"] = {
        "matcher": "score_case, gold = marker A, prediction = marker B, before adjudication",
        "micro_f1_all_28_incl_zero_bend_FP": f1c(tot), "tp_fp_fn_all_28": [tot["tp"], tot["fp"], tot["fn"]],
        "micro_f1_line_scans_only": f1c(tot_scored),
        "n_A": tot["nA"], "n_B": tot["nB"], "pairs_accepted_within_1p5mm_2deg": tot["accepted"],
        "matched_outside_1p5mm_2deg": tot["outside"],
        "zero_bend_items_agreed": zero["agree"], "zero_bend_items_disagreed": zero["disagree"],
        "per_item_f1": {g: v["ab_f1"] for g, v in summary["per_item"].items()},
        "scored_27_only": {"items": len(sc), "macro_item_f1": round(sum(v["ab_f1"] for v in sc.values()) / len(sc), 4),
                           "micro_f1_incl_zero_bend_FP": f1c(tot27), "tp_fp_fn": [tot27["tp"], tot27["fp"], tot27["fn"]],
                           "micro_f1_line_scans_only": f1c(tot27_lines),
                           "tp_fp_fn_line_scans": [tot27_lines["tp"], tot27_lines["fp"], tot27_lines["fn"]]},
        "note": "hem lines are in neither marker's lines.json (the tool refused parallel planes), so they do not enter "
                "this agreement; both markers flagged the gb_1baa22/gb_d11e33/gb_231669/gb_ac2239 hems in their notes."}
    summary["final_line_counts"] = {g: v["final_lines"] for g, v in summary["per_item"].items() if v["role"] == "SCORED"}
    summary["final_total_lines_scored"] = sum(summary["final_line_counts"].values())
    (ADJ / "ADJUDICATION_FE1.partial.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("canary", "ab_agreement", "final_line_counts", "final_total_lines_scored")},
                     indent=1))


def asset_for(gid: str, v: dict, log: list) -> Path:
    from research.bench_labeling.custody import CustodyError, check_path
    try:
        check_path(v["original_path"])
        return Path(v["original_path"])
    except CustodyError as e:
        from research.zeroshot.common import run_allowlist as RA
        dst = RA.stage(STAGE / f"{gid}.ply", ASSET_COPIES, name=f"{gid}.ply", by="FE1/adjudicator write-label asset")
        if sha(dst) != v["sha256"]:
            raise SystemExit(f"{gid}: staged asset sha256 differs")
        log.append({"gid": gid, "original_refused_by_name": str(e), "asset_copy": str(dst.relative_to(REPO))})
        return dst


def labels_and_freeze():
    from research.bench_labeling.label import write_label
    summary = json.loads((ADJ / "ADJUDICATION_FE1.partial.json").read_text())
    K = key()
    ROOT.mkdir(parents=True, exist_ok=True)
    staged = []
    summary["frozen"] = {}
    for gid, v in sorted(K.items()):
        if v["role"] != "SCORED":
            continue
        d = ADJ / gid
        fdir = d / "final/lines"
        adj = json.loads((d / "adjudication.json").read_text())
        asset = asset_for(gid, v, staged)
        res = write_label(fdir, asset, labels_root=ROOT, confirm_no_bends=adj["no_bends_present"])
        lp, pp = Path(res["label_path"]), Path(res["provenance_path"])
        lsha = sha(lp)
        prov = json.loads(pp.read_text())
        assert prov["asset_sha256"] == v["sha256"] and prov["coordinate_match"] == "sha256", gid
        assert len(json.loads(lp.read_text())["bends"]) == adj["final_line_count"], gid
        renders = {}
        for sub in ("marker_A", "marker_B"):
            for png in sorted((FE1 / gid / sub).rglob("*.png")):
                renders[str(png.relative_to(D))] = sha(png)
        for png in sorted(d.rglob("*.png")):
            renders[str(png.relative_to(D))] = sha(png)
        mk = {}
        for m in ("A", "B"):
            md = FE1 / gid / f"marker_{m}"
            mk[f"marker_{m}"] = {"lines_json": json.loads((md / "lines/lines.json").read_text()),
                                 "lines_sha256": sha(md / "lines/lines.json"),
                                 "pairs": json.loads((md / "pairs.json").read_text()), "pairs_sha256": sha(md / "pairs.json"),
                                 "notes_md": (md / "notes.md").read_text(), "dir": str(md.relative_to(REPO))}
        prov["final_eval_gold"] = {"schema": "sherman.final_eval.gold_provenance.v1", "batch": "FE1", "gid": gid,
                                   "scan_id": v["scan_id"], "subset": v["subset"], "part_id": adj["part_id"],
                                   "original_path": v["original_path"], "asset_path": str(asset), **mk,
                                   "adjudication": adj, "adjudication_sha256": sha(d / "adjudication.json"),
                                   "final_lines_sha256": sha(fdir / "lines.json"), "render_sha256": renders,
                                   "label_sha256": lsha}
        pp.write_text(json.dumps(prov, indent=2, allow_nan=False) + "\n")
        out = subprocess.run([str(REPO / ".venv/bin/python"), "-B", str(D / "tools/gold_freeze_fe1.py"), "freeze",
                              "--scan", v["scan_id"], "--label-path", str(lp), "--label-sha256", lsha, "--by", BY,
                              "--note", f"FE1 adjudicated gold ({adj['final_line_count']} lines, source "
                                        f"{adj['final_source']}); A/B F1 {adj['ab_match']['f1']:.4f}; canary F1 A/B "
                                        f"{summary['canary']['marker_A_f1']}/{summary['canary']['marker_B_f1']}; "
                                        f"adjudication {(d / 'adjudication.json').relative_to(REPO)}"],
                             cwd=REPO, capture_output=True, text=True)
        if out.returncode:
            raise SystemExit(out.stderr + out.stdout)
        e = json.loads(out.stdout.splitlines()[0])
        summary["frozen"][gid] = {"scan_id": v["scan_id"], "part_id": adj["part_id"], "final_lines": adj["final_line_count"],
                                  "label_path": str(lp.relative_to(REPO)), "label_sha256": lsha,
                                  "provenance": str(pp.relative_to(REPO)), "asset_path": str(asset),
                                  "gold_ledger_seq": e["seq"], "entry_sha256": e["entry_sha256"]}
        print(gid, summary["frozen"][gid]["final_lines"], e["seq"])
    manifest = {"schema": "sherman.final_eval.gold_root_manifest.v1", "labels_root": str(ROOT.relative_to(REPO)),
                "created": NOW, "n_labels": len(summary["frozen"]),
                "labels": {g: {k: f[k] for k in ("scan_id", "final_lines", "label_path", "label_sha256", "gold_ledger_seq")}
                           | {"provenance_sha256": sha(REPO / f["provenance"])} for g, f in sorted(summary["frozen"].items())},
                "total_lines": sum(f["final_lines"] for f in summary["frozen"].values()),
                "staged_asset_copies": staged}
    mp = ADJ / "GOLD_ROOT_MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    summary["labels_root"] = str(ROOT.relative_to(REPO))
    summary["gold_root_manifest"] = {"path": str(mp.relative_to(REPO)), "sha256": sha(mp)}
    summary["staged_asset_copies"] = staged
    summary["ledger_mechanism"] = ("tools/gold_freeze_fe1.py (FROZEN for FINAL_EVAL_HELDOUT/FE1; gold_guard.py untouched, "
                                   "its pinned hash intact)")
    summary["gold_ledger_sha256_after_freeze"] = sha(GOLD_LEDGER)
    (ADJ / "ADJUDICATION_FE1.json").write_text(json.dumps(summary, indent=1) + "\n")
    (ADJ / "ADJUDICATION_FE1.partial.json").unlink()


def ledger_note():
    from domains.production.engine.eval import freeze
    s = json.loads((ADJ / "ADJUDICATION_FE1.json").read_text())
    entries = freeze.read_ledger(FREEZE_LEDGER)
    if any(e.get("kind") == "RUN" and "final_eval_2026-10-28" in json.dumps(e) for e in entries):
        raise SystemExit("an FE1 RUN entry already exists; the gold NOTE must precede it")
    seqs = sorted(f["gold_ledger_seq"] for f in s["frozen"].values())
    items = {
        "gold_root": s["labels_root"],
        "gold_root_manifest": {**s["gold_root_manifest"], "n_labels": len(s["frozen"]),
                               "total_lines": s["final_total_lines_scored"]},
        "gold_ledger": {"path": str(GOLD_LEDGER.relative_to(REPO)), "sha256": sha(GOLD_LEDGER),
                        "frozen_seqs": [seqs[0], seqs[-1]], "n_frozen": len(seqs)},
        "adjudication": {"path": str((ADJ / "ADJUDICATION_FE1.json").relative_to(REPO)),
                         "sha256": sha(ADJ / "ADJUDICATION_FE1.json"),
                         "report": str((D / "ADJUDICATION.md").relative_to(REPO)),
                         "report_sha256": sha(D / "ADJUDICATION.md") if (D / "ADJUDICATION.md").is_file() else None},
        "per_item_line_counts": s["final_line_counts"],
        "canary": s["canary"],
        "ab_agreement": {k: s["ab_agreement"][k] for k in ("micro_f1_all_28_incl_zero_bend_FP", "tp_fp_fn_all_28",
                                                           "micro_f1_line_scans_only", "zero_bend_items_disagreed",
                                                           "scored_27_only", "per_item_f1")},
        "new_conventions_for_owner": [{k: c[k] for k in ("id", "items", "decision")} for c in s["new_conventions"]],
        "staged_asset_copies": len(s["staged_asset_copies"]),
        "prereg_seq": 37, "refers_to": [35, 37, 39, 40],
        "rule": "gold FROZEN before any FE1 engine RUN entry; gold is never edited after an engine output exists",
    }
    e = freeze.append_entry(FREEZE_LEDGER, kind="NOTE", recorded_by=BY, items=items,
                            label="FE1 gold FROZEN: final held-out evaluation adjudicated gold (27 scans), canary QC, A/B "
                                  "agreement and new convention decisions; before any engine RUN")
    print(json.dumps({"seq": e["seq"], "entry_id": e["entry_id"], "entry_sha256": e["entry_sha256"]}))
    print(json.dumps(freeze.verify_ledger(FREEZE_LEDGER)))


if __name__ == "__main__":
    cmd = sys.argv[1:] or ["main"]
    {"main": main, "labels-and-freeze": labels_and_freeze, "ledger-note": ledger_note}[cmd[0]]()
