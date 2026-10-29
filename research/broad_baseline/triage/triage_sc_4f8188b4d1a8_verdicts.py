#!/usr/bin/env python3
"""Write the triage verdicts + a PROPOSED (not applied) gold correction for sc_4f8188b4d1a8."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import triage_sc_4f8188b4d1a8 as T  # noqa: E402

pts, _ = T.load_ply(T.SCAN)
gold = {b["bend_id"]: np.array([b["line_start"], b["line_end"]], float) for b in json.load(open(T.GOLD))["manual_bends"]}
a = T.unit(gold["B02"][1] - gold["B02"][0]); b = T.unit(gold["B01"][1] - gold["B01"][0]); b = T.unit(b - (b @ a) * a); n = np.cross(a, b)
Rm = np.stack([a, b, n]); c0 = pts.mean(0)
back = lambda q: (np.asarray(q, float) @ Rm + c0).round(4).tolist()
# hinge of the upper-strip end flanges (strip mid-skin n~30.0, end-piece mid-skin b~-111.5/+112.5), end piece spans a -116..-72
prop = {"C1_like_b_neg": [back([-116.0, -111.3, 30.0]), back([-72.0, -111.3, 30.0])],
        "C9_like_b_pos": [back([-116.0, 112.3, 30.6]), back([-72.0, 112.3, 30.6])]}
out = {
  "scan_id": T.CASE, "gold_source": str(T.GOLD), "frame": "scan_raw",
  "finding": "Gold B07/B08 (15 mm, along plate normal at the B06-flange/end-piece corners) sit on butt seams, not bends: "
             "zoom slices show the B06 flange and the end piece as separate capped sheet ends with a gap. The end pieces are "
             "bent from the upper strip (continuous 1.5 mm rounded L along a at n~30), matching STEP bends C1/C9 (90 deg, 44.2 mm). "
             "B09/B10 are real 90 deg slot tabs (STEP C6/C8, 78 mm).",
  "proposed_gold_replacement_not_applied": {"remove": ["B07", "B08"], "add": prop},
  "evidence_pngs": ["zoom_endpiece_bend_vs_seam.png", "slices_corners_constant_n.png", "slices_ends_constant_b.png",
                    "slices_slot_tabs.png", "part_frame_overview.png", "part_frame_corner_B03end_CAD_B5_CAD_B1.png"],
}
(T.OUT / "triage_verdicts.json").write_text(json.dumps(out, indent=1))
print(json.dumps(prop))
