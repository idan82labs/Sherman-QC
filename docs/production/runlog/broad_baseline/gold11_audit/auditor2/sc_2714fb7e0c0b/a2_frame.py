#!/usr/bin/env python3
"""Auditor 2, sc_2714fb7e0c0b: load scan once, build a part frame, cache (read-only on inputs).

Frame: u = B01 direction (long edges), v = B03 direction (Z-step bends), w = u x v (main-plate normal).
Origin = gold B03 start (corner of B01/B03). Gold read from the human_bend_lines_v1 mirror (identical to
catalog shaked_xv/C06, frame identity). Cache goes to the session scratchpad, not the repo.
"""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
ROOT = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(ROOT / "research/broad_baseline"))
import bb_common as bb  # noqa

SCAN = ROOT / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_V.ply"
GOLD = ROOT / "data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json"
CACHE = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_2714/pts_frame.npz")
OUT = Path(__file__).resolve().parent

def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)

def main():
    gold = json.loads(GOLD.read_text())
    G = {b["bend_id"]: (np.array(b["line_start"], float), np.array(b["line_end"], float)) for b in gold["bends"]}
    ex = unit(G["B01"][1] - G["B01"][0])
    ey0 = unit(G["B03"][1] - G["B03"][0])
    ey = unit(ey0 - (ey0 @ ex) * ex)
    ez = np.cross(ex, ey)
    O = G["B03"][0].copy()
    R = np.vstack([ex, ey, ez])
    pts = bb.load_scan_vertices(SCAN)
    q = (pts - O) @ R.T
    np.savez_compressed(CACHE, raw=pts.astype(np.float32), q=q.astype(np.float32), R=R, O=O)
    frame = {"origin_raw": O.tolist(), "R_rows_u_v_w": R.tolist(), "n_points": int(len(pts)),
             "angle_B01_B03_deg": float(np.degrees(np.arccos(abs(ex @ ey0)))),
             "q_min": q.min(0).tolist(), "q_max": q.max(0).tolist(), "gold_uvw": {}}
    for k, (s, e) in G.items():
        frame["gold_uvw"][k] = [((s - O) @ R.T).round(2).tolist(), ((e - O) @ R.T).round(2).tolist()]
    (OUT / "a2_frame.json").write_text(json.dumps(frame, indent=1))
    print(json.dumps(frame, indent=1))

if __name__ == "__main__":
    main()
