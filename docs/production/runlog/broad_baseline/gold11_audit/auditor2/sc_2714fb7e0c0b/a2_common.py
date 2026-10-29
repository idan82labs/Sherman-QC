"""Auditor 2 helpers for sc_2714fb7e0c0b (light, read-only)."""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent
CACHE = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_2714/pts_frame.npz")
GOLD = Path("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json")
PRED = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_repro/runs/sc_2714fb7e0c0b/best_f1")

def load():
    d = np.load(CACHE)
    return d["q"].astype(np.float64), d["R"], d["O"], d["raw"].astype(np.float64)

def to_frame(x, R, O):
    return (np.asarray(x, float) - O) @ R.T

def gold_lines(R, O):
    g = json.loads(GOLD.read_text())
    return {b["bend_id"]: (to_frame(b["line_start"], R, O), to_frame(b["line_end"], R, O)) for b in g["bends"]}

def pred_lines(R, O, fname="predictions_best_f1.json"):
    p = PRED / fname
    if not p.is_file():
        return {}
    rows = json.loads(p.read_text()).get("owned_bend_regions") or []
    out = {}
    for r in rows:
        if r.get("line_start") and r.get("line_end"):
            out[str(r["bend_id"])] = (to_frame(r["line_start"], R, O), to_frame(r["line_end"], R, O))
    return out
