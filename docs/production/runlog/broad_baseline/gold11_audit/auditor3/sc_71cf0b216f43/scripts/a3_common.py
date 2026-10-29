"""Auditor-3 helpers for sc_71cf0b216f43 (read-only on scan + gold; writes only under auditor3/)."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
from pathlib import Path

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
OUT = ROOT / "docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_71cf0b216f43"
SCR = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_71cf")
SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01_full_scan.ply")
GOLD_MIRROR = ROOT / "data/bend_corpus/annotations/human_bend_lines_v1/c615c4cd283605201afac7e8.json"


def load_pts():
    return np.load(SCR / "pts.npy"), np.load(SCR / "vn.npy")


def load_gold():
    d = json.loads(GOLD_MIRROR.read_text())
    return [(b["bend_id"], np.asarray(b["line_start"], float), np.asarray(b["line_end"], float)) for b in d["bends"]]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# part frame (from area-weighted normal clustering of the scan mesh)
H0 = unit([0.2583, -0.3327, 0.907])
L0 = unit([-0.5985, -0.7915, -0.1236])
L0 = unit(L0 - H0 * (L0 @ H0))
S0 = np.cross(H0, L0)


def to_frame(P, c=None):
    P = np.asarray(P, float)
    c = np.zeros(3) if c is None else c
    R = np.stack([L0, S0, H0])
    return (P - c) @ R.T
