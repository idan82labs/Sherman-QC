"""Shared helpers for the N1 47085000 diagnosis (gold access is for diagnosis / scoring only)."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "research/broad_baseline"), str(REPO / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
PROG = REPO / "docs/production/runlog/zeroshot_2026-10-25"
N1 = PROG / "newscans_N1"
GOLD = REPO / "data/bend_corpus/annotations/ai_bend_lines_newscans_n1_v1/47bce7ae44bc19b53aef78aa.json"
FAIL = ["B1", "B4", "B14", "B25", "B26"]


def gold():
    import score as S
    G = S.load_gold({"kind": "ai_studio", "path": str(GOLD)}, frame_records=None, locus="apex")["lines"]
    return {g["bend_id"]: np.array([g["line_start"], g["line_end"]], float) for g in G}


def pm(g, p):
    from scripts.evaluate_f1_against_manual_gold import pair_metrics
    return pair_metrics(np.asarray(g, float), np.asarray(p, float))


def is_match(m):
    return (m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 4 and m["gold_span_overlap"] >= 0.6
            and m["prediction_span_overlap"] >= 0.6)


def eval_lines(rel):
    d = json.loads((N1 / rel / "sc_3d191f5f8ff8/eval_lines.json").read_text())
    return [np.array([r["line_start"], r["line_end"]], float) for r in d["owned_bend_regions"]]


def seg_dist(P, a, b):
    """distance of points P to segment ab, and the along-parameter s (mm from a)."""
    d = b - a; L = np.linalg.norm(d); u = d / L
    s = (P - a) @ u
    sc = np.clip(s, 0, L)
    return np.linalg.norm(P - (a + sc[:, None] * u), axis=1), s
