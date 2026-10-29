"""DEV gold, folds, candidate labels and scoring for Z4/Z5 (frozen matcher; DEV only).

- Gold: INVENTORY gold path of DEV scans, loaded by research/broad_baseline/score.load_gold (apex locus).
- Folds: z4/dev_folds.json, 5 folds grouped by the 5-digit family group (family_group5), balanced greedily by
  gold-line count, made once and re-used (the integrator's dev_folds.json wins if it appears).
- Candidate label: matches at least one gold line under the frozen thresholds (5 deg, 4 mm, 0.6 of both spans).
- Scoring: score.score_case (one-to-one Hungarian), micro / family-macro with the phase-1 bootstrap convention.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.zeroshot.z4_selector.custody import OUT, REPO, RUNLOG, dev_ids, dev_inventory

for p in (str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)
import score as S  # noqa: E402
from research.jev.pool_v2 import pair_metrics_matrix  # noqa: E402

FOLDS_FILE = OUT / "dev_folds.json"
INTEGRATOR_FOLDS = RUNLOG / "dev_folds.json"  # integrator-owned, frozen 2026-10-25
N_BOOT, SEED = 10000, 20261018


def case(sid: str, inv: dict) -> dict:
    r = inv[sid]
    g = r["gold"]
    return {"case_id": sid, "family_id": r["family"], "tier": g["set"], "gold": {"kind": "ai_studio", "path": g["path"]},
            "label_complete": g.get("label_complete", True)}


def load_gold_lines(sid: str, inv: dict) -> dict:
    g = S.load_gold(case(sid, inv)["gold"], frame_records=None, locus="apex")
    return {"lines": g["lines"], "zero_bend_control": g["zero_bend_control"],
            "label_complete": inv[sid]["gold"].get("label_complete", True)}


def make_folds(inv: dict, k: int = 5) -> dict:
    if INTEGRATOR_FOLDS.is_file():
        d = json.loads(INTEGRATOR_FOLDS.read_text())
        return {"k": int(d["n_folds"]), "scan_fold": {s: int(f) for s, f in d["scan_fold"].items()},
                "source": str(INTEGRATOR_FOLDS), "split_sha256": d.get("split_sha256")}
    if FOLDS_FILE.is_file():
        return json.loads(FOLDS_FILE.read_text())
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in dev_ids():
        groups[inv[sid]["family_group5"]].append(sid)
    weight = {gk: sum(max(1, inv[s]["gold"].get("n_lines") or 0) for s in v) for gk, v in groups.items()}
    load = [0] * k
    assign: dict[str, int] = {}
    for gk in sorted(groups, key=lambda x: (-weight[x], x)):
        f = int(np.argmin(load))
        assign[gk] = f
        load[f] += weight[gk]
    folds = {"schema": "sherman.zeroshot.z4.dev_folds.v1", "k": k, "group_key": "family_group5",
             "method": "greedy balance by gold-line count (max(1,n_lines) per scan), groups sorted by weight desc",
             "made_by": "Z4 (no integrator dev_folds.json existed on 2026-10-25)",
             "group_fold": assign, "scan_fold": {s: assign[inv[s]["family_group5"]] for s in dev_ids()},
             "fold_gold_weight": load}
    FOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FOLDS_FILE.write_text(json.dumps(folds, indent=1, sort_keys=True))
    return folds


def match_matrix(ends: np.ndarray, gold: list[dict]) -> np.ndarray:
    """bool [n_cand, n_gold]: frozen-matcher match (5 deg, 4 mm, 0.6 of both spans)."""
    if len(ends) == 0 or not gold:
        return np.zeros((len(ends), len(gold)), bool)
    g = np.array([[x["line_start"], x["line_end"]] for x in gold], float)
    m = pair_metrics_matrix(g, ends)
    ok = ((m["axis_angle_deg"] <= 5.0) & (m["transverse_offset_mm"] <= 4.0)
          & (m["gold_span_overlap"] >= 0.6) & (m["prediction_span_overlap"] >= 0.6))
    return ok.T


def score_lines(sid: str, inv: dict, lines: list[dict]) -> dict:
    preds = {"owned_bend_regions": [{"bend_id": f"P{i}", "line_start": list(map(float, l["line_start"])),
                                      "line_end": list(map(float, l["line_end"]))} for i, l in enumerate(lines)],
             "status": "OK" if lines else "NO_PREDICTIONS", "scan": {}, "lane": "z4"}
    rec = S.score_case(case(sid, inv), frame_records=None, preloaded_predictions=preds)
    rec["family_group5"] = inv[sid]["family_group5"]
    return rec


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def aggregate(recs: list[dict], boot: bool = True) -> dict:
    rows = [x for x in recs if x.get("disposition") == "SCORED_COMPLETE_LABEL"]
    ctrl = [x for x in recs if x.get("disposition") == "ZERO_BEND_CONTROL"]
    fam = defaultdict(lambda: [0, 0, 0])
    for x in rows:
        f = fam[x["family_id"]]
        f[0] += x["true_positive"]; f[1] += x["false_positive"]; f[2] += x["false_negative"]
    if not fam:
        return {"scans": 0}
    tp, fp, fn = (sum(v[i] for v in fam.values()) for i in range(3))
    p, r, f1 = prf(tp, fp, fn)
    arr = np.array(list(fam.values()), float)
    ff = np.array([prf(*v)[2] for v in arr])
    out = {"scans": len(rows), "families": len(fam), "gold_lines": tp + fn, "tp": tp, "fp": fp, "fn": fn,
           "P": round(p, 3), "R": round(r, 3), "micro_f1": round(f1, 3), "family_macro_f1": round(float(ff.mean()), 3),
           "zero_bend_controls": len(ctrl), "control_fp_lines": sum(x.get("false_positive_lines", 0) for x in ctrl),
           "empty_output_scans": sum(1 for x in rows if not x.get("prediction_count"))}
    if boot:
        rng = np.random.default_rng(SEED)
        mic, mac = [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(arr), len(arr))
            mic.append(prf(*arr[idx].sum(0))[2]); mac.append(ff[idx].mean())
        ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
        out["micro_f1_ci"], out["family_macro_f1_ci"] = ci(mic), ci(mac)
    return out


def a0_records() -> dict[str, dict]:
    d = json.loads((RUNLOG / "baseline/scores_dev_per_case.json").read_text())["a0"]
    keep = set(dev_ids())
    return {k: v for k, v in d.items() if k in keep}
