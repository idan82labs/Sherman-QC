"""zs_lane_v4 evaluation kit: the v4 development pool, the frozen matcher, family aggregates and paired bootstraps.

Everything here is evaluation-side (labels, gold, families); none of it is imported by the v4 runtime.

* Pool: docs/production/runlog/zs_v4/POOL.json (tools/build_pool.py; freeze-ledger NOTE seq 48).
* Matcher: the frozen research/broad_baseline score_case through research/zeroshot/zs_lane_v1/scoring.score_scan
  (5 deg / 4 mm / 0.6 / 0.6, one-to-one Hungarian, apex locus, raw frame), unchanged.
* Family (bootstrap and LOFO unit): family5 = part_id[:5] (POOL.json).
* Bootstrap: family bootstrap n 10000, seed 20261018 (scoring.aggregate / bootstrap with explicit n, seed).
* Paired difference: zs_lane_v2.train.paired_bootstrap (b - a over families scored in both; same n, seed).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.zeroshot.zs_lane_v1 import scoring as SC
from research.zeroshot.zs_lane_v1.core import REPO

V4RUN = REPO / "docs/production/runlog/zs_v4"
POOL_PATH = V4RUN / "POOL.json"
N_BOOT, SEED = 10000, 20261018


def load_pool() -> list[dict]:
    return json.loads(POOL_PATH.read_text())["scans"]


def spec_of(s: dict) -> dict:
    return {"case_id": s["case_id"], "family_id": s["family5"], "tier": f"V4POOL_{s['origin']}",
            "gold": s["gold"], "label_complete": s.get("label_complete", True)}


def gold_lines(s: dict) -> list[dict]:
    """Gold lines (apex locus, raw frame) exactly as the frozen scorer loads them."""
    return SC.S.load_gold(s["gold"], frame_records=None, locus="apex")["lines"]


def gold_raw(s: dict) -> list[dict]:
    return json.loads((REPO / s["gold"]["path"]).read_text()).get("bends") or []


def pinned_lines(s: dict, lane: str) -> list[dict] | None:
    """Stored output of a pinned lane on the pool scan ([{'start','end',...}]); None = missing / failed."""
    ref = s["pinned_outputs"][lane]
    if "#" in ref:
        p, key = ref.split("#")
        lr = (json.loads((REPO / p).read_text()).get("lanes") or {}).get(key)
    else:
        lr = json.loads((REPO / ref).read_text())
    if not lr or lr.get("status") != "OK":
        return None
    return [{"start": l["start_mm"], "end": l["end_mm"], "src": l.get("src"), "score": l.get("score"),
             "op": l.get("op"), "fix": l.get("fix"), "incl": l.get("incl_deg")} for l in lr["lines"]]


def pinned_run(s: dict, lane: str) -> dict | None:
    ref = s["pinned_outputs"][lane]
    if "#" in ref:
        p, key = ref.split("#")
        return (json.loads((REPO / p).read_text()).get("lanes") or {}).get(key)
    return json.loads((REPO / ref).read_text())


class Scorer:
    """Cached frozen-matcher scoring of line sets per pool scan."""

    def __init__(self, pool: list[dict]):
        self.by_id = {s["case_id"]: s for s in pool}
        self.cache: dict = {}

    def score(self, case_id: str, lines: list[dict] | None, diagnose: bool = False) -> dict:
        key = (case_id, diagnose, None if lines is None else tuple(
            tuple(np.round(np.r_[l["start"], l["end"]], 5)) for l in lines))
        if key not in self.cache:
            rec = SC.score_scan(spec_of(self.by_id[case_id]), lines, diagnose=diagnose)
            rec["origin"] = self.by_id[case_id]["origin"]
            self.cache[key] = rec
        return self.cache[key]

    def counts(self, per_scan: dict[str, list[dict] | None]) -> tuple[int, int, int, int]:
        """(tp, fp, fn, control_fp_lines) summed over the scans given."""
        tp = fp = fn = cfp = 0
        for s, lines in per_scan.items():
            r = self.score(s, lines)
            if r.get("disposition") == "SCORED_COMPLETE_LABEL":
                tp += r["true_positive"]; fp += r["false_positive"]; fn += r["false_negative"]
            elif r.get("disposition") == "ZERO_BEND_CONTROL":
                cfp += int(r.get("false_positive_lines") or 0)
        return tp, fp, fn, cfp

    def micro(self, per_scan: dict[str, list[dict] | None]) -> float:
        tp, fp, fn, _ = self.counts(per_scan)
        return SC.prf(tp, fp, fn)[2]

    def macro(self, per_scan: dict[str, list[dict] | None]) -> float:
        fam = defaultdict(lambda: [0, 0, 0])
        for s, lines in per_scan.items():
            r = self.score(s, lines)
            if r.get("disposition") == "SCORED_COMPLETE_LABEL":
                f = fam[r["family_id"]]
                f[0] += r["true_positive"]; f[1] += r["false_positive"]; f[2] += r["false_negative"]
        return float(np.mean([SC.prf(*v)[2] for v in fam.values()])) if fam else 0.0


def aggregate(recs: list[dict], ci: bool = True) -> dict:
    return SC.aggregate(recs, ci=ci, n_boot=N_BOOT, seed=SEED)


def paired(recs_a: list[dict], recs_b: list[dict]) -> dict:
    from research.zeroshot.zs_lane_v2.train import paired_bootstrap
    return paired_bootstrap(recs_a, recs_b, n_boot=N_BOOT, seed=SEED)


def family_table(recs: list[dict]) -> dict[str, list[int]]:
    fam = defaultdict(lambda: [0, 0, 0])
    for r in recs:
        if r.get("disposition") == "SCORED_COMPLETE_LABEL":
            f = fam[r["family_id"]]
            f[0] += r["true_positive"]; f[1] += r["false_positive"]; f[2] += r["false_negative"]
    return dict(fam)


def slim(a: dict) -> dict:
    keys = ["scans", "families", "gold", "pred", "tp", "fp", "fn", "P", "R", "micro_f1", "micro_ci95",
            "family_macro_f1", "macro_ci95", "failed_or_missing_scans", "controls", "fn_causes", "fp_causes"]
    return {k: a.get(k) for k in keys}


def match_label(cand_lines: list[tuple[Any, Any]], gold: list[dict]) -> np.ndarray:
    """Per candidate segment (start, end): 1 when it matches some gold line under the frozen gates (not one-to-one;
    the v1/v2 training label)."""
    from scripts.evaluate_f1_against_manual_gold import pair_metrics
    y = np.zeros(len(cand_lines))
    G = [np.array([g["line_start"], g["line_end"]], float) for g in gold]
    for j, (a, b) in enumerate(cand_lines):
        pe = np.array([a, b], float)
        if not np.isfinite(pe).all() or np.linalg.norm(pe[1] - pe[0]) < 1e-6:
            continue
        for g in G:
            m = pair_metrics(g, pe)
            if (m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 4 and m["gold_span_overlap"] >= 0.6
                    and m["prediction_span_overlap"] >= 0.6):
                y[j] = 1
                break
    return y


class FastScorer:
    """A vectorised re-implementation of the frozen matcher (pair_metrics, is_match, the 1e9-sentinel
    linear_sum_assignment) for inner-CV selection only. Every reported number uses the frozen ``Scorer``;
    ``tools/check_fastscorer.py`` asserts that both give identical TP/FP/FN."""

    def __init__(self, items):
        self.gold = {}
        self.zero = {}
        self.family = {}
        for it in items:
            G = np.array([[g["line_start"], g["line_end"]] for g in it.gold], float).reshape(-1, 2, 3)
            self.gold[it.case_id] = G
            self.zero[it.case_id] = bool(it.zero_bend) or len(G) == 0
            self.family[it.case_id] = it.family

    @staticmethod
    def admissible(G: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(admissible bool [g, p], cost [g, p]) under 5 deg / 4 mm / 0.6 / 0.6."""
        gm = G.mean(1)
        pm = P.mean(1)
        gd = G[:, 1] - G[:, 0]
        pd = P[:, 1] - P[:, 0]
        gl = np.linalg.norm(gd, axis=1)
        pl = np.linalg.norm(pd, axis=1)
        ga = gd / gl[:, None]
        pa = pd / pl[:, None]
        dot = np.clip(ga @ pa.T, -1.0, 1.0)
        ang = np.degrees(np.arccos(np.abs(dot)))
        md = pm[None, :, :] - gm[:, None, :]
        go = np.linalg.norm(md - ga[:, None, :] * np.einsum("gpk,gk->gp", md, ga)[..., None], axis=2)
        po = np.linalg.norm(md - pa[None, :, :] * np.einsum("gpk,pk->gp", md, pa)[..., None], axis=2)
        off = np.maximum(go, po)
        s0 = np.einsum("gpk,gk->gp", P[None, :, 0, :] - gm[:, None, :], ga)
        s1 = np.einsum("gpk,gk->gp", P[None, :, 1, :] - gm[:, None, :], ga)
        g0 = np.einsum("gk,gk->g", G[:, 0] - gm, ga)
        g1 = np.einsum("gk,gk->g", G[:, 1] - gm, ga)
        glo, ghi = np.minimum(g0, g1)[:, None], np.maximum(g0, g1)[:, None]
        plo, phi = np.minimum(s0, s1), np.maximum(s0, s1)
        ov = np.maximum(0.0, np.minimum(ghi, phi) - np.maximum(glo, plo))
        gov = ov / gl[:, None]
        pov = ov / pl[None, :]
        adm = (ang <= 5.0) & (off <= 4.0) & (gov >= 0.6) & (pov >= 0.6)
        cost = ang / 5.0 + off / 4.0 + (1 - np.minimum(gov, 1.0)) + (1 - np.minimum(pov, 1.0))
        return adm, cost

    def counts(self, case_id: str, lines: list[dict] | None) -> tuple[int, int, int, int]:
        """(tp, fp, fn, control_fp) for one scan."""
        from scipy.optimize import linear_sum_assignment
        G = self.gold[case_id]
        P = np.array([[l["start"], l["end"]] for l in (lines or [])], float).reshape(-1, 2, 3)
        if len(P):
            P = P[np.linalg.norm(P[:, 1] - P[:, 0], axis=1) > 1e-9]
        if self.zero[case_id]:
            return 0, 0, 0, len(P)
        if not len(P):
            return 0, 0, len(G), 0
        adm, cost = self.admissible(G, P)
        M = np.where(adm, cost, 1e9)
        r, c = linear_sum_assignment(M)
        tp = int(np.sum(M[r, c] < 1e9))
        return tp, len(P) - tp, len(G) - tp, 0

    def micro(self, pred: dict[str, list[dict] | None]) -> float:
        tp = fp = fn = 0
        for k, v in pred.items():
            a, b, c, _ = self.counts(k, v)
            tp += a; fp += b; fn += c
        return SC.prf(tp, fp, fn)[2]

    def micro_macro(self, pred: dict[str, list[dict] | None]) -> tuple[float, float, int]:
        fam = defaultdict(lambda: [0, 0, 0])
        cfp = 0
        for k, v in pred.items():
            a, b, c, d = self.counts(k, v)
            cfp += d
            if not self.zero[k]:
                f = fam[self.family[k]]
                f[0] += a; f[1] += b; f[2] += c
        tp, fp, fn = (sum(v[i] for v in fam.values()) for i in range(3))
        return SC.prf(tp, fp, fn)[2], float(np.mean([SC.prf(*v)[2] for v in fam.values()])) if fam else 0.0, cfp
