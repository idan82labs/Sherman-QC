"""zs_lane_v1 scoring: the frozen matcher (research/broad_baseline/score.py score_case: 5 deg / 4 mm / 0.6 of both
spans, one-to-one Hungarian, apex locus, raw frame) and the programme aggregates.

Rules (pre-registered in testb/PREREG.json):
  * every scan of the cohort is scored; a missing or FAILED output is scored as an empty prediction (all gold lines
    false negatives) -- never dropped;
  * micro = summed TP/FP/FN over complete-label line scans; family-macro = mean over part families of the
    family-micro F1; 95 % CIs = percentile bootstrap over families (default n 10000, seed 20261025);
  * zero-bend controls are reported separately as FP lines;
  * a scan the matcher cannot score (EXCLUDED_BEFORE_SCORING) is an error, not a silent drop.
Gold of record: DEV = INVENTORY gold (121 SPLIT_FROZEN dev scans; the 18 FRESH-DEV labels read under freeze-ledger
seq 8) + the 4 newly frozen ai_bend_lines_zs_v1 labels (gold ledger); TEST-B = the frozen re-audit labels
(ai_bend_lines_zs_v1_SEALED, gold ledger), with the old FRESH labels (INVENTORY) as a secondary column.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.zeroshot.zs_lane_v1.core import REPO  # noqa: F401  (sets sys.path)
import bb_common as bb  # noqa: E402
import score as S  # noqa: E402  (research/broad_baseline/score.py, frozen)
import score_broad as SB  # noqa: E402

PROG = REPO / "docs/production/runlog/zeroshot_2026-10-25"
SCORER_FILES = ["research/broad_baseline/score.py", "research/broad_baseline/score_broad.py",
                "research/broad_baseline/bb_common.py", "scripts/evaluate_f1_against_manual_gold.py",
                "research/zeroshot/zs_lane_v1/scoring.py"]
BOOT_N, BOOT_SEED = 10000, 20261025


def _inv() -> dict[str, dict]:
    return {r["scan_id"]: r for r in json.loads((PROG / "INVENTORY.json").read_text())["scans"]}


def _ledger_frozen() -> dict[str, dict]:
    last = {}
    for x in (PROG / "gold/GOLD_LEDGER.jsonl").read_text().splitlines():
        if x.strip():
            e = json.loads(x)
            last[e["scan_id"]] = e
    return {k: v for k, v in last.items() if v["status"] == "FROZEN"}


def split() -> dict:
    return json.loads((PROG / "SPLIT_FROZEN.json").read_text())


def dev_ids() -> list[str]:
    led = _ledger_frozen()
    pend = sorted(s for s, e in led.items() if e["portion"] == "DEV_PENDING_GOLD")
    return list(split()["dev"]["scans"]) + pend


def dev_folds() -> dict[str, int]:
    d = json.loads((PROG / "dev_folds.json").read_text())
    f = {k: int(v) for k, v in d["scan_fold"].items()}
    for s, v in d.get("pending_gold", {}).items():
        f[s] = int(v["fold"])
    return f


def case_specs(cohort: str) -> dict[str, dict]:
    """cohort: 'dev' (125), 'testb' (61, re-audit gold of record), 'testb_old' (61, old FRESH labels)."""
    inv, led = _inv(), _ledger_frozen()
    out = {}
    if cohort == "dev":
        for s in split()["dev"]["scans"]:
            g = inv[s]["gold"]
            out[s] = {"case_id": s, "family_id": inv[s]["family"], "tier": "DEV",
                      "gold": {"kind": "ai_studio", "path": g["path"]}, "label_complete": g.get("label_complete", True),
                      "scan_path": inv[s]["path"]}
        for s, e in led.items():
            if e["portion"] == "DEV_PENDING_GOLD":
                out[s] = {"case_id": s, "family_id": inv[s]["family"] if s in inv else e["part_id"],
                          "tier": "DEV_ZS_V1", "gold": {"kind": "ai_studio", "path": e["label_path"]},
                          "label_complete": True, "scan_path": e["path"], "label_sha256": e["label_sha256"]}
        return out
    tb = split()["test_b"]["scans"]
    for s in tb:
        e = led[s]
        fam = inv[s]["family"] if s in inv else e["part_id"]
        if cohort == "testb":
            gold, sha = {"kind": "ai_studio", "path": e["label_path"]}, e["label_sha256"]
        elif cohort == "testb_old":
            gold, sha = {"kind": "ai_studio", "path": inv[s]["gold"]["path"]}, None
        else:
            raise ValueError(cohort)
        out[s] = {"case_id": s, "family_id": fam, "tier": cohort.upper(), "gold": gold, "label_complete": True,
                  "scan_path": e["path"], "label_sha256": sha}
    return out


def score_scan(spec: dict, lines: list[dict] | None, diagnose: bool = True) -> dict:
    """lines: [{'start','end'}] (None = missing/failed output -> empty prediction, all FN)."""
    regs = [bb.as_line(f"P{i + 1:03d}", c["start"], c["end"]) for i, c in enumerate(lines or [])]
    payload = {"owned_bend_regions": regs, "status": "OK", "scan": {}, "lane": "zs"}
    case = {k: spec[k] for k in ("case_id", "family_id", "tier", "gold", "label_complete")}
    rec = S.score_case(case, frame_records=None, preloaded_predictions=payload)
    rec["output_missing_or_failed"] = lines is None
    if rec.get("disposition") == "EXCLUDED_BEFORE_SCORING":
        raise RuntimeError(f"{spec['case_id']}: matcher excluded the scan ({rec.get('reason')})")
    if diagnose and rec.get("disposition") == "SCORED_COMPLETE_LABEL":
        gl = S.load_gold(case["gold"], frame_records=None, locus="apex")["lines"]
        dg = SB.diagnose(gl, regs, rec)
        rec["fn_causes"], rec["fp_causes"] = dg["fn_cause_counts"], dg["fp_cause_counts"]
        rec["fn_rows"], rec["fp_rows"] = dg["false_negatives"], dg["false_positives"]
    rec["n_pred"] = len(regs)
    return rec


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _fam(recs):
    fam = defaultdict(lambda: [0, 0, 0])
    for x in recs:
        if x.get("disposition") == "SCORED_COMPLETE_LABEL":
            f = fam[x["family_id"]]
            f[0] += x["true_positive"]; f[1] += x["false_positive"]; f[2] += x["false_negative"]
    return fam


def aggregate(recs: list[dict], ci: bool = True, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    rows = [x for x in recs if x.get("disposition") == "SCORED_COMPLETE_LABEL"]
    ctrl = [x for x in recs if x.get("disposition") == "ZERO_BEND_CONTROL"]
    other = Counter(x.get("disposition") for x in recs if x not in rows and x not in ctrl)
    fam = _fam(recs)
    tp, fp, fn = (sum(v[i] for v in fam.values()) for i in range(3))
    p, r, f1 = prf(tp, fp, fn)
    macro = float(np.mean([prf(*v)[2] for v in fam.values()])) if fam else 0.0
    fnc, fpc = Counter(), Counter()
    for x in rows:
        fnc.update(x.get("fn_causes") or {})
        fpc.update(x.get("fp_causes") or {})
    worst = sorted(((round(prf(*v)[2], 3), k) for k, v in fam.items()))[:5]
    out = {"scans": len(rows), "families": len(fam), "gold": tp + fn, "pred": tp + fp, "tp": tp, "fp": fp, "fn": fn,
           "P": round(p, 4), "R": round(r, 4), "micro_f1": round(f1, 4), "family_macro_f1": round(macro, 4),
           "fn_causes": dict(fnc), "fp_causes": dict(fpc), "worst_families": worst,
           "failed_or_missing_scans": sum(1 for x in recs if x.get("output_missing_or_failed")),
           "controls": {"scans": len(ctrl), "fp_lines": sum(x.get("false_positive_lines", 0) for x in ctrl)},
           "other_dispositions": dict(other)}
    if ci and fam:
        out.update(bootstrap(recs, n_boot, seed))
    return out


def bootstrap(recs: list[dict], n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    arr = np.array(list(_fam(recs).values()), float)
    ff = np.array([prf(*v)[2] for v in arr])
    rng = np.random.default_rng(seed)
    mic, mac = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        mic[b] = prf(*arr[idx].sum(0))[2]
        mac[b] = ff[idx].mean()
    q = lambda x: [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]  # noqa: E731
    return {"micro_ci95": q(mic), "macro_ci95": q(mac), "bootstrap": {"n": n_boot, "seed": seed, "unit": "family"}}


def eval_lines_of(path: Path) -> list[dict] | None:
    """Lines of an eval_lines.json (None when missing). A FAILED payload has no lines -> scored as all FN."""
    if not path.is_file():
        return None
    d = bb.load_eval_lines(path)
    if d.get("status") != "OK":
        return None
    out = []
    for r in d.get("owned_bend_regions") or []:
        if bb.valid_line(r.get("line_start"), r.get("line_end")):
            out.append({"start": r["line_start"], "end": r["line_end"]})
    return out
