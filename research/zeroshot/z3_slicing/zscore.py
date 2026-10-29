"""Scoring helpers for Z3: the frozen matcher (research/broad_baseline/score.py score_case, 5 deg / 4 mm /
0.6 of both spans, one-to-one Hungarian, apex locus, raw frame) and the diagnostic error budget
(score_broad.diagnose). Family-macro = mean over part families of family-micro F1."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "research/broad_baseline", REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import bb_common as bb  # noqa: E402
import score as S  # noqa: E402
import score_broad as SB  # noqa: E402

from research.zeroshot.z3_slicing.custody import RUNLOG  # noqa: E402

EVAL_SCHEMA = bb.EVAL_SCHEMA


def case_spec(sid: str, row: dict) -> dict:
    g = row["gold"]
    return {"case_id": sid, "family_id": row["family"], "tier": g["set"],
            "gold": {"kind": "ai_studio", "path": g["path"]}, "label_complete": g.get("label_complete", True)}


def to_payload(lines: list[dict[str, Any]], *, lane: str, scan_path: str | None, scan_sha: str | None,
               status: str = "OK", extra: dict | None = None) -> dict:
    regs = []
    for i, c in enumerate(lines):
        regs.append(bb.as_line(f"Z{i + 1:03d}", c["start"], c["end"],
                               **{k: v for k, v in c.items() if k in ("score", "incl", "family", "n_support")}))
    return bb.eval_payload(lane=lane, lines=regs, scan_path=Path(scan_path) if scan_path else None,
                           scan_sha256=scan_sha, source_kind="z3_axis_slicing", source_path=None,
                           conversion=extra or {}, status=status)


def score_lines(sid: str, row: dict, lines: list[dict[str, Any]], diagnose: bool = True) -> dict:
    payload = {"owned_bend_regions": [bb.as_line(f"Z{i + 1:03d}", c["start"], c["end"]) for i, c in enumerate(lines)],
               "status": "OK", "scan": {}, "lane": "z3"}
    case = case_spec(sid, row)
    rec = S.score_case(case, frame_records=None, preloaded_predictions=payload)
    if diagnose and rec.get("disposition") == "SCORED_COMPLETE_LABEL":
        gl = S.load_gold(case["gold"], frame_records=None, locus="apex")["lines"]
        pl = payload["owned_bend_regions"]
        dg = SB.diagnose(gl, pl, rec)
        rec["fn_causes"], rec["fp_causes"] = dg["fn_cause_counts"], dg["fp_cause_counts"]
        rec["fn_rows"], rec["fp_rows"] = dg["false_negatives"], dg["false_positives"]
    return rec


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def aggregate(recs: list[dict]) -> dict:
    rows = [x for x in recs if x.get("disposition") == "SCORED_COMPLETE_LABEL"]
    ctrl = [x for x in recs if x.get("disposition") == "ZERO_BEND_CONTROL"]
    fam = defaultdict(lambda: [0, 0, 0])
    for x in rows:
        f = fam[x["family_id"]]
        f[0] += x["true_positive"]
        f[1] += x["false_positive"]
        f[2] += x["false_negative"]
    tp, fp, fn = (sum(v[i] for v in fam.values()) for i in range(3))
    p, r, f1 = prf(tp, fp, fn)
    macro = float(np.mean([prf(*v)[2] for v in fam.values()])) if fam else 0.0
    fnc, fpc = Counter(), Counter()
    for x in rows:
        fnc.update(x.get("fn_causes") or {})
        fpc.update(x.get("fp_causes") or {})
    return {"scans": len(rows), "families": len(fam), "gold": tp + fn, "pred": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "P": round(p, 4), "R": round(r, 4), "micro_f1": round(f1, 4), "family_macro_f1": round(macro, 4),
            "fn_causes": dict(fnc), "fp_causes": dict(fpc),
            "controls": {"scans": len(ctrl), "fp_lines": sum(x.get("false_positive_lines", 0) for x in ctrl)}}


def bootstrap_ci(recs: list[dict], n_boot: int = 10000, seed: int = 20261018) -> dict:
    rows = [x for x in recs if x.get("disposition") == "SCORED_COMPLETE_LABEL"]
    fam = defaultdict(lambda: [0, 0, 0])
    for x in rows:
        f = fam[x["family_id"]]
        f[0] += x["true_positive"]; f[1] += x["false_positive"]; f[2] += x["false_negative"]
    arr = np.array(list(fam.values()), float)
    ff = np.array([prf(*v)[2] for v in arr])
    rng = np.random.default_rng(seed)
    mic, mac = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        mic.append(prf(*arr[idx].sum(0))[2])
        mac.append(ff[idx].mean())
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]  # noqa: E731
    return {"micro_ci": ci(mic), "macro_ci": ci(mac)}


def folds(inv: dict[str, dict], dev: list[str]) -> dict[str, int]:
    """Prefer the integrator's dev_folds.json; else the Z3 family-grouped 5-fold file (made once, recorded)."""
    top = RUNLOG / "dev_folds.json"
    mine = RUNLOG / "z3" / "dev_folds_z3.json"
    for p in (top, mine):
        if p.is_file():
            d = json.loads(p.read_text())
            fm = d.get("scan_fold") or d.get("fold_of_scan") or d
            if isinstance(fm, dict) and all(s in fm for s in dev):
                return {s: int(fm[s]) for s in dev}
    groups = sorted({inv[s]["family_group5"] for s in dev})
    rng = np.random.default_rng(20261025)
    perm = rng.permutation(len(groups))
    # balance by scan count: greedy assignment of groups (random order) to the lightest fold
    cnt = Counter(inv[s]["family_group5"] for s in dev)
    load = [0] * 5
    gfold = {}
    for gi in sorted(perm, key=lambda i: -cnt[groups[i]]):
        f = int(np.argmin(load))
        gfold[groups[gi]] = f
        load[f] += cnt[groups[gi]]
    fm = {s: gfold[inv[s]["family_group5"]] for s in dev}
    mine.parent.mkdir(parents=True, exist_ok=True)
    mine.write_text(json.dumps({"schema": "sherman.zeroshot.z3.dev_folds.v1",
                                "rule": "5 folds grouped by INVENTORY family_group5 (5-digit), greedy balanced by "
                                        "scan count over a seeded (20261025) group order; DEV ids of SPLIT_FROZEN.json only",
                                "fold_sizes": load, "fold_of_scan": fm}, indent=1))
    return fm
