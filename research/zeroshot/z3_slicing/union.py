"""Union of Z3 lines with other lanes' eval_lines (A0 baseline, Z2) on DEV, deduplicated by a geometric NMS
(4 deg / 2.5 mm / 50 % of the shorter span; the earlier source wins), scored with the frozen matcher.

Sources are read-only: A0 = the phase-1 baseline predictions (score_baseline.a0_path logic), Z2 = the other WP's
eval_lines under z2/runs/<tag>/<sid>/eval_lines.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from research.zeroshot.z3_slicing import custody, slicer, zscore

REPO = custody.REPO
H = custody.RUNLOG
RUNS_BF1 = REPO / "docs/production/runlog/broad_baseline/runs/best_f1"


def a0_lines(sid: str, inv: dict) -> list[dict[str, Any]]:
    pid = inv[sid].get("a0_prediction_case_id") or sid
    for p in (RUNS_BF1 / pid / "predictions_a0.json", H / "baseline/a0_runs" / sid / "predictions_a0.json"):
        if p.is_file():
            custody.check_source_path(p)
            return _eval_lines(p)
    return []


def z2_lines(sid: str, tag: str) -> list[dict[str, Any]]:
    p = H / "z2/runs" / tag / sid / "eval_lines.json"
    if not p.is_file():
        return []
    custody.check_source_path(p)
    return _eval_lines(p)


def _eval_lines(p: Path) -> list[dict[str, Any]]:
    d = zscore.bb.load_eval_lines(p)
    out = []
    for r in d.get("owned_bend_regions") or []:
        try:
            e = zscore.bb.endpoints(r)
        except Exception:  # noqa: BLE001
            continue
        if not zscore.bb.valid_line(e[0], e[1]):
            continue
        out.append({"start": np.asarray(e[0], float).tolist(), "end": np.asarray(e[1], float).tolist()})
    return out


def union(*sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for src in sources:
        for c in src:
            if any(_dup(c, k) for k in kept):
                continue
            kept.append(c)
    return kept


def _dup(a, b) -> bool:
    ang, off, ov = slicer._pair(a, b)
    return ang < 4.0 and off < 2.5 and ov > 0.5


def main(argv=None) -> int:
    """usage: python -m research.zeroshot.z3_slicing.union SELECTOR_OOF.json [--z2-tag it00] [--json OUT]"""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    ap.add_argument("--z2-tag", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    inv = custody.dev_inventory()
    sel = json.loads(Path(a.oof).read_text())["selected"]
    ids = [s for s in custody.dev_ids() if s in sel]
    src = {"z3": {s: sel[s] for s in ids}, "a0": {s: a0_lines(s, inv) for s in ids}}
    if a.z2_tag:
        src["z2"] = {s: z2_lines(s, a.z2_tag) for s in ids}
        have = sum(1 for s in ids if src["z2"][s] or (H / "z2/runs" / a.z2_tag / s / "eval_lines.json").is_file())
        print(f"z2 runs present for {have}/{len(ids)} scans")
    combos = [("z3",), ("a0",), ("z3", "a0"), ("a0", "z3")]
    if "z2" in src:
        combos += [("z2",), ("z3", "z2"), ("z2", "z3"), ("z3", "z2", "a0")]
    out = {}
    for combo in combos:
        recs = [zscore.score_lines(s, inv[s], union(*[src[k][s] for k in combo])) for s in ids]
        agg = zscore.aggregate(recs)
        ci = zscore.bootstrap_ci(recs, n_boot=2000)
        name = "+".join(combo)
        out[name] = {**agg, **ci}
        print(f"{name:12s} P={agg['P']:.3f} R={agg['R']:.3f} F1={agg['micro_f1']:.3f} {ci['micro_ci']} "
              f"macro={agg['family_macro_f1']:.3f} tp={agg['tp']} fp={agg['fp']} fn={agg['fn']} ctrlFP={agg['controls']['fp_lines']}")
        print(f"{'':12s} FN {agg['fn_causes']} FP {agg['fp_causes']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
