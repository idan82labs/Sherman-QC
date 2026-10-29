"""CAD-assisted path prototype (DEV only): rotation-invariant CAD facts as priors on the zero-shot search.

CAD facts per part (compiled_part.json of the E1 compile; no registration, no CAD pose, no CAD lines):
  n_bends, the multiset of nominal included angles, thickness t, and K = number of distinct bend-axis directions.
Priors applied to zs_lane_v1 (fold models from the DEV out-of-fold run, so every number is out-of-fold):
  T  thickness: Z3 detect uses the CAD t (skips the scan thickness estimate) and the D1 snap gate uses it;
  F  family cap: Z3 slices only the K + 2 best-supported axis families (search-time cut);
  A  angle window: a pooled candidate survives only within +-6 deg of some CAD included angle;
  N  count cap: at most n_bends lines, highest selector probability first.
Arms: zs (no CAD), zs+A, zs+A+N, zs+T+F+A+N (re-run Z3 with priors), and e3_else_zs (E3 lines when the stored E3
baseline run is VALID, else zs+T+F+A+N). The 5 OD-3 DEV-CHECK cases are refused by the guard (cad_path=True).

usage: sherman_slot.sh .venv/bin/python -B -m research.zeroshot.zs_lane_v1.cad_prior --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.zeroshot.zs_lane_v1 import core, scoring, train as T
from research.zeroshot.common.gold_guard import require_runnable, GoldGuardRefusal

COMPILE = Path("/private/tmp/claude-501/sherman_zeroshot_work/compile")
INTEG = scoring.PROG / "integration"
ANGLE_WIN = 6.0


def cad_facts(sku: str) -> dict | None:
    p = COMPILE / sku / "compiled/compiled_part.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    bends = [b for b in d.get("bends", []) if b.get("supported", True)]
    if not bends:
        return None
    axes = []
    for b in bends:
        u = np.asarray(b["axis"]["direction"], float)
        u /= np.linalg.norm(u)
        if not any(abs(u @ a) > math.cos(math.radians(3)) for a in axes):
            axes.append(u)
    return {"n_bends": len(bends), "incl": sorted(float(b["nominal_included_deg"]) for b in bends),
            "t_mm": float(d["thickness"]["t_mm"]), "K": len(axes)}


def e3_lines(sid: str):
    p = scoring.PROG / "baseline/e3_runs" / sid / "predictions_e3.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    if d.get("status") != "OK":
        return None
    return [{"start": r["line_start"], "end": r["line_end"]} for r in d.get("owned_bend_regions") or []]


def apply_priors(C, prob, facts, angle=True, count=True, thr=0.5):
    keep = []
    for c, q in zip(C, prob):
        if q < thr:
            continue
        if angle and min(abs(c["incl"] - a) for a in facts["incl"]) > ANGLE_WIN:
            continue
        keep.append((q, c))
    keep.sort(key=lambda x: -x[0])
    if count:
        keep = keep[:facts["n_bends"]]
    return [{"start": c["start_s"], "end": c["end_s"]} for _, c in keep]


def rerun_z3(path: str, facts: dict, z2_lines: list[dict]):
    """Z3 with CAD t and the K + 2 family cap, pooled with the stored Z2 lines, snapped with the CAD t."""
    from research.zeroshot.z3_slicing import slicer
    t0 = time.time()
    mesh = slicer.load_mesh(path)
    t_load = time.time() - t0
    prm = slicer.Params(max_families=min(20, facts["K"] + 2))
    t1 = time.time()
    res = slicer.detect(mesh, prm, thickness={"t_mm": facts["t_mm"], "source": "CAD"})
    t_det = time.time() - t1
    res["_mesh"] = mesh
    C, X = core.pool(res, z2_lines, True)
    core.snap(C, (np.asarray(mesh.V, float), np.asarray(mesh.vn, float), facts["t_mm"]))
    return C, X, {"load_s": t_load, "detect_s": t_det}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inv = scoring._inv()
    specs = scoring.case_specs("dev")
    folds = scoring.dev_folds()
    FM = pickle.loads((INTEG / "oof_v1/fold_models.pkl").read_bytes())
    P = pickle.loads((INTEG / "oof_v1/zs_snap_oof_probs.pkl").read_bytes())["P"]
    dumps = T.load_dumps(INTEG / "dev_dump", list(specs))
    arms = {k: {} for k in ("zs", "zs+A", "zs+A+N", "zs+T+F+A+N", "e3_else_zs", "e3", "e3_else_none")}
    rt, refused, cohort = [], [], []
    for s in scoring.dev_ids():
        sku = inv[s]["cad"].get("sku") if s in inv else None
        facts = cad_facts(sku) if sku else None
        if facts is None:
            continue
        try:
            require_runnable(s, cad_path=True)
        except GoldGuardRefusal as r:
            refused.append((s, r.reason))
            continue
        cohort.append(s)
        d = dumps[s]
        C = d["candidates"]
        arms["zs"][s] = [{"start": c["start_s"], "end": c["end_s"]} for c, q in zip(C, P[s]) if q >= 0.5]
        arms["zs+A"][s] = apply_priors(C, P[s], facts, angle=True, count=False)
        arms["zs+A+N"][s] = apply_priors(C, P[s], facts, angle=True, count=True)
        C2, X2, tm = rerun_z3(os.path.expanduser(specs[s]["scan_path"]), facts, d["z2_lines"])
        m = FM[f"zs|_s|{folds[s]}"]
        p2 = m.predict(X2) if len(C2) else np.zeros(0)
        arms["zs+T+F+A+N"][s] = apply_priors(C2, p2, facts, angle=True, count=True)
        e3 = e3_lines(s)
        arms["e3"][s] = e3 if e3 is not None else []
        arms["e3_else_none"][s] = e3 if e3 is not None else []
        arms["e3_else_zs"][s] = e3 if e3 is not None else arms["zs+T+F+A+N"][s]
        rt.append({"scan": s, "z3_default_s": d["stages"]["z3"]["wall"], "z3_prior_s": tm["load_s"] + tm["detect_s"],
                   "z2_s": d["stages"]["z2"]["wall"], "e3_valid": e3 is not None, **facts})
        print(s, facts["n_bends"], facts["K"], round(d["stages"]["z3"]["wall"], 2), round(tm["load_s"] + tm["detect_s"], 2),
              flush=True)
    res = {"schema": "sherman.zeroshot.zs_lane_v1.cad_prior_proto.v1", "cohort_scans": len(cohort), "refused": refused,
           "angle_window_deg": ANGLE_WIN, "arms": {}}
    for name, sel in arms.items():
        sub = {s: specs[s] for s in cohort}
        recs = [scoring.score_scan(sub[s], sel.get(s)) for s in cohort]
        agg = scoring.aggregate(recs, n_boot=2000)
        res["arms"][name] = {k: agg[k] for k in ("scans", "families", "gold", "tp", "fp", "fn", "P", "R", "micro_f1",
                                                  "family_macro_f1", "micro_ci95", "macro_ci95", "controls",
                                                  "fn_causes", "fp_causes")}
        print(name, agg["micro_f1"], agg["family_macro_f1"], agg["tp"], agg["fp"], agg["fn"], flush=True)
    z3d = np.array([r["z3_default_s"] for r in rt]); z3p = np.array([r["z3_prior_s"] for r in rt])
    z2 = np.array([r["z2_s"] for r in rt])
    q = lambda x: {"median_s": round(float(np.median(x)), 2), "p95_s": round(float(np.percentile(x, 95)), 2),  # noqa: E731
                   "max_s": round(float(x.max()), 2)}
    res["runtime"] = {"z3_default": q(z3d), "z3_with_priors": q(z3p), "zs_default(z3+z2)": q(z3d + z2),
                      "zs_with_priors(z3p+z2)": q(z3p + z2), "per_scan": rt}
    (out / "CAD_PRIOR_PROTO.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["runtime"], default=str)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
