"""zs_lane_v3 DEV evaluation (out-of-fold on dev_folds.json) and final model fit.

Inputs: a v3 DEV dump (dev_dump.py, run through the gold guard and the geometry slot). Folds: dev_folds.json.

Per outer fold f (exactly zs_lane_v2's honest procedure, reused from zs_lane_v2/train.py):
  * the pooled GBM (21 Z3 + 5 pool features) is fit on the 4 training folds;
  * the v2 post-processing setting is chosen by inner grouped CV over v2's 'final' grid (ties -> first);
  * NEW in v3: the Z3-alone GBM (21 Z3 features, has_z3 rows, the zs_lane_v1 z3 recipe) is fit on the same 4
    training folds and predicts fold f (Z3R input);
  * fold f's lines = postprocess(outer probabilities, chosen setting) -> EXT -> Z3R with the frozen v3 FixParams.
The EXT / Z3R parameters themselves were set on all-DEV OOF prototypes (newscans_N1/diag_47085000), not nested:
the v3 DEV numbers are selection-aware for those 4 values (see README.md).

Arms written (P/v3/oof/OOF_<arm>.json, _per_case.json, _lines.json):
  v3            primary: nested v2 procedure + EXT + Z3R
  v2_repro      the v2 base reproduced from the v3 dump (must equal v2/oof/OOF_base__final)
  v3_ext_only, v3_z3r_only            ablations
  v3_min_in_<x> EXT containment sweep (disclosure of the DEV plateau)
  v3_frozen_post  the frozen v2 post setting for every fold + fixes (selection-biased, diagnostic)
  v1base_fixes  the fixes on zs_lane_v1's frozen OOF selection (prototype cross-check)
Paired family bootstraps (10 000 resamples, seed 20261025) against v1's and v2's frozen DEV OOF per-case records.

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3.train --dump DIR --out DIR [--fit-final MODELDIR]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from research.zeroshot.zs_lane_v2 import scoring
from research.zeroshot.zs_lane_v2 import train as T2
from research.zeroshot.zs_lane_v2.postproc import PostParams
from research.zeroshot.zs_lane_v3 import core
from research.zeroshot.zs_lane_v3.fixes import FixParams
from research.zeroshot.z3_slicing.selector import GBM

PROG = scoring.PROG
V1_PER_CASE = PROG / "integration/oof_v1/DEV_OOF_per_case.json"
V2_PER_CASE = PROG / "v2/oof/OOF_base__final_per_case.json"
V2_CONFIG = core.REPO / "research/zeroshot/zs_lane_v2/config_v2.json"
FROZEN_V2_POST = PostParams(**json.loads(V2_CONFIG.read_text())["variants"]["zs"]["post"])
FIX = FixParams()
Z3 = core.Z3_FEATS


def z3_rows(r: dict) -> list[int]:
    return [i for i, c in enumerate(r["candidates"]) if c.get("has_z3")]


def z3_fold_probs(dumps, ids, folds, Y):
    """Out-of-fold Z3-alone probabilities (NaN for candidates without Z3 features) and the fold models."""
    P3, models = {}, {}
    for f in sorted(set(folds[s] for s in ids)):
        tr = [s for s in ids if folds[s] != f and z3_rows(dumps[s])]
        X = np.vstack([dumps[s]["X"][z3_rows(dumps[s])][:, :Z3] for s in tr])
        y = np.concatenate([Y[s][z3_rows(dumps[s])] for s in tr])
        m = GBM().fit(X, y)
        models[f] = m
        for s in ids:
            if folds[s] == f:
                p = np.full(len(dumps[s]["candidates"]), np.nan)
                rows = z3_rows(dumps[s])
                if rows:
                    p[rows] = m.predict(dumps[s]["X"][rows][:, :Z3])
                P3[s] = p
    return P3, models


def lines_for(r: dict, prob, prob_z3, post: PostParams, fix: FixParams) -> list[dict]:
    v = {"threshold": T2.FIXED_THR, "post": asdict(post), "fix": asdict(fix)}
    out, _, _ = core.select(r["candidates"], r["X"], r.get("evidence"), r.get("z2_lines") or [], v, None, None,
                            prob=prob, prob_z3=prob_z3)
    return out


def as_eval(lines: list[dict]) -> list[dict]:
    return [{"start": l["start_s"], "end": l["end_s"], "op": l.get("op"), "fix": l.get("fix"),
             "parts": l.get("parts")} for l in lines]


def per_scan_fp(recs: list[dict], ids: set[str]) -> dict[str, int]:
    out = {}
    for x in recs:
        if x["case_id"] in ids:
            out[x["case_id"]] = int(x.get("false_positive_lines", x.get("false_positive", 0)) or 0)
    return out


def rolled_dev(specs) -> list[str]:
    inv = {x["scan_id"]: x for x in json.loads((PROG / "INVENTORY.json").read_text())["scans"]}
    return sorted(s for s in specs if "ROLLED" in str(((inv.get(s) or {}).get("cad") or {}).get("compile_reason") or ""))


def v1_base_lines() -> dict[str, list[dict]]:
    """zs_lane_v1's frozen DEV OOF selection (p >= 0.5 on the snapped candidates of the v1 dump)."""
    oof = pickle.loads((PROG / "integration/oof_v1/zs_snap_oof_probs.pkl").read_bytes())["P"]
    out = {}
    for s, p in oof.items():
        d = PROG / "integration/dev_dump" / f"{s}.json"
        if not d.is_file():
            continue
        C = json.loads(d.read_text())["candidates"]
        out[s] = [{"start_s": c["start_s"], "end_s": c["end_s"], "score": float(q), "src": c.get("src", []),
                   "incl": c.get("incl"), "parts": [i], "op": "keep"} for i, (c, q) in enumerate(zip(C, p)) if q >= 0.5]
    return out


def main(argv=None) -> int:
    from research.zeroshot.zs_lane_v3.fixes import apply
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-final", default="")
    ap.add_argument("--boot", type=int, default=scoring.BOOT_N)
    a = ap.parse_args(argv)
    specs = scoring.case_specs("dev")
    folds = scoring.dev_folds()
    dumps = T2.load_dumps(Path(a.dump), list(specs))
    missing = [s for s in specs if s not in dumps]
    ids = [s for s in specs if s in dumps]
    sc = T2.Scorer(specs)
    print(f"dumps {len(dumps)} missing {missing}", flush=True)
    t0 = time.time()
    sel_v2, P, chosen, _ = T2.nested_oof(dumps, specs, folds, "base", T2.GRIDS["final"], sc)
    Y = {s: T2.labels(dumps[s]["candidates"], T2.gold_lines(specs[s])) for s in ids}
    P3, z3_models = z3_fold_probs(dumps, ids, folds, Y)
    print(f"OOF models fit in {time.time() - t0:.0f}s", flush=True)

    # cross-check: the v3 Z3-alone fold models vs zs_lane_v1's frozen z3 fold models
    fm1 = pickle.loads((PROG / "integration/oof_v1/fold_models.pkl").read_bytes())
    z3_dev = []
    for s in ids:
        rows = z3_rows(dumps[s])
        if rows:
            p1 = fm1[f"z3|_s|{folds[s]}"].predict(dumps[s]["X"][rows][:, :Z3])
            z3_dev.append(float(np.max(np.abs(p1 - P3[s][rows]))))
    z3_check = {"max_abs_diff_vs_v1_z3_fold_models": max(z3_dev) if z3_dev else None}
    print("z3 fold models vs v1:", z3_check, flush=True)

    post_of = {s: chosen[folds[s]][1] for s in ids}
    arms: dict[str, dict[str, list[dict]]] = {}
    sel_time = []
    arms["v3"] = {}
    for s in ids:
        t1 = time.perf_counter()
        arms["v3"][s] = lines_for(dumps[s], P[s], P3[s], post_of[s], FIX)
        sel_time.append(time.perf_counter() - t1)
    arms["v2_repro"] = {s: lines_for(dumps[s], P[s], P3[s], post_of[s], replace(FIX, ext=False, z3r=False)) for s in ids}
    arms["v3_ext_only"] = {s: lines_for(dumps[s], P[s], P3[s], post_of[s], replace(FIX, z3r=False)) for s in ids}
    arms["v3_z3r_only"] = {s: lines_for(dumps[s], P[s], P3[s], post_of[s], replace(FIX, ext=False)) for s in ids}
    for x in (0.5, 0.6, 0.75, 0.9):
        arms[f"v3_min_in_{x}"] = {s: lines_for(dumps[s], P[s], P3[s], post_of[s], replace(FIX, ext_min_in=x))
                                  for s in ids}
    arms["v3_frozen_post"] = {s: lines_for(dumps[s], P[s], P3[s], FROZEN_V2_POST, FIX) for s in ids}
    v1b = v1_base_lines()
    arms["v1base_fixes"] = {}
    for s in ids:
        r = dumps[s]
        base = v1b.get(s, [])
        if len(json.loads((PROG / "integration/dev_dump" / f"{s}.json").read_text())["candidates"]) != len(r["candidates"]):
            raise RuntimeError(f"{s}: v1 and v3 dumps differ in candidate count")
        arms["v1base_fixes"][s] = apply(base, r["candidates"], r.get("z2_lines") or [],
                                        np.zeros(len(r["candidates"])), P3[s],
                                        (r.get("evidence") or {}).get("sweep") if r.get("evidence") else None, FIX)
    v1base = {s: v1b.get(s, []) for s in ids}

    v1 = json.loads(V1_PER_CASE.read_text())["zs|fixed|snap"]
    v2 = json.loads(V2_PER_CASE.read_text())
    ctrl_ids = {x["case_id"] for x in v1 if x.get("disposition") == "ZERO_BEND_CONTROL"}
    rolled = set(rolled_dev(specs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"dump": a.dump, "missing_dumps": missing, "fix_params": asdict(FIX), "z3_check": z3_check,
               "chosen_post": {str(f): {"index": k, "params": p.key(), "inner_scores": sc_} for f, (k, p, sc_) in chosen.items()},
               "control_scans": sorted(ctrl_ids), "rolled_dev_scans": sorted(rolled), "arms": {}}
    recs_of = {}
    for name, sel in list(arms.items()) + [("v1base_repro", v1base)]:
        n_boot = a.boot if name in ("v3", "v2_repro", "v3_frozen_post", "v1base_fixes") else 2000
        ev = {s: as_eval(sel[s]) for s in sel}
        agg, recs = T2.evaluate(specs, ev, sc, n_boot=n_boot)
        recs_of[name] = recs
        res = {"agg": agg, "vs_v1": T2.paired_bootstrap(v1, recs, n_boot=n_boot),
               "vs_v2": T2.paired_bootstrap(v2, recs, n_boot=n_boot),
               "classes": T2.class_budget(recs, specs),
               "fixes": dict(Counter(l.get("fix") for v in sel.values() for l in v if l.get("fix"))),
               "control_fp_lines": per_scan_fp(recs, ctrl_ids), "rolled_fp": per_scan_fp(recs, rolled)}
        summary["arms"][name] = res
        print(f"{name}: F1={agg['micro_f1']} {agg.get('micro_ci95')} macro={agg['family_macro_f1']} "
              f"tp/fp/fn={agg['tp']}/{agg['fp']}/{agg['fn']} ctrlFP={agg['controls']['fp_lines']} fixes={res['fixes']} "
              f"| vs v1 {res['vs_v1']['d_micro']} {res['vs_v1']['d_micro_ci95']} {res['vs_v1']['families_better']}/"
              f"{res['vs_v1']['families_worse']} | vs v2 {res['vs_v2']['d_micro']} {res['vs_v2']['d_micro_ci95']} "
              f"{res['vs_v2']['families_better']}/{res['vs_v2']['families_worse']}", flush=True)
        if name in ("v3", "v2_repro", "v3_frozen_post", "v1base_fixes"):
            (out / f"OOF_{name}_per_case.json").write_text(json.dumps(T2.slim(recs), indent=1, default=str))
            (out / f"OOF_{name}_lines.json").write_text(json.dumps(ev, default=str))
    # the v1-base arm vs its own reproduced base (prototype quoted the fixes on v1 against v1's records)
    summary["arms"]["v1base_fixes"]["vs_v1base_repro"] = T2.paired_bootstrap(recs_of["v1base_repro"],
                                                                            recs_of["v1base_fixes"], n_boot=a.boot)
    summary["controls_and_rolled"] = {
        "v1_frozen": {"control_fp_lines": per_scan_fp(v1, ctrl_ids), "rolled_fp": per_scan_fp(v1, rolled)},
        "v2_frozen": {"control_fp_lines": per_scan_fp(v2, ctrl_ids), "rolled_fp": per_scan_fp(v2, rolled)},
        "v3": {"control_fp_lines": summary["arms"]["v3"]["control_fp_lines"],
               "rolled_fp": summary["arms"]["v3"]["rolled_fp"]}}
    added = {s: [l for l in arms["v3"][s] if l.get("fix")] for s in ids}
    summary["fix_events"] = {s: [{"fix": l["fix"], "parts": l["parts"], "ext_mm": l.get("ext_mm"), "p_z3": l.get("p_z3"),
                                  "len_mm": round(float(np.linalg.norm(np.subtract(l["end_s"], l["start_s"]))), 2)}
                                 for l in v] for s, v in added.items() if v}
    v2r = {s: [(tuple(np.round(l["start_s"], 4)), tuple(np.round(l["end_s"], 4))) for l in arms["v2_repro"][s]] for s in ids}
    v2f = json.loads((PROG / "v2/oof/OOF_base__final_lines.json").read_text())
    v2f = {s: [(tuple(np.round(l["start"], 4)), tuple(np.round(l["end"], 4))) for l in v2f.get(s, [])] for s in ids}
    summary["v2_repro_identical_scans"] = sum(1 for s in ids if v2r[s] == v2f[s])
    summary["select_fix_time_s"] = {"median": float(np.median(sel_time)), "p95": float(np.percentile(sel_time, 95)),
                                    "max": float(np.max(sel_time))}
    (out / "OOF_SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "OOF_v3_prob.json").write_text(json.dumps({s: {"p_zs": np.round(P[s], 5).tolist(),
                                                          "p_z3": [None if x != x else round(float(x), 5) for x in P3[s]]}
                                                      for s in ids}))
    print("v2 base reproduced identically on", summary["v2_repro_identical_scans"], "of", len(ids), "scans", flush=True)
    if a.fit_final:
        md = Path(a.fit_final)
        md.mkdir(parents=True, exist_ok=True)
        fit = [s for s in ids if len(dumps[s]["candidates"])]
        X = np.vstack([T2.features(dumps[s], "base") for s in fit])
        Yall = np.concatenate([Y[s] for s in fit])
        mz = GBM().fit(X, Yall)
        (md / "zs_lane_v3_zs_gbm.pkl").write_bytes(pickle.dumps(
            {"model": mz, "threshold": T2.FIXED_THR, "variant": "zs", "feats": "base", "n_features": int(X.shape[1]),
             "trained_on": f"DEV {len(fit)} scans (all folds), labels at snapped place (zs_lane_v2 recipe)",
             "rows": int(len(Yall)), "positives": int(Yall.sum())}))
        f3 = [s for s in ids if z3_rows(dumps[s])]
        X3 = np.vstack([dumps[s]["X"][z3_rows(dumps[s])][:, :Z3] for s in f3])
        Y3 = np.concatenate([Y[s][z3_rows(dumps[s])] for s in f3])
        m3 = GBM().fit(X3, Y3)
        (md / "zs_lane_v3_z3_gbm.pkl").write_bytes(pickle.dumps(
            {"model": m3, "threshold": T2.FIXED_THR, "variant": "z3", "feats": "z3_21", "n_features": Z3,
             "trained_on": f"DEV {len(f3)} scans (all folds), has_z3 rows, labels at snapped place (zs_lane_v1 z3 recipe)",
             "rows": int(len(Y3)), "positives": int(Y3.sum())}))
        # equivalence with the frozen v2 zs model and the frozen v1 z3 model on every DEV row
        m2 = pickle.loads((PROG / "v2/models/zs_lane_v2_zs_gbm.pkl").read_bytes())["model"]
        m1 = pickle.loads((PROG / "integration/models/zs_lane_v1_z3_gbm.pkl").read_bytes())["model"]
        eq = {"zs_vs_v2_frozen_max_abs": float(np.max(np.abs(m2.predict(X) - mz.predict(X)))),
              "z3_vs_v1_frozen_max_abs": float(np.max(np.abs(m1.predict(X3) - m3.predict(X3)))),
              "rows_zs": int(len(Yall)), "rows_z3": int(len(Y3))}
        (md / "FINAL_MODELS.json").write_text(json.dumps(eq, indent=1))
        print("final models:", md, eq, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
