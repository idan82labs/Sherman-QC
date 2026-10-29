"""zs_lane_v1 DEV evaluation (out-of-fold) and final selector fit.

Inputs: the DEV dump (dev_dump.py; same code path as the lane). Folds: dev_folds.json (5 family-group folds; the 4
newly frozen scans take their recorded pending fold). Variants:
  z3 : Z3 candidates, 21 Z3 features, GBM                       (Z3 alone)
  zs : Z3 candidates + Z2-only lines, 21 Z3 + 5 pool features, GBM (integrated)
  z2 : Z2 lines as emitted (no selector)                         (Z2 alone)
Selector: z3_slicing.selector.GBM (HistGradientBoosting depth 3, lr 0.05, 300 iterations, l2 1, seed 0). Candidate
label = matches a gold line under the frozen gates (5 deg / 4 mm / 0.6 of both spans) at its final (snapped) place.
Threshold rules compared: 'fixed' (0.5, no tuning) and 'tuned' (per fold, max micro-F1 on the 4 training folds over
0.15..0.8 step 0.025, the Z3 WP rule). Placement: 'snap' (D1 gated outer-apex snap, the lane) and 'raw' (ablation).

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v1.train --dump DIR --out DIR [--fit-final MODELDIR]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

from research.zeroshot.zs_lane_v1 import scoring
from research.zeroshot.zs_lane_v1.lane import Z3_FEATS
from research.zeroshot.z3_slicing.selector import GBM
from scripts.evaluate_f1_against_manual_gold import pair_metrics

THR_GRID = np.round(np.arange(0.15, 0.8, 0.025), 3)
FIXED_THR = 0.5


def load_dumps(d: Path, ids: list[str]) -> dict[str, dict]:
    out = {}
    for s in ids:
        p = d / f"{s}.json"
        if p.is_file():
            r = json.loads(p.read_text())
            r["X"] = np.array([[np.nan if x is None else x for x in row] for row in r["X"]], float).reshape(
                len(r["candidates"]), -1) if r["candidates"] else np.zeros((0, Z3_FEATS + 5))
            out[s] = r
    return out


def gold_lines(spec: dict) -> list[dict]:
    return scoring.S.load_gold(spec["gold"], frame_records=None, locus="apex")["lines"]


def labels(cands: list[dict], gold: list[dict], key: str) -> np.ndarray:
    y = np.zeros(len(cands))
    for j, c in enumerate(cands):
        pe = np.array([c["start" + key], c["end" + key]])
        for g in gold:
            m = pair_metrics(np.array([g["line_start"], g["line_end"]]), pe)
            if (m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 4 and m["gold_span_overlap"] >= 0.6
                    and m["prediction_span_overlap"] >= 0.6):
                y[j] = 1
                break
    return y


def view(r: dict, variant: str) -> tuple[list[dict], np.ndarray]:
    C, X = r["candidates"], r["X"]
    if variant == "z3":
        keep = [i for i, c in enumerate(C) if c["has_z3"]]
        return [C[i] for i in keep], (X[keep][:, :Z3_FEATS] if keep else np.zeros((0, Z3_FEATS)))
    return C, X


def lines_of(C, prob, thr, key):
    return [{"start": c["start" + key], "end": c["end" + key]} for c, q in zip(C, prob) if q >= thr]


def micro(specs, ids, sel_fn):
    tp = fp = fn = 0
    for s in ids:
        r = scoring.score_scan(specs[s], sel_fn(s), diagnose=False)
        if r.get("disposition") == "SCORED_COMPLETE_LABEL":
            tp += r["true_positive"]; fp += r["false_positive"]; fn += r["false_negative"]
    return scoring.prf(tp, fp, fn)[2]


FOLD_MODELS: dict = {}


def oof(dumps, specs, folds, variant, key):
    """Out-of-fold probabilities per scan (+ the per-fold tuned threshold). Fold models kept in FOLD_MODELS."""
    ids = list(specs)
    data = {s: view(dumps[s], variant) for s in ids if s in dumps}
    Y = {s: labels(data[s][0], gold_lines(specs[s]), key) for s in data}
    P, tuned = {}, {}
    for f in sorted(set(folds[s] for s in ids)):
        tr = [s for s in data if folds[s] != f and len(data[s][0])]
        te = [s for s in data if folds[s] == f]
        mdl = GBM().fit(np.vstack([data[s][1] for s in tr]), np.concatenate([Y[s] for s in tr]))
        FOLD_MODELS[(variant, key, f)] = mdl
        ptr = {s: mdl.predict(data[s][1]) for s in tr}
        best = (-1.0, FIXED_THR)
        for thr in THR_GRID:
            f1 = micro(specs, tr, lambda s: lines_of(data[s][0], ptr[s], thr, key))
            if f1 > best[0] + 1e-12:
                best = (f1, float(thr))
        tuned[f] = best[1]
        for s in te:
            P[s] = mdl.predict(data[s][1]) if len(data[s][0]) else np.zeros(0)
    return data, P, tuned, Y


def evaluate(specs, per_scan_lines, n_boot=scoring.BOOT_N):
    recs = [scoring.score_scan(specs[s], per_scan_lines.get(s)) for s in specs]
    return scoring.aggregate(recs, n_boot=n_boot), recs


def runtime(dumps, variant, n_sel):
    """Per-scan lane runtime estimate from the dump stages: detection stages of the variant + pooling + snap of the
    selected lines (snap time per candidate measured in the dump)."""
    out = []
    for s, r in dumps.items():
        st = r.get("stages", {})
        z3 = st.get("z3", {}).get("wall", 0.0)
        z2 = st.get("z2", {}).get("wall", 0.0)
        sa = st.get("snap_all", {})
        per = sa.get("wall", 0.0) / max(sa.get("n", 1), 1)
        if variant == "z2":
            out.append(z2)
        elif variant == "z3":
            out.append(z3 + per * n_sel.get(s, 0))
        else:
            out.append(z3 + z2 + st.get("pool", {}).get("wall", 0.0) + per * n_sel.get(s, 0))
    a = np.array(out)
    return {"median_s": round(float(np.median(a)), 2), "mean_s": round(float(a.mean()), 2),
            "p95_s": round(float(np.percentile(a, 95)), 2), "max_s": round(float(a.max()), 2),
            "note": "single thread (OMP 1), excludes interpreter start-up and the sha256 of the scan"}


def slim(recs):
    keep = ("case_id", "family_id", "tier", "disposition", "gold_count", "prediction_count", "true_positive",
            "false_positive", "false_negative", "false_positive_lines", "fn_causes", "fp_causes", "fn_rows",
            "fp_rows", "output_missing_or_failed", "f1")
    return [{k: r[k] for k in keep if k in r} for r in recs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-final", default="")
    ap.add_argument("--variants", default="z3,zs")
    a = ap.parse_args(argv)
    specs = scoring.case_specs("dev")
    folds = scoring.dev_folds()
    dumps = load_dumps(Path(a.dump), list(specs))
    missing = [s for s in specs if s not in dumps]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"schema": "sherman.zeroshot.zs_lane_v1.dev_oof.v1", "cohort": {
        "scans": len(specs), "dumped": len(dumps), "missing_dump_scored_as_empty": missing,
        "folds": "dev_folds.json (5 family-group folds; 4 pending scans at their recorded fold)"}, "arms": {}}
    per_case = {}
    # Z2 alone (no selector)
    z2l = {s: [{"start": l["start"], "end": l["end"]} for l in dumps[s]["z2_lines"]] for s in dumps
           if "z2" not in dumps[s].get("errors", {})}
    agg, recs = evaluate(specs, z2l)
    agg["runtime"] = runtime(dumps, "z2", {})
    res["arms"]["z2|none|z2snap"] = agg
    per_case["z2|none|z2snap"] = slim(recs)
    print(f"z2 alone: {agg['micro_f1']} / {agg['family_macro_f1']} {agg.get('micro_ci95')} ctrlFP={agg['controls']['fp_lines']}",
          flush=True)
    for variant in a.variants.split(","):
        for key, kname in (("_s", "snap"), ("", "raw")):
            data, P, tuned, Y = oof(dumps, specs, folds, variant, key)
            for rule in ("fixed", "tuned"):
                thr_of = (lambda s: FIXED_THR) if rule == "fixed" else (lambda s: tuned[folds[s]])
                sel = {s: lines_of(data[s][0], P[s], thr_of(s), key) for s in data}
                agg, recs = evaluate(specs, sel, n_boot=scoring.BOOT_N if (kname == "snap") else 2000)
                agg["thresholds"] = {"rule": rule, "by_fold": tuned if rule == "tuned" else FIXED_THR,
                                     "median_tuned": float(np.median(list(tuned.values())))}
                agg["runtime"] = runtime(dumps, variant, {s: len(v) for s, v in sel.items()})
                name = f"{variant}|{rule}|{kname}"
                res["arms"][name] = agg
                per_case[name] = slim(recs)
                print(f"{name}: P={agg['P']} R={agg['R']} F1={agg['micro_f1']} {agg.get('micro_ci95')} "
                      f"macro={agg['family_macro_f1']} {agg.get('macro_ci95')} tp/fp/fn={agg['tp']}/{agg['fp']}/{agg['fn']} "
                      f"ctrlFP={agg['controls']['fp_lines']} thr={agg['thresholds']['by_fold']}", flush=True)
                print(f"   FN {agg['fn_causes']} FP {agg['fp_causes']}", flush=True)
    (out / "DEV_OOF.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "fold_models.pkl").write_bytes(pickle.dumps({f"{v}|{k or 'raw'}|{f}": m for (v, k, f), m in FOLD_MODELS.items()}))
    (out / "DEV_OOF_per_case.json").write_text(json.dumps(per_case, indent=1, default=str))
    if a.fit_final:
        md = Path(a.fit_final)
        md.mkdir(parents=True, exist_ok=True)
        for variant in a.variants.split(","):
            data = {s: view(dumps[s], variant) for s in specs if s in dumps}
            Y = {s: labels(data[s][0], gold_lines(specs[s]), "_s") for s in data}
            tr = [s for s in data if len(data[s][0])]
            mdl = GBM().fit(np.vstack([data[s][1] for s in tr]), np.concatenate([Y[s] for s in tr]))
            pth = md / f"zs_lane_v1_{variant}_gbm.pkl"
            pth.write_bytes(pickle.dumps({"model": mdl, "threshold": FIXED_THR, "variant": variant,
                                          "n_features": int(data[tr[0]][1].shape[1]),
                                          "trained_on": f"DEV {len(tr)} scans (all folds), labels at snapped place",
                                          "rows": int(sum(len(Y[s]) for s in tr)),
                                          "positives": int(sum(Y[s].sum() for s in tr))}))
            print(f"final {variant}: {pth}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
