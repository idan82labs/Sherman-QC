"""zs_lane_v2 DEV evaluation (out-of-fold, nested grouped CV) and final selector fit.

Inputs: a v2 DEV dump (dev_dump.py; same code path as the lane, with the v2 evidence). Folds: dev_folds.json
(5 family-group folds; the 4 newly frozen scans at their recorded fold).

Per outer fold f:
  * the GBM selector (z3_slicing.selector.GBM: HistGradientBoosting depth 3, lr 0.05, 300 it, l2 1, seed 0) is
    fit on the 4 training folds; candidate label = matches a gold line under the frozen gates at its snapped place;
  * the post-processing configuration (postproc.PostParams) is chosen by INNER grouped CV on the 4 training folds:
    inner out-of-fold probabilities (GBM fit on 3 of the 4 training folds, predict the 4th), max pooled micro-F1
    over the grid, ties broken toward the first (simplest) configuration;
  * the outer model + the chosen configuration are applied to fold f.
The threshold is the pre-registered fixed 0.5 (as v1). Paired family bootstrap vs v1 uses v1's frozen DEV OOF
per-case records (integration/oof_v1/DEV_OOF_per_case.json, arm zs|fixed|snap).

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v2.train --dump DIR --out DIR --feats base[,sweep]
          --grid NAME [--fit-final MODELDIR]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research.zeroshot.zs_lane_v2 import scoring
from research.zeroshot.zs_lane_v2.evidence import SWEEP_FEATS
from research.zeroshot.zs_lane_v2.postproc import PostParams, postprocess, features as _features, sweep_matrix
from research.zeroshot.z3_slicing.selector import GBM
from scripts.evaluate_f1_against_manual_gold import pair_metrics

FIXED_THR = 0.5
Z3_FEATS = 21
V1_PER_CASE = scoring.PROG / "integration/oof_v1/DEV_OOF_per_case.json"

GRIDS: dict[str, list[PostParams]] = {
    "none": [PostParams()],
    "split": [PostParams()] + [PostParams(split=True, split_min_gap=g) for g in (1.0, 2.0, 4.0)],
    "join": [PostParams(), PostParams(join=True), PostParams(join=True, join_low=0.2),
             PostParams(join=True, join_mode="notch", notch_cov=0.8),
             PostParams(join=True, join_mode="notch", notch_cov=0.8, join_low=0.2)],
    "sweep": [PostParams()] + [PostParams(sweep_min=x) for x in (0.25, 0.5)],
    "final": [PostParams(sweep_min=0.25)] + [PostParams(join=True, join_low=jl, sweep_min=0.25, dedupe_join=dd)
                                             for dd in (False, True) for jl in (1.0, 0.2, 0.0)],
    "j0sweep": [PostParams(join=True, join_low=0.0, sweep_min=x) for x in (0.0, 0.25, 0.5, 0.75)],
    "j0sweep25": [PostParams(join=True, join_low=0.0, sweep_min=0.25)],
    "j0": [PostParams(join=True, join_low=0.0)],
    "joinlow": [PostParams(), PostParams(join=True)] + [PostParams(join=True, join_low=x) for x in (0.2, 0.1, 0.05, 0.0)],
}
GRIDS["full"] = [PostParams()] + [PostParams(join=True, join_low=jl, split=sg > 0, split_min_gap=max(sg, 1.0),
                                             sweep_min=sm)
                                  for jl in (1.0, 0.2) for sg in (0.0, 1.0, 2.0) for sm in (0.0, 0.25, 0.5)]


def load_dumps(d: Path, ids: list[str], n_base: int = Z3_FEATS + 5) -> dict[str, dict]:
    out = {}
    for s in ids:
        p = d / f"{s}.json"
        if p.is_file():
            r = json.loads(p.read_text())
            n = len(r["candidates"])
            r["X"] = (np.array([[np.nan if x is None else x for x in row] for row in r["X"]], float).reshape(n, -1)
                      if n else np.zeros((0, n_base)))
            r["SW"] = sweep_matrix((r.get("evidence") or {}).get("sweep"), n)
            out[s] = r
    return out


def features(r: dict, feats: str) -> np.ndarray:
    return _features(r["X"], r["SW"], feats)


def gold_lines(spec: dict) -> list[dict]:
    return scoring.S.load_gold(spec["gold"], frame_records=None, locus="apex")["lines"]


def labels(cands: list[dict], gold: list[dict], key: str = "_s") -> np.ndarray:
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


class Scorer:
    """Cached frozen-matcher scoring of line sets per scan."""

    def __init__(self, specs):
        self.specs = specs
        self.cache: dict = {}

    def score(self, s: str, lines: list[dict] | None, diagnose: bool = False) -> dict:
        key = (s, diagnose, None if lines is None else tuple(
            tuple(np.round(np.r_[l["start"], l["end"]], 4)) for l in lines))
        if key not in self.cache:
            self.cache[key] = scoring.score_scan(self.specs[s], lines, diagnose=diagnose)
        return self.cache[key]

    def micro(self, per_scan: dict[str, list[dict]]) -> float:
        tp = fp = fn = 0
        for s, lines in per_scan.items():
            r = self.score(s, lines)
            if r.get("disposition") == "SCORED_COMPLETE_LABEL":
                tp += r["true_positive"]; fp += r["false_positive"]; fn += r["false_negative"]
        return scoring.prf(tp, fp, fn)[2]


def final_lines(r: dict, prob: np.ndarray, prm: PostParams, thr: float = FIXED_THR) -> list[dict]:
    out = postprocess(r["candidates"], prob, thr, r.get("evidence"), prm)
    return [{"start": l["start_s"], "end": l["end_s"], "op": l["op"], "parts": l["parts"]} for l in out]


def nested_oof(dumps, specs, folds, feats: str, grid: list[PostParams], sc: Scorer, log=print):
    ids = [s for s in specs if s in dumps]
    X = {s: features(dumps[s], feats) for s in ids}
    Y = {s: labels(dumps[s]["candidates"], gold_lines(specs[s])) for s in ids}
    F = sorted(set(folds[s] for s in ids))
    P, chosen, models = {}, {}, {}
    for f in F:
        tr = [s for s in ids if folds[s] != f]
        te = [s for s in ids if folds[s] == f]
        if len(grid) > 1:
            inner = {}
            for g in sorted(set(folds[s] for s in tr)):
                itr = [s for s in tr if folds[s] != g and len(X[s])]
                ite = [s for s in tr if folds[s] == g]
                m = GBM().fit(np.vstack([X[s] for s in itr]), np.concatenate([Y[s] for s in itr]))
                for s in ite:
                    inner[s] = m.predict(X[s]) if len(X[s]) else np.zeros(0)
            scores = []
            for prm in grid:
                scores.append(sc.micro({s: final_lines(dumps[s], inner[s], prm) for s in tr}))
            k = int(np.argmax(np.round(scores, 6)))
            chosen[f] = (k, grid[k], [round(x, 4) for x in scores])
            log(f"  fold {f}: inner-CV picks #{k} {grid[k].key()}  scores={[round(x, 4) for x in scores]}")
        else:
            chosen[f] = (0, grid[0], [])
        trf = [s for s in tr if len(X[s])]
        mdl = GBM().fit(np.vstack([X[s] for s in trf]), np.concatenate([Y[s] for s in trf]))
        models[f] = mdl
        for s in te:
            P[s] = mdl.predict(X[s]) if len(X[s]) else np.zeros(0)
    sel = {s: final_lines(dumps[s], P[s], chosen[folds[s]][1]) for s in ids}
    return sel, P, chosen, models


def evaluate(specs, sel: dict, sc: Scorer, n_boot=scoring.BOOT_N):
    recs = [sc.score(s, sel.get(s), diagnose=True) for s in specs]
    return scoring.aggregate(recs, n_boot=n_boot), recs


def v1_records() -> list[dict]:
    return json.loads(V1_PER_CASE.read_text())["zs|fixed|snap"]


def paired_bootstrap(recs_a: list[dict], recs_b: list[dict], n_boot=scoring.BOOT_N, seed=scoring.BOOT_SEED) -> dict:
    """b - a, family bootstrap (micro and family-macro), on the families scored in both."""
    def fam(recs):
        f = defaultdict(lambda: [0, 0, 0])
        for x in recs:
            if x.get("disposition") == "SCORED_COMPLETE_LABEL":
                v = f[x["family_id"]]
                v[0] += x["true_positive"]; v[1] += x["false_positive"]; v[2] += x["false_negative"]
        return f
    fa, fb = fam(recs_a), fam(recs_b)
    keys = sorted(set(fa) & set(fb))
    A = np.array([fa[k] for k in keys], float)
    B = np.array([fb[k] for k in keys], float)
    f1 = lambda v: scoring.prf(*v)[2]  # noqa: E731
    fA = np.array([f1(v) for v in A])
    fB = np.array([f1(v) for v in B])
    rng = np.random.default_rng(seed)
    dm, dM = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        dm[b] = f1(B[idx].sum(0)) - f1(A[idx].sum(0))
        dM[b] = fB[idx].mean() - fA[idx].mean()
    q = lambda x: [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]  # noqa: E731
    better = int((fB > fA + 1e-9).sum())
    worse = int((fB < fA - 1e-9).sum())
    return {"d_micro": round(f1(B.sum(0)) - f1(A.sum(0)), 4), "d_micro_ci95": q(dm),
            "d_macro": round(float(fB.mean() - fA.mean()), 4), "d_macro_ci95": q(dM),
            "families": len(keys), "families_better": better, "families_worse": worse,
            "changed": sorted([(k, round(float(fA[i]), 3), round(float(fB[i]), 3)) for i, k in enumerate(keys)
                               if abs(fB[i] - fA[i]) > 1e-9], key=lambda x: x[2] - x[1])}


def class_budget(recs: list[dict], specs: dict) -> dict:
    """Error classes of the TEST-B post-hoc analysis, counted on DEV: SPAN (FN/FP), MISSED, OFFSET, SPURIOUS,
    TAKEN, plus tiny-bend FNs (gold <= 10 mm) and empty-output line scans."""
    fn, fp = Counter(), Counter()
    tiny_fn = empty = 0
    for x in recs:
        if x.get("disposition") != "SCORED_COMPLETE_LABEL":
            continue
        for r in x.get("fn_rows") or []:
            fn[r["cause"].split("+")[0]] += 1
            if (r.get("gold_length_mm") or 99) <= 10.0:
                tiny_fn += 1
        for r in x.get("fp_rows") or []:
            fp[r["cause"].split("+")[0]] += 1
        if x.get("n_pred", x.get("prediction_count", 1)) == 0 and x.get("gold_count", 0) > 0:
            empty += 1
    return {"FN": dict(fn), "FP": dict(fp), "tiny_fn_le10mm": tiny_fn, "empty_line_scans": empty}


def slim(recs):
    keep = ("case_id", "family_id", "tier", "disposition", "gold_count", "prediction_count", "true_positive",
            "false_positive", "false_negative", "false_positive_lines", "fn_causes", "fp_causes", "fn_rows",
            "fp_rows", "output_missing_or_failed", "f1", "n_pred")
    return [{k: r[k] for k in keep if k in r} for r in recs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feats", default="base")
    ap.add_argument("--grid", default="none")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fit-final", default="")
    ap.add_argument("--final-post", default="", help="PostParams JSON for the final model record")
    ap.add_argument("--boot", type=int, default=scoring.BOOT_N)
    a = ap.parse_args(argv)
    specs = scoring.case_specs("dev")
    folds = scoring.dev_folds()
    dumps = load_dumps(Path(a.dump), list(specs))
    missing = [s for s in specs if s not in dumps]
    sc = Scorer(specs)
    tag = a.tag or f"{a.feats}|{a.grid}"
    print(f"== {tag}: dumps {len(dumps)} missing {missing}", flush=True)
    sel, P, chosen, models = nested_oof(dumps, specs, folds, a.feats, GRIDS[a.grid], sc)
    agg, recs = evaluate(specs, sel, sc, n_boot=a.boot)
    v1 = v1_records()
    pb = paired_bootstrap(v1, recs, n_boot=a.boot)
    ops = Counter(l["op"] for v in sel.values() for l in v)
    print(f"{tag}: P={agg['P']} R={agg['R']} F1={agg['micro_f1']} {agg.get('micro_ci95')} macro={agg['family_macro_f1']} "
          f"{agg.get('macro_ci95')} tp/fp/fn={agg['tp']}/{agg['fp']}/{agg['fn']} ctrlFP={agg['controls']['fp_lines']} ops={dict(ops)}",
          flush=True)
    print(f"   FN {agg['fn_causes']} FP {agg['fp_causes']}", flush=True)
    print(f"   vs v1: d_micro {pb['d_micro']} {pb['d_micro_ci95']} d_macro {pb['d_macro']} {pb['d_macro_ci95']} "
          f"better/worse {pb['families_better']}/{pb['families_worse']}", flush=True)
    print(f"   changed families: {pb['changed']}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    safe = tag.replace("|", "__").replace(",", "+")
    res = {"tag": tag, "feats": a.feats, "grid": a.grid, "dump": a.dump, "missing_dumps": missing, "agg": agg,
           "vs_v1": pb, "classes": class_budget(recs, specs), "classes_v1": class_budget(v1, specs),
           "chosen": {str(f): {"index": k, "params": p.key(), "inner_scores": s} for f, (k, p, s) in chosen.items()},
           "ops": dict(ops)}
    (out / f"OOF_{safe}.json").write_text(json.dumps(res, indent=1, default=str))
    (out / f"OOF_{safe}_per_case.json").write_text(json.dumps(slim(recs), indent=1, default=str))
    (out / f"OOF_{safe}_lines.json").write_text(json.dumps(sel, default=str))
    (out / f"OOF_{safe}_prob.json").write_text(json.dumps({s: np.round(p, 5).tolist() for s, p in P.items()}))
    if a.fit_final:
        ids = [s for s in specs if s in dumps and len(dumps[s]["candidates"])]
        X = np.vstack([features(dumps[s], a.feats) for s in ids])
        Y = np.concatenate([labels(dumps[s]["candidates"], gold_lines(specs[s])) for s in ids])
        mdl = GBM().fit(X, Y)
        md = Path(a.fit_final)
        md.mkdir(parents=True, exist_ok=True)
        pth = md / "zs_lane_v2_zs_gbm.pkl"
        pth.write_bytes(pickle.dumps({"model": mdl, "threshold": FIXED_THR, "variant": "zs", "feats": a.feats,
                                      "n_features": int(X.shape[1]), "post": json.loads(a.final_post or "{}"),
                                      "trained_on": f"DEV {len(ids)} scans (all folds), labels at snapped place",
                                      "rows": int(len(Y)), "positives": int(Y.sum())}))
        print(f"final model: {pth}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
