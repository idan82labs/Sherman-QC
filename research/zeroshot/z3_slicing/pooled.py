"""Evidence-gated union: Z3 candidates pooled with other lanes' lines (Z2, A0), one out-of-fold selector.

Every external line is attached to the Z3 candidate it duplicates (4 deg / 2.5 mm / 50 % of the shorter span);
that candidate gets the flags in_z2 / in_a0. External lines with no Z3 counterpart become extra candidates with
Z3 features missing (NaN; HistGradientBoosting handles them) and the flags src_z2 / src_a0. A GBM on the pooled
features is fit on 4 folds (family groups, dev_folds.json), its threshold chosen on the same 4 folds, applied to
the held-out fold. usage: python -m research.zeroshot.z3_slicing.pooled RUN_DIR --z2-tag it00 [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import custody, zscore, slicer  # noqa: E402
from research.zeroshot.z3_slicing.score_run import load_run, evaluate  # noqa: E402
from research.zeroshot.z3_slicing.selector import feats, FEATS_BASE as FEATS, GBM, labels  # noqa: E402
from research.zeroshot.z3_slicing.union import a0_lines, z2_lines  # noqa: E402

EXTRA = ["has_z3", "in_z2", "in_a0", "n_src", "ext_log_len"]


def pool_scan(d: dict, ext: dict[str, list[dict]]) -> tuple[list[dict], np.ndarray]:
    C = [dict(c) for c in d.get("candidates", [])]
    F = feats(d) if C else np.zeros((0, len(FEATS)))
    flags = np.zeros((len(C), len(EXTRA)))
    flags[:, 0] = 1
    for c in C:
        c["_src"] = {"z3"}
    extra_c, extra_f = [], []
    for name, lines in ext.items():
        col = EXTRA.index(f"in_{name}")
        for ln in lines:
            best, bj = None, -1
            for j, c in enumerate(C):
                ang, off, ov = slicer._pair(ln, c)
                if ang < 4.0 and off < 2.5 and ov > 0.5:
                    sc = ang / 4 + off / 2.5 - ov
                    if best is None or sc < best:
                        best, bj = sc, j
            if bj >= 0:
                flags[bj, col] = 1
                C[bj]["_src"].add(name)
                continue
            # external-only: dedupe against other external-only lines
            dup = False
            for k, c in enumerate(extra_c):
                ang, off, ov = slicer._pair(ln, c)
                if ang < 4.0 and off < 2.5 and ov > 0.5:
                    extra_f[k][EXTRA.index(f"in_{name}")] = 1
                    c["_src"].add(name)
                    dup = True
                    break
            if dup:
                continue
            L = float(np.linalg.norm(np.subtract(ln["end"], ln["start"])))
            row = np.zeros(len(EXTRA))
            row[col] = 1
            row[EXTRA.index("ext_log_len")] = math.log(L + 1)
            extra_c.append({"start": ln["start"], "end": ln["end"], "_src": {name}})
            extra_f.append(row)
    for j, c in enumerate(C):
        flags[j, EXTRA.index("n_src")] = len(c["_src"])
        flags[j, EXTRA.index("ext_log_len")] = math.log(c["length"] + 1)
    for k, c in enumerate(extra_c):
        extra_f[k][EXTRA.index("n_src")] = len(c["_src"])
    X = np.hstack([F, flags]) if len(C) else np.zeros((0, len(FEATS) + len(EXTRA)))
    if extra_c:
        Xe = np.hstack([np.full((len(extra_c), len(FEATS)), np.nan), np.array(extra_f)])
        X = np.vstack([X, Xe])
    return C + extra_c, X


def run(run_dir: str, z2_tag: str, use: tuple[str, ...], out_json: str = "") -> dict:
    inv = custody.dev_inventory()
    ids = custody.dev_ids()
    data = load_run(Path(run_dir), ids)
    fold = zscore.folds(inv, ids)
    P, X, Y = {}, {}, {}
    for s, d in data.items():
        ext = {}
        if "z2" in use:
            ext["z2"] = z2_lines(s, z2_tag)
        if "a0" in use:
            ext["a0"] = a0_lines(s, inv)
        C, Xs = pool_scan(d, ext)
        P[s] = {"candidates": C}
        X[s] = Xs
        Y[s] = labels(s, inv[s], P[s])
    pred, thr_used = {}, {}
    for f in sorted(set(fold.values())):
        tr = [s for s in data if fold[s] != f and len(X[s])]
        te = [s for s in data if fold[s] == f]
        mdl = GBM().fit(np.vstack([X[s] for s in tr]), np.concatenate([Y[s] for s in tr]))
        ptr = {s: mdl.predict(X[s]) for s in tr}
        best = (-1, 0.5)
        for thr in np.arange(0.15, 0.8, 0.025):
            agg, _ = evaluate({s: P[s] for s in tr}, inv,
                              lambda dd, s: [c for c, q in zip(dd["candidates"], ptr[s]) if q >= thr], diagnose=False)
            if agg["micro_f1"] > best[0]:
                best = (agg["micro_f1"], thr)
        thr_used[f] = round(float(best[1]), 3)
        for s in te:
            pred[s] = (mdl.predict(X[s]) if len(X[s]) else np.zeros(0), best[1])
    sel = {s: [c for c, q in zip(P[s]["candidates"], pred[s][0]) if q >= pred[s][1]] for s in data}
    agg, recs = evaluate(data, inv, lambda dd, s: sel[s])
    ci = zscore.bootstrap_ci(recs)
    name = "+".join(("z3",) + use)
    print(f"POOLED {name}: P={agg['P']:.3f} R={agg['R']:.3f} F1={agg['micro_f1']:.3f} {ci['micro_ci']} "
          f"macro={agg['family_macro_f1']:.3f} {ci['macro_ci']} tp={agg['tp']} fp={agg['fp']} fn={agg['fn']} thr={thr_used}")
    print("  FN", agg["fn_causes"], " FP", agg["fp_causes"], " controls", agg["controls"])
    res = {"pool": name, "z2_tag": z2_tag, "thresholds_by_fold": thr_used, **agg, **ci}
    if out_json:
        Path(out_json).write_text(json.dumps({"summary": res, "per_case": recs,
                                              "selected": {s: [{k: v for k, v in c.items() if k != "_src"} | {"src": sorted(c["_src"])}
                                                               for c in sel[s]] for s in sel}}, indent=1, default=str))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--z2-tag", default="it00")
    ap.add_argument("--use", default="z2,a0")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    run(a.run, a.z2_tag, tuple(x for x in a.use.split(",") if x), a.json)
