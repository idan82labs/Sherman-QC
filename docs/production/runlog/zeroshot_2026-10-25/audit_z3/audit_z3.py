"""Independent audit of the Z3 claim. Read-only on Z3 code/results; writes to audit_z3/.
Never reads gold files whose path contains _SEALED (those 18 DEV scans are excluded from independent checks)."""
import glob, json, re, sys, collections, hashlib
from pathlib import Path
import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
for p in (REPO / "research/broad_baseline", REPO, REPO / "backend"):
    sys.path.insert(0, str(p))
import score as S  # frozen scorer, unmodified
import bb_common as bb
from scripts.evaluate_f1_against_manual_gold import pair_metrics
from research.zeroshot.z3_slicing.selector import feats  # read: candidate-only geometric features

RL = REPO / "docs/production/runlog/zeroshot_2026-10-25"
Z3 = RL / "z3"
OUT = RL / "audit_z3"
W = Path("/private/tmp/claude-501/z3work")
sp = json.loads((RL / "SPLIT_FROZEN.json").read_text())
DEV = sp["dev"]["scans"]; TB = set(sp["test_b"]["scans"])
INV = {r["scan_id"]: r for r in json.loads((RL / "INVENTORY.json").read_text())["scans"] if r["scan_id"] in set(DEV)}
FOLD = json.loads((RL / "dev_folds.json").read_text())["scan_fold"]
SEALED = {s for s in DEV if "sealed" in INV[s]["gold"]["path"].lower()}
OPEN = [s for s in DEV if s not in SEALED]
THR, MATCH = S._matcher()
report = {}


def case(s, pred=None):
    g = INV[s]["gold"]
    c = {"case_id": s, "family_id": INV[s]["family"], "tier": g["set"],
         "gold": {"kind": "ai_studio", "path": g["path"]}, "label_complete": g.get("label_complete", True)}
    if pred:
        c["predictions"] = str(pred)
    return c


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return p, r, 2 * p * r / (p + r) if p + r else 0


def agg(recs):
    rows = [r for r in recs if r["disposition"] == "SCORED_COMPLETE_LABEL"]
    tp = sum(r["true_positive"] for r in rows); fp = sum(r["false_positive"] for r in rows); fn = sum(r["false_negative"] for r in rows)
    fam = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        f = fam[r["family_id"]]; f[0] += r["true_positive"]; f[1] += r["false_positive"]; f[2] += r["false_negative"]
    ctrl = [r for r in recs if r["disposition"] == "ZERO_BEND_CONTROL"]
    p, rr, f1 = prf(tp, fp, fn)
    return {"scans": len(rows), "tp": tp, "fp": fp, "fn": fn, "gold": tp + fn, "P": round(p, 4), "R": round(rr, 4),
            "micro_f1": round(f1, 4), "family_macro_f1": round(float(np.mean([prf(*v)[2] for v in fam.values()])), 4) if fam else None,
            "controls": len(ctrl), "control_fp": sum(r["false_positive_lines"] for r in ctrl),
            "other_dispositions": collections.Counter(r["disposition"] for r in recs if r["disposition"] not in ("SCORED_COMPLETE_LABEL", "ZERO_BEND_CONTROL"))}

# ---------------------------------------------------------------- 1. eval_lines == OOF selections, thresholds per fold
oof = json.loads((Z3 / "oof_it7_gbm.json").read_text())
tbf = {int(k): v for k, v in oof["summary"]["thresholds_by_fold"].items()}
mism, thr_bad, path_bad = [], [], []
for s in DEV:
    ev = json.loads((Z3 / "runs/it7" / s / "eval_lines.json").read_text())
    a = sorted([tuple(np.round(r["line_start"] + r["line_end"], 4)) for r in ev["owned_bend_regions"]])
    b = sorted([tuple(np.round(list(c["start"]) + list(c["end"]), 4)) for c in oof["selected"][s]])
    if a != b:
        mism.append(s)
    if abs(ev["conversion"]["threshold"] - tbf[FOLD[s]]) > 1e-6:
        thr_bad.append((s, ev["conversion"]["threshold"], tbf[FOLD[s]]))
    if Path(ev["scan"]["path"]).resolve() != Path(INV[s]["path"]).expanduser().resolve():
        path_bad.append(s)
report["1_eval_equals_oof_selected"] = {"mismatch": mism, "threshold_not_equal_fold_threshold": thr_bad,
                                        "scan_path_not_inventory": path_bad, "n": len(DEV)}

# ---------------------------------------------------------------- 2. independent rescoring of runs/it7 eval_lines
recs = {}
for s in OPEN:
    recs[s] = S.score_case(case(s, Z3 / "runs/it7" / s / "eval_lines.json"), frame_records=None)
theirs = {r["case_id"]: r for r in oof["per_case"]}
diff = [s for s in OPEN if recs[s]["disposition"] == "SCORED_COMPLETE_LABEL" and
        (recs[s]["true_positive"], recs[s]["false_positive"], recs[s]["false_negative"]) !=
        (theirs[s]["true_positive"], theirs[s]["false_positive"], theirs[s]["false_negative"])]
mine_open = agg(list(recs.values()))
their_open = agg([theirs[s] for s in OPEN])
their_sealed = agg([theirs[s] for s in SEALED])
comb = {k: mine_open[k] + their_sealed[k] for k in ("tp", "fp", "fn")}
report["2_rescore"] = {"open_scans_mine": mine_open, "open_scans_theirs": their_open, "per_scan_count_diffs": diff,
                       "sealed_scans_theirs_not_reverified": their_sealed, "n_sealed": len(SEALED),
                       "combined_micro_f1": round(prf(comb["tp"], comb["fp"], comb["fn"])[2], 4), "combined": comb,
                       "gold_lines_open_from_inventory": sum(INV[s]["gold"]["n_lines"] for s in OPEN)}

# ---------------------------------------------------------------- 3. own OOF re-runs (open scans only)
GOLD = {}
for s in OPEN:
    g = S.load_gold(case(s)["gold"], frame_records=None, locus="apex")
    GOLD[s] = g
LINE = [s for s in OPEN if not GOLD[s]["zero_bend_control"]]
CTRL = [s for s in OPEN if GOLD[s]["zero_bend_control"]]


def load_cands(tag):
    D = {}
    for s in OPEN:
        p = W / tag / f"{s}.json"
        D[s] = json.loads(p.read_text()) if p.is_file() else {"candidates": []}
    return D


def labels(s, C):
    y = np.zeros(len(C))
    for j, c in enumerate(C):
        pe = np.array([c["start"], c["end"]])
        for g in GOLD[s]["lines"]:
            m = pair_metrics(np.array([g["line_start"], g["line_end"]]), pe)
            if m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 4 and m["gold_span_overlap"] >= .6 and m["prediction_span_overlap"] >= .6:
                y[j] = 1; break
    return y


def counts(s, C, keep):
    pl = [bb.as_line(f"Z{i}", c["start"], c["end"]) for i, (c, k) in enumerate(zip(C, keep)) if k]
    if GOLD[s]["zero_bend_control"]:
        return (0, len(pl), 0)
    m = MATCH(GOLD[s]["lines"], pl, THR)
    return (m["true_positive"], m["false_positive"], m["false_negative"])


def gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0)


def cv(D, X, Y, fold, fixed_thr=None, scans=None):
    scans = scans or OPEN
    per = {}
    for f in sorted(set(fold[s] for s in scans)):
        tr = [s for s in scans if fold[s] != f and len(X[s])]
        te = [s for s in scans if fold[s] == f]
        m = gbm().fit(np.vstack([X[s] for s in tr]), np.concatenate([Y[s] for s in tr]))
        P = {s: (m.predict_proba(X[s])[:, 1] if len(X[s]) else np.zeros(0)) for s in tr + te}
        if fixed_thr is None:
            best = (-1, .5)
            for t in np.arange(.15, .8, .025):
                c = np.sum([counts(s, D[s]["candidates"], P[s] >= t) for s in tr if s in LINE], axis=0)
                f1 = prf(*c)[2]
                if f1 > best[0]: best = (f1, t)
            thr = best[1]
        else:
            thr = fixed_thr
        for s in te:
            per[s] = counts(s, D[s]["candidates"], P[s] >= thr)
    lc = np.sum([per[s] for s in scans if s in LINE], axis=0)
    fam = collections.defaultdict(lambda: np.zeros(3))
    for s in scans:
        if s in LINE: fam[INV[s]["family"]] += per[s]
    return {"micro_f1": round(prf(*lc)[2], 4), "tp_fp_fn": [int(x) for x in lc],
            "macro": round(float(np.mean([prf(*v)[2] for v in fam.values()])), 4),
            "ctrl_fp": int(sum(per[s][1] for s in scans if s in CTRL))}, per


def regroup(key, seed):
    groups = sorted({key(s) for s in DEV})
    rng = np.random.default_rng(seed); rng.shuffle(groups)
    cnt = collections.Counter(key(s) for s in DEV); load = [0] * 5; gf = {}
    for g in sorted(groups, key=lambda g: -cnt[g]):   # stable sort keeps random order among ties
        f = int(np.argmin(load)); gf[g] = f; load[f] += cnt[g]
    return {s: gf[key(s)] for s in DEV}


g5 = lambda s: INV[s]["family_group5"]
g4 = lambda s: INV[s]["family_group5"][:4] if not INV[s]["family_group5"].startswith("UNKNOWN") else INV[s]["family_group5"]
g3 = lambda s: INV[s]["family_group5"][:3] if not INV[s]["family_group5"].startswith("UNKNOWN") else INV[s]["family_group5"]

res3 = {}
D7 = load_cands("dev_it7"); X7 = {s: feats(D7[s]) for s in OPEN}; Y7 = {s: labels(s, D7[s]["candidates"]) for s in OPEN}
r, per7 = cv(D7, X7, Y7, FOLD); res3["it7_devfolds_open"] = r
res3["it7_devfolds_open_fixed_thr_0.5"] = cv(D7, X7, Y7, FOLD, fixed_thr=0.5)[0]
for name, key in (("g5", g5), ("g4", g4), ("g3", g3)):
    vals = [cv(D7, X7, Y7, regroup(key, sd))[0]["micro_f1"] for sd in range(10)]
    res3[f"it7_random_{name}_folds_x10"] = {"mean": round(float(np.mean(vals)), 4), "min": min(vals), "max": max(vals), "all": vals}
# leave-one-group-out (g4)
logo = {s: g4(s) for s in DEV}
res3["it7_leave_one_g4_group_out"] = cv(D7, X7, Y7, logo)[0]
# per-iteration (code it1..it7) with the same OOF selector, per fold
it = {}
for tag in ("dev_it1", "dev_it2", "dev_it4", "dev_it6", "dev_it7"):
    D = D7 if tag == "dev_it7" else load_cands(tag)
    X = {s: feats(D[s]) for s in OPEN}; Y = {s: labels(s, D[s]["candidates"]) for s in OPEN}
    r, per = cv(D, X, Y, FOLD)
    byf = {}
    for f in range(5):
        c = np.sum([per[s] for s in LINE if FOLD[s] == f], axis=0); byf[f] = round(prf(*c)[2], 4)
    allc = np.sum([counts(s, D[s]["candidates"], np.ones(len(D[s]["candidates"]), bool)) for s in LINE], axis=0)
    it[tag] = {**r, "by_fold": byf, "cand_recall": round(allc[0] / (allc[0] + allc[2]), 4),
               "n_failed": sum(D[s].get("status") == "FAILED" for s in OPEN)}
res3["per_iteration_oof_gbm_open"] = it

# touched vs untouched scans (debug subsets, named families in logs, overlays)
touched = set()
for f in glob.glob(str(W / "sub*/*.json")) + glob.glob(str(W / "pc1/*.json")) + glob.glob(str(W / "overlays_it6/*.png")) + glob.glob(str(Z3 / "overlays/*.png")):
    touched.add(Path(f).stem)
for f in (W / "m4.txt", W / "m5.txt", W / "m5full.txt"):
    touched |= set(re.findall(r"sc_[0-9a-f]{12}", f.read_text()))
txt = (Z3 / "ITER_LOG.md").read_text() + (Z3 / "README.md").read_text()
named = set(re.findall(r"\b\d{8}\b", txt))
touched |= {s for s in DEV if INV[s]["family"] in named or INV[s]["part_id"] in named}
touched &= set(DEV)
tl = [s for s in LINE if s in touched]; ul = [s for s in LINE if s not in touched]
def sub(scans, per):
    c = np.sum([per[s] for s in scans], axis=0); return {"scans": len(scans), "tp_fp_fn": [int(x) for x in c], "micro_f1": round(prf(*c)[2], 4)}
# using the delivered it7 eval_lines (their OOF) and also all-DEV counts from their per_case
def sub_theirs(scans):
    c = np.sum([(theirs[s]["true_positive"], theirs[s]["false_positive"], theirs[s]["false_negative"]) for s in scans], axis=0)
    return {"scans": len(scans), "tp_fp_fn": [int(x) for x in c], "micro_f1": round(prf(*c)[2], 4)}
allline = [s for s in DEV if theirs[s]["disposition"] == "SCORED_COMPLETE_LABEL"]
res3["touched_vs_untouched"] = {"n_touched_dev": len(touched), "named_families": sorted(named),
                                "open_touched_it7_myoof": sub(tl, per7), "open_untouched_it7_myoof": sub(ul, per7),
                                "all_touched_theirs": sub_theirs([s for s in allline if s in touched]),
                                "all_untouched_theirs": sub_theirs([s for s in allline if s not in touched]),
                                "touched_ids": sorted(touched)}
# it1 (code tuned on iteration-0 subset only) on scans untouched at it1 time (sub1..sub5 debug subset)
sub0 = set()
for d in ("sub1", "sub2", "sub3", "sub4", "sub5"):
    sub0 |= {Path(f).stem for f in glob.glob(str(W / d / "*.json"))}
D1 = load_cands("dev_it1"); X1 = {s: feats(D1[s]) for s in OPEN}; Y1 = {s: labels(s, D1[s]["candidates"]) for s in OPEN}
r1, per1 = cv(D1, X1, Y1, FOLD)
res3["it1_on_non_iteration0_scans"] = {**sub([s for s in LINE if s not in sub0], per1), "iteration0_subset": sorted(sub0)}
res3["it7_on_non_iteration0_scans"] = sub([s for s in LINE if s not in sub0], per7)
report["3_cv"] = res3
report["meta"] = {"n_dev": len(DEV), "n_open": len(OPEN), "n_sealed_excluded": len(SEALED), "line_open": len(LINE), "ctrl_open": len(CTRL),
                  "testb_in_dev": len(set(DEV) & TB)}
(OUT / "audit_results.json").write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(report, indent=1, default=str)[:12000])
