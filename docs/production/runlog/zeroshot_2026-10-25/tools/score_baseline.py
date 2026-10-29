"""Score the phase-1 baseline with the frozen matcher (score.py score_case: 5 deg / 4 mm / 0.6 of both spans,
one-to-one Hungarian, apex locus, raw frame). Family-macro = mean over families of family-micro F1 (gold_v2_blind_score
convention). CIs: 95 % percentile bootstrap over families, 10 000 reps, seed 20261018.
Arms: a0 (zero-shot A0 = best_f1 without CAD), best_f1 (A0 U gated CAD, existing GOLD_SET_V2 runs only),
e3 (e3-1.12.0 on DEV scans with STEP), e3_else_a0 (E3 lines when E3 returned a VALID inspection, else A0).
TEST-B per-case rows go to baseline/scores_TESTB_SEALED.json; only aggregates are written to BASELINE.json.
usage: .venv/bin/python -B score_baseline.py"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
import score as S  # noqa: E402
import bb_common as bb  # noqa: E402
import score_broad as SB  # noqa: E402
from collections import Counter
H = Path(__file__).resolve().parents[1]
RUNS = REPO / "docs/production/runlog/broad_baseline/runs/best_f1"
inv = {r["scan_id"]: r for r in json.loads((H / "INVENTORY.json").read_text())["scans"]}
split = json.loads((H / "SPLIT_FROZEN.json").read_text())
TEST = set(split["test_b"]["scans"]); DEV = set(split["dev"]["scans"])
NR = {"sc_eaa89c6d1f1f", "sc_152f1b1aacb2", "sc_c623d2e45bff", "sc_9ef2cf4e52cb", "sc_7a1773eca9fa"}
N_BOOT, SEED = 10000, 20261018


def a0_path(sid):
    pid = inv[sid].get("a0_prediction_case_id") or sid
    for p in (RUNS / pid / "predictions_a0.json", H / "baseline/a0_runs" / sid / "predictions_a0.json",
              H / "baseline/a0_runs_TESTB_SEALED" / sid / "predictions_a0.json"):
        if p.is_file():
            return p
    return None


def load(p):
    if p is None or not Path(p).is_file():
        return {"owned_bend_regions": [], "status": "PREDICTIONS_MISSING", "scan": {}}
    d = bb.load_eval_lines(Path(p))
    d = dict(d); d["scan"] = {}  # PLY/STL twins: vertex sets identical (gold_set_v1/twin_vertex_check.json)
    return d


def preds(arm, sid):
    if arm == "a0":
        return load(a0_path(sid))
    if arm == "best_f1":
        pid = inv[sid].get("a0_prediction_case_id") or sid
        return load(RUNS / pid / "predictions_best_f1.json") if (RUNS / pid / "predictions_best_f1.json").is_file() else None
    e3p = H / "baseline/e3_runs" / sid / "predictions_e3.json"
    if arm == "e3":
        return load(e3p) if e3p.is_file() else None
    if arm == "e3_else_a0":
        if e3p.is_file():
            d = json.loads(e3p.read_text())
            if d.get("status") == "OK" and d.get("inspection_state") == "VALID":
                return load(e3p)
            return load(a0_path(sid))
        return None


def case(sid):
    r = inv[sid]; g = r["gold"]
    return {"case_id": sid, "family_id": r["family"], "tier": g["set"], "gold": {"kind": "ai_studio", "path": g["path"]},
            "label_complete": g.get("label_complete", True)}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; rr = tp / (tp + fn) if tp + fn else 0.0
    return p, rr, (2 * p * rr / (p + rr) if p + rr else 0.0)


def metrics(rows, rng):
    rows = [x for x in rows if x.get("disposition") == "SCORED_COMPLETE_LABEL"]
    if not rows:
        return {"scans": 0}
    fam = defaultdict(lambda: [0, 0, 0])
    for x in rows:
        f = fam[x["family_id"]]; f[0] += x["true_positive"]; f[1] += x["false_positive"]; f[2] += x["false_negative"]
    tp, fp, fn = (sum(v[i] for v in fam.values()) for i in range(3))
    p, r_, f1 = prf(tp, fp, fn)
    arr = np.array(list(fam.values()), float); ff = np.array([prf(*v)[2] for v in arr])
    mic, mac = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(arr), len(arr)); a = arr[idx].sum(0); mic.append(prf(*a)[2]); mac.append(ff[idx].mean())
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    fnc, fpc = Counter(), Counter()
    for x in rows:
        fnc.update(x.get("fn_causes") or {}); fpc.update(x.get("fp_causes") or {})
    return {"fn_causes": dict(fnc), "fp_causes": dict(fpc), "scans": len(rows), "families": len(fam), "gold_lines": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "P": round(p, 3), "R": round(r_, 3), "micro_f1": round(f1, 3), "micro_f1_ci": ci(mic),
            "family_macro_f1": round(float(ff.mean()), 3), "family_macro_f1_ci": ci(mac),
            "lane_failures": sorted(x["case_id"] for x in rows if x.get("lane_status") not in ("OK", "A0_ONLY_NO_CAD", None)
                                    and not x.get("prediction_count"))}


rows = {arm: {} for arm in ("a0", "best_f1", "e3", "e3_else_a0")}
WITH_TEST = "--with-test" in sys.argv
for sid in sorted(DEV | (TEST if WITH_TEST else set())):
    c = case(sid)
    for arm in rows:
        if sid in TEST and arm != "a0":
            continue
        pr = preds(arm, sid)
        if pr is None:
            continue
        rec = S.score_case(c, frame_records=None, preloaded_predictions=pr)
        if sid in DEV and arm in ("a0", "e3") and rec.get("disposition") == "SCORED_COMPLETE_LABEL":
            gl = S.load_gold(c["gold"], frame_records=None, locus="apex")["lines"]
            pl = [x for x in pr.get("owned_bend_regions") or [] if bb.valid_line(x.get("line_start"), x.get("line_end"))]
            dg = SB.diagnose(gl, pl, rec)
            rec["fn_causes"], rec["fp_causes"] = dg["fn_cause_counts"], dg["fp_cause_counts"]
        rec["custody"] = inv[sid]["custody"]; rec["stage"] = inv[sid]["stage"]
        rec["has_step"] = inv[sid]["cad"]["has_step"]; rec["e3_eligible"] = bool(inv[sid]["cad"].get("sku")) and sid not in NR
        rows[arm][sid] = rec

cohorts = {
    "DEV_all": lambda s, r: s in DEV,
    "DEV_A1+other_consumed": lambda s, r: s in DEV and r["custody"] == "DEV_CONSUMED",
    "DEV_consumed_DEVCHECK": lambda s, r: s in DEV and r["custody"] == "CONSUMED_DEVCHECK",
    "DEV_consumed_FRESH": lambda s, r: s in DEV and r["custody"] == "CONSUMED_FRESH",
    "DEV_with_STEP": lambda s, r: s in DEV and r["has_step"],
    "DEV_no_CAD": lambda s, r: s in DEV and not r["has_step"],
    "DEV_E3_eligible": lambda s, r: s in DEV and r["e3_eligible"],
    "DEV_full": lambda s, r: s in DEV and r["stage"] == "full",
    "DEV_partial": lambda s, r: s in DEV and r["stage"] == "partial",
    "TEST_B_aggregate_only": lambda s, r: s in TEST,
}
out = {"schema": "sherman.zeroshot.baseline.v1", "matcher": "score.py score_case (5 deg / 4 mm / 0.6, Hungarian, apex, raw)",
       "split_sha256_file": "SPLIT_FROZEN.json", "arms": {}}
for arm, rr in rows.items():
    out["arms"][arm] = {}
    for name, fn in cohorts.items():
        sub = [r for s, r in rr.items() if fn(s, r)]
        if not sub:
            continue
        rng = np.random.default_rng(SEED)
        m = metrics(sub, rng)
        m["zero_bend_controls"] = {"scans": sum(r.get("disposition") == "ZERO_BEND_CONTROL" for r in sub),
                                   "fp_lines": sum(r.get("false_positive_lines", 0) for r in sub if r.get("disposition") == "ZERO_BEND_CONTROL")}
        m["partial_recall_only"] = {"scans": sum(r.get("disposition") == "SCORED_RECALL_ONLY_PARTIAL_LABEL" for r in sub)}
        m["excluded"] = [(r["case_id"], r.get("reason")) for r in sub if r.get("disposition") == "EXCLUDED_BEFORE_SCORING"]
        if name.startswith("TEST"):
            m.pop("lane_failures", None)
            m["lane_failure_count"] = sum(1 for r in sub if r.get("lane_status") not in ("OK", "A0_ONLY_NO_CAD", "NO_PREDICTIONS"))
        out["arms"][arm][name] = m
(H / "BASELINE.json").write_text(json.dumps(out, indent=1, default=str))
dev_rows = {arm: {s: r for s, r in rr.items() if s in DEV} for arm, rr in rows.items()}
(H / "baseline/scores_dev_per_case.json").write_text(json.dumps(dev_rows, indent=1, default=str))
if WITH_TEST:
    (H / "baseline/scores_TESTB_SEALED.json").write_text(json.dumps({s: r for s, r in rows["a0"].items() if s in TEST}, indent=1, default=str))
for arm, d in out["arms"].items():
    for name, m in d.items():
        if m.get("scans"):
            print(f"{arm:11s} {name:24s} n={m['scans']:3d} fam={m['families']:3d} gold={m['gold_lines']:4d} "
                  f"P={m['P']:.3f} R={m['R']:.3f} F1={m['micro_f1']:.3f} {m['micro_f1_ci']} macro={m['family_macro_f1']:.3f} {m['family_macro_f1_ci']}")
