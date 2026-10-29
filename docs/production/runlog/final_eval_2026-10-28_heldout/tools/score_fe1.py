"""Score the FE1 one-time run once (PREREG.md s5; freeze seqs 36, 43, RUN 44).

Frozen matcher research/broad_baseline/score.py score_case through research/zeroshot/zs_lane_v1/scoring.py
(score_scan, aggregate, bootstrap; unchanged), family = part_id[:5], bootstrap n 10000 seed 20261018 passed explicitly.
Paired v3 - v1: research/zeroshot/zs_lane_v2/train.py paired_bootstrap (same n, seed).
Sensitivity (seq 43): S1 drops NC1 open-hem lines and the NC4 adjudicator-only hem; S2 drops every 180 deg hem line;
S3 = unseen families (drop group5 38173, 26687). S1/S2 gold copies are derived files under runs/derived_gold/ (the
frozen gold is never edited).
Pooled v1 held-out: TEST-B 'zs|testb' + N1 'zs_lane_v1:zs (primary)' recorded per-case records + FE1 v1 records.
Writes RESULTS.json (results.json) + per-case records. usage (repo root): .venv/bin/python -B <this>
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v1 import scoring as SC  # noqa: E402
from research.zeroshot.zs_lane_v2.train import paired_bootstrap  # noqa: E402

D = REPO / "docs/production/runlog/final_eval_2026-10-28_heldout"
ZS = REPO / "docs/production/runlog/zeroshot_2026-10-25"
N, SEED = 10000, 20261018
ARMS = ["zs_lane_v3", "zs_lane_v1"]
SEEN = {"38173", "26687"}
INCL = re.compile(r"incl ([0-9.]+) deg")
HEM = re.compile(r"180 deg hem \((\w+)\)")
NC1_GIDS = {"gb_1baa22", "gb_d11e33", "gb_231669", "gb_ac2239"}   # open hems (ADJUDICATION s5)
NC4 = {("gb_68cd06", "68cd06_hem9")}                                 # adjudicator-only patch-9 end hem
JOGGLE_GIDS = {"gb_437879"}
OK = "OK"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def agg(recs):
    return SC.aggregate(recs, ci=True, n_boot=N, seed=SEED)


def slim(a):
    keys = ["scans", "families", "gold", "pred", "tp", "fp", "fn", "P", "R", "micro_f1", "micro_ci95", "family_macro_f1",
            "macro_ci95", "failed_or_missing_scans", "controls", "fn_causes", "fp_causes", "worst_families"]
    return {k: a.get(k) for k in keys}


def table():
    man = json.loads((D / "adjudication/GOLD_ROOT_MANIFEST.json").read_text())["labels"]
    sm = {s["scan_id"]: s for s in json.loads((D / "SET_MANIFEST.json").read_text())["scored"]}
    out = {}
    for gid, m in sorted(man.items()):
        assert sha(REPO / m["label_path"]) == m["label_sha256"], gid
        adj = json.loads((D / "adjudication" / gid / "adjudication.json").read_text())
        s = sm[m["scan_id"]]
        out[gid] = {"gid": gid, "scan_id": m["scan_id"], "label": m["label_path"], "group5": s["family_group5"],
                    "subset": s["subset"], "pair_side": s.get("pair_side"), "pair_key": s["part_id"] + "_" + str(s.get("rev")),
                    "part_class": adj["part_class"], "zero_bend": adj["zero_bend"], "label_complete": adj["label_complete"]}
    return out


def gold_bends(path):
    return json.loads((REPO / path).read_text())["bends"]


def derived_gold(t, variant):
    """S1/S2 gold copies (frozen gold untouched). Returns {gid: path, ...}, dropped lines."""
    paths, dropped = {}, []
    root = D / "runs/derived_gold" / variant
    root.mkdir(parents=True, exist_ok=True)
    for gid, r in t.items():
        g = json.loads((REPO / r["label"]).read_text())
        keep = []
        for b in g["bends"]:
            h = HEM.search(b["notes"])
            drop = False
            if h and variant == "S2_no_hems":
                drop = True
            if h and variant == "S1_no_new_conventions" and (gid in NC1_GIDS or (gid, h.group(1)) in NC4):
                drop = True
            (dropped.append(f"{gid}:{b['bend_id']}:{h.group(1)}") if drop else keep.append(b))
        if len(keep) == len(g["bends"]):
            paths[gid] = r["label"]
            continue
        g["bends"] = keep
        g["notes"] = (g.get("notes") or "") + f" | DERIVED {variant} copy for FE1 sensitivity (seq 43); frozen gold unchanged"
        p = root / f"{gid}.json"
        p.write_text(json.dumps(g, indent=1))
        paths[gid] = str(p.relative_to(REPO))
    return paths, dropped


def run_rec(arm, gid):
    p = D / "runs" / arm / f"{gid}.json"
    return json.loads(p.read_text()) if p.exists() else {"status": "NO_OUTPUT_FILE", "lines": []}


def lines_of(r):
    if r.get("status") != OK:
        return None
    return [{"start": l["start_mm"], "end": l["end_mm"]} for l in r["lines"]]


def score_all(t, arm, gold_paths):
    recs = []
    for gid, r in t.items():
        spec = {"case_id": gid, "family_id": r["group5"], "tier": "FE1", "gold": {"kind": "ai_studio", "path": gold_paths[gid]},
                "label_complete": r["label_complete"]}
        rr = run_rec(arm, gid)
        try:
            rec = SC.score_scan(spec, lines_of(rr), diagnose=True)
        except RuntimeError as exc:   # EXCLUDED_BEFORE_SCORING: evaluation failure, reported, gold counted FN in strict
            rec = {"case_id": gid, "family_id": r["group5"], "disposition": "EXCLUDED_BEFORE_SCORING", "error": str(exc),
                   "gold_count": len(gold_bends(gold_paths[gid])), "output_missing_or_failed": lines_of(rr) is None}
        rec["subset"] = r["subset"]
        recs.append(rec)
    return recs


def micro_incl_controls(recs):
    a = SC.aggregate(recs, ci=False)
    cfp = a["controls"]["fp_lines"]
    excl_fn = sum(x.get("gold_count", 0) for x in recs if x.get("disposition") == "EXCLUDED_BEFORE_SCORING")
    p, r, f = SC.prf(a["tp"], a["fp"] + cfp, a["fn"])
    ps, rs, fs = SC.prf(a["tp"], a["fp"] + cfp, a["fn"] + excl_fn)
    return {"tp": a["tp"], "fp": a["fp"] + cfp, "fn": a["fn"], "P": round(p, 4), "R": round(r, 4), "micro_f1": round(f, 4),
            "strict_micro_f1_excluded_as_fn": round(fs, 4), "excluded_before_scoring": excl_fn}


def per_scan_rows(t, arm, recs):
    rows = []
    for x in recs:
        gid = x["case_id"]
        rr = run_rec(arm, gid)
        gb = {b["bend_id"]: b for b in gold_bends(t[gid]["label"])}
        row = {"gid": gid, "subset": t[gid]["subset"], "family": t[gid]["group5"], "part_class": t[gid]["part_class"],
               "disposition": x["disposition"], "status": rr.get("status"), "gold": len(gb), "pred": x.get("n_pred", 0),
               "wall_s": rr.get("wall_s"), "peak_rss_gib": rr.get("peak_rss_gib"), "rss_limit_gib": rr.get("rss_limit_gib"),
               "t_mm": rr.get("t_mm_scan"), "failure_reason": None if rr.get("status") == OK else
               (json.dumps(rr.get("errors"))[:400] if rr.get("errors") else rr.get("status"))}
        if x["disposition"] == "ZERO_BEND_CONTROL":
            row.update(tp=0, fp=x["false_positive_lines"], fn=0, f1=None, matches=[],
                       fp_ids=[f"P{i + 1:03d}" for i in range(x.get("n_pred", 0))], fn_ids=[])
        elif x["disposition"] == "SCORED_COMPLETE_LABEL":
            ms = []
            for m in x["matches"]:
                i = int(m["prediction_id"][1:]) - 1
                e = rr["lines"][i].get("incl_deg")
                b = gb[m["gold_id"]]
                gi = INCL.search(b["notes"])
                g = float(gi.group(1)) if gi else None
                hem = bool(HEM.search(b["notes"]))
                ms.append({"gold_id": m["gold_id"], "pred_id": m["prediction_id"], "engine_line_id": rr["lines"][i].get("line_id"),
                           "offset_mm": round(m["transverse_offset_mm"], 3), "axis_angle_deg": round(m["axis_angle_deg"], 3),
                           "gold_incl_deg": g, "engine_incl_deg": None if e is None else round(float(e), 2),
                           "incl_diff_deg": None if (e is None or g is None) else round(float(e) - g, 2),
                           "hem_gold_construction": hem, "joggle": gid in JOGGLE_GIDS})
            row.update(tp=x["true_positive"], fp=x["false_positive"], fn=x["false_negative"], f1=round(x["f1"], 4),
                       matches=ms, fp_ids=x["unmatched_prediction_ids"], fn_ids=x["unmatched_gold_ids"],
                       fn_causes=x.get("fn_causes"), fp_causes=x.get("fp_causes"))
        else:
            row.update(tp=0, fp=0, fn=row["gold"], f1=None, matches=[], fp_ids=[], fn_ids=list(gb), error=x.get("error"))
        rows.append(row)
    return rows


def angle_summary(rows):
    all_d = [m["incl_diff_deg"] for r in rows for m in r["matches"] if m["incl_diff_deg"] is not None]
    nd = [m["incl_diff_deg"] for r in rows for m in r["matches"]
          if m["incl_diff_deg"] is not None and not m["hem_gold_construction"] and not m["joggle"]]
    ad = sorted(abs(d) for d in nd)
    q = lambda p: round(float(np.percentile(ad, p)), 2) if ad else None  # noqa: E731
    return {"n_matched_with_angles": len(all_d), "n_excl_hem_and_joggle": len(nd),
            "rule": "engine included angle - gold included angle; hem gold lines (tangent-band construction, gold angle is not the hem angle) and the joggle scan are flagged and left out of the summary",
            "median_abs": q(50), "p90_abs": q(90), "max_abs": round(ad[-1], 2) if ad else None,
            "mean_signed": round(float(np.mean(nd)), 2) if nd else None,
            "within_1deg": sum(1 for d in ad if d <= 1), "within_2deg": sum(1 for d in ad if d <= 2),
            "within_5deg": sum(1 for d in ad if d <= 5),
            "flagged": [{"gid": r["gid"], "gold_id": m["gold_id"], "diff": m["incl_diff_deg"],
                         "why": "hem" if m["hem_gold_construction"] else "joggle"}
                        for r in rows for m in r["matches"] if m["hem_gold_construction"] or m["joggle"]]}


def runtime(t, arm):
    w, rs, fails = [], [], []
    for gid in t:
        r = run_rec(arm, gid)
        if r.get("wall_s") is not None:
            w.append(r["wall_s"]); rs.append(r["peak_rss_gib"])
        if r.get("status") != OK:
            fails.append({"gid": gid, "status": r.get("status"), "reason": json.dumps(r.get("errors"))[:400],
                          "wall_s": r.get("wall_s"), "rc": r.get("driver_rc")})
    q = lambda a, p: round(float(np.percentile(a, p)), 1) if a else None  # noqa: E731
    return {"wall_s": {"p50": q(w, 50), "p95": q(w, 95), "max": max(w) if w else None, "total": round(sum(w), 1), "n": len(w)},
            "peak_rss_gib": {"p50": q(rs, 50), "max": max(rs) if rs else None},
            "ok": sum(1 for gid in t if run_rec(arm, gid).get("status") == OK), "failures": fails}


def main():
    t = table()
    frozen_paths = {g: r["label"] for g, r in t.items()}
    variants = {"PRIMARY": (frozen_paths, [])}
    for v in ("S1_no_new_conventions", "S2_no_hems"):
        variants[v] = derived_gold(t, v)
    res = {"schema": "sherman.final_eval.fe1_results.v1", "run": "final_eval_2026-10-28_heldout FE1",
           "ledger": {"run_entry_seq": 44, "freeze_seq": 36, "gold_seq": 42, "sensitivity_seq": 43, "prereg_sha256": sha(D / "PREREG.md")},
           "matcher": "frozen score_case 5 deg / 4 mm / 0.6 / 0.6, apex", "bootstrap": {"n": N, "seed": SEED, "unit": "family (part_id[:5])"},
           "gold_root_manifest_sha256": sha(D / "adjudication/GOLD_ROOT_MANIFEST.json"), "arms": {}, "sensitivity": {}}
    per_case = {}
    recs_by = {}
    for v, (paths, dropped) in variants.items():
        for arm in ARMS:
            recs_by[(v, arm)] = score_all(t, arm, paths)
        if v != "PRIMARY":
            res["sensitivity"][v] = {"dropped_gold_lines": dropped, "n_dropped": len(dropped),
                                     "derived_gold_sha256": {g: sha(REPO / p) for g, p in paths.items() if p != frozen_paths[g]},
                                     "arms": {arm: {"all": slim(agg(recs_by[(v, arm)])),
                                                    "micro_incl_control_fp": micro_incl_controls(recs_by[(v, arm)]),
                                                    "unseen_families": slim(agg([x for x in recs_by[(v, arm)] if x["family_id"] not in SEEN]))}
                                              for arm in ARMS},
                                     "paired_v3_minus_v1": paired_bootstrap(recs_by[(v, "zs_lane_v1")], recs_by[(v, "zs_lane_v3")], n_boot=N, seed=SEED)}
    for arm in ARMS:
        recs = recs_by[("PRIMARY", arm)]
        rows = per_scan_rows(t, arm, recs)
        zb = [r for r in rows if r["disposition"] == "ZERO_BEND_CONTROL"]
        rf = [r for r in rows if r["part_class"] in ("rolled", "flat")]
        pairs = defaultdict(dict)
        for r in rows:
            if t[r["gid"]]["pair_side"] in ("V", "X"):
                pairs[t[r["gid"]]["pair_key"]][t[r["gid"]]["pair_side"]] = (r["gid"], r["pred"], r["gold"])
        res["arms"][arm] = {
            "primary": slim(agg(recs)),
            "micro_incl_control_fp": micro_incl_controls(recs),
            "by_subset": {s: slim(agg([x for x in recs if x["subset"] == s])) for s in ("HOLDOUT10", "N001_N016", "EHO4")},
            "zero_bend": {"scans": len(zb), "per_scan_fp": {r["gid"]: r["fp"] for r in zb}, "fp_total": sum(r["fp"] for r in zb)},
            "rolled_or_flat_fp": {"scans": {r["gid"]: {"class": r["part_class"], "fp": r["fp"]} for r in rf},
                                  "total": sum(r["fp"] for r in rf)},
            "unseen_families_S3": slim(agg([x for x in recs if x["family_id"] not in SEEN])),
            "seen_families_only": slim(agg([x for x in recs if x["family_id"] in SEEN])),
            "angles": angle_summary(rows),
            "runtime": runtime(t, arm),
            "vx_pairs_context_only": [{"V": v.get("V", (None,))[0], "X": v.get("X", (None,))[0],
                                       "engine_lines_V": v.get("V", (None, None))[1], "engine_lines_X": v.get("X", (None, None))[1],
                                       "gold_V": v.get("V", (None, None, None))[2], "gold_X": v.get("X", (None, None, None))[2]}
                                      for v in pairs.values()],
            "per_scan": rows}
        per_case[arm] = [{k: x.get(k) for k in ("case_id", "family_id", "subset", "disposition", "gold_count", "prediction_count",
                                                "true_positive", "false_positive", "false_negative", "false_positive_lines",
                                                "output_missing_or_failed", "matches", "unmatched_gold_ids", "unmatched_prediction_ids")}
                         for x in recs]
    res["paired_v3_minus_v1"] = paired_bootstrap(recs_by[("PRIMARY", "zs_lane_v1")], recs_by[("PRIMARY", "zs_lane_v3")], n_boot=N, seed=SEED)
    res["paired_v3_minus_v1_unseen_S3"] = paired_bootstrap(
        [x for x in recs_by[("PRIMARY", "zs_lane_v1")] if x["family_id"] not in SEEN],
        [x for x in recs_by[("PRIMARY", "zs_lane_v3")] if x["family_id"] not in SEEN], n_boot=N, seed=SEED)
    res["sensitivity"]["S3_unseen_families_only"] = {
        "excluded_group5": sorted(SEEN), "arms": {a: res["arms"][a]["unseen_families_S3"] for a in ARMS},
        "paired_v3_minus_v1": res["paired_v3_minus_v1_unseen_S3"]}

    # pooled v1 held-out: TEST-B (61) + N1 (4) + FE1 (27); recorded per-case records, family = part_id[:5]
    tb = json.loads((ZS / "testb/RESULTS_per_case.json").read_text())["zs|testb"]
    n1 = json.loads((ZS / "newscans_N1/RESULTS_per_case.json").read_text())["zs_lane_v1:zs (primary)"]
    fe = recs_by[("PRIMARY", "zs_lane_v1")]
    g5 = lambda rs: [dict(x, family_id=str(x["family_id"])[:5]) for x in rs]  # noqa: E731
    pool = g5(tb) + g5(n1) + g5(fe)
    res["pooled_v1_heldout"] = {
        "sources": {"TEST-B": {"path": "docs/production/runlog/zeroshot_2026-10-25/testb/RESULTS_per_case.json", "key": "zs|testb",
                               "sha256": sha(ZS / "testb/RESULTS_per_case.json"), "n": len(tb)},
                    "N1": {"path": "docs/production/runlog/zeroshot_2026-10-25/newscans_N1/RESULTS_per_case.json",
                           "key": "zs_lane_v1:zs (primary)", "sha256": sha(ZS / "newscans_N1/RESULTS_per_case.json"), "n": len(n1)},
                    "FE1": {"n": len(fe)}},
        "family_unit": "part_id[:5] for all three (TEST-B/N1 records carry full part_id; truncated here)",
        "pooled": slim(agg(pool)), "micro_incl_control_fp": micro_incl_controls(pool),
        "pooled_recorded_family_ids": slim(agg(tb + n1 + fe)),
        "components": {"TEST-B": slim(agg(g5(tb))), "N1": slim(agg(g5(n1))), "FE1": slim(agg(g5(fe)))},
        "caveats": ["TEST-B/N1 v1 runs used the research lane CLI under the then pin (code fingerprint af87df2f; lane code, models and config unchanged by the 10-26/10-28 re-pins, which re-hashed E3 files only); FE1 used lane_cli.run_lane blind",
                    "families shared between TEST-B and FE1 at group5 level (26687) merge into one bootstrap unit"]}
    adj = json.loads((D / "adjudication/ADJUDICATION_FE1.json").read_text())
    res["markers_and_canary_from_adjudication"] = {"canary": adj["canary"], "ab_agreement": adj["ab_agreement"],
                                                   "source": "adjudication/ADJUDICATION_FE1.json", "sha256": sha(D / "adjudication/ADJUDICATION_FE1.json")}
    (D / "runs/RESULTS_per_case.json").write_text(json.dumps(per_case, indent=1, default=str) + "\n")
    res["per_case_sha256"] = sha(D / "runs/RESULTS_per_case.json")
    res["run_outputs_sha256"] = {f"{a}/{g}": sha(D / "runs" / a / f"{g}.json") for a in ARMS for g in t if (D / "runs" / a / f"{g}.json").exists()}
    (D / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(sha(D / "results.json"))
    for arm in ARMS:
        a = res["arms"][arm]
        print(arm, {k: a["primary"][k] for k in ("tp", "fp", "fn", "micro_f1", "micro_ci95", "family_macro_f1", "macro_ci95", "failed_or_missing_scans")})


if __name__ == "__main__":
    main()
