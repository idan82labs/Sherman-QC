"""New production engine (E3) x the frozen Jev v2 designs (research only).

Takes one completed A1 iteration's physical work directory (``W/physical/E3/<case>/record.json``,
``W/compiled/dev/<sku>``) and scores these arms on the frozen BROAD A1 cases
(``docs/production/runlog/freeze/broad_A1.json``, frozen BROAD gold, frozen physical scorer) and on
the GOLD_SET_V1 cases in the A1 view (same selection as ``a1/iter_N/measure/tools/goldset.py``):

  a  engine alone
  b  engine + Jev composite filter        (score ``jev_composite_combined``, frozen lane threshold)
  c  engine + v2 single-feature filter    (pool two-sided transition score, frozen threshold;
                                          a line with no matching pool cluster is dropped, as frozen)
  d  engine + liberal pool selection      (``apply_v2 --add-from-pool`` logic):
       d_lr        engine lines kept, pool lines added by the frozen logistic regression + NMS
       d_lr_jev    as d_lr, and the added pool line must also pass the frozen Jev composite filter
       d_jevclf    engine lines kept, pool lines added by the frozen Jev+classifier ranker
       d_apply_v2  apply_v2 literally: engine + pool lines ranked together by the logistic regression,
                   NMS, frozen threshold (engine lines can be dropped)

Every threshold is read from ``jev_ab_v2/design_choice.json`` (fitted on v2 calibration families);
nothing is fitted on these cases. Trained scores use the v2 protocol: models fitted on the v2
calibration-family pool rows (leave-one-family-out when the target's family is a calibration family).

Subcommands: ``build`` (heavy, under the geometry lock), ``jev`` (dry run, or ``--live``; cached by
body sha), ``score`` (offline). Engine code is imported read-only; nothing under domains/ is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from research.jev import arms_v2 as A
from research.jev.input_opt_v2 import (CASE_LIST, OLD_SPLITS, REPO_ROOT, _fkey, _key_from_env_file, _read,
                                       load_truth, match_mask, neighbour_context, read_candidates)
from research.jev.pool_v2 import cluster_lines, load_pool_lines, nms_select, pair_metrics_matrix, pool_file

STUDY = REPO_ROOT / "docs" / "production" / "runlog" / "jev_ab_v2"
FZ = REPO_ROOT / "docs" / "production" / "runlog" / "freeze"
GOLD_LIST = REPO_ROOT / "docs" / "production" / "runlog" / "gold_set_v1" / "case_list_gold_v1.json"
MODEL = "jev-1.13.0"

# arm -> (filter score, filter threshold key) applied to engine lines; (pool score, pool threshold key,
# extra gate on added pool lines); and whether engine lines are kept unconditionally
ARMS = {
    "a_engine_alone": {},
    "b_engine+jev_composite_filter": {"filter": ("jev_composite_combined", "lane+jev_composite_combined_filter")},
    "c_engine+single_feature_filter": {"filter": ("best_feature", "lane+best_feature_filter")},
    "c_sens_missing_kept": {"filter": ("best_feature", "lane+best_feature_filter"), "missing_kept": True},
    "d_lr": {"pool": ("clf_logreg", "pool+clf_logreg")},
    "d_lr_jev": {"pool": ("clf_logreg", "pool+clf_logreg"),
                 "gate": ("jev_composite_combined", "lane+jev_composite_combined_filter")},
    "d_jevclf": {"pool": ("jev_plus_clf", "pool+jev_plus_clf")},
    "d_apply_v2": {"pool": ("clf_logreg", "pool+clf_logreg"), "rank_engine": True},
    "ref_pool_lr_alone": {"pool": ("clf_logreg", "pool+clf_logreg"), "no_engine": True},
}


def frozen_thresholds(path: Path = STUDY / "design_choice.json") -> dict[str, float]:
    d = _read(path)
    if d.get("jev_design") != "composite" or d.get("jev_score") != "jev_composite_combined":
        raise SystemExit("design_choice.json is not the frozen composite design")
    return {k: float(v) for k, v in d["calib_thresholds"].items() if v is not None}


# --------------------------------------------------------------------------
# selection (pure)
# --------------------------------------------------------------------------


def select_arm(spec: Mapping[str, Any], eng: Sequence[int], pool: Sequence[int], ends: np.ndarray,
               scores: Mapping[str, Mapping[int, float]], thr: Mapping[str, float]) -> list[int]:
    """Row indices kept for one scan. ``eng``/``pool``: row indices of the engine lines / pool candidates."""
    kept_eng = [] if spec.get("no_engine") else list(eng)
    if "filter" in spec:
        name, tkey = spec["filter"]
        t = thr[tkey]
        s = scores[name]

        def keep(i):
            v = s.get(i)
            if v is None:  # no score at all (no Jev answer): kept, as the v2 lane filter did
                return True
            if v == -math.inf:  # feature undefined (no matching pool cluster): dropped, as frozen
                return bool(spec.get("missing_kept"))
            return v >= t
        kept_eng = [i for i in kept_eng if keep(i)]
    if "pool" not in spec:
        return kept_eng
    name, tkey = spec["pool"]
    t = thr[tkey]
    s = scores[name]
    cand = [i for i in pool if i in s and math.isfinite(s[i])]
    if "gate" in spec:
        gname, gkey = spec["gate"]
        gs = scores[gname]
        cand = [i for i in cand if i in gs and gs[i] >= thr[gkey]]
    if spec.get("rank_engine"):  # apply_v2 literal: engine lines are ordinary candidates
        cand = cand + [i for i in kept_eng if i in s and math.isfinite(s[i])]
        order = list(dict.fromkeys(cand))
        local = nms_select(ends[order], [s[i] for i in order], t)
        return [order[k] for k in local]
    # rescue: engine lines first (always kept), then pool lines above threshold that duplicate nothing kept
    order = kept_eng + cand
    sc = [math.inf] * len(kept_eng) + [s[i] for i in cand]
    if not order:
        return []
    local = nms_select(ends[order], sc, t)
    return [order[k] for k in local]


# --------------------------------------------------------------------------
# build (heavy)
# --------------------------------------------------------------------------


def a1_cases() -> tuple[dict[str, Any], list[str]]:
    doc = _read(FZ / "cases_A1.json")
    broad = _read(FZ / "broad_A1.json")["cases"]
    return doc, broad


def engine_lines_for(W: Path, doc, cid: str) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    from domains.production.engine.eval.freeze.physical import _engine_line_predictions

    c = next(x for x in doc["cases"] if x["case_ref"] == cid)
    rp = W / "physical" / "E3" / cid / "record.json"
    info: dict[str, Any] = {"case_ref": cid, "record": rp.is_file()}
    if not rp.is_file():
        info["status"] = "NOT_RUN"
        return None, info
    rec = _read(rp)
    info["status"] = rec.get("status")
    info["result_sha256"] = rec.get("result_sha256")
    res = Path(rec.get("engine_dir") or "") / "result.json"
    if res.is_file():
        eng = (_read(res).get("engine") or {})
        info["engine_version"] = eng.get("engine_version")
        info["config_sha256"] = eng.get("config_sha256")
        info["algorithm_versions"] = eng.get("algorithm_versions")
    if rec.get("status") != "OK":
        return None, info
    preds = _engine_line_predictions([cid], {cid: rec}, {cid: W / "compiled" / "dev" / c["recipe_sku"]})
    if cid not in preds:
        info["status"] = "NO_RESULT"
        return None, info
    return preds[cid], info


def _pool_block(clusters, lines, k):
    if k is None:
        return {}
    x = clusters[k]
    rep = lines[x["rep"]]
    radii = [lines[j].radius_mm for j in x["members"] if lines[j].radius_mm is not None]
    return {"pool_member_count": x["member_count"], "pool_generator_count": len(x["generators"]),
            "pool_generator_kind": "+".join(x["generators"]),
            "pool_radius_mm": float(np.median(radii)) if radii else None,
            "pool_radius_votes_count": len(radii),
            **{f"pool_{key}": val for key, val in rep.descriptors.items()}}


def cmd_build(a: argparse.Namespace) -> int:
    from research.jev.features_v2 import FEATURE_VERSION, line_features_v2, load_clouds

    W = Path(a.a1_work)
    out = Path(a.out)
    (out / "features_cache").mkdir(parents=True, exist_ok=True)
    doc, broad = a1_cases()
    cl = {c["case_id"]: c for c in _read(CASE_LIST)["cases"]}
    splits = _read(OLD_SPLITS)["families"]
    manifest, rows = [], []
    t0 = time.time()
    for cid in broad:
        c = next(x for x in doc["cases"] if x["case_ref"] == cid)
        bid = (c.get("broad") or {}).get("broad_case_id")
        lines, info = engine_lines_for(W, doc, cid)
        info.update(broad_case_id=bid, family=c["family"])
        b = cl.get(bid)
        info["v2_split"] = splits.get(b["family_id"], {}).get("split") if b else None
        if lines is None or b is None:
            info["lines"] = 0 if lines is None else len(lines)
            manifest.append(info)
            continue
        if b["scan_sha256"] != c["scan_sha256"]:
            raise SystemExit(f"{cid}: A1 scan sha differs from the broad case list")
        ends = np.array([[l["line_start"], l["line_end"]] for l in lines], float).reshape(-1, 2, 3)
        pf = pool_file(bid)
        plines, _meta = load_pool_lines(pf) if pf else ([], {})
        clusters = cluster_lines(plines)
        pool_ends = np.array([[plines[x["rep"]].start, plines[x["rep"]].end] for x in clusters]).reshape(-1, 2, 3)
        ctx = neighbour_context(ends, pool_ends, [None] * len(ends))
        to_pool = [None] * len(ends)
        if len(ends) and len(pool_ends):
            mm = pair_metrics_matrix(ends, pool_ends)
            ok = match_mask(mm)
            cost = mm["axis_angle_deg"] / 5 + mm["transverse_offset_mm"] / 4
            for i in range(len(ends)):
                if ok[i].any():
                    to_pool[i] = int(np.argmin(np.where(ok[i], cost[i], 1e9)))
        cf = out / "features_cache" / f"{bid}.json"
        cache = _read(cf) if cf.is_file() else {}
        v2c = STUDY / "build" / "features_cache" / f"{bid}.json"
        seed = _read(v2c) if v2c.is_file() else {}
        cache["__version__"] = FEATURE_VERSION
        need = []
        for e in ends:
            k = _fkey(e[0], e[1])
            if k not in cache:
                if k in seed and seed.get("__version__") == FEATURE_VERSION:
                    cache[k] = seed[k]
                else:
                    need.append(e)
        if need:
            clouds = load_clouds(Path(b["scan_path"]))
            for e in need:
                cache[_fkey(e[0], e[1])] = line_features_v2(clouds, e[0], e[1])
            del clouds
        cf.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
        for k, (e, l) in enumerate(zip(ends, lines)):
            rows.append({"case_id": bid, "a1_case": cid, "family_id": b["family_id"], "a1_family": c["family"],
                         "tier": b["tier"], "split": info["v2_split"], "scan_sha256": b["scan_sha256"],
                         "set": "e3", "cand": str(l["bend_id"]), "line_start": e[0].tolist(),
                         "line_end": e[1].tolist(), "feat": cache[_fkey(e[0], e[1])], "ctx": ctx[k],
                         "pool": _pool_block(clusters, plines, to_pool[k]), "in_lane": False,
                         "pool_size_count": len(pool_ends)})
        info.update(lines=len(lines), computed=len(need), pool_clusters=len(pool_ends),
                    lines_matching_a_pool_cluster=sum(x is not None for x in to_pool))
        manifest.append(info)
        print(f"{cid} {bid}: lines {len(lines)} computed {len(need)} pool {len(pool_ends)} t={time.time() - t0:.0f}s",
              flush=True)
    with (out / "e3_candidates.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    vers = sorted({(m.get("engine_version"), m.get("config_sha256")) for m in manifest if m.get("status") == "OK"},
                  key=str)
    (out / "engine_manifest.json").write_text(json.dumps(
        {"a1_work": str(W), "engine_versions_configs": vers, "cases": manifest}, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "versions": vers, "seconds": round(time.time() - t0, 1)}))
    return 0


# --------------------------------------------------------------------------
# Jev (composite design on the engine's lines)
# --------------------------------------------------------------------------


def _states(rows, design="composite"):
    from research.jev.states_v2 import build_state_v2, record_id_v2

    seen, out = set(), []
    for r in rows:
        rid = record_id_v2(r)
        if rid not in seen:
            seen.add(rid)
            out.append((rid, build_state_v2(r, design)))
    return out


def cmd_jev(a: argparse.Namespace) -> int:
    from research.jev.runner_v2 import dry_run, live_run

    rows = read_candidates(Path(a.build) / "e3_candidates.jsonl")
    states = _states(rows)
    ad = Path(a.answers) / "composite"
    print(json.dumps(dry_run(states, ad / f"dry_run_{Path(a.build).name}", model=MODEL, variant="composite_v2")))
    if not a.live:
        return 0
    env = _key_from_env_file(Path(a.env_file)) if a.env_file else None
    res = live_run(states, ad / "cache", live=True, model=MODEL, variant="composite_v2", budget_usd=a.budget,
                   workers=a.workers, env=env)
    with (ad / "live_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"build": str(a.build), "time": time.time(), **res}) + "\n")
    print(json.dumps(res))
    return 0 if res["error"] is None else 3


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


ATOMIC = ("is_real_bend", "has_two_flat_flanges", "is_cut_edge_or_hole", "is_inner_or_duplicate_skin",
          "is_gentle_curvature_or_artifact")


def compute_scores(v2rows, e3rows, a1_bids: set[str], e3_answers: Path, bootstrap_note=None):
    """Scores on targets = v2 pool rows of the A1 scans + E3 rows, with the v2 protocol."""
    from research.jev.score_v2 import cv_scores, jev_scores

    case_list = _read(CASE_LIST)
    truth = load_truth(case_list, "ai_relabel")
    rows = list(v2rows) + list(e3rows)
    y = A.any_match_labels(rows, {c: t["lines"] for c, t in truth.items()})
    fam = [r["family_id"] for r in rows]
    nv = len(v2rows)
    Pc = [i for i in range(nv) if rows[i]["set"] == "pool" and rows[i]["split"] == "calib"]
    calib_fams = {fam[i] for i in Pc}
    T = [i for i in range(nv) if rows[i]["set"] == "pool" and rows[i]["case_id"] in a1_bids] + \
        list(range(nv, len(rows)))
    scores: dict[str, dict[int, float]] = {}
    for kind in ("logreg", "gbm"):
        def fp(tr, te, kind=kind):
            return A.fit_predict(kind, [rows[i] for i in tr], y[tr], [rows[i] for i in te])
        scores[f"clf_{kind}"] = cv_scores(fp, Pc, [fam[i] for i in Pc], T, [fam[i] for i in T], calib_fams)
    col = A.raw_matrix(rows, ["pool.pool_two_sided_transition_score"])[:, 0]
    scores["best_feature"] = {i: (float(col[i]) if math.isfinite(col[i]) else -math.inf) for i in T}
    jr = jev_scores(rows[:nv], STUDY / "answers" / "composite", "composite", MODEL)
    je = jev_scores(rows[nv:], e3_answers, "composite", MODEL)
    jr.update({nv + k: v for k, v in je.items()})
    have_T = [i for i in T if i in jr]
    Pc_have = [i for i in Pc if i in jr]

    def fpc(tr, te):
        from sklearn.linear_model import LogisticRegression
        Xtr = np.array([[_logit(jr[i][k]) for k in ATOMIC] for i in tr])
        Xte = np.array([[_logit(jr[i][k]) for k in ATOMIC] for i in te])
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
        return m.predict_proba(Xte)[:, 1]
    scores["jev_composite_combined"] = cv_scores(fpc, Pc_have, [fam[i] for i in Pc_have], have_T,
                                                 [fam[i] for i in have_T], calib_fams) if have_T else {}
    s1, s2 = scores["jev_composite_combined"], scores["clf_gbm"]
    scores["jev_plus_clf"] = {i: 0.5 * (s1[i] + s2[i]) for i in s1 if i in s2}
    missing = {"pool_rows_without_jev": sum(1 for i in T if i < nv and i not in jr),
               "e3_rows_without_jev": sum(1 for i in T if i >= nv and i not in jr)}
    return rows, scores, T, missing


def _prf(tp, fp, fn):
    return A._prf(tp, fp, fn)


def summarise(per_case: Mapping[str, Mapping[str, Any]], base: Mapping[str, Mapping[str, Any]] | None,
              rounds: int) -> dict[str, Any]:
    """per_case: {case: {family, tp, fp, fn, control, fp_lines, recall_only}} -> micro/macro + paired CIs."""
    comp = {c: v for c, v in per_case.items() if not v["control"] and not v["recall_only"]}
    tp = sum(v["tp"] for v in comp.values())
    fp = sum(v["fp"] for v in comp.values())
    fn = sum(v["fn"] for v in comp.values())
    p, r, f = _prf(tp, fp, fn)

    def tab(pc):
        t = defaultdict(lambda: [0, 0, 0])
        for v in pc.values():
            if v["control"] or v["recall_only"]:
                continue
            s = t[v["family"]]
            s[0] += v["tp"]; s[1] += v["fp"]; s[2] += v["fn"]
        return {k: tuple(x) for k, x in t.items()}
    ta = tab(per_case)
    ff = [_prf(*x)[2] for x in ta.values()]
    ro = [v for v in per_case.values() if v["recall_only"]]
    out = {"scans": len(comp), "families": len(ta), "tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
           "recall": round(r, 4), "micro_f1": round(f, 4),
           "family_macro_f1": round(float(np.mean(ff)), 4) if ff else None,
           "control_fp_lines": sum(v.get("fp_lines", 0) for v in per_case.values() if v["control"]),
           "recall_only_cases": {"cases": len(ro), "tp": sum(v["tp"] for v in ro), "fn": sum(v["fn"] for v in ro)}}
    if ta:
        out["ci95"] = A.family_bootstrap(ta, tab(base) if base is not None else None, rounds=rounds)
    return out


def broad_per_case(gold_b, preds_pc: Mapping[str, list]) -> dict[str, dict[str, Any]]:
    from domains.production.engine.eval.freeze.physical import broad_line_scores

    rep = broad_line_scores(gold_b, preds_pc)
    out = {}
    for r in rep["cases"]:
        ctrl = r["disposition"] == "ZERO_BEND_CONTROL"
        out[r["case_id"]] = {"family": r["family"], "control": ctrl,
                             "recall_only": r["disposition"] == "SCORED_RECALL_ONLY_PARTIAL_LABEL",
                             "tp": r.get("true_positive", 0), "fp": r.get("false_positive") or 0,
                             "fn": r.get("false_negative", 0), "fp_lines": r.get("false_positive_lines", 0),
                             "unscored_fp": r.get("unscored_unmatched_predictions"),
                             "unmatched_gold_ids": r.get("unmatched_gold_ids")}
    return out, {k: rep[k] for k in ("line_cases", "families", "true_positive", "false_positive",
                                     "false_negative", "micro", "family_macro_f1")}


def goldset_per_case(doc, ok_bids: set[str], preds_bid: Mapping[str, list], scan_of: Mapping[str, dict]):
    sys.path[:0] = [str(REPO_ROOT / "research/broad_baseline")]
    import score as sc  # noqa: WPS433

    a1 = {(c.get("broad") or {}).get("broad_case_id"): c for c in doc["cases"]
          if (c.get("broad") or {}).get("broad_case_id")}
    out = {}
    for g in _read(GOLD_LIST)["cases"]:
        pid = g.get("prediction_case_id") or g["case_id"]
        if pid not in a1 or not g.get("cad_path") or pid not in ok_bids:
            continue
        twin = pid != g["case_id"]
        pred = {"schema": "sherman.broad_baseline.eval_lines.v1", "status": "OK", "frame": "scan_raw",
                "scan": {} if twin else scan_of.get(pid, {}),
                "owned_bend_regions": [{"bend_id": f"l{k}", "line_start": l["line_start"], "line_end": l["line_end"]}
                                       for k, l in enumerate(preds_bid.get(pid) or [])]}
        r = sc.score_case({"case_id": g["case_id"], "tier": g.get("tier"), "family_id": g["family_id"],
                           "gold": g["gold"], "label_complete": g.get("label_complete", True)},
                          frame_records=None, preloaded_predictions=pred)
        d = r.get("disposition")
        out[g["case_id"]] = {"family": g["family_id"], "control": d == "ZERO_BEND_CONTROL",
                             "recall_only": d not in ("SCORED_COMPLETE_LABEL", "ZERO_BEND_CONTROL"),
                             "disposition": d, "tp": r.get("true_positive", 0) or 0,
                             "fp": r.get("false_positive", 0) or 0, "fn": r.get("false_negative", 0) or 0,
                             "fp_lines": r.get("false_positive_lines", 0), "prediction_case": pid}
    return out


def cmd_score(a: argparse.Namespace) -> int:
    build = Path(a.build)
    thr = frozen_thresholds()
    doc, broad = a1_cases()
    man = _read(build / "engine_manifest.json")
    e3rows = read_candidates(build / "e3_candidates.jsonl")
    bid_of = {m["case_ref"]: m["broad_case_id"] for m in man["cases"]}
    ok_pc = [m["case_ref"] for m in man["cases"] if m.get("status") == "OK"]
    a1_bids = {bid_of[c] for c in broad}
    v2rows = read_candidates(STUDY / "build" / "candidates.jsonl")
    rows, scores, T, missing = compute_scores(v2rows, e3rows, a1_bids, Path(a.answers) / "composite")
    nv = len(v2rows)
    by_case_pool, by_case_e3 = defaultdict(list), defaultdict(list)
    for i in T:
        (by_case_e3 if i >= nv else by_case_pool)[rows[i]["case_id"]].append(i)
    ends = np.array([[r["line_start"], r["line_end"]] for r in rows], float).reshape(-1, 2, 3)
    ran_bids = {bid_of[c] for c in ok_pc}
    selections = {}
    for arm, spec in ARMS.items():
        sel = {}
        for bid in a1_bids:
            eng = by_case_e3.get(bid, []) if bid in ran_bids else []
            sel[bid] = select_arm(spec, eng, by_case_pool.get(bid, []), ends, scores, thr)
        selections[arm] = sel
    line = lambda i: {"bend_id": rows[i]["cand"], "line_start": rows[i]["line_start"], "line_end": rows[i]["line_end"]}
    gold_all = [g for g in doc["broad_gold_cases"] if g["case_id"] in set(broad)]
    v2split = {m["case_ref"]: m.get("v2_split") for m in man["cases"]}
    scan_of = {m["broad_case_id"]: {"sha256": next(c["scan_sha256"] for c in doc["cases"] if c["case_ref"] == m["case_ref"])}
               for m in man["cases"]}
    pc_of_bid = {v: k for k, v in bid_of.items()}
    cohorts_b = {"BROAD_all26": list(broad), "BROAD_engine_ran": ok_pc,
                 "BROAD_engine_ran_v2test": [c for c in ok_pc if v2split.get(c) == "eval"],
                 "BROAD_engine_ran_v2calib": [c for c in ok_pc if v2split.get(c) == "calib"]}
    res = {"schema": "jev_ab_e3_results_v1", "flags": ["exploratory_not_unseen (CONSUMED development scans)"],
           "engine": man["engine_versions_configs"], "a1_work": man["a1_work"],
           "thresholds_frozen_from": "docs/production/runlog/jev_ab_v2/design_choice.json",
           "thresholds": {k: thr[v] for k, v in {"b": "lane+jev_composite_combined_filter",
                                                 "c": "lane+best_feature_filter", "d_lr": "pool+clf_logreg",
                                                 "d_jevclf": "pool+jev_plus_clf"}.items()},
           "jev_missing": missing, "arms": {}, "per_case": {}, "cohorts": {}}
    base = {}
    for arm in ARMS:
        preds_pc = {pc_of_bid[b]: [line(i) for i in v] for b, v in selections[arm].items()}
        pcb, frozen_agg = broad_per_case(gold_all, preds_pc)
        preds_bid = {b: [line(i) for i in v] for b, v in selections[arm].items()}
        gs = goldset_per_case(doc, ran_bids, preds_bid, scan_of)
        if arm == "a_engine_alone":
            base = {"broad": pcb, "gold": gs}
        out = {"frozen_scorer_BROAD_all26": frozen_agg}
        for cname, cc in cohorts_b.items():
            sub = {c: pcb[c] for c in cc if c in pcb}
            out[cname] = summarise(sub, {c: base["broad"][c] for c in sub}, a.bootstrap)
            res["cohorts"][cname] = len(sub)
        out["GOLD_SET_V1_A1_view_engine_ran"] = summarise(gs, {c: base["gold"][c] for c in gs}, a.bootstrap)
        res["cohorts"]["GOLD_SET_V1_A1_view_engine_ran"] = len(gs)
        n_e3 = sum(1 for v in selections[arm].values() for i in v if i >= nv)
        n_pool = sum(1 for v in selections[arm].values() for i in v if i < nv)
        out["kept_lines"] = {"engine": n_e3, "pool_added": n_pool}
        res["arms"][arm] = out
        res["per_case"][arm] = {"BROAD": pcb, "GOLD_SET_V1": gs}
    # AUCs of the filter scores on the engine's own lines (label: line matches a frozen BROAD gold line)
    from domains.production.engine.eval.line_metric import match_lines, prepare_gold
    from domains.production.engine.eval.freeze.physical import _gold_case_objects

    usable, _ = prepare_gold(_gold_case_objects(gold_all))
    lab = {}
    for b, idx in by_case_e3.items():
        pc = pc_of_bid[b]
        g = next(x for x in gold_all if x["case_id"] == pc)
        if pc not in usable or g.get("zero_bend_control") or not g.get("label_complete", True):
            continue
        m = match_lines(usable[pc], [line(i) for i in idx])
        hit = {x["prediction_id"] for x in m["matches"]}
        for i in idx:
            lab[i] = int(rows[i]["cand"] in hit)
    auc = {}
    for name in ("jev_composite_combined", "best_feature", "clf_logreg", "clf_gbm", "jev_plus_clf"):
        idx = [i for i in lab if i in scores[name] and math.isfinite(scores[name][i])]
        yy = np.array([lab[i] for i in idx])
        auc[name] = {"n": len(idx), "pos": int(yy.sum()) if len(idx) else 0,
                     "auc": None if not len(idx) or A.auc(yy, [scores[name][i] for i in idx]) is None
                     else round(A.auc(yy, [scores[name][i] for i in idx]), 4)}
    res["auc_on_engine_lines_vs_BROAD_gold"] = auc
    # rescue ceiling: of the gold lines the engine missed on scans it ran, how many does any pool
    # candidate match (frozen matcher criteria, not one-to-one), and how many does each arm recover
    ceil = {"engine_missed_gold_lines": 0, "matched_by_some_pool_candidate": 0, "per_case": {}}
    for pc in ok_pc:
        r0 = base["broad"].get(pc)
        if not r0 or r0["control"] or pc not in usable:
            continue
        miss = set(r0.get("unmatched_gold_ids") or [])
        gl = [g for g in usable[pc] if str(g.get("bend_id")) in miss]
        pidx = by_case_pool.get(bid_of[pc], [])
        hit = 0
        if gl and pidx:
            m = pair_metrics_matrix(np.array([[g["line_start"], g["line_end"]] for g in gl], float), ends[pidx])
            hit = int(match_mask(m).any(axis=1).sum())
        ceil["engine_missed_gold_lines"] += len(gl)
        ceil["matched_by_some_pool_candidate"] += hit
        ceil["per_case"][pc] = {"missed": len(gl), "in_pool": hit}
    res["rescue_ceiling_engine_ran"] = ceil
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    for arm, o in res["arms"].items():
        s = " | ".join(f"{c}: F1 {o[c]['micro_f1']} fm {o[c]['family_macro_f1']} P {o[c]['precision']} R {o[c]['recall']}"
                       for c in ("BROAD_all26", "BROAD_engine_ran", "GOLD_SET_V1_A1_view_engine_ran"))
        print(f"{arm:32s} {s}")
    print(json.dumps({"missing": missing, "auc": auc}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m research.jev.e3_ab_v1")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--a1-work", required=True, help="A1 physical work dir (contains physical/E3, compiled/dev)")
    b.add_argument("--out", required=True)
    j = sub.add_parser("jev")
    j.add_argument("--build", required=True)
    j.add_argument("--answers", required=True)
    j.add_argument("--live", action="store_true")
    j.add_argument("--env-file", default="")
    j.add_argument("--budget", type=float, default=1.0)
    j.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("--build", required=True)
    s.add_argument("--answers", required=True)
    s.add_argument("--bootstrap", type=int, default=2000)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"build": cmd_build, "jev": cmd_jev, "score": cmd_score}[a.command](a)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
