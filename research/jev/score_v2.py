"""Arms, thresholds and results for the Jev input-optimisation study (v2).

Run after ``input_opt_v2 build``/``states`` and the Jev calls. Offline: reads the
candidate table, the cached Jev answers and the truth, and writes results.json.

Protocol (all choices on calibration families only):

- trained scores (classifiers, the composite-Jev combiner) on calibration rows are
  leave-one-family-out; test-family rows are scored by a model fitted on all
  calibration rows;
- the single best feature and its sign are chosen by calibration AUC;
- the Jev input design is chosen by calibration pool AUC (frozen in
  ``design_choice.json`` before any test-family call);
- every keep threshold maximises calibration micro line-F1 of its arm;
- test families are reported separately with family-level bootstrap CIs, deltas
  paired against "lane alone" on the same resampled families.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.jev import arms_v2 as A
from research.jev.input_opt_v2 import CASE_LIST, REPO_ROOT, _read, load_truth, read_candidates, sha_file
from research.jev.labels_v1 import read_jsonl
from research.jev.states_v2 import record_id_v2

OLD_AB = REPO_ROOT / "docs" / "production" / "runlog" / "jev_ab"
JEV_DESIGNS = ("typed", "typed_reading", "composite")
ATOMIC = ("has_two_flat_flanges", "is_cut_edge_or_hole", "is_inner_or_duplicate_skin",
          "is_gentle_curvature_or_artifact")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cv_scores(fit_predict, train_idx, groups, target_idx, target_groups, calib_fams):
    """Leave-one-calibration-family-out for calibration targets; full-calibration fit for the rest."""
    out = {}
    tg = np.asarray(target_groups)
    trg = np.asarray(groups)
    for fam in sorted(set(tg.tolist())):
        tsel = [target_idx[k] for k in np.flatnonzero(tg == fam)]
        trsel = [train_idx[k] for k in np.flatnonzero(trg != fam)] if fam in calib_fams else list(train_idx)
        pred = fit_predict(trsel, tsel)
        out.update(dict(zip(tsel, pred)))
    return out


def old_jev_scores(rows) -> tuple[dict[int, float], float]:
    cands = {(c["case_id"], c["bend_id"]): c["record_id"] for c in read_jsonl(OLD_AB / "export" / "candidates.jsonl")
             if c["lane"] == "best_f1"}
    ans = {a["record_id"]: float(a["answers"]["is_real_bend"]["noul"])
           for a in read_jsonl(OLD_AB / "answers" / "normal" / "answers.jsonl")}
    t = float(_read(OLD_AB / "operating_points.json")["thresholds"]["best_f1"]["jev"])
    out = {}
    for i, r in enumerate(rows):
        rid = cands.get((r["case_id"], r["cand"]))
        if rid in ans:
            out[i] = ans[rid]
    return out, t


def jev_scores(rows, answers_dir: Path, design: str, model: str) -> dict[int, dict[str, float]]:
    from research.jev.runner_v2 import load_answers
    from research.jev.states_v2 import build_state_v2

    variant = "composite_v2" if design == "composite" else "v2"
    states = [(record_id_v2(r), build_state_v2(r, design)) for r in rows]
    ans = load_answers(states, answers_dir / "cache", model=model, variant=variant)
    out = {}
    for i, (rid, _s) in enumerate(states):
        if rid in ans:
            out[i] = {k: float(v["noul"]) for k, v in ans[rid]["answers"].items()}
    return out


def cmd_score(args: argparse.Namespace) -> int:
    rows = read_candidates(Path(args.build) / "candidates.jsonl")
    case_list = _read(args.case_list)
    cases = [c["case_id"] for c in case_list["cases"]]
    family_of = {c["case_id"]: c["family_id"] for c in case_list["cases"]}
    split_of = {r["case_id"]: r["split"] for r in rows}
    for c in case_list["cases"]:
        split_of.setdefault(c["case_id"], "eval")
    truths = {"ai_relabel": load_truth(case_list, "ai_relabel")}
    gs_dir = REPO_ROOT / "data" / "bend_corpus" / "annotations" / "ai_bend_lines_gold_v1"
    if gs_dir.is_dir() and any(gs_dir.glob("*.json")):
        truths["gold_set_v1"] = load_truth(case_list, "gold_set_v1")
    truth = truths["ai_relabel"]
    gold_lines = {c: truth[c]["lines"] for c in cases}
    calib_cases = [c for c in cases if split_of[c] == "calib"]
    test_cases = [c for c in cases if split_of[c] == "eval"]
    calib_fams = {family_of[c] for c in calib_cases}
    y = A.any_match_labels(rows, gold_lines)
    P = [i for i, r in enumerate(rows) if r["set"] == "pool"]
    L = [i for i, r in enumerate(rows) if r["set"] == "lane"]
    Pc = [i for i in P if split_of[rows[i]["case_id"]] == "calib"]
    fam = [family_of[rows[i]["case_id"]] for i in range(len(rows))]

    # ---- scores --------------------------------------------------------------------------
    scores: dict[str, dict[int, float]] = {}
    meta: dict[str, Any] = {}
    for kind in ("logreg", "gbm"):
        def fp(tr, te, kind=kind):
            return A.fit_predict(kind, [rows[i] for i in tr], y[tr], [rows[i] for i in te])
        scores[f"clf_{kind}"] = cv_scores(fp, Pc, [fam[i] for i in Pc], P + L, [fam[i] for i in P + L], calib_fams)
    # single best feature (sign and choice on calibration pool rows)
    Xc = A.raw_matrix([rows[i] for i in Pc])
    best = None
    for j, name in enumerate(A.FEATURE_NAMES):
        col = np.where(np.isnan(Xc[:, j]), np.nanmin(Xc[:, j]) if np.isfinite(np.nanmin(Xc[:, j])) else 0, Xc[:, j])
        a = A.auc(y[Pc], col)
        if a is None:
            continue
        sign = 1.0 if a >= 0.5 else -1.0
        aa = max(a, 1 - a)
        if best is None or aa > best[0]:
            best = (aa, name, sign, j)
    Xall = A.raw_matrix(rows)
    colall = Xall[:, best[3]] * best[2]
    worst = np.nanmin(colall)
    colall = np.where(np.isnan(colall), worst - 1.0, colall)
    scores["best_feature"] = {i: float(colall[i]) for i in P + L}
    meta["best_feature"] = {"name": best[1], "sign": best[2], "calib_pool_auc": round(best[0], 4)}
    # pool generator prior (reference: the pool's own descriptors)
    prior = {}
    for i in P + L:
        p = rows[i].get("pool") or {}
        ts = p.get("pool_two_sided_transition_score") or 0.0
        prior[i] = float(ts * math.log1p(rows[i]["feat"]["line_length_mm"]) * math.log1p(p.get("pool_member_count") or 0))
    scores["pool_prior"] = prior
    # Jev designs
    model = args.model
    jev_raw = {}
    for d in JEV_DESIGNS:
        ad = Path(args.answers) / d
        if not ad.is_dir():
            continue
        jr = jev_scores(rows, ad, d, model)
        if not jr:
            continue
        jev_raw[d] = jr
        scores[f"jev_{d}"] = {i: v["is_real_bend"] for i, v in jr.items()}
        if d == "composite":
            have = [i for i in P + L if i in jr]
            Pc_have = [i for i in Pc if i in jr]

            def fpc(tr, te, jr=jr):
                from sklearn.linear_model import LogisticRegression
                Xtr = np.array([[_logit(jr[i][k]) for k in ("is_real_bend", *ATOMIC)] for i in tr])
                Xte = np.array([[_logit(jr[i][k]) for k in ("is_real_bend", *ATOMIC)] for i in te])
                m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
                return m.predict_proba(Xte)[:, 1]
            scores["jev_composite_combined"] = cv_scores(fpc, Pc_have, [fam[i] for i in Pc_have], have,
                                                         [fam[i] for i in have], calib_fams)
    old, old_t = old_jev_scores(rows)
    scores["old_jev_v1"] = old

    # ---- design choice (calibration pool AUC) ------------------------------------------------
    def calib_auc(name):
        s = scores.get(name, {})
        idx = [i for i in Pc if i in s]
        return A.auc(y[idx], [s[i] for i in idx]) if idx else None
    cand_jev = [n for n in scores if n.startswith("jev_") and n != "jev_composite"]
    jev_choice = json.loads(Path(args.design_choice).read_text())["jev_score"] if args.design_choice and \
        Path(args.design_choice).is_file() else (max(cand_jev, key=lambda n: calib_auc(n) or 0) if cand_jev else None)
    clf_choice = max(("clf_logreg", "clf_gbm"), key=lambda n: calib_auc(n) or 0)
    meta["calib_pool_auc"] = {n: (round(calib_auc(n), 4) if calib_auc(n) is not None else None) for n in scores}
    meta["jev_choice"] = jev_choice
    meta["clf_choice"] = clf_choice
    if jev_choice:
        s1, s2 = scores[jev_choice], scores[clf_choice]
        scores["jev_plus_clf"] = {i: 0.5 * (s1[i] + s2[i]) for i in s1 if i in s2}

    # ---- arms --------------------------------------------------------------------------------
    arms_spec = [("lane_alone", "lane", None, None)]
    arms_spec.append(("lane+old_jev_v1_filter", "lane", "old_jev_v1", old_t))
    for n in [k for k in scores if k.startswith("jev_")] + ["jev_plus_clf", clf_choice, "best_feature"]:
        if n in scores:
            arms_spec.append((f"lane+{n}_filter", "lane", n, "fit"))
    arms_spec.append(("pool+pool_prior", "pool", "pool_prior", "fit"))
    for n in [k for k in scores if k.startswith("jev_")] + ["jev_plus_clf", "clf_logreg", "clf_gbm", "best_feature"]:
        if n in scores:
            arms_spec.append((f"pool+{n}", "pool", n, "fit"))
    results = {"schema": "jev_input_opt_v2_results_v1",
               "flags": ["exploratory_not_unseen (CONSUMED development scans)"],
               "inputs": {"candidates_sha256": sha_file(Path(args.build) / "candidates.jsonl"),
                          "case_list_sha256": sha_file(args.case_list), "truths": sorted(truths),
                          "jev_model": model},
               "split": {"calib_families": sorted(calib_fams),
                         "test_families": sorted({family_of[c] for c in test_cases}),
                         "calib_scans": len(calib_cases), "test_scans": len(test_cases)},
               "meta": meta, "arms": {}, "auc": {}, "ceilings": {}}
    # Jev answers may be incomplete on test families (the TypeSafe credit balance ran out mid-run):
    # Jev arms are reported only on the test scans where every candidate of every Jev design is answered.
    jev_names = [n for n in scores if n.startswith("jev_")]
    rows_of = defaultdict(list)
    for i in P + L:
        rows_of[rows[i]["case_id"]].append(i)
    jev_complete = [c for c in test_cases if jev_names and all(all(i in scores[n] for i in rows_of[c])
                                                                for n in jev_names)]
    results["split"]["test_scans_jev_complete"] = len(jev_complete)
    results["split"]["test_families_jev_complete"] = sorted({family_of[c] for c in jev_complete})
    cohorts = {"TEST": test_cases, "TEST_JEV_COMPLETE": jev_complete, "CALIB": calib_cases, "ALL": cases}

    def jev_dependent(name):
        return name is not None and (name.startswith("jev_") or name == "jev_plus_clf")

    def cohort_complete(name, cc):
        return not jev_dependent(name) or all(all(i in scores[name] for i in rows_of[c]) for c in cc)
    # ceilings: recall with every candidate of a set kept
    for set_name, idxs in (("lane", L), ("pool", P)):
        sel = defaultdict(list)
        for i in idxs:
            sel[rows[i]["case_id"]].append(i)
        for tname, tr in truths.items():
            pc = A.score_cases(sel, rows, tr, cases)
            for cname, cc in cohorts.items():
                agg = A.aggregate({c: pc[c] for c in cc}, family_of)
                results["ceilings"].setdefault(tname, {}).setdefault(set_name, {})[cname] = {
                    "recall_at_set": agg["recall"], "candidates": sum(1 for i in idxs if rows[i]["case_id"] in cc),
                    "perfect_selector_f1": round(2 * agg["recall"] / (1 + agg["recall"]), 4) if agg["recall"] else 0.0}
    base_tables = {}
    for arm, set_name, sname, thr in arms_spec:
        idxs = L if set_name == "lane" else P
        dedupe = set_name == "pool"
        if sname is None:
            sel = defaultdict(list)
            for i in idxs:
                sel[rows[i]["case_id"]].append(i)
            t = None
        else:
            s = scores[sname]
            sub_rows = [rows[i] for i in idxs if i in s]
            sub_idx = [i for i in idxs if i in s]
            sub_s = [s[i] for i in sub_idx]
            if thr == "fit":
                t, trace = A.fit_threshold(sub_rows, sub_s, truth, calib_cases, family_of, dedupe=dedupe)
            else:
                t = thr
            local = A.select(sub_rows, sub_s, t, dedupe=dedupe)
            sel = {c: [sub_idx[k] for k in v] for c, v in local.items()}
            if set_name == "lane":  # lane rows without a score (should not happen) are kept, as a filter would
                missing = [i for i in idxs if i not in s]
                for i in missing:
                    sel.setdefault(rows[i]["case_id"], []).append(i)
        out = {"candidate_set": set_name, "score": sname, "keep_threshold": t, "dedupe_nms": dedupe}
        for tname, tr in truths.items():
            pc = A.score_cases(sel, rows, tr, cases)
            for cname, cc in cohorts.items():
                if not cohort_complete(sname, cc):
                    out.setdefault(tname, {})[cname] = {"not_reported": "Jev answers incomplete in this cohort "
                                                        "(TypeSafe credits ran out)"}
                    continue
                sub = {c: pc[c] for c in cc}
                agg = A.aggregate(sub, family_of)
                tab = A.fam_table(sub, family_of)
                key = (tname, cname)
                if arm == "lane_alone":
                    base_tables[key] = tab
                if cname != "CALIB":
                    agg["ci95"] = A.family_bootstrap(tab, None if arm == "lane_alone" else base_tables[key],
                                                     rounds=args.bootstrap)
                out.setdefault(tname, {})[cname] = agg
        results["arms"][arm] = out
    # AUCs (any-match label), per candidate set, test and calib cohorts
    for sname, s in scores.items():
        for set_name, idxs in (("pool", P), ("lane", L)):
            for cname, cc in (("TEST", set(test_cases)), ("TEST_JEV_COMPLETE", set(jev_complete)),
                              ("CALIB", set(calib_cases))):
                if cname == "TEST" and not cohort_complete(sname, test_cases):
                    continue
                idx = [i for i in idxs if i in s and rows[i]["case_id"] in cc]
                if not idx:
                    continue
                yy = y[idx]
                ss = [s[i] for i in idx]
                a = A.auc(yy, ss)
                results["auc"].setdefault(sname, {}).setdefault(set_name, {})[cname] = {
                    "n": len(idx), "pos": int(yy.sum()), "auc": None if a is None else round(a, 4),
                    "ci95": A.auc_ci(yy, ss, [fam[i] for i in idx], rounds=args.bootstrap) if cname != "CALIB" else None}
    spend = {}
    for d in JEV_DESIGNS:
        cdir = Path(args.answers) / d / "cache"
        if cdir.is_dir():
            toks = [int(json.loads(f.read_text()).get("usage", {}).get("input_tokens", 0)) for f in cdir.glob("*.json")]
            spend[d] = {"responses": len(toks), "input_tokens": sum(toks), "usd": round(sum(toks) / 1e6 * 0.042, 4)}
    results["jev_spend_all_cached_responses"] = spend
    Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    for arm, o in results["arms"].items():
        line = f"{arm:42s} t={o['keep_threshold']!s:10.8s}"
        for cname in ("TEST", "TEST_JEV_COMPLETE", "CALIB"):
            t = o["ai_relabel"][cname]
            line += f" | {cname} " + ("n/a" if "f1" not in t else f"F1={t['f1']:.3f} P={t['precision']:.2f} R={t['recall']:.2f}")
        print(line)
    print(json.dumps(meta, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m research.jev.score_v2")
    p.add_argument("--build", required=True)
    p.add_argument("--answers", required=True, help="dir with <design>/cache/")
    p.add_argument("--case-list", default=str(CASE_LIST))
    p.add_argument("--model", default="jev-1.13.0")
    p.add_argument("--design-choice")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out", required=True)
    return cmd_score(p.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
