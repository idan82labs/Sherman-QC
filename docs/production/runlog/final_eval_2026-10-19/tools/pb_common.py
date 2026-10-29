"""Shared per-bend correctness (a1/iter_7 per_bend_all.py definition) + family bootstrap. Report-only."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

FORMED = {"FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED"}


def per_bend_rows(cid, family, result, truth_of, line_match_ids, line_src):
    """Rows for one engine-OK case. truth_of(uid) -> FORMED/NOT_FORMED/NOT_DETERMINABLE/None(skip).
    line_match_ids: set of engine bend_uids whose published line matches gold (None = criterion not applied)."""
    rows = []
    for b in result["bends"]:
        uid = b["bend_uid"]
        t = truth_of(uid)
        if t is None or t == "NOT_DETERMINABLE":
            continue
        chs = b["characteristics"]
        ang = [c for c in chs if c["kind"] == "INCLUDED_ANGLE"]
        fl = [c for c in chs if c["kind"] == "FLANGE_LENGTH"]
        decs = [c.get("decision") for c in chs]
        line_ok = None if line_match_ids is None else uid in line_match_ids
        st = b["state"]
        if t == "FORMED":
            ang_ok = bool(ang) and ang[0].get("state") == "MEASURED"
            fl_strict = all(c.get("state") == "MEASURED" for c in fl)
            fl_riser = all(c.get("state") == "MEASURED" or c.get("reason") == "UNSUPPORTED_FEATURE" for c in fl)
            base = st in FORMED and ang_ok and line_ok is not False
            ok_s, ok_r = base and fl_strict, base and fl_riser
        else:
            ok_s = ok_r = st == "NOT_YET_FORMED"
        fp = "PASS" in decs and (t == "NOT_FORMED" or line_ok is False)
        rows.append({"case": cid, "family": family, "label": b["label"], "truth": t, "state": st,
                     "line_ok": line_ok, "line_src": line_src, "correct_strict": ok_s, "correct_riser_aware": ok_r,
                     "false_pass_proxy": fp, "any_pass": "PASS" in decs})
    return rows


def summarize(rows, missing):
    """missing: list of {case, family, bends} for cases the engine did not run/refused (all bends incorrect)."""
    n_run = len(rows)
    n_miss = sum(m["bends"] for m in missing)
    fam = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        fam[r["family"]][0] += 1
        fam[r["family"]][1] += int(r["correct_riser_aware"])
        fam[r["family"]][2] += int(r["correct_strict"])
    famh = {k: list(v) for k, v in fam.items()}
    for m in missing:
        famh.setdefault(m["family"], [0, 0, 0])
        famh[m["family"]][0] += m["bends"]
    cs = sum(r["correct_strict"] for r in rows)
    cr = sum(r["correct_riser_aware"] for r in rows)
    out = {
        "engine_ran": {"bends": n_run, "correct_riser_aware": cr, "correct_strict": cs,
                       "rate_riser_aware": cr / n_run if n_run else None, "rate_strict": cs / n_run if n_run else None,
                       "family_macro_riser_aware": (sum(v[1] / v[0] for v in fam.values()) / len(fam)) if fam else None},
        "headline_with_misses": {"bends": n_run + n_miss, "correct_riser_aware": cr, "correct_strict": cs,
                                 "rate_riser_aware": cr / (n_run + n_miss) if n_run + n_miss else None,
                                 "rate_strict": cs / (n_run + n_miss) if n_run + n_miss else None,
                                 "family_macro_riser_aware": (sum(v[1] / v[0] for v in famh.values() if v[0]) /
                                                              len([v for v in famh.values() if v[0]])) if famh else None},
        "missing_cases": missing,
        "false_pass_proxy": sum(r["false_pass_proxy"] for r in rows),
        "false_pass_bends": [f"{r['case']}:{r['label']}:{r['truth'][:3]}->{r['state'][:3]}" for r in rows if r["false_pass_proxy"]],
        "bends_with_any_PASS": sum(r["any_pass"] for r in rows),
        "per_family": {k: {"bends": v[0], "correct_riser_aware": v[1]} for k, v in sorted(famh.items())},
    }
    out["_fam_counts"] = famh
    return out


def line_stats(case_rows):
    """case_rows: [{family, tp, fp, fn}] complete-label line cases -> micro + family-macro F1."""
    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    tp = sum(r["tp"] for r in case_rows); fp = sum(r["fp"] for r in case_rows); fn = sum(r["fn"] for r in case_rows)
    fam = defaultdict(lambda: [0, 0, 0])
    for r in case_rows:
        fam[r["family"]][0] += r["tp"]; fam[r["family"]][1] += r["fp"]; fam[r["family"]][2] += r["fn"]
    return {"cases": len(case_rows), "families": len(fam), "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": tp / (tp + fp) if tp + fp else None, "micro_recall": tp / (tp + fn) if tp + fn else None,
            "micro_f1": f1(tp, fp, fn), "family_macro_f1": (sum(f1(*v) for v in fam.values()) / len(fam)) if fam else None}


def bootstrap_lines(case_rows, n_boot=10000, seed=20261018):
    fams = sorted({r["family"] for r in case_rows})
    by = defaultdict(list)
    for r in case_rows:
        by[r["family"]].append(r)
    rng = random.Random(seed)
    mi, ma = [], []
    for _ in range(n_boot):
        pick = [rng.choice(fams) for _ in fams]
        rows = []
        for i, f in enumerate(pick):
            rows += [dict(r, family=f"{f}#{i}") for r in by[f]]
        s = line_stats(rows)
        mi.append(s["micro_f1"]); ma.append(s["family_macro_f1"])
    q = lambda xs: [round(sorted(xs)[int(0.025 * len(xs))], 4), round(sorted(xs)[int(0.975 * len(xs)) - 1], 4)]
    return {"resample_unit": "family", "n_boot": n_boot, "seed": seed, "micro_f1_95ci": q(mi), "family_macro_f1_95ci": q(ma)}


def bootstrap_bends(fam_counts, n_boot=10000, seed=20261018):
    """fam_counts: {family: [bends, correct_riser, correct_strict]} -> CI of pooled riser-aware rate."""
    fams = sorted(k for k, v in fam_counts.items() if v[0])
    rng = random.Random(seed)
    xs = []
    for _ in range(n_boot):
        pick = [rng.choice(fams) for _ in fams]
        n = sum(fam_counts[f][0] for f in pick); c = sum(fam_counts[f][1] for f in pick)
        xs.append(c / n if n else 0.0)
    xs.sort()
    return {"rate_riser_aware_95ci": [round(xs[int(0.025 * n_boot)], 4), round(xs[int(0.975 * n_boot) - 1], 4)],
            "resample_unit": "family", "n_boot": n_boot, "seed": seed}
