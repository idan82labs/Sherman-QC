"""Per-line error taxonomy of the pinned lanes (v1, v3 rev3) over the v4 pool, from the candidate cache.

Each FN gold line and each FP line of a lane is attributed to the pipeline component that produced it, by replaying
the pinned selection on the cache step by step (no geometry; the replay reproduces the stored outputs, see
validate_cache.py). Components:

FN (gold line unmatched by the lane):
  NO_CAND[:SPAN|:OFFSET|:ANGLE|:FAR]  no pooled candidate matches it at its snapped placement (generator / extent miss;
                                      the suffix is the gate the nearest candidate fails)
  SNAP_LOST                           a candidate matches at its raw placement but not after the D1 snap
  CLS_REJECT                          a matching candidate exists but the pooled GBM gives p < 0.5
  SWEEP_GATE   (v3)                   p >= 0.5 but v2's sweep gate (station sweep < 0.25) removed it
  JOIN_ALTERED (v3)                   selected, then the v2 join merged it into a line that no longer matches
  EXT_ALTERED  (v3)                   selected, then EXT extended it past the gates
  EXT_DEDUPE   (v3)                   selected, then dropped by EXT's dedupe (an extended line "duplicates" it)
  JOIN_DEDUPE  (v3)                   selected, then dropped by the join dedupe
  TAKEN                               a matching line was output but the one-to-one assignment gave it to another gold
FP (output line unmatched): by origin  CLS (a kept candidate) / JOIN / EXT / Z3R, by candidate source (z3, z2, z2+z3)
  and by the frozen diagnosis cause (SPURIOUS, SPAN_LT_60PCT, OFFSET_GT_4MM, TAKEN_BY_OTHER = duplicate).
Writes results/DIAGNOSIS.json.  usage (repo root): .venv/bin/python -B docs/production/runlog/zs_v4/tools/diagnose.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v4 import cache_io as CI, evalkit as EK  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import pair_metrics  # noqa: E402

GATES = dict(angle=5.0, off=4.0, gov=0.6, pov=0.6)


def gate_fail(m: dict) -> str | None:
    if m["axis_angle_deg"] > GATES["angle"]:
        return "ANGLE"
    if m["transverse_offset_mm"] > GATES["off"]:
        return "OFFSET"
    if m["gold_span_overlap"] < GATES["gov"] or m["prediction_span_overlap"] < GATES["pov"]:
        return "SPAN"
    return None


def metrics(g: dict, a, b) -> dict:
    return pair_metrics(np.array([g["line_start"], g["line_end"]], float), np.array([a, b], float))


def matches(g: dict, a, b) -> bool:
    try:
        return gate_fail(metrics(g, a, b)) is None
    except ValueError:
        return False


def nearest_fail(g: dict, segs: list) -> str:
    best, cause = None, "FAR"
    for a, b in segs:
        try:
            m = metrics(g, a, b)
        except ValueError:
            continue
        if m["axis_angle_deg"] > 20 or m["transverse_offset_mm"] > 15:
            continue
        c = m["axis_angle_deg"] / 5 + m["transverse_offset_mm"] / 4 + (1 - min(m["gold_span_overlap"], 1)) + (
            1 - min(m["prediction_span_overlap"], 1))
        if best is None or c < best:
            best, cause = c, gate_fail(m) or "TAKEN"
    return cause


def v3_steps(r: dict) -> dict:
    """Replay v3: probs, sweep-gated selection, postprocess (join/dedupe), EXT, Z3R."""
    from research.zeroshot.zs_lane_v2.postproc import PostParams, postprocess, features, sweep_matrix
    from research.zeroshot.zs_lane_v3.fixes import FixParams, extend, rescue
    from research.zeroshot.zs_lane_v3.core import z3_probs
    M = CI.pinned_models()
    v = M[("zs_lane_v3", "params")]["variant"]
    C, X, ev = r["candidates"], r["X"], r.get("evidence")
    if not C:
        return {"prob": np.zeros(0), "post": [], "ext": [], "final": [], "sel": []}
    SW = sweep_matrix((ev or {}).get("sweep"), len(C))
    prob = M[("zs_lane_v3", "zs")]["model"].predict(features(X, SW, "base"))
    p3 = z3_probs(M[("zs_lane_v3", "z3")], C, X)
    prm = PostParams(**v["post"])
    sel = [i for i in range(len(C)) if prob[i] >= 0.5]
    sw = (ev or {}).get("sweep")
    gated = [i for i in sel if not (sw is not None and sw[i] is not None and sw[i][0] < prm.sweep_min)]
    post = postprocess(C, prob, 0.5, ev, prm)
    for o in post:
        o.pop("_i", None)
    fx = FixParams(**v["fix"])
    ext = extend(post, C, r.get("z2_lines") or [], fx)
    fin = rescue(ext, C, prob, p3, sw, fx)
    return {"prob": prob, "p3": p3, "sel": sel, "gated": gated, "post": post, "ext": ext, "final": fin}


def main() -> None:
    pool = EK.load_pool()
    sc = EK.Scorer(pool)
    rows_fn, rows_fp, per_scan = [], [], {}
    for s in pool:
        cid = s["case_id"]
        d = CI.load(cid)
        if d is None:
            continue
        gold = EK.gold_lines(s)
        out = {}
        # ---------------- v1
        r1 = d["v1"]
        C1 = r1.get("candidates") or []
        M = CI.pinned_models()
        p1 = M[("zs_lane_v1", "zs")]["model"].predict(r1["X"]) if C1 else np.zeros(0)
        lines1 = [{"start": c["start_s"], "end": c["end_s"], "_i": i} for i, c in enumerate(C1) if p1[i] >= 0.5]
        # ---------------- v3
        rv = d.get("v3x") or d["v4"]
        st = v3_steps(rv)
        C3 = rv.get("candidates") or []
        lines3 = [{"start": l["start_s"], "end": l["end_s"], "op": l.get("op"), "fix": l.get("fix"),
                   "parts": l.get("parts")} for l in st["final"]]
        for lane, C, lines, prob in (("v1", C1, lines1, p1), ("v3", C3, lines3, st["prob"])):
            rec = sc.score(cid, [{"start": l["start"], "end": l["end"]} for l in lines], diagnose=True)
            out[lane] = {k: rec.get(k) for k in ("true_positive", "false_positive", "false_negative", "disposition")}
            if rec.get("disposition") not in ("SCORED_COMPLETE_LABEL", "ZERO_BEND_CONTROL"):
                continue
            matched_g = {m["gold_id"] for m in rec.get("matches") or []}
            matched_p = {m["prediction_id"] for m in rec.get("matches") or []}
            snapped = [(c["start_s"], c["end_s"]) for c in C]
            raw = [(c["start"], c["end"]) for c in C]
            for gi, g in enumerate(gold):
                gid = str(g.get("bend_id") or f"G{gi + 1:02d}")
                if gid in matched_g:
                    continue
                cm = [i for i, (a, b) in enumerate(snapped) if matches(g, a, b)]
                if not cm:
                    if any(matches(g, a, b) for a, b in raw):
                        comp = "SNAP_LOST"
                    else:
                        comp = "NO_CAND:" + nearest_fail(g, snapped)
                elif all(prob[i] < 0.5 for i in cm):
                    comp = "CLS_REJECT"
                elif lane == "v1":
                    comp = "TAKEN"
                else:
                    keep = [i for i in cm if prob[i] >= 0.5]
                    if not any(i in st["gated"] for i in keep):
                        comp = "SWEEP_GATE"
                    else:
                        po = [l for l in st["post"] if any(i in l["parts"] for i in keep)]
                        if not po:
                            comp = "JOIN_DEDUPE"
                        elif not any(matches(g, l["start_s"], l["end_s"]) for l in po):
                            comp = "JOIN_ALTERED"
                        else:
                            ex = [l for l in st["ext"] if any(i in l["parts"] for i in keep)]
                            if not ex:
                                comp = "EXT_DEDUPE"
                            elif not any(matches(g, l["start_s"], l["end_s"]) for l in ex):
                                comp = "EXT_ALTERED"
                            else:
                                comp = "TAKEN"
                cause = next((x["cause"] for x in rec.get("fn_rows") or [] if x["gold_id"] == gid), None)
                rows_fn.append({"case_id": cid, "family": s["family5"], "origin": s["origin"], "lane": lane,
                                "gold_id": gid, "component": comp, "cause": cause,
                                "gold_len": round(float(np.linalg.norm(np.subtract(g["line_end"], g["line_start"]))), 1),
                                "best_p": round(float(max((prob[i] for i in cm), default=float("nan"))), 3) if cm else None})
            for pj, l in enumerate(lines):
                pid = f"P{pj + 1:03d}"
                if pid in matched_p:
                    continue
                if lane == "v1":
                    origin, srcs = "CLS", C[l["_i"]].get("src")
                else:
                    origin = l.get("fix") or ("JOIN" if l.get("op") == "join" else "CLS")
                    srcs = sorted(set(x for i in (l.get("parts") or []) for x in (C[i].get("src") or [])))
                cause = next((x["cause"] for x in rec.get("fp_rows") or [] if x["prediction_id"] == pid), "CONTROL")
                rows_fp.append({"case_id": cid, "family": s["family5"], "origin": s["origin"], "lane": lane,
                                "pred_id": pid, "component": origin, "src": "+".join(srcs or []), "cause": cause,
                                "len": round(float(np.linalg.norm(np.subtract(l["end"], l["start"]))), 1)})
        per_scan[cid] = out
    summ = {}
    for lane in ("v1", "v3"):
        fn = [r for r in rows_fn if r["lane"] == lane]
        fp = [r for r in rows_fp if r["lane"] == lane]
        summ[lane] = {"FN_total": len(fn), "FN_by_component": dict(Counter(r["component"] for r in fn).most_common()),
                      "FP_total": len(fp), "FP_by_component": dict(Counter(r["component"] for r in fp).most_common()),
                      "FP_by_src": dict(Counter(r["src"] for r in fp).most_common()),
                      "FP_by_cause": dict(Counter(r["cause"] for r in fp).most_common()),
                      "FN_by_origin": dict(Counter(r["origin"] for r in fn)), "FP_by_origin": dict(Counter(r["origin"] for r in fp))}
    # lines where the lanes differ
    diff = defaultdict(lambda: {"v1_only_FN": [], "v3_only_FN": [], "v1_only_FP": [], "v3_only_FP": []})
    k1 = {(r["case_id"], r["gold_id"]) for r in rows_fn if r["lane"] == "v1"}
    k3 = {(r["case_id"], r["gold_id"]) for r in rows_fn if r["lane"] == "v3"}
    for r in rows_fn:
        key = (r["case_id"], r["gold_id"])
        if r["lane"] == "v3" and key not in k1:
            diff[r["family"]]["v3_only_FN"].append(f"{r['case_id']}:{r['gold_id']}:{r['component']}")
        if r["lane"] == "v1" and key not in k3:
            diff[r["family"]]["v1_only_FN"].append(f"{r['case_id']}:{r['gold_id']}:{r['component']}")
    doc = {"schema": "sherman.zs_v4.diagnosis.v1", "summary": summ, "lane_differences_by_family": diff,
           "fn_rows": rows_fn, "fp_rows": rows_fp, "per_scan": per_scan}
    (EK.V4RUN / "results/DIAGNOSIS.json").write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps(summ, indent=1))
    print(json.dumps(diff, indent=1)[:4000])


if __name__ == "__main__":
    main()
