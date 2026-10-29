"""Angle truth check (DEV, CAD path cohort): is the zs_lane_v1 included angle true?

For every zs line the frozen matcher (research/broad_baseline/score.py via zs_lane_v1.scoring) matches to a gold
line, the gold line is tied to its CAD bend through E3's registration (E3 published OUTER_APEX line, frozen match;
else E3's decoded-stage CAD locus mapped to raw, 5 deg / 6 mm / 0.5). The zs included angle is then compared with
  (a) the CAD nominal, (b) E3's measured INCLUDED_ANGLE (the metrology authority), (c) the gold line's angle.
Inputs are stored outputs only (no engine run):
  * cohort facts  : COHORT.json from cad_path/tools/build_cohort.py (CADPATH_E3_DIR = re-run e3-1.12.0 'base')
  * E3 results    : <E3_DIR>/<scan>/engine/result.json (INCLUDED_ANGLE characteristic)
  * zs lines      : integration/dev_cli/zs/<scan>/eval_lines.json (frozen CLI, field incl); secondary: the
                    out-of-fold selection (cohort cand, p >= 0.5)
  * gold          : read through the gold guard (cad_path=True) by the frozen scorer; gold angle = 'incl X deg' in
                    the label's notes.
usage (repo root): .venv/bin/python -B .../angle_check.py COHORT.json E3_DIR OUT.json
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend"), str(REPO / "research/broad_baseline")]
import numpy as np
from research.zeroshot.zs_lane_v1 import scoring
from research.zeroshot.common.gold_guard import require_runnable
from domains.production.engine.eval.line_metric import pair_metrics
import bb_common as bb

W = Path("/private/tmp/claude-501/sherman_zeroshot_work")
CLI = scoring.PROG / "integration/dev_cli/zs"
TOL = 1.0


def pm(a, b):
    return pair_metrics(np.asarray(a, float), np.asarray(b, float))


def frozen_ok(m):
    return m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 4 and m["gold_span_overlap"] >= .6 \
        and m["prediction_span_overlap"] >= .6


def assoc_ok(m):
    return m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 6 and m["prediction_span_overlap"] >= .5


def joggle_uids(cj):
    """Joggle = two bends sharing a panel, opposite directions, equal sweep <= 60 deg, axes <= 10 t apart."""
    t = float(cj["thickness"]["t_mm"])
    bs = [b for b in cj["bends"] if b.get("supported", True)]
    out = set()
    for i, a in enumerate(bs):
        for b in bs[i + 1:]:
            pa, pb = {a["parent_panel_id"], a["child_panel_id"]}, {b["parent_panel_id"], b["child_panel_id"]}
            if not (pa & pb) or a["direction"] == b["direction"]:
                continue
            if abs(a["sweep_deg"] - b["sweep_deg"]) > 2 or a["sweep_deg"] > 60:
                continue
            p, q = np.asarray(a["axis"]["point_mm"], float), np.asarray(b["axis"]["point_mm"], float)
            d = np.asarray(a["axis"]["direction"], float); d /= np.linalg.norm(d)
            v = q - p
            if np.linalg.norm(v - v.dot(d) * d) <= 10 * t:
                out |= {a["bend_uid"], b["bend_uid"]}
    return out


def cls(nom, jog):
    if jog:
        return "joggle"
    if abs(nom - 90) <= 5:
        return "~90"
    return "acute" if nom < 90 else "obtuse"


def stats(x):
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return {"n": 0}
    a = np.abs(x)
    return {"n": int(len(x)), "median_abs": round(float(np.median(a)), 3), "p90_abs": round(float(np.percentile(a, 90)), 3),
            "max_abs": round(float(a.max()), 3), "bias_mean": round(float(x.mean()), 3),
            "bias_median": round(float(np.median(x)), 3), "mae": round(float(a.mean()), 3),
            "within_1deg": round(float((a <= 1).mean()), 3), "within_2deg": round(float((a <= 2).mean()), 3)}


def zs_lines_cli(s):
    p = CLI / s / "eval_lines.json"
    d = json.loads(p.read_text())
    if d.get("status") != "OK":
        return []
    return [{"start": r["line_start"], "end": r["line_end"], "incl": r.get("incl"), "id": r.get("bend_id")}
            for r in d.get("owned_bend_regions") or [] if bb.valid_line(r.get("line_start"), r.get("line_end"))]


def zs_lines_oof(row):
    k = sorted([c for c in row["cand"] if c["p"] >= 0.5], key=lambda c: -c["p"])
    return [{"start": c["start"], "end": c["end"], "incl": c["incl"], "id": f"OOF{i}"} for i, c in enumerate(k)]


def run(rows, e3dir, specs, src):
    recs, skipped = [], []
    for row in rows:
        s = row["scan"]
        require_runnable(s, cad_path=True)
        e3 = row["e3"]
        if e3.get("state") != "VALID":
            skipped.append({"scan": s, "why": f"E3 {e3.get('state')} {e3.get('reasons')}: no registration -> no CAD bend"})
            continue
        zl = zs_lines_cli(s) if src == "cli" else zs_lines_oof(row)
        rec = scoring.score_scan(specs[s], [{"start": z["start"], "end": z["end"]} for z in zl], diagnose=False)
        if rec.get("disposition") != "SCORED_COMPLETE_LABEL":
            skipped.append({"scan": s, "why": rec.get("disposition")}); continue
        gpath = bb.expand(specs[s]["gold"]["path"])
        graw = json.loads(Path(gpath).read_text())["bends"]
        gold = {}
        for i, gb in enumerate(graw, 1):
            gid = str(gb.get("bend_id") or f"B{i:02d}")
            mm = re.search(r"incl\s+([0-9.]+)\s*deg", gb.get("notes") or "")
            e = bb.endpoints(gb)
            gold[gid] = {"line": [list(e[0]), list(e[1])], "incl": float(mm.group(1)) if mm else None}
        res = json.loads((e3dir / s / "engine/result.json").read_text())
        e3ang = {}
        for b in res.get("bends") or []:
            for ch in b["characteristics"]:
                if ch["kind"] == "INCLUDED_ANGLE":
                    e3ang[b["bend_uid"]] = ch
        cj = json.loads((W / "compile" / row["sku"] / "compiled/compiled_part.json").read_text())
        nom = {b["bend_uid"]: float(b["nominal_included_deg"]) for b in cj["bends"]}
        jog = joggle_uids(cj)
        # gold -> E3 bend, one-to-one: published line (frozen) first, then mapped nominal (assoc)
        cand = []
        for gid, g in gold.items():
            for b in e3["bends"]:
                if b.get("published"):
                    m = pm(g["line"], b["published"])
                    if frozen_ok(m):
                        cand.append((0, m["transverse_offset_mm"], gid, b["uid"], "E3_PUBLISHED"))
                m = pm(g["line"], b["nominal_raw"])
                if assoc_ok(m):
                    cand.append((1, m["transverse_offset_mm"], gid, b["uid"], "E3_MAPPED_NOMINAL"))
        cand.sort()
        g2b, used = {}, set()
        for _, off, gid, uid, how in cand:
            if gid in g2b or uid in used:
                continue
            g2b[gid] = (uid, how, round(off, 2)); used.add(uid)
        zid = {f"P{i + 1:03d}": z for i, z in enumerate(zl)}
        for mt in rec.get("matches") or []:
            gid = mt.get("gold_id") or mt.get("gold"); pid = mt.get("prediction_id") or mt.get("prediction")
            z = zid[pid]; g = gold[gid]
            r = {"scan": s, "family": row["family"], "sku": row["sku"], "t_mm": row["t_mm"], "gold_id": gid,
                 "zs_id": z["id"], "zs_incl": z["incl"], "gold_incl": g["incl"], "zs_line": [z["start"], z["end"]]}
            if gid in g2b:
                uid, how, off = g2b[gid]
                ch = e3ang.get(uid) or {}
                lab = next(b["label"] for b in e3["bends"] if b["uid"] == uid)
                st = next(b["state"] for b in e3["bends"] if b["uid"] == uid)
                r.update({"bend_uid": uid, "bend_label": lab, "e3_bend_state": st, "corr": how, "corr_offset_mm": off,
                          "nominal": nom[uid], "class": cls(nom[uid], uid in jog), "e3_state": ch.get("state"),
                          "e3_decision": ch.get("decision"), "e3_incl": ch.get("actual") if ch.get("state") == "MEASURED" else None,
                          "e3_uncertainty": ch.get("uncertainty")})
            else:
                r.update({"bend_uid": None, "class": None, "nominal": None, "e3_incl": None})
            r["d_cad"] = None if r["nominal"] is None or z["incl"] is None else z["incl"] - r["nominal"]
            r["d_e3"] = None if r["e3_incl"] is None or z["incl"] is None else z["incl"] - r["e3_incl"]
            r["d_gold"] = None if g["incl"] is None or z["incl"] is None else z["incl"] - g["incl"]
            r["e3_minus_cad"] = None if r["e3_incl"] is None else r["e3_incl"] - r["nominal"]
            recs.append(r)
    return recs, skipped


def summarize(recs):
    out = {"all": {k: stats([r[k] for r in recs]) for k in ("d_e3", "d_cad", "d_gold", "e3_minus_cad")}}
    for key, f in (("by_class", lambda r: r["class"]), ("by_thickness", lambda r: r["t_mm"])):
        grp = defaultdict(list)
        for r in recs:
            if r["bend_uid"] is not None:
                grp[f(r)].append(r)
        out[key] = {str(k): {"n_bends": len(v), **{d: stats([r[d] for r in v]) for d in ("d_e3", "d_cad", "d_gold", "e3_minus_cad")}}
                    for k, v in sorted(grp.items(), key=lambda kv: str(kv[0]))}
    # tolerance side agreement (+-1 deg about nominal), on bends E3 measured
    tt = [r for r in recs if r["e3_incl"] is not None and r["zs_incl"] is not None]
    a = [(abs(r["e3_incl"] - r["nominal"]) <= TOL, abs(r["zs_incl"] - r["nominal"]) <= TOL) for r in tt]
    out["tolerance_pm1"] = {"n": len(a), "agree": sum(x == y for x, y in a),
                            "agree_rate": round(sum(x == y for x, y in a) / len(a), 3) if a else None,
                            "both_in": sum(x and y for x, y in a), "both_out": sum((not x) and (not y) for x, y in a),
                            "e3_in_zs_out": sum(x and not y for x, y in a), "e3_out_zs_in": sum((not x) and y for x, y in a)}
    # with E3's decision: E3 PASS/FAIL only (REVIEW = within uncertainty band)
    pf = [r for r in tt if r.get("e3_decision") in ("PASS", "FAIL")]
    b = [(r["e3_decision"] == "PASS", abs(r["zs_incl"] - r["nominal"]) <= TOL) for r in pf]
    out["vs_e3_decision_pass_fail"] = {"n": len(b), "agree": sum(x == y for x, y in b),
                                       "e3_pass_zs_out": sum(x and not y for x, y in b),
                                       "e3_fail_zs_in": sum((not x) and y for x, y in b)}
    return out


def main(cohort, e3dir, outp):
    rows = json.loads(Path(cohort).read_text())["rows"]
    specs = scoring.case_specs("dev")
    e3dir = Path(e3dir)
    out = {"inputs": {"cohort": cohort, "e3_dir": str(e3dir), "zs_cli": str(CLI), "tol_deg": TOL}}
    for src in ("cli", "oof"):
        recs, skipped = run(rows, e3dir, specs, src)
        out[src] = {"n_matched_zs_gold": len(recs), "n_with_cad_bend": sum(r["bend_uid"] is not None for r in recs),
                    "n_with_e3_angle": sum(r["e3_incl"] is not None for r in recs),
                    "skipped": skipped, "summary": summarize(recs), "records": recs}
        print(src, len(recs), json.dumps(out[src]["summary"]["all"]["d_e3"]), json.dumps(out[src]["summary"]["tolerance_pm1"]))
    Path(outp).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
