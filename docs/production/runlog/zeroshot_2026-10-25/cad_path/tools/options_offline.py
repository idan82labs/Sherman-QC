"""CAD path (DEV) options 1-3, offline from build_cohort.py facts. Frozen matcher (5 deg / 4 mm / 0.6, apex, raw).
Lines scored are for display / REVIEW only: none of these arms changes an E3 characteristic, so none can create a PASS.
usage: .venv/bin/python -B .../options_offline.py COHORT.json OUT.json"""
import json, sys
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend"), str(REPO / "research/broad_baseline")]
import numpy as np
from research.zeroshot.zs_lane_v1 import scoring
from domains.production.engine.eval.line_metric import pair_metrics

THR = 0.5
WIN_LO, WIN_HI = 6.0, 15.0          # one-sided window: [nominal - 6, nominal + 15] (toward 180 = under-bent)
EXPLAIN = dict(ang=5.0, off=6.0, ov=0.5)   # zs line explained by a CAD locus (offset 6 mm: a withheld in-plane
                                           # direction can leave the mapped nominal ~1-2 mm off; zs SPAN fragments ok)


def sel(row, cap=False, win=None):
    nom = sorted({b["incl"] for st in row["cad_stages"].values() for b in st})
    keep = []
    for c in row["cand"]:
        if c["p"] < THR:
            continue
        if win and c.get("incl") is not None and nom:
            lo, hi = win
            if not any(a - lo <= c["incl"] <= a + hi for a in nom):
                continue
        keep.append(c)
    keep.sort(key=lambda c: -c["p"])
    if cap:
        keep = keep[:row["n_bends"]]
    return [{"start": c["start"], "end": c["end"], "p": c["p"]} for c in keep]


def explained(z, ref, ang=EXPLAIN["ang"], off=EXPLAIN["off"], ov=EXPLAIN["ov"]):
    m = pair_metrics(np.asarray(ref, float), np.asarray([z["start"], z["end"]], float))
    return m["axis_angle_deg"] <= ang and m["transverse_offset_mm"] <= off and m["prediction_span_overlap"] >= ov, m


def frozen_match(a, b):
    m = pair_metrics(np.asarray(a, float), np.asarray(b, float))
    return m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 4 and m["gold_span_overlap"] >= .6 and m["prediction_span_overlap"] >= .6


def main(cohort, out):
    rows = json.loads(Path(cohort).read_text())["rows"]
    specs = scoring.case_specs("dev")
    arms = {k: {} for k in ("e3", "zs", "zs+N", "zs+A1side+N", "zs+Asym6+N", "opt2_e3_gapfill", "opt2_e3_gapfill_bends_only",
                            "opt1_e3_union_unexplained")}
    flags = []
    gap_rows = []
    loo = {"bends": 0, "flagged": 0}
    for r in rows:
        s = r["scan"]; e3 = r["e3"]
        zsN = sel(r, cap=True)
        arms["zs"][s] = sel(r)
        arms["zs+N"][s] = zsN
        arms["zs+A1side+N"][s] = sel(r, cap=True, win=(WIN_LO, WIN_HI))
        arms["zs+Asym6+N"][s] = sel(r, cap=True, win=(6.0, 6.0))
        valid = e3.get("state") == "VALID"
        pub = [{"start": b["published"][0], "end": b["published"][1]} for b in e3.get("bends", []) if b.get("published")]
        arms["e3"][s] = pub
        if not valid:
            # refused / no run: zs + count cap, display only (no registration, no CAD association, REVIEW)
            arms["opt2_e3_gapfill"][s] = zsN
            arms["opt2_e3_gapfill_bends_only"][s] = pub
            arms["opt1_e3_union_unexplained"][s] = pub
            gap_rows.append({"scan": s, "e3": e3.get("state"), "reasons": e3.get("reasons"), "zs_added": len(zsN), "kind": "REFUSED_SCAN"})
            continue
        # option 2: bends E3 decoded as formed (or UNKNOWN) without a published position -> zs line explained by
        # that bend's own CAD locus (mapped through E3's registration), not matching a published line
        used, add = set(), []
        for b in e3["bends"]:
            if b.get("published") or b["state"] == "NOT_YET_FORMED":
                continue
            best = None
            for i, z in enumerate(zsN):
                if i in used or any(frozen_match(p["start"] and [p["start"], p["end"]], [z["start"], z["end"]]) for p in pub):
                    continue
                ok, m = explained(z, b["nominal_raw"])
                if ok and (best is None or m["transverse_offset_mm"] < best[1]):
                    best = (i, m["transverse_offset_mm"])
            gap_rows.append({"scan": s, "bend": b["label"], "state": b["state"], "pos_state": b["pos_state"],
                             "pos_reason": b["pos_reason"], "zs_found": best is not None,
                             "offset_mm": None if best is None else round(best[1], 2), "kind": "WITHHELD_BEND"})
            if best is not None:
                used.add(best[0]); add.append(zsN[best[0]])
        arms["opt2_e3_gapfill"][s] = pub + [{"start": z["start"], "end": z["end"]} for z in add]
        arms["opt2_e3_gapfill_bends_only"][s] = arms["opt2_e3_gapfill"][s]
        # option 1: zs lines no CAD bend locus (any bend, decoded stage, through E3's registration) explains
        # what the recipe explains: E3's published (measured) lines, every bend's decoded-stage locus, and every
        # stage model's formed-bend loci (a partial scan whose stage E3 left undetermined), all through E3's own T
        Ti = np.linalg.inv(np.asarray(e3["T"], float))
        stage_loci = [(np.asarray([l["start"], l["end"]], float) @ Ti[:3, :3].T + Ti[:3, 3]).tolist()
                      for st in r["cad_stages"].values() for l in st]
        noms = [b["nominal_raw"] for b in e3["bends"]] + [[p["start"], p["end"]] for p in pub] + stage_loci
        unexpl = [z for z in zsN if not any(explained(z, n)[0] for n in noms)]
        arms["opt1_e3_union_unexplained"][s] = pub + [{"start": z["start"], "end": z["end"]} for z in unexpl]
        flags.append({"scan": s, "n_zs": len(zsN), "n_flags": len(unexpl), "flags": unexpl})
        # leave-one-bend-out: a zs line explained by exactly one CAD bend is flagged when that bend is removed
        for b in e3["bends"]:
            zz = [z for z in zsN if explained(z, b["nominal_raw"])[0]]
            if not zz:
                continue
            others = [x["nominal_raw"] for x in e3["bends"] if x is not b]
            loo["bends"] += 1
            loo["flagged"] += int(any(not any(explained(z, n)[0] for n in others) for z in zz))
    res = {"arms": {}, "notes": "every arm is display/REVIEW geometry; E3 characteristics are untouched (0 PASS created)"}
    for name, selx in arms.items():
        recs = [scoring.score_scan(specs[r["scan"]], [{"start": l["start"], "end": l["end"]} for l in selx[r["scan"]]]) for r in rows]
        agg = scoring.aggregate(recs, n_boot=2000)
        res["arms"][name] = {k: agg.get(k) for k in ("scans", "families", "gold", "tp", "fp", "fn", "P", "R", "micro_f1",
                                                      "family_macro_f1", "micro_ci95", "macro_ci95", "controls")}
        print(f"{name:28s} F1 {agg['micro_f1']:.4f} macro {agg['family_macro_f1']:.4f} TP {agg['tp']} FP {agg['fp']} FN {agg['fn']} ctrl {agg['controls']}")
    # option 1 flag quality against gold (diagnostic only)
    nflag = sum(f["n_flags"] for f in flags); sc_flag = sum(1 for f in flags if f["n_flags"])
    real = 0
    for f in flags:
        if not f["n_flags"]:
            continue
        rec = scoring.score_scan(specs[f["scan"]], [{"start": z["start"], "end": z["end"]} for z in f["flags"]])
        real += rec.get("true_positive", 0)
    res["option1"] = {"valid_scans": len(flags), "scans_flagged": sc_flag, "flags": nflag, "flags_on_a_gold_line": real,
                      "leave_one_bend_out": loo,
                      "per_scan": [{k: v for k, v in f.items() if k != "flags"} for f in flags if f["n_flags"]]}
    print("option1", json.dumps({k: v for k, v in res["option1"].items() if k != "per_scan"}))
    res["option2_rows"] = gap_rows
    wb = [g for g in gap_rows if g["kind"] == "WITHHELD_BEND"]
    print("option2 withheld/unknown bends", len(wb), "zs found", sum(g["zs_found"] for g in wb),
          "refused scans", sum(1 for g in gap_rows if g["kind"] == "REFUSED_SCAN"))
    Path(out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
