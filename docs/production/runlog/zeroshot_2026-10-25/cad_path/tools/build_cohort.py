"""CAD path (DEV): per-scan facts for the options study. Light (no geometry, no engine run).

For every DEV scan with a compiled CAD that the gold guard allows on the CAD path (cad_path=True refuses the
OD-3 DEV-CHECK cases BEFORE any per-scan file is read):
  * zs: every pooled+snapped candidate with its OUT-OF-FOLD selector probability (no CAD used in training);
  * CAD: per stage model, the formed bends' OUTER_APEX lines in the cad_model_mm frame (+ nominal angle, length);
  * E3 (stored e3-1.12.0 runs): validity, registration raw_to_frame, decoded stage, per bend: state, position
    state / reason, published OUTER_APEX line (raw), the decoded-stage CAD locus mapped to raw through E3's own
    registration, characteristic decisions.
usage (repo root; CADPATH_E3_DIR=<runs>/base to read re-run E3 outputs): .venv/bin/python -B docs/production/runlog/zeroshot_2026-10-25/cad_path/tools/build_cohort.py OUT.json
"""
import json, os, pickle, sys
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend"), str(REPO / "research/broad_baseline")]
import numpy as np
from research.zeroshot.zs_lane_v1 import scoring
from research.zeroshot.common.gold_guard import require_runnable, GoldGuardRefusal
from domains.production.engine.eval.engine_output import (load_compiled, load_recipe, load_engine_output,
                                                          decoded_stage, stage_locus, frame_to_raw, row_characteristic,
                                                          measured_outer_apex)

W = Path("/private/tmp/claude-501/sherman_zeroshot_work")
INTEG = scoring.PROG / "integration"


def main(out):
    inv = scoring._inv()
    folds = scoring.dev_folds()
    P = pickle.loads((INTEG / "oof_v1/zs_snap_oof_probs.pkl").read_bytes())["P"]
    rows, refused = [], []
    for s in scoring.dev_ids():
        r = inv.get(s)
        sku = r["cad"].get("sku") if r else None
        if not sku or not (W / "compile" / sku / "compiled/compiled_part.json").is_file():
            continue
        try:
            require_runnable(s, cad_path=True)
        except GoldGuardRefusal as e:
            refused.append([s, e.reason]); continue
        cj0 = json.loads((W / "compile" / sku / "compiled/compiled_part.json").read_text())
        if not cj0.get("thickness") or not [b for b in cj0.get("bends") or [] if b.get("supported", True)]:
            continue  # same cohort rule as the cad_prior prototype (no CAD facts -> not on the CAD path)
        d = json.loads((INTEG / "dev_dump" / f"{s}.json").read_text())
        cand = [{"start": c["start_s"], "end": c["end_s"], "incl": c.get("incl"), "p": float(q)}
                for c, q in zip(d["candidates"], P[s])]
        cdir = W / "compile" / sku / "compiled"
        comp = load_compiled(cdir / "compiled_part.json")
        cj = json.loads((cdir / "compiled_part.json").read_text())
        bends = [b for b in cj["bends"] if b.get("supported", True)]
        stages = {}
        for st in comp.stages:
            ls = []
            for uid in st.formed_bend_uids:
                b = next(x for x in comp.bends if x.bend_uid == uid)
                p = stage_locus(comp, st, uid, "outer_apex")
                ls.append({"uid": uid, "label": b.label, "start": p[0].tolist(), "end": p[1].tolist(),
                           "incl": float(b.nominal_included_deg)})
            stages[st.stage_model_id] = ls
        e3 = {"state": None}
        edir = Path(os.environ.get("CADPATH_E3_DIR", str(W / "e3"))) / s / "engine"
        if (edir / "result.json").is_file():
            rec = load_recipe(W / "compile" / sku / "dev.recipe.json")
            o = load_engine_output(edir, compiled=comp, recipe=rec)
            res = o.result
            if res is not None:
                iv = res.inspection_validity
                e3 = {"state": iv.state, "reasons": list(iv.reasons or []),
                      "T": None if iv.registration is None or iv.registration.raw_to_frame is None else
                      np.asarray(iv.registration.raw_to_frame, float).reshape(4, 4).tolist(),
                      "stage": res.stage_decision.stage_model_id, "bends": []}
                if iv.state == "VALID":
                    st = decoded_stage(res, comp)
                    for row in res.bends:
                        pos = row_characteristic(row, "BEND_LINE_POSITION")
                        nom = frame_to_raw(res, stage_locus(comp, st, row.bend_uid, "outer_apex"))
                        line = row.display.measured_polyline_mm
                        pub = None
                        if pos is not None and pos.state == "MEASURED" and line is not None and row.state != "NOT_YET_FORMED":
                            apex, _ = measured_outer_apex(comp, st, row, fallback="NOMINAL")
                            pub = frame_to_raw(res, apex).tolist()
                        e3["bends"].append({
                            "uid": row.bend_uid, "label": row.label, "state": row.state,
                            "pos_state": None if pos is None else pos.state,
                            "pos_reason": None if pos is None else getattr(pos, "reason", None),
                            "published": pub, "nominal_raw": nom.tolist(),
                            "decisions": [[c.kind, c.side, c.state, c.decision] for c in row.characteristics]})
        rows.append({"scan": s, "family": r["family"], "fold": folds.get(s), "stage": r.get("stage"), "sku": sku,
                     "t_mm": float(cj["thickness"]["t_mm"]), "n_bends": len(bends), "cand": cand,
                     "zs_wall_s": {"z3": d["stages"]["z3"]["wall"], "z2": d["stages"]["z2"]["wall"]},
                     "cad_stages": stages, "final_stage": f"S{len(comp.bends)}", "e3": e3,
                     "scan_path": r["path"]})
        print(s, e3["state"], len(cand), flush=True)
    Path(out).write_text(json.dumps({"rows": rows, "refused": refused}, default=str))
    print("rows", len(rows), "refused", refused)


if __name__ == "__main__":
    main(sys.argv[1])
