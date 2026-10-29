"""Option 4 comparison: base vs add vs only (runtime, VALID rate, line F1 vs gold, characteristic diff vs base).
usage: compare_arms.py COHORT.json RUNS_DIR OUT.json"""
import json, sys, re
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend"), str(REPO / "research/broad_baseline")]
import numpy as np
from research.zeroshot.zs_lane_v1 import scoring
from research.zeroshot.common.gold_guard import require_runnable
from domains.production.engine.eval.engine_output import load_compiled, load_recipe, load_engine_output
from domains.production.engine.eval.line_metric import engine_lines
W = Path("/private/tmp/claude-501/sherman_zeroshot_work")


def chars(res):
    out = {}
    for b in res.bends:
        out[(b.bend_uid, "STATE", None)] = (b.state, None, None)
        for c in b.characteristics:
            a = None if c.actual is None else round(float(c.actual), 2)
            out[(b.bend_uid, c.kind, c.side)] = (c.state, c.decision, a)
    return out


def reg_time(log):
    for l in log.splitlines():
        if "pose margin" in l or "validity" in l:
            return float(re.match(r"\[\s*([0-9.]+)s\]", l).group(1))
    return None


def main(cohort, runs, out):
    rows = json.loads(Path(cohort).read_text())["rows"]
    specs = scoring.case_specs("dev")
    arms = ("base", "add", "only")
    data = {a: {} for a in arms}
    for r in rows:
        s = r["scan"]; require_runnable(s, cad_path=True)
        cdir = W / "compile" / r["sku"] / "compiled"
        comp, rec = load_compiled(cdir / "compiled_part.json"), load_recipe(W / "compile" / r["sku"] / "dev.recipe.json")
        for a in arms:
            d = Path(runs) / a / s
            sm = json.loads((d / "summary.json").read_text()) if (d / "summary.json").is_file() else None
            if sm is None:
                continue
            o = load_engine_output(d / "engine", compiled=comp, recipe=rec)
            res = o.result if o.ok else None
            lines = engine_lines(res, comp) if res is not None else []
            log = (d / "engine/engine.log").read_text() if (d / "engine/engine.log").is_file() else ""
            amb = None
            if res is not None and res.inspection_validity.registration is not None:
                amb = bool(getattr(res.inspection_validity.registration, "pose_ambiguous", False))
            data[a][s] = {"wall": sm["wall"], "state": sm["state"], "reasons": sm["reasons"], "lines": lines,
                          "chars": chars(res) if res is not None else {}, "reg_s": reg_time(log), "ambiguous": amb,
                          "T": None if res is None or res.inspection_validity.registration is None or
                          res.inspection_validity.registration.raw_to_frame is None else
                          np.asarray(res.inspection_validity.registration.raw_to_frame, float).reshape(4, 4)}
    common = [r["scan"] for r in rows if all(r["scan"] in data[a] for a in arms)]
    res = {"scans": len(common), "arms": {}}
    q = lambda x: {"median": round(float(np.median(x)), 2), "p95": round(float(np.percentile(x, 95)), 2),  # noqa
                   "max": round(float(np.max(x)), 2), "sum": round(float(np.sum(x)), 1)}
    for a in arms:
        recs = [scoring.score_scan(specs[s], [{"start": l["line_start"], "end": l["line_end"]} for l in data[a][s]["lines"]])
                for s in common]
        agg = scoring.aggregate(recs, n_boot=2000)
        res["arms"][a] = {"valid": sum(1 for s in common if data[a][s]["state"] == "VALID"),
                          "states": {s: data[a][s]["state"] for s in common if data[a][s]["state"] != "VALID"},
                          "wall": q([data[a][s]["wall"] for s in common]),
                          "registration_s": q([data[a][s]["reg_s"] or 0 for s in common]),
                          "ambiguous_scans": sum(1 for s in common if data[a][s]["ambiguous"]),
                          "line": {k: agg.get(k) for k in ("tp", "fp", "fn", "micro_f1", "family_macro_f1", "micro_ci95")}}
        if a != "base":
            diff, newpass, statechg, posechg = [], 0, [], []
            for s in common:
                b, x = data["base"][s], data[a][s]
                if b["state"] != x["state"]:
                    statechg.append([s, b["state"], x["state"]])
                if b["ambiguous"] != x["ambiguous"]:
                    statechg.append([s, f"ambiguous {b['ambiguous']}", f"ambiguous {x['ambiguous']}"])
                if b["T"] is not None and x["T"] is not None:
                    dR = b["T"][:3, :3] @ x["T"][:3, :3].T
                    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
                    if ang > 0.5 or np.linalg.norm(b["T"][:3, 3] - x["T"][:3, 3]) > 50:
                        posechg.append([s, round(ang, 2)])
                for k in set(b["chars"]) | set(x["chars"]):
                    vb, vx = b["chars"].get(k), x["chars"].get(k)
                    if vb is None or vx is None or vb[0] != vx[0] or vb[1] != vx[1] or (
                            vb[2] is not None and vx[2] is not None and abs(vb[2] - vx[2]) > 0.05):
                        diff.append([s, *[str(t) for t in k], str(vb), str(vx)])
                        if vx is not None and vx[1] == "PASS" and (vb is None or vb[1] != "PASS"):
                            newpass += 1
            res["arms"][a]["vs_base"] = {"state_or_ambiguity_changes": statechg, "pose_changes_gt_0.5deg": posechg,
                                         "characteristic_changes": len(diff), "scans_with_changes": len({d[0] for d in diff}),
                                         "new_PASS_decisions": newpass, "diff_rows": diff[:400]}
        print(a, json.dumps({k: v for k, v in res["arms"][a].items() if k not in ("vs_base",)}), flush=True)
        if a != "base":
            vb = res["arms"][a]["vs_base"]
            print("  vs base:", vb["state_or_ambiguity_changes"], "pose", vb["pose_changes_gt_0.5deg"], "chars", vb["characteristic_changes"],
                  "scans", vb["scans_with_changes"], "newPASS", vb["new_PASS_decisions"])
    res["per_scan"] = {s: {a: {"wall": data[a][s]["wall"], "reg_s": data[a][s]["reg_s"], "state": data[a][s]["state"]}
                           for a in arms} for s in common}
    Path(out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
