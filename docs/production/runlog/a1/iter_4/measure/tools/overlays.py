"""Overlays for the 20 worst bends of A1 iteration 0 (report-only; frozen render code used read-only).

Selection: per_bend.json rows ranked by badness, at most 3 bends per case. For each bend:
  1. the frozen G4 close-up (render.render_closeup, first tile) when the E3 output yields a scene;
  2. always: a raw-frame context overlay (scan points near the bend, engine apex line, every gold line
     of the case, the matched gold line highlighted) so that a bend without a published line is visible too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from domains.production.engine.eval import render as R  # noqa: E402
from domains.production.engine.eval.engine_output import load_compiled  # noqa: E402
from domains.production.engine.eval.freeze.physical import _gold_case_objects  # noqa: E402
from domains.production.engine.eval.line_metric import prepare_gold  # noqa: E402
from domains.production.engine.eval.plyio import read_ply  # noqa: E402

W = Path(sys.argv[1])
OUTD = Path(sys.argv[2])
OUTD.mkdir(parents=True, exist_ok=True)
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "cases_A1.json").read_text())
cases = {c["case_ref"]: c for c in doc["cases"]}
pb = json.loads((W / "analysis" / "per_bend_all.json").read_text())  # iter-4: every A1 case with an expected state
rows = sorted(pb["rows"], key=lambda r: -r["badness"])
picked, per_case = [], {}
for r in rows:
    if per_case.get(r["case_id"], 0) >= 3:
        continue
    per_case[r["case_id"]] = per_case.get(r["case_id"], 0) + 1
    picked.append(r)
    if len(picked) == 20:
        break
usable, _ = prepare_gold(_gold_case_objects(doc["broad_gold_cases"]))
from domains.production.engine.eval.freeze.gold import relabel_gold_cases  # noqa: E402
_rl, _ = relabel_gold_cases(doc["gold_cases"], only_files=[f["file"] for f in doc["relabel_files"]])
usable_rel, _ = prepare_gold(_gold_case_objects(_rl))
subs = json.loads((W / "analysis" / "subset_scores.json").read_text())


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".ply":
        return np.asarray(read_ply(path, with_faces=False).vertices, dtype=np.float64)
    import trimesh  # STL
    return np.asarray(trimesh.load(path, force="mesh").vertices, dtype=np.float64)


cache_pts: dict[str, np.ndarray] = {}
index = []
for i, r in enumerate(picked, start=1):
    cid, uid = r["case_id"], r["bend_uid"]
    c = cases[cid]
    stem = f"{i:02d}_{cid}_{r['label']}"
    rec = {"rank": i, **{k: r[k] for k in ("case_id", "family", "bend_uid", "label", "engine_state", "line_published",
                                            "line_matches_gold", "angle_dev_deg", "flange_devs_mm", "decisions", "badness", "truth", "line_src")}}
    edir = W / "physical" / "E3" / cid / "engine"
    cdir = W / "compiled" / "dev" / c["recipe_sku"]
    recipe = FZ / "recipes" / f"{c['recipe_sku']}.recipe.json"
    try:
        scene = R.scene_from_engine_output(edir, cdir / "compiled_part.json", recipe, title=f"{cid} {r['label']}")
        plans = [p for p in R.plan_closeups(scene) if p.bend_uid == uid]
        if plans:
            R.render_closeup(scene, plans[0], OUTD / f"{stem}_closeup.png")
            rec["closeup"] = f"{stem}_closeup.png"
    except Exception as e:  # noqa: BLE001 - report and continue
        rec["closeup_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    # raw-frame context overlay
    try:
        if cid not in cache_pts:
            cache_pts[cid] = load_points(Path(c["scan_path"]))
        P = cache_pts[cid]
        bid = (c.get("broad") or {}).get("broad_case_id")
        pred_file = W / "analysis" / "lanes" / "e3" / bid / "predictions_e3.json"
        preds = json.loads(pred_file.read_text())["owned_bend_regions"] if pred_file.is_file() else []
        mine = [p for p in preds if p["bend_id"] == uid]
        gold = usable.get(cid) or []
        bf = json.loads((FZ / "broad_reference_lanes_A1.json").read_text())["lanes"]["best_f1"].get(cid, {}).get("lines") or []
        ctr = P.mean(0)
        X = P - ctr
        _, _, vt = np.linalg.svd(X[:: max(1, len(X) // 20000)], full_matrices=False)
        n = vt[2]  # view along the thinnest principal axis
        u, v = vt[0], vt[1]
        proj = lambda Q: np.c_[(np.asarray(Q) - ctr) @ u, (np.asarray(Q) - ctr) @ v]  # noqa: E731
        sub = X[:: max(1, len(X) // 60000)]
        fig, ax = plt.subplots(figsize=(10, 7), dpi=110)
        q = proj(sub + ctr)
        ax.scatter(q[:, 0], q[:, 1], s=0.3, c=(sub @ n), cmap="Greys", alpha=0.5, linewidths=0)
        for g in usable_rel.get(cid) or []:
            a = proj([g["line_start"], g["line_end"]])
            ax.plot(a[:, 0], a[:, 1], "--", color="#65a30d", lw=1.6, alpha=0.8)
        for g in gold:
            a = proj([g["line_start"], g["line_end"]])
            ax.plot(a[:, 0], a[:, 1], "-", color="#16a34a", lw=2.2, alpha=0.8)
            ax.text(a[:, 0].mean(), a[:, 1].mean(), str(g["bend_id"]), color="#166534", fontsize=7)
        for p in bf:
            a = proj([p["line_start"], p["line_end"]])
            ax.plot(a[:, 0], a[:, 1], ":", color="#2563eb", lw=1.4)
        for p in preds:
            a = proj([p["line_start"], p["line_end"]])
            ax.plot(a[:, 0], a[:, 1], "-", color="#f97316" if p["bend_id"] != uid else "#dc2626",
                    lw=1.3 if p["bend_id"] != uid else 3.0)
        title = (f"#{i} {cid} fam {r['family']} {r['label']}  truth={r.get('truth')}  state={r['engine_state']}  line_published={r['line_published']}  "
                 f"matches_gold={r['line_matches_gold']}\nangle dev={r['angle_dev_deg']}  flange devs={r['flange_devs_mm']}  "
                 f"decisions={r['decisions']}")
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        ax.legend(handles=[plt.Line2D([], [], color="#16a34a", lw=2, label="BROAD gold (apex)"),
                           plt.Line2D([], [], color="#65a30d", ls="--", lw=1.6, label="Gold11 relabel"),
                           plt.Line2D([], [], color="#dc2626", lw=3, label="engine: this bend"),
                           plt.Line2D([], [], color="#f97316", lw=1.3, label="engine: other bends"),
                           plt.Line2D([], [], color="#2563eb", ls=":", lw=1.4, label="best_f1 lane")],
                  loc="lower right", fontsize=7)
        if not mine:
            ax.text(0.01, 0.98, "engine published NO line for this bend", transform=ax.transAxes, color="#dc2626",
                    fontsize=9, va="top")
        fig.tight_layout()
        fig.savefig(OUTD / f"{stem}_context.png")
        plt.close(fig)
        rec["context"] = f"{stem}_context.png"
    except Exception as e:  # noqa: BLE001
        rec["context_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    index.append(rec)
    print(json.dumps({k: rec.get(k) for k in ("rank", "case_id", "label", "closeup", "closeup_error", "context", "context_error")}))
(OUTD / "INDEX.json").write_text(json.dumps({"selection": "per_bend_all badness rank (state error vs G3 truth / complete-scan FORMED +12, line not confirmed by gold of record +10, false-PASS proxy +8, not correct +4, angle/flange deviations), max 3 bends per case, every A1 case the engine ran", "bends": index}, indent=1))
