"""Apply the v2 winning design to another lane's candidates (e.g. the new production engine).

Input, one scan per call:

- ``--engine-dir DIR --compiled COMPILED.json --recipe RECIPE.json``: an E3 output
  directory. Lines are the engine's measured outer-apex loci mapped to the raw scan
  frame by the engine's own eval helpers (``domains.production.engine.eval.line_metric.
  engine_lines``, read-only import; nothing in domains/ is edited); or
- ``--lines EVAL_LINES.json``: any ``sherman.broad_baseline.eval_lines.v1`` file
  (``frame: scan_raw``, ``owned_bend_regions[]`` with ``line_start``/``line_end``).

plus ``--scan`` (the raw scan the lines are in) and optionally ``--pool`` (the scan's
``f1_uniform_scan_only_candidate_pool_v3`` file). With a pool, neighbour context and
pool descriptors are measured against it exactly as in the study; ``--add-from-pool``
also proposes pool lines the engine missed (the liberal arm). Without a pool, context is
measured against the input lines and pool descriptors are missing (the model has
missing-value indicators, but this is a distribution shift and is flagged).

Decision (``frozen.json``): in pool mode the frozen logistic regression (refitted
deterministically from the study's calibration families, or ``--train-on all``) ranks the
engine lines plus the pool and keeps NMS-deduped lines above the calibration threshold; in
filter mode the single scan feature (pool two-sided transition score) filters the engine's
lines. ``--jev-design`` with ``--live`` adds Jev's primary Noul to the output as a
diagnostic only; it never changes the keep decision (Jev did not earn a place). Geometry is heavy: run under the shared geometry lock.

Writes ``OUT.json``: per line, the scores and the keep decision, and an eval-lines file
``OUT.lines.json`` with the kept lines (scorable by research/broad_baseline/score.py).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from research.jev import arms_v2 as A
from research.jev.input_opt_v2 import CASE_LIST, REPO_ROOT, _read, load_truth, neighbour_context, read_candidates
from research.jev.pool_v2 import cluster_lines, load_pool_lines, nms_select, pair_metrics_matrix

STUDY = REPO_ROOT / "docs" / "production" / "runlog" / "jev_ab_v2"


def engine_raw_lines(engine_dir: Path, compiled: Path, recipe: Path) -> list[dict[str, Any]]:
    from domains.production.engine.eval.engine_output import load_compiled, load_engine_output, load_recipe
    from domains.production.engine.eval.line_metric import engine_lines

    comp, rec = load_compiled(compiled), load_recipe(recipe)
    out = load_engine_output(engine_dir, compiled=comp, recipe=rec)
    if not out.ok:
        raise SystemExit(f"engine output not usable: {out.status} {out.error or out.problems}")
    return engine_lines(out.result, comp, locus="OUTER_APEX", frame="raw")


def train_rows(train_on: str):
    rows = read_candidates(STUDY / "build" / "candidates.jsonl")
    truth = load_truth(_read(CASE_LIST), "ai_relabel")
    y = A.any_match_labels(rows, {c: t["lines"] for c, t in truth.items()})
    keep = [i for i, r in enumerate(rows) if r["set"] == "pool" and (train_on == "all" or r["split"] == "calib")]
    return [rows[i] for i in keep], y[keep]


def build_rows(scan: Path, lines: list[dict[str, Any]], pool_path: Path | None, add_from_pool: bool):
    from research.jev.features_v2 import line_features_v2, load_clouds

    ends = np.array([[l["line_start"], l["line_end"]] for l in lines], float).reshape(-1, 2, 3)
    pool_ends = np.zeros((0, 2, 3))
    clusters, plines = [], []
    if pool_path:
        plines, _ = load_pool_lines(pool_path)
        clusters = cluster_lines(plines)
        pool_ends = np.array([[plines[x["rep"]].start, plines[x["rep"]].end] for x in clusters]).reshape(-1, 2, 3)
    ref = pool_ends if len(pool_ends) else ends
    ctx = neighbour_context(ends, ref, [None] * len(ends) if len(pool_ends) else list(range(len(ends))))
    to_pool = [None] * len(ends)
    if len(pool_ends) and len(ends):
        mm = pair_metrics_matrix(ends, pool_ends)
        ok = ((mm["axis_angle_deg"] <= 5) & (mm["transverse_offset_mm"] <= 4) & (mm["gold_span_overlap"] >= 0.6)
              & (mm["prediction_span_overlap"] >= 0.6))
        cost = mm["axis_angle_deg"] / 5 + mm["transverse_offset_mm"] / 4
        for i in range(len(ends)):
            if ok[i].any():
                to_pool[i] = int(np.argmin(np.where(ok[i], cost[i], 1e9)))

    def pool_block(k):
        if k is None:
            return {}
        x = clusters[k]
        radii = [plines[j].radius_mm for j in x["members"] if plines[j].radius_mm is not None]
        return {"pool_member_count": x["member_count"], "pool_radius_mm": float(np.median(radii)) if radii else None,
                "pool_radius_votes_count": len(radii),
                **{f"pool_{k2}": v for k2, v in plines[x["rep"]].descriptors.items()}}

    clouds = load_clouds(scan)
    rows = []
    for i, (e, l) in enumerate(zip(ends, lines)):
        rows.append({"case_id": "scan", "set": "input", "cand": str(l.get("bend_id", i)), "scan_sha256": "x",
                     "line_start": e[0].tolist(), "line_end": e[1].tolist(),
                     "feat": line_features_v2(clouds, e[0], e[1]), "ctx": ctx[i], "pool": pool_block(to_pool[i])})
    if add_from_pool and len(pool_ends):
        pctx = neighbour_context(pool_ends, pool_ends, list(range(len(pool_ends))))
        for k, e in enumerate(pool_ends):
            rows.append({"case_id": "scan", "set": "pool", "cand": f"P{k}", "scan_sha256": "x",
                         "line_start": e[0].tolist(), "line_end": e[1].tolist(),
                         "feat": line_features_v2(clouds, e[0], e[1]), "ctx": pctx[k], "pool": pool_block(k)})
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m research.jev.apply_v2", description=__doc__.split("\n\n")[0])
    p.add_argument("--scan", required=True)
    p.add_argument("--lines")
    p.add_argument("--engine-dir")
    p.add_argument("--compiled")
    p.add_argument("--recipe")
    p.add_argument("--pool")
    p.add_argument("--add-from-pool", action="store_true")
    p.add_argument("--frozen", default=str(STUDY / "frozen.json"))
    p.add_argument("--train-on", choices=("calib", "all"), default="calib")
    p.add_argument("--jev-design", choices=("typed", "typed_reading", "composite"))
    p.add_argument("--live", action="store_true")
    p.add_argument("--env-file", default="")
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    frozen = _read(a.frozen)
    if a.engine_dir:
        lines = engine_raw_lines(Path(a.engine_dir), Path(a.compiled), Path(a.recipe))
    elif a.lines:
        doc = _read(a.lines)
        if doc.get("frame") != "scan_raw":
            raise SystemExit("--lines must be an eval-lines file in the scan_raw frame")
        lines = [r for r in doc.get("owned_bend_regions") or [] if r.get("line_start") and r.get("line_end")]
    else:
        raise SystemExit("give --engine-dir/--compiled/--recipe or --lines")
    rows = build_rows(Path(a.scan), lines, Path(a.pool) if a.pool else None, a.add_from_pool)
    mode = "pool" if a.add_from_pool else "filter"
    op = frozen[mode]
    if op["score"] == "best_feature":
        col = A.raw_matrix(rows, [op["feature"]])[:, 0] * float(op["sign"])
        primary = np.where(np.isnan(col), -np.inf, col)
    else:
        tr, ytr = train_rows(a.train_on)
        primary = A.fit_predict(op["score"].removeprefix("clf_"), tr, ytr, rows)
    scores = {op["score"]: np.where(np.isfinite(primary), primary, np.nan).tolist()}
    combined = primary
    if a.jev_design:
        from research.jev.input_opt_v2 import _key_from_env_file
        from research.jev.runner_v2 import live_run, load_answers
        from research.jev.states_v2 import build_state_v2
        import hashlib

        variant = "composite_v2" if a.jev_design == "composite" else "v2"
        states = [(hashlib.sha256(json.dumps(r["line_start"] + r["line_end"]).encode()).hexdigest()[:24],
                   build_state_v2(r, a.jev_design)) for r in rows]
        cache = Path(a.out).with_suffix(".jev_cache")
        if a.live:
            live_run(states, cache, live=True, model="jev-1.13.0", variant=variant, budget_usd=a.budget,
                     env=_key_from_env_file(Path(a.env_file)) if a.env_file else None)
        ans = load_answers(states, cache, model="jev-1.13.0", variant=variant)
        jev = np.array([ans[rid]["answers"]["is_real_bend"]["noul"] if rid in ans else math.nan for rid, _ in states])
        scores["jev"] = jev.tolist()
    t = float(op["threshold"])
    ends = [[r["line_start"], r["line_end"]] for r in rows]
    kept = nms_select(ends, combined, t) if mode == "pool" else [i for i in range(len(rows)) if combined[i] >= t]
    out = {"schema": "jev_v2_apply_v1", "frozen": frozen, "mode": mode, "threshold": t,
           "pool_given": bool(a.pool), "context_reference": "pool" if a.pool else "input_lines (distribution shift)",
           "lines": [{"bend_id": r["cand"], "set": r["set"], "keep": i in set(kept),
                      **{k: (None if not math.isfinite(float(v[i])) else round(float(v[i]), 4)) for k, v in scores.items()}}
                     for i, r in enumerate(rows)]}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    kept_lines = [{"bend_id": rows[i]["cand"], "line_start": rows[i]["line_start"], "line_end": rows[i]["line_end"]}
                  for i in kept]
    Path(a.out).with_suffix(".lines.json").write_text(json.dumps(
        {"schema": "sherman.broad_baseline.eval_lines.v1", "lane": f"jev_v2_{mode}", "status": "OK",
         "frame": "scan_raw", "owned_bend_regions": kept_lines}, indent=2) + "\n")
    print(json.dumps({"lines_in": len(lines), "candidates": len(rows), "kept": len(kept), "threshold": t}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
