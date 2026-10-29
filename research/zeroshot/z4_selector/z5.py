"""Z5 global-consistency selection on top of the Z4 scores (CP-SAT in .venv-f1-structured).

Per scan, over the candidates whose Z4 out-of-fold score is >= low_frac * threshold (top 80):
  - weight w_i = logit(p_i) - logit(threshold)  (so with no constraints Z5 == threshold selection);
  - near-duplicate conflicts (matcher 5 deg / 4 mm / 0.6 shorter span), i.e. a globally optimal NMS;
  - the two-skin duplicate rule: parallel (<= 5 deg), overlapping (>= 0.5 shorter span) lines closer than
    t / sin(alpha/2) + 1 mm (t = D1 thickness, alpha = measured included angle) cannot both be kept;
  - flange graph: flange nodes = A0 pool plane_table planes, two skins merged (parallel normals, offset <= t+1.5);
    a candidate joins flanges (a, b) when both planes run along it (line parallel to the plane, plane within
    reach of the line, line inside the plane's extent) and their dihedral matches the measured bend angle;
    a candidate with no flange pair pays `nopair_penalty` (strip evidence); at most one line per flange pair
    and carrier (collinear duplicates); near-tree bound #edges <= #touched nodes - 1 + tree_slack.
Parameters (low_frac, nopair_penalty, tree_slack, two_skin) are chosen per outer fold on the training folds'
inner-CV scores only, then applied to the held-out fold's out-of-fold scores.

  .venv/bin/python -B -m research.zeroshot.z4_selector.z5 --z4 iter1 --out-name iter1_z5
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from research.zeroshot.z4_selector import data as D
from research.zeroshot.z4_selector.candidates import FEAT_ROOT, pool_path
from research.zeroshot.z4_selector.custody import OUT, REPO, dev_ids, dev_inventory
from research.jev.pool_v2 import pair_metrics_matrix

SOLVER_PY = str(REPO / ".venv-f1-structured/bin/python")
SOLVER = str(Path(__file__).with_name("z5_solve.py"))
MAX_CAND = 80


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def load_planes(sid: str, t: float | None):
    pp = pool_path(sid)
    if pp is None:
        return []
    pool = json.loads(pp.read_text())
    planes = []
    for pl in pool.get("plane_table") or []:
        n = np.asarray(pl["normal"], float)
        n /= np.linalg.norm(n)
        B = np.asarray(pl["tangent_basis"], float)
        ext = np.asarray(pl["extent_uv"], float)
        planes.append({"n": n, "d": float(pl["offset"]), "B": B, "ext": ext, "support": pl.get("support_count", 0)})
    # merge two skins: parallel normals (either sign) and |offset difference| <= t + 1.5 mm
    tt = (t or 2.0) + 1.5
    node = list(range(len(planes)))
    for i in range(len(planes)):
        for j in range(i):
            c = float(planes[i]["n"] @ planes[j]["n"])
            if abs(c) >= math.cos(math.radians(5)):
                di = planes[i]["d"] * (1 if c > 0 else -1)
                if abs(di - planes[j]["d"]) <= tt:
                    node[i] = node[j]
                    break
    ids = {v: k for k, v in enumerate(sorted(set(node)))}
    for p, v in zip(planes, node):
        p["node"] = ids[v]
    return planes


def flange_pair(a, b, bend_deg, planes, t):
    """(node_a, node_b) of the two flanges this line separates, or None."""
    if not planes:
        return None
    ax = b - a
    L = np.linalg.norm(ax)
    ax /= L
    mid = 0.5 * (a + b)
    reach = max(8.0, 4.0 * (t or 2.0))
    near = []
    for p in planes:
        if abs(float(p["n"] @ ax)) > math.sin(math.radians(8)):
            continue
        dist = abs(float(p["n"] @ mid + p["d"]))
        if dist > reach:
            continue
        uv = p["B"] @ mid if p["B"].shape == (2, 3) else None
        if uv is not None:
            lo, hi = p["ext"][0] - 15.0, p["ext"][1] + 15.0
            if np.any(uv < lo) or np.any(uv > hi):
                continue
        near.append(p)
    best, cost = None, 1e9
    for p, q in itertools.combinations(near, 2):
        if p["node"] == q["node"]:
            continue
        dih = math.degrees(math.acos(min(1.0, abs(float(p["n"] @ q["n"])))))  # 0..90
        if bend_deg is None:
            c = 0.0 if dih >= 8 else 1e9
        else:
            bd = bend_deg if bend_deg <= 90 else 180 - bend_deg
            c = abs(dih - bd)
            if c > 15:
                continue
        if c < cost:
            best, cost = tuple(sorted((p["node"], q["node"]))), c
    return best


def build_scan(sid, rows, scores, thr, t, planes, low_frac, nopair_pen, tree_slack, two_skin, dup_mm=4.0):
    # candidates = greedy-NMS survivors at low_frac * threshold (so with no extra terms Z5 == Z4 exactly)
    from research.zeroshot.z4_selector.selector import nms
    allends = np.array([[r["line_start"], r["line_end"]] for r in rows], float).reshape(-1, 2, 3)
    order = nms(allends, scores, low_frac * thr, dup_mm)[:MAX_CAND] if len(rows) else []
    if not order:
        return {"id": sid, "w": []}, []
    ends = np.array([[rows[i]["line_start"], rows[i]["line_end"]] for i in order], float)
    w = [int(round(1000 * (logit(float(scores[i])) - logit(thr)))) for i in order]
    m = pair_metrics_matrix(ends, ends)
    ang, off, ov = m["axis_angle_deg"], m["transverse_offset_mm"], m["shorter_span_overlap"]
    conflicts = []
    n = len(order)
    inc = [rows[i]["f"].get("x_median_included_angle_deg") for i in order]
    for i in range(n):
        for j in range(i):
            if ang[i, j] > 5.0:
                continue
            if off[i, j] <= dup_mm and ov[i, j] >= 0.6:
                conflicts.append([i, j]); continue
            if two_skin and t and ov[i, j] >= 0.5:
                al = [x for x in (inc[i], inc[j]) if isinstance(x, (int, float)) and x > 1]
                alpha = min(al) if al else 90.0
                if off[i, j] <= t / max(0.05, math.sin(math.radians(alpha) / 2)) + 1.0:
                    conflicts.append([i, j])
    edges, pairs = [], {}
    if nopair_pen or tree_slack is not None:
        for i, ci in enumerate(order):
            r = rows[ci]
            fp = flange_pair(np.asarray(r["line_start"]), np.asarray(r["line_end"]),
                             r["f"].get("x_median_bend_angle_deg"), planes, t)
            if fp:
                edges.append([i, fp[0], fp[1]])
                pairs.setdefault(fp, []).append(i)
    atmost1 = []
    for fp, lst in pairs.items():  # same flange pair, same carrier (collinear within 4 mm) -> one line
        for i, j in itertools.combinations(lst, 2):
            if ang[i, j] <= 5.0 and off[i, j] <= 4.0:
                atmost1.append([i, j])
    nn = (max([max(e[1], e[2]) for e in edges]) + 1) if edges else 0
    return ({"id": sid, "w": w, "conflicts": conflicts, "atmost1": atmost1, "edges": edges, "n_nodes": nn,
             "tree_slack": tree_slack, "nopair_penalty": int(1000 * nopair_pen)}, order)


def solve_many(problems):
    with tempfile.TemporaryDirectory() as td:
        fi, fo = Path(td) / "in.jsonl", Path(td) / "out.jsonl"
        fi.write_text("\n".join(json.dumps(p) for p in problems) + "\n")
        subprocess.run([SOLVER_PY, "-B", SOLVER, "--in", str(fi), "--out", str(fo)], check=True,
                       env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
        return {r["id"]: r for r in map(json.loads, fo.read_text().splitlines())}


GRID = [dict(low_frac=lf, nopair_pen=npn, tree_slack=ts, two_skin=tw)
        for lf in (1.0, 0.6) for npn in (0.0, 0.5, 1.5) for ts in (None, 1, 3) for tw in (False, True)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--z4", required=True, help="selector output name under z4/")
    ap.add_argument("--tag", action="append", required=True, help="feature tag(s) used by that selector run")
    ap.add_argument("--out-name", required=True)
    a = ap.parse_args(argv)
    from scipy.optimize import linear_sum_assignment

    inv = dev_inventory()
    folds = D.make_folds(inv)
    sfold = folds["scan_fold"]
    summ = json.loads((OUT / a.z4 / "summary.json").read_text())
    thr_k = {int(k): v["threshold"] for k, v in summ["thresholds"].items()}
    dup_k = {int(k): float(v.get("dedupe_mm", 4.0)) for k, v in summ["thresholds"].items()}
    Z = np.load(OUT / a.z4 / "oof_scores.npz")
    sid_arr = Z["sid"]
    rows_by, gold, planes, tmm = {}, {}, {}, {}
    for sid in dev_ids():
        d = None
        for tag in a.tag:
            f = FEAT_ROOT / tag / f"{sid}.json"
            if f.is_file():
                d = json.loads(f.read_text())
        rows_by[sid] = d["rows"] if d else []
        tmm[sid] = (d or {}).get("meta", {}).get("thickness_mm")
        planes[sid] = load_planes(sid, tmm[sid])
        gold[sid] = D.load_gold_lines(sid, inv)
    idx = {sid: np.flatnonzero(sid_arr == sid) for sid in dev_ids()}

    def counts(sid, keep_rows):
        g = gold[sid]
        if g["zero_bend_control"]:
            return 0, len(keep_rows), 0
        if not keep_rows:
            return 0, 0, len(g["lines"])
        e = np.array([[r["line_start"], r["line_end"]] for r in keep_rows], float)
        mm = D.match_matrix(e, g["lines"])
        if not mm.any():
            return 0, len(keep_rows), len(g["lines"])
        r, c = linear_sum_assignment(-mm.astype(float))
        tp = int(mm[r, c].sum())
        return tp, len(keep_rows) - tp, len(g["lines"]) - tp

    def run_cfg(sids, score_key, k, cfg):
        probs, orders = [], {}
        for sid in sids:
            sc = Z[score_key][idx[sid]] if len(idx[sid]) else np.zeros(0)
            p, order = build_scan(sid, rows_by[sid], sc, thr_k[k], tmm[sid], planes[sid], dup_mm=dup_k[k], **cfg)
            probs.append(p); orders[sid] = order
        sol = solve_many(probs)
        out = {}
        for sid in sids:
            out[sid] = [rows_by[sid][orders[sid][i]] for i in sol[sid]["keep"]] if orders[sid] else []
        return out

    chosen, recs = {}, []
    ev = OUT / a.out_name / "eval_lines"
    ev.mkdir(parents=True, exist_ok=True)
    for k in range(folds["k"]):
        train = [s for s in dev_ids() if sfold[s] != k]
        test = [s for s in dev_ids() if sfold[s] == k]
        best = (-1, None)
        for cfg in GRID:
            sel = run_cfg(train, f"inner_{k}", k, cfg)
            tp = fp = fn = 0
            for s in train:
                if gold[s]["zero_bend_control"]:
                    continue
                x = counts(s, sel[s]); tp += x[0]; fp += x[1]; fn += x[2]
            f1 = D.prf(tp, fp, fn)[2]
            if f1 > best[0] + 1e-9:
                best = (f1, cfg)
        chosen[k] = {"cfg": best[1], "inner_f1": round(best[0], 4)}
        sel = run_cfg(test, "oof", k, best[1])
        for s in test:
            lines = [{"line_start": r["line_start"], "line_end": r["line_end"]} for r in sel[s]]
            recs.append(D.score_lines(s, inv, lines))
            (ev / f"{s}.json").write_text(json.dumps({"schema": "sherman.broad_baseline.eval_lines.v1",
                                                       "lane": f"z5_{a.out_name}", "frame": "scan_raw",
                                                       "status": "OK" if lines else "NO_PREDICTIONS",
                                                       "owned_bend_regions": [{"bend_id": f"Z{i+1}", **l} for i, l in enumerate(lines)]}))
        print(k, chosen[k], flush=True)
    res = {"z4": a.z4, "chosen": chosen, "oof": D.aggregate(recs), "grid_size": len(GRID)}
    (OUT / a.out_name / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / a.out_name / "per_case.json").write_text(json.dumps({r["case_id"]: {k2: r.get(k2) for k2 in (
        "disposition", "gold_count", "prediction_count", "true_positive", "false_positive", "false_negative")} for r in recs}, indent=1))
    print(json.dumps(res["oof"], indent=1))


if __name__ == "__main__":
    main()
