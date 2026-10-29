"""Build the v4 candidate cache for ONE pool scan (run through sherman_slot.sh; one job at a time).

Writes docs/production/runlog/zs_v4/cache/<case_id>.json.gz with:
  * "v1": the zs_lane_v1 pinned blind pipeline's pooled candidates, every one D1-snapped, with the 26 pooled
          features (lane_cli.run_lane blind config: Z2 it08 thickness on the scan, z3_slicing.slicer with that t and
          max_families 20, pooling, snap). Selecting with the pinned GBM at 0.5 reproduces the pinned v1 output;
          refitting the GBM in folds is "v1 retrained".
  * "v4": research/zeroshot/zs_lane_v4/core.process (v3 rev3 detection stages + v4 profiles / metadata).
  * "v3x": only for scans above the face budget: core.process with the v1 job's thickness (exactly the pinned v3
          inputs, for the reproduction check); profiles off.
The scan bytes are sha256-checked against POOL.json before anything runs. No gold, no labels are read here.
usage (repo root): sherman_slot.sh .venv/bin/python -B docs/production/runlog/zs_v4/tools/cache_build.py <case_id>
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
CACHE = REPO / "docs/production/runlog/zs_v4/cache"


def jdefault(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def xlist(X) -> list:
    X = np.asarray(X, float)
    return [[None if not np.isfinite(x) else float(x) for x in row] for row in X.tolist()] if X.size else []


def slim_cands(C: list[dict]) -> list[dict]:
    keep = ("start", "end", "start_s", "end_s", "src", "has_z3", "incl", "da", "db", "flank_a", "flank_b", "length",
            "family", "n_support")
    out = []
    for c in C:
        o = {k: c[k] for k in keep if k in c}
        sn = c.get("snap") or {}
        o["snap"] = {k: sn[k] for k in ("moved", "locus", "shift_mm", "reason") if k in sn}
        out.append(o)
    return out


def main(case_id: str) -> int:
    from research.zeroshot.zs_lane_v4 import core as C4
    from research.zeroshot.zs_lane_v1 import core as C1
    from research.zeroshot.z3_slicing import slicer
    pool = {s["case_id"]: s for s in json.loads((REPO / "docs/production/runlog/zs_v4/POOL.json").read_text())["scans"]}
    s = pool[case_id]
    path = s["path"]
    t0 = time.time()
    doc = {"schema": "sherman.zs_v4.cache.v1", "case_id": case_id, "scan_id": s["scan_id"], "sha256": s["sha256"],
           "face_count": s.get("face_count"), "errors": {}}
    sha = C1.sha256_file(path)
    if sha != s["sha256"]:
        raise SystemExit(f"sha256 mismatch for {case_id}: {sha}")
    # ---- Z2 once on the scan as given (it08 + v2 fallback, payload attributes kept)
    tz = time.time()
    z2 = C4.run_z2(path)
    z2_wall = time.time() - tz
    lines4, scan4, info4 = z2
    if "fallback" in info4:                 # the plain it08 run found nothing: v1 sees the plain result
        l1, _s1, i1 = C1.run_z2(path)
        z2_v1, t1 = l1, float(i1["t_mm"])
    else:
        z2_v1, t1 = [{k: l[k] for k in ("start", "end", "incl")} for l in lines4], float(info4["t_mm"])
    # ---- v1 pinned blind pipeline, every candidate snapped
    v1 = {"t_mm": t1, "z2_wall": round(z2_wall, 3)}
    try:
        ta = time.time()
        mesh = slicer.load_mesh(path)
        z3 = slicer.detect(mesh, slicer.Params(max_families=20), thickness={"t_mm": t1, "source": "CAD"})
        z3["_mesh"] = mesh
        v1["z3_wall"] = round(time.time() - ta, 3)
        cands, X = C1.pool(z3, z2_v1, True)
        tb = time.time()
        C1.snap(cands, (np.asarray(mesh.V, float), np.asarray(mesh.vn, float), t1))
        v1["snap_wall"] = round(time.time() - tb, 3)
        v1.update(status="OK", n_families=len(z3["families"]), candidates=slim_cands(cands), X=xlist(X),
                  z2_lines=z2_v1)
        del mesh, z3
    except Exception as e:  # noqa: BLE001
        v1.update(status="FAILED", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
    doc["v1"] = v1
    # ---- v4 record (v3 rev3 detection stages + profiles)
    tc = time.time()
    rec = C4.process(path, z2_given=z2)
    del z2, scan4
    rec.pop("_snap_pts", None)
    doc["v4"] = {"status": rec["status"], "stages": rec["stages"], "errors": {k: v for k, v in rec["errors"].items()},
                 "meta": rec.get("meta"), "candidates": slim_cands(rec["candidates"]), "X": xlist(rec["X"]),
                 "z2_lines": rec["z2_lines"], "evidence": rec.get("evidence"), "profiles": rec.get("profiles"),
                 "wall_s": round(time.time() - tc, 3)}
    budgeted = bool((rec["stages"].get("budget") or {}).get("applied"))
    del rec
    if budgeted:
        td = time.time()
        r3 = C4.process(path, thickness=t1, profiles=False)
        doc["v3x"] = {"status": r3["status"], "stages": r3["stages"], "t_mm": t1,
                      "candidates": slim_cands(r3["candidates"]), "X": xlist(r3["X"]), "z2_lines": r3["z2_lines"],
                      "evidence": r3.get("evidence"), "wall_s": round(time.time() - td, 3)}
        del r3
    doc["wall_s"] = round(time.time() - t0, 3)
    import resource
    doc["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 3)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"{case_id}.json.gz.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh, default=jdefault)
    os.replace(tmp, CACHE / f"{case_id}.json.gz")
    print(f"{case_id} v1={v1.get('status')}:{len(v1.get('candidates') or [])} v4={doc['v4']['status']}:"
          f"{len(doc['v4']['candidates'])} budget={budgeted} wall={doc['wall_s']:.1f}s rss={doc['peak_rss_gib']:.2f}G",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
