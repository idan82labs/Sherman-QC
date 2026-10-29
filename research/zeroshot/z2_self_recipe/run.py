"""Z2 lane: scan.ply -> eval_lines.json (sherman.broad_baseline.eval_lines.v1, raw scan frame).

usage (repo root):
  .venv/bin/python -m research.zeroshot.z2_self_recipe.run --scan <sc_id> [--out DIR]
  .venv/bin/python -m research.zeroshot.z2_self_recipe.run --batch --tag it01 [--only sc_a,sc_b] [--workers 1]

Batch runs loop over the Z2 DEV cohort (common.cohort()) inside one process; heavy runs go through
/private/tmp/claude-501/sherman_slot.sh. No CAD, no registration, no labels are read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
import traceback
from pathlib import Path

from research.zeroshot.z2_self_recipe import common as C
from research.zeroshot.z2_self_recipe.segment import segment, SegParams
from research.zeroshot.z2_self_recipe.graph import build_graph, nms, suppress_facets, break_triangles, discover_risers, GraphParams
import bb_common as bb

LANE = "z2_self_recipe"

# grouped-CV grid over the numeric graph thresholds that were read off DEV (it01-it05)
GRID = {f"ro{r}_sp{sp}": GraphParams(max_ro_t=float(r), min_span_mm=float(sp), d1_snap=False)
        for r in (12, 17, 25, 1000) for sp in (4, 6, 8)}


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _d1_snap(scan, kept):
    """Final placement: D1's gated outer-apex snap (read-only call; gate t / sin(theta/2) + 1 mm)."""
    import math
    import numpy as np
    from research.zeroshot.d1_two_skin import two_skin as D1
    out = []
    for e in kept:
        th = 180.0 - e.sweep_deg
        try:
            prof = D1.bend_profile(scan.P, scan.N, np.asarray(e.line_start), np.asarray(e.line_end), scan.t)
            gate = scan.t / max(math.sin(math.radians(th) / 2.0), 1e-3) + 1.0
            out.append(D1.snap_line_to_outer_apex(e.line_start, e.line_end, prof, gate))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def run_scan(path: Path, seg: SegParams | None = None, gp: GraphParams | None = None, scan=None) -> tuple[dict, dict, object]:
    t0 = time.time()
    scan = scan if scan is not None else segment(str(path), seg)
    t1 = time.time()
    risers = discover_risers(scan, gp) if (gp is not None and gp.risers) else []
    edges, ginfo = build_graph(scan, gp)
    ginfo["risers"] = risers
    gp = gp or GraphParams()
    facets = suppress_facets(scan, edges, gp)
    ginfo["facets"] = sorted(facets)
    kept = nms(edges, gp, facets)
    kept, ginfo["triangle_drops"] = break_triangles(kept)
    snaps = _d1_snap(scan, kept) if gp.d1_snap else [None] * len(kept)
    t2 = time.time()
    lines = []
    for i, e in enumerate(kept):
        if snaps[i] is not None and snaps[i][2].get("moved"):
            e.line_start, e.line_end = snaps[i][0].tolist(), snaps[i][1].tolist()
        lines.append(bb.as_line(f"Z2_{i:02d}", e.line_start, e.line_end, origin="z2_self_recipe",
                                flanges=[e.a, e.b], sweep_deg=round(e.sweep_deg, 3),
                                included_angle_deg=round(180.0 - e.sweep_deg, 3), span_mm=round(e.span_mm, 2),
                                cover=round(e.cover, 3), score=round(e.score, 4), ro_mm=e.ro_mm,
                                stations=e.stations, sharp_start=e.sharp_start, sharp_end=e.sharp_end))
    diag = {"t_mm": scan.t, "t_conf": scan.t_conf, "points": len(scan.P), "spacing": scan.spacing,
            "regions": len(scan.regions), "flanges": [
                {"fid": f.fid, "area": round(f.area, 1), "two_sided": f.two_sided, "n": [round(x, 4) for x in f.normal],
                 "width": round(f.width, 1), "length": round(f.length, 1),
                 "skins": [{"rid": r, "n": [round(x, 5) for x in pl.n], "d": round(float(pl.d), 4)} for r, pl in f.skins]}
                for f in scan.flanges],
            "labelled_fraction": float((scan.label >= 0).mean()),
            "edges": [{"a": e.a, "b": e.b, "ok": e.ok, "reason": e.reason, "sweep": round(e.sweep_deg, 2),
                       "span": round(e.span_mm, 1), "arc_pts": e.arc_points, "arc_slices": e.arc_slices,
                       "cover": round(e.cover, 3), "ro": e.ro_mm, "line": [e.line_start, e.line_end] if e.ok else None,
                       "info": {k: v for k, v in e.info.items() if isinstance(v, (int, float, str, bool, list, tuple))}}
                      for e in edges],
            "graph": ginfo, "timing_s": {**{k: round(v, 3) for k, v in scan.timing.items()},
                                         "segment": round(t1 - t0, 3), "graph": round(t2 - t1, 3), "total": round(t2 - t0, 3)}}
    sha = bb.sha256_file(path)
    payload = bb.eval_payload(lane=LANE, lines=lines, scan_path=path, scan_sha256=sha, source_kind="z2_flange_graph",
                              source_path=None, conversion={"rule": "z2 scan-derived flange graph + E3 local metrology",
                                                            "frame_in": "scan_raw", "transform": "identity",
                                                            "code": code_fingerprint()})
    return payload, diag, scan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", action="store_true", help="graph-parameter grid for the grouped-CV check (one segmentation per scan)")
    x = ap.parse_args(argv)
    inv = C.inventory()
    ids = [x.scan] if x.scan else (x.only.split(",") if x.only else C.cohort())
    allowed = set(C.cohort())
    root = Path(x.out) if x.out else C.OUT / "runs" / x.tag
    root.mkdir(parents=True, exist_ok=True)
    rec = {"tag": x.tag, "code": code_fingerprint(), "started": bb.utc_now(), "scans": {}}
    for sid in ids:
        if sid not in allowed:
            print(f"{sid}: not in the Z2 DEV cohort, skipped", flush=True)
            continue
        d = root / sid
        d.mkdir(exist_ok=True)
        t0 = time.time()
        try:
            path = C.scan_path(sid, inv)
            payload, diag, scan = run_scan(path)
            bb.write_json(d / "eval_lines.json", payload)
            bb.write_json(d / "diag.json", diag)
            if x.grid:
                for name, gp in GRID.items():
                    gd = root.parent / f"{x.tag}_{name}" / sid
                    gd.mkdir(parents=True, exist_ok=True)
                    pg, dg, _ = run_scan(path, gp=gp, scan=scan)
                    bb.write_json(gd / "eval_lines.json", pg)
                    bb.write_json(gd / "diag.json", dg)
            st = "OK"
        except Exception as err:  # noqa: BLE001
            st = f"FAIL:{type(err).__name__}:{err}"
            (d / "error.txt").write_text(traceback.format_exc())
            bb.write_json(d / "eval_lines.json", bb.eval_payload(lane=LANE, lines=[], scan_path=None, scan_sha256=None,
                                                                 source_kind="z2_flange_graph", source_path=None,
                                                                 conversion={}, status="FAILED",
                                                                 failure={"error": st}))
        dt = time.time() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS: bytes
        rec["scans"][sid] = {"status": st, "wall_s": round(dt, 2), "peak_rss_gib": round(rss, 3)}
        print(f"{sid} {st[:80]} {dt:.1f}s rss={rss:.2f}G", flush=True)
    rec["finished"] = bb.utc_now()
    bb.write_json(root / "run_record.json", rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
