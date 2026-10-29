"""zs_lane_v2 core: one scan -> pooled candidates -> evidence -> selector -> span post-processing -> eval_lines.

Pipeline (no CAD, no labels, no part-specific inputs):
  1. Z3 axis-family slicing (vendored copy zs_lane_v2/z3s.py of z3_slicing/slicer.py, with the v2 changes listed
     there): every candidate with its 21 plain geometric features.
  2. Z2 scan-derived flange graph (research/zeroshot/z2_self_recipe, final it08 config, D1 snap inside): lines.
  3. Pooling (z3_slicing.pooled.pool_scan, read-only reuse): each Z2 line becomes an agreement flag on the Z3
     candidate it duplicates (4 deg / 2.5 mm / 50 %); Z2 lines with no Z3 counterpart become extra candidates whose
     Z3 features are NaN. Features = 21 Z3 + 5 pool flags.
  4. One GBM selector (fit by train.py; DEV out-of-fold numbers only) and a pre-registered fixed threshold.
  5. D1 gated outer-apex snap on every pooled candidate (shift <= t / sin(theta/2) + 1 mm), on the Z3 mesh.
  6. v2 evidence (evidence.py) on the mesh: per-candidate station profile (normal sweep across the bisector,
     flank presence), collinear-pair plate-continuity evidence, interior interruption runs with plate
     connectivity; the span post-processing (postproc.py) joins / splits the selected lines on that evidence.
A stage that fails is recorded; if no stage produces candidates the payload is empty with status FAILED (the scorer
counts every gold line of that scan as a false negative; failed scans are never dropped).
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "research/broad_baseline"), str(REPO / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LANE = "zs_lane_v2"
SNAP_PAD_MM = 1.0    # D1 gate: shift <= t / sin(theta/2) + 1 mm (D1_REPORT section 5)
Z3_PARAMS: dict | None = None     # overrides of z3s.Params (set by lane / dev_dump from the config)
Z2_FALLBACK: bool = False         # v2: rerun Z2 with scale-relative settings when it finds no line
EVIDENCE_SPLITS: bool = True      # interior interruption runs (only used by the split option; off in the lane
                                  # when the frozen config does not split)
KEEP = ("da", "db", "flank_a", "flank_b", "length", "family", "n_support")


def _rss_gib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS: bytes


def run_z3(path: str) -> dict:
    from research.zeroshot.zs_lane_v2 import z3s as slicer
    t0 = time.time()
    mesh = slicer.load_mesh(path)
    res = slicer.detect(mesh, slicer.Params(**(Z3_PARAMS or {})))
    res["_mesh"] = mesh
    res["timing_s"]["wall"] = time.time() - t0
    return res


def run_z2(path: str) -> tuple[list[dict], Any, dict]:
    from research.zeroshot.z2_self_recipe.run import run_scan
    t0 = time.time()
    payload, diag, scan = run_scan(Path(path))
    fallback = None
    if Z2_FALLBACK and not (payload.get("owned_bend_regions") or []):
        payload, diag, scan, fallback = z2_fallback(path, scan, diag)
    lines = []
    for r in payload.get("owned_bend_regions") or []:
        lines.append({"start": [float(v) for v in r["line_start"]], "end": [float(v) for v in r["line_end"]],
                      "incl": float(r.get("included_angle_deg") or 90.0)})
    info = {"wall": time.time() - t0, "t_mm": float(diag["t_mm"]), "n_lines": len(lines)}
    if fallback is not None:
        info["fallback"] = fallback
    return lines, scan, info


def z2_fallback(path: str, scan, diag) -> tuple[dict, dict, Any, dict]:
    """v2 Z2 fallback for scans where the it08 flange graph finds no line (small / coarse parts): the
    segmentation and graph floors that are absolute mm / mm^2 become scale-relative (fractions of the part
    size), floored at the scan's point spacing. Rerun once; the first result is kept when the rerun fails."""
    from dataclasses import replace
    from research.zeroshot.z2_self_recipe.run import run_scan
    from research.zeroshot.z2_self_recipe.segment import segment, SegParams
    from research.zeroshot.z2_self_recipe.graph import GraphParams
    P = np.asarray(scan.P, float)
    ext = np.percentile(P, 99, axis=0) - np.percentile(P, 1, axis=0)
    diag_mm = float(np.linalg.norm(ext))
    sp = float(getattr(scan, "spacing", 0.5) or 0.5)
    seg = replace(SegParams(), min_region_area_mm2=float(max(4.0 * sp * sp, min(12.0, 1e-3 * diag_mm ** 2))),
                  min_seed_points=int(max(10, min(30, (diag_mm / max(sp, 1e-3)) ** 2 * 1e-3))))
    gp = replace(GraphParams(), min_span_mm=float(max(3.0 * sp, min(8.0, 0.08 * diag_mm))))
    info = {"diag_mm": round(diag_mm, 1), "spacing": round(sp, 3), "min_region_area_mm2": seg.min_region_area_mm2,
            "min_seed_points": seg.min_seed_points, "min_span_mm": gp.min_span_mm}
    try:
        scan2 = segment(str(path), seg)
        payload2, diag2, scan2 = run_scan(Path(path), seg=seg, gp=gp, scan=scan2)
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        return {"owned_bend_regions": []}, diag, scan, info
    info["n_lines"] = len(payload2.get("owned_bend_regions") or [])
    return payload2, diag2, scan2, info


def pool(z3: dict | None, z2_lines: list[dict] | None, use_z2: bool) -> tuple[list[dict], np.ndarray]:
    """Pooled candidates + feature matrix (z3_slicing.pooled.pool_scan). Extra (Z2-only) candidates get their
    included angle from the Z2 line they came from."""
    from research.zeroshot.z3_slicing.pooled import pool_scan
    d = z3 if z3 is not None else {"candidates": []}
    ext = {"z2": [{"start": [float(v) for v in l["start"]], "end": [float(v) for v in l["end"]]}
                  for l in (z2_lines or [])]} if use_z2 else {}
    C, X = pool_scan(d, ext)
    key = {(tuple(np.round(l["start"], 6)), tuple(np.round(l["end"], 6))): l["incl"] for l in (z2_lines or [])}
    out = []
    for c in C:
        o = {"start": [float(v) for v in c["start"]], "end": [float(v) for v in c["end"]],
             "src": sorted(c.get("_src", [])), "has_z3": "incl" in c}
        for k in KEEP:
            if k in c:
                o[k] = c[k].tolist() if hasattr(c[k], "tolist") else c[k]
        if "incl" in c:
            o["incl"] = float(c["incl"])
        else:
            o["incl"] = float(key.get((tuple(np.round(c["start"], 6)), tuple(np.round(c["end"], 6))), 90.0))
        out.append(o)
    return out, np.asarray(X, float)


def snap_points(z2_scan, z3: dict | None) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Point set for the D1 snap: the Z3 mesh vertices / area-weighted outward normals / Z3 t (the same for every
    variant, with or without Z2), else the Z2 scan points / outward normals / t."""
    if z3 is not None and z3.get("_mesh") is not None:
        m = z3["_mesh"]
        return np.asarray(m.V, float), np.asarray(m.vn, float), float(z3["t_mm"])
    if z2_scan is not None:
        return np.asarray(z2_scan.P, float), np.asarray(z2_scan.N, float), float(z2_scan.t)
    return None


def snap(cands: list[dict], pts) -> list[dict]:
    """D1 gated outer-apex snap of each line (axis and span kept). Adds start_s / end_s / snap info."""
    from research.zeroshot.d1_two_skin import two_skin as D1
    for c in cands:
        c["start_s"], c["end_s"], c["snap"] = c["start"], c["end"], {"moved": False}
        if pts is None:
            c["snap"]["reason"] = "NO_POINTS"
            continue
        P, N, t = pts
        th = float(c.get("incl") or 90.0)
        try:
            prof = D1.bend_profile(P, N, np.asarray(c["start"]), np.asarray(c["end"]), t)
            gate = t / max(math.sin(math.radians(th) / 2.0), 1e-3) + SNAP_PAD_MM
            a, b, info = D1.snap_line_to_outer_apex(c["start"], c["end"], prof, gate)
        except Exception as e:  # noqa: BLE001
            c["snap"] = {"moved": False, "reason": f"ERROR:{type(e).__name__}"}
            continue
        c["start_s"], c["end_s"] = [float(v) for v in a], [float(v) for v in b]
        c["snap"] = {k: v for k, v in info.items() if isinstance(v, (bool, int, float, str))}
    return cands


def process(path: str, *, use_z2: bool = True, use_z3: bool = True, snap_all: bool = False) -> dict:
    """Run the detection stages on one scan and pool the candidates (selection is lane.select_lines). With
    ``snap_all`` (DEV dump) every pooled candidate is snapped so out-of-fold selection can reuse the placements;
    otherwise only the selected lines are snapped (lane.py), with the point set kept in rec['_snap_pts']."""
    rec: dict[str, Any] = {"stages": {}, "errors": {}}
    t0 = time.time()
    z3 = z2_lines = z2_scan = None
    try:
        if not use_z3:
            raise RuntimeError("Z3 disabled for this variant")
        z3 = run_z3(path)
        rec["stages"]["z3"] = {"wall": round(z3["timing_s"]["wall"], 3), "t_mm": z3["t_mm"],
                               "n_candidates": len(z3["candidates"]), "area_total": z3.get("area_total"),
                               "n_families": len(z3.get("families", [])),
                               "family_fallback": bool(z3.get("family_fallback"))}
    except Exception as e:  # noqa: BLE001
        if use_z3:
            rec["errors"]["z3"] = f"{type(e).__name__}: {e}"
            rec["errors"]["z3_trace"] = traceback.format_exc()[-2000:]
    if use_z2:
        try:
            z2_lines, z2_scan, info = run_z2(path)
            rec["stages"]["z2"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}
        except Exception as e:  # noqa: BLE001
            rec["errors"]["z2"] = f"{type(e).__name__}: {e}"
            rec["errors"]["z2_trace"] = traceback.format_exc()[-2000:]
    t1 = time.time()
    cands, X = pool(z3, z2_lines, use_z2)
    rec["stages"]["pool"] = {"wall": round(time.time() - t1, 3), "n": len(cands)}
    rec["z2_lines"] = z2_lines or []
    pts = snap_points(z2_scan, z3)
    rec["_snap_pts"] = pts
    if snap_all:
        t2 = time.time()
        snap(cands, pts)
        rec["stages"]["snap_all"] = {"wall": round(time.time() - t2, 3), "n": len(cands)}
    t3 = time.time()
    try:
        rec["evidence"] = evidence_stage(cands, z3)
    except Exception as e:  # noqa: BLE001
        rec["errors"]["evidence"] = f"{type(e).__name__}: {e}"
        rec["errors"]["evidence_trace"] = traceback.format_exc()[-2000:]
        rec["evidence"] = None
    rec["stages"]["evidence"] = {"wall": round(time.time() - t3, 3)}
    rec["candidates"] = cands
    rec["X"] = X
    rec["wall_s"] = round(time.time() - t0, 3)
    rec["peak_rss_gib_process"] = round(_rss_gib(), 3)
    rec["status"] = "OK" if (z3 is not None or z2_lines is not None) else "FAILED"
    return rec


def evidence_stage(cands: list[dict], z3: dict | None) -> dict | None:
    """v2 evidence on the Z3 mesh for every pooled candidate (raw placement): station-profile summaries
    (normal sweep, flank presence), interior interruption runs with plate connectivity (lines >= 10 mm), and
    plate-continuity evidence for consecutive collinear pairs. None when there is no mesh."""
    from research.zeroshot.zs_lane_v2 import evidence as E
    if z3 is None or z3.get("_mesh") is None:
        return None
    m = z3["_mesh"]
    ctx = E.MeshCtx(m.V, m.vn, m.F, float(z3["t_mm"]))
    sweep, splits = [], {}
    for i, c in enumerate(cands):
        if "da" not in c or float(np.linalg.norm(np.subtract(c["end"], c["start"]))) < 1.0:
            sweep.append(None)
            continue
        prof = E.station_profile(ctx, c)
        sweep.append([E.sweep_features(prof)[k] for k in E.SWEEP_FEATS])
        if EVIDENCE_SPLITS and float(np.linalg.norm(np.subtract(c["end"], c["start"]))) >= 10.0:
            r = E.split_runs(ctx, c, prof)
            if r:
                splits[str(i)] = r
    pairs = []
    for i, j, g0, g1, ov in E.collinear_pairs(cands, ctx.t):
        if "da" not in cands[i] and "da" in cands[j]:
            # i has no flank directions (a Z2-only line): measure the gap along j instead
            q0 = np.asarray(cands[j]["start"], float)
            e = np.asarray(cands[j]["end"], float) - q0
            Lj = float(np.linalg.norm(e))
            e /= max(Lj, 1e-9)
            sb = sorted([float((np.asarray(cands[i]["start"], float) - q0) @ e),
                         float((np.asarray(cands[i]["end"], float) - q0) @ e)])
            h0, h1 = (sb[1], 0.0) if sb[1] <= 0.5 * Lj else (Lj, sb[0])
            if h1 < h0:
                h0, h1 = h1, h1
            ev = E.pair_evidence(ctx, cands[j], cands[i], h0, h1)
            ev["len_i"], ev["len_j"] = ev["len_j"], ev["len_i"]
            ev["flank_i"], ev["flank_j"] = ev["flank_j"], ev["flank_i"]
            ev["ref"] = "j"
        else:
            ev = E.pair_evidence(ctx, cands[i], cands[j], g0, g1)
        ev.update(i=int(i), j=int(j), g0=float(g0), g1=float(g1), overlap=float(ov))
        pairs.append(ev)
    return {"sweep": sweep, "splits": splits, "pairs": pairs, "spacing": ctx.spacing, "cell": ctx.cell,
            "t_mm": ctx.t}


def payload(path: str, scan_sha: str | None, lines: list[dict], *, status: str, conversion: dict,
            lane: str = LANE) -> dict:
    import bb_common as bb
    regs = []
    for i, c in enumerate(lines):
        meta = {k: c[k] for k in ("score", "incl", "src", "op", "parts") if k in c}
        if "snap" in c:
            meta["d1_snap"] = c["snap"]
        regs.append(bb.as_line(f"ZS{i + 1:03d}", c["start_s"] if "start_s" in c else c["start"],
                               c["end_s"] if "end_s" in c else c["end"], **meta))
    failure = None if status == "OK" else {"error": conversion.get("errors")}
    return bb.eval_payload(lane=lane, lines=regs, scan_path=Path(path), scan_sha256=scan_sha,
                           source_kind=lane, source_path=None, conversion=conversion, status=status, failure=failure)


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
