"""zs_lane_v4 core: one scan -> pooled candidates + evidence + compact support profiles (the record that the v4
selector and post-processing read).

Stages 1-6 are zs_lane_v3 revision 3's detection stages, called through the frozen modules (imported read-only,
never edited): the density-invariant face budget (zs_lane_v3.budget), Z2 (z2_self_recipe it08, with zs_lane_v2's
fallback), Z3 axis-family slicing (zs_lane_v2.z3s with fallback_frac_only, through zs_lane_v3.z3_merge's exact
bounded merge), pooling (z3_slicing.pooled), the D1 gated outer-apex snap of every pooled candidate and the v2
evidence (station sweep summaries, interior interruption runs, collinear-pair plate continuity). With the same
thickness these stages give exactly the zs_lane_v3 candidates, features and evidence.

v4 adds, all label-free and part-agnostic (mm, degrees, multiples of the measured t):
  7. compact per-candidate support profiles (``profiles.py``): per 1 mm station along the candidate's raw line,
     extended over the extent of every Z2 line pooled onto it (+5 mm), the bend-zone / flank-A / flank-B /
     normal-sweep station tests of zs_lane_v2.evidence and the tight apex-touch test at 0.5 mm stations;
  8. Z2 line attributes kept from the Z2 payload (span, cover, score, sweep) for features;
  9. scan-level metadata (thickness, mesh spacing, area, face count, family count, budget applied).
Thickness: the Z2 self-recipe estimate on the (budgeted) scan unless an external one is given (production
CAD prior) -- the blind production convention of the coverage / FE1 runs.
"""
from __future__ import annotations

import math
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from research.zeroshot.zs_lane_v2 import core as V2
from research.zeroshot.zs_lane_v2.core import REPO, sha256_file  # noqa: F401  (REPO also sets sys.path)

LANE = "zs_lane_v4"
Z3_PARAMS = {"fallback_frac_only": True}
BUDGET_MAX_FACES = 3_000_000
BUDGET_CELL_MM = 0.30
MAX_FAMILIES = 20


def configure() -> None:
    """The frozen v3 detection settings on the reused v2 stage functions (Z3 fallback, Z2 fallback, splits on)."""
    V2.Z3_PARAMS = dict(Z3_PARAMS)
    V2.Z2_FALLBACK = True
    V2.EVIDENCE_SPLITS = True


def run_z2(path: str) -> tuple[list[dict], Any, dict]:
    """zs_lane_v2.core.run_z2 (it08 + the v2 fallback) keeping the Z2 payload attributes of every line."""
    from research.zeroshot.z2_self_recipe.run import run_scan
    t0 = time.time()
    payload, diag, scan = run_scan(Path(path))
    fallback = None
    if not (payload.get("owned_bend_regions") or []):
        payload, diag, scan, fallback = V2.z2_fallback(path, scan, diag)
    lines = []
    for r in payload.get("owned_bend_regions") or []:
        lines.append({"start": [float(v) for v in r["line_start"]], "end": [float(v) for v in r["line_end"]],
                      "incl": float(r.get("included_angle_deg") or 90.0),
                      "span_mm": _f(r.get("span_mm")), "cover": _f(r.get("cover")), "z2_score": _f(r.get("score")),
                      "ro_mm": _f(r.get("ro_mm")), "stations": _f(r.get("stations"))})
    info = {"wall": time.time() - t0, "t_mm": float(diag["t_mm"]), "n_lines": len(lines),
            "spacing": _f(diag.get("spacing")), "points": int(diag.get("points") or 0),
            "n_flanges": len(diag.get("flanges") or []), "labelled_fraction": _f(diag.get("labelled_fraction"))}
    if fallback is not None:
        info["fallback"] = fallback
    return lines, scan, info


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def run_z3(path: str, t_mm: float, max_families: int = MAX_FAMILIES) -> dict:
    from research.zeroshot.zs_lane_v3 import core as V3
    return V3.run_z3(path, thickness={"t_mm": float(t_mm), "source": "given"}, max_families=int(max_families))


def process(path: str, *, thickness: float | None = None, max_families: int = MAX_FAMILIES,
            z2_given: tuple | None = None, keep_mesh: bool = False, profiles: bool = True) -> dict:
    """Detection, pooling, snap of every pooled candidate, v2 evidence and the v4 support profiles.
    ``z2_given`` = a run_z2 result on the same bytes (cache builder only; skips a second identical Z2 run)."""
    configure()
    with tempfile.TemporaryDirectory(prefix="zs_v4_budget_") as work:
        return _process(path, work, thickness, max_families, z2_given, keep_mesh, profiles)


def _process(path: str, work: str, thickness, max_families, z2_given, keep_mesh, profiles) -> dict:
    from research.zeroshot.zs_lane_v3 import budget
    rec: dict[str, Any] = {"stages": {}, "errors": {}, "lane": LANE}
    t0 = time.time()
    src_path = path
    try:
        path, binfo = budget.budget_scan(path, BUDGET_MAX_FACES, work, BUDGET_CELL_MM)
    except Exception as e:  # noqa: BLE001
        binfo = {"applied": False, "error": f"{type(e).__name__}: {e}"}
    rec["stages"]["budget"] = binfo
    z3 = z2_lines = z2_scan = None
    try:
        if z2_given is not None and path == src_path:
            z2_lines, scan, info = z2_given
        else:
            z2_lines, scan, info = run_z2(path)
        from types import SimpleNamespace
        z2_scan = SimpleNamespace(P=scan.P, N=scan.N, t=scan.t)
        del scan
        rec["stages"]["z2"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}
    except Exception as e:  # noqa: BLE001
        rec["errors"]["z2"] = f"{type(e).__name__}: {e}"
        rec["errors"]["z2_trace"] = traceback.format_exc()[-2000:]
    t_mm = thickness if thickness is not None else (rec["stages"].get("z2", {}).get("t_mm"))
    try:
        if t_mm is None:
            raise RuntimeError("no thickness (Z2 failed and none given)")
        z3 = run_z3(path, float(t_mm), max_families)
        rec["stages"]["z3"] = {"wall": round(z3["timing_s"]["wall"], 3), "t_mm": z3["t_mm"],
                               "n_candidates": len(z3["candidates"]), "area_total": z3.get("area_total"),
                               "n_families": len(z3.get("families", [])),
                               "family_fallback": bool(z3.get("family_fallback"))}
    except Exception as e:  # noqa: BLE001
        rec["errors"]["z3"] = f"{type(e).__name__}: {e}"
        rec["errors"]["z3_trace"] = traceback.format_exc()[-2000:]
    t1 = time.time()
    v2lines = [{k: l[k] for k in ("start", "end", "incl")} for l in (z2_lines or [])]
    cands, X = V2.pool(z3, v2lines if z2_lines is not None else None, True)
    rec["stages"]["pool"] = {"wall": round(time.time() - t1, 3), "n": len(cands)}
    rec["z2_lines"] = z2_lines or []
    pts = V2.snap_points(z2_scan, z3)
    t2 = time.time()
    V2.snap(cands, pts)
    rec["stages"]["snap_all"] = {"wall": round(time.time() - t2, 3), "n": len(cands)}
    t3 = time.time()
    try:
        rec["evidence"] = V2.evidence_stage(cands, z3)
    except Exception as e:  # noqa: BLE001
        rec["errors"]["evidence"] = f"{type(e).__name__}: {e}"
        rec["errors"]["evidence_trace"] = traceback.format_exc()[-2000:]
        rec["evidence"] = None
    rec["stages"]["evidence"] = {"wall": round(time.time() - t3, 3)}
    mesh = z3.get("_mesh") if z3 is not None else None
    if profiles and mesh is not None:
        t4 = time.time()
        try:
            from research.zeroshot.zs_lane_v4 import profiles as PR
            ctx = PR.mesh_ctx(mesh, float(z3["t_mm"]), rec.get("evidence"))
            rec["profiles"] = PR.candidate_profiles(ctx, cands, z2_lines or [])
        except Exception as e:  # noqa: BLE001
            rec["errors"]["profiles"] = f"{type(e).__name__}: {e}"
            rec["errors"]["profiles_trace"] = traceback.format_exc()[-2000:]
            rec["profiles"] = None
        rec["stages"]["profiles"] = {"wall": round(time.time() - t4, 3)}
    rec["meta"] = scan_meta(mesh, z3, rec, binfo)
    rec["candidates"] = cands
    rec["X"] = X
    if keep_mesh:
        rec["_mesh"] = mesh
        rec["_t_mm"] = float(z3["t_mm"]) if z3 is not None else None
    rec["wall_s"] = round(time.time() - t0, 3)
    rec["peak_rss_gib_process"] = round(V2._rss_gib(), 3)
    rec["status"] = "OK" if (z3 is not None or z2_lines is not None) else "FAILED"
    return rec


def scan_meta(mesh, z3: dict | None, rec: dict, binfo: dict) -> dict:
    """Scan-level, label-free metadata (features / diagnostics)."""
    out: dict[str, Any] = {"budget_applied": bool(binfo.get("applied")), "faces_in": binfo.get("faces")}
    z2 = rec["stages"].get("z2") or {}
    out.update(t_z2=z2.get("t_mm"), z2_spacing=z2.get("spacing"), z2_points=z2.get("points"),
               z2_n_flanges=z2.get("n_flanges"), z2_labelled=z2.get("labelled_fraction"),
               z2_n_lines=z2.get("n_lines"), z2_fallback=bool(z2.get("fallback")))
    if z3 is not None:
        out.update(t_mm=float(z3["t_mm"]), area_total=_f(z3.get("area_total")), n_families=len(z3.get("families", [])),
                   family_fallback=bool(z3.get("family_fallback")), n_z3=len(z3.get("candidates", [])))
    if mesh is not None:
        V = np.asarray(mesh.V)
        ext = np.percentile(V, 99, axis=0) - np.percentile(V, 1, axis=0) if len(V) else np.zeros(3)
        out.update(n_vertices=int(len(V)), n_faces=int(len(mesh.F)), bbox_diag=float(np.linalg.norm(ext)),
                   spacing=(rec.get("evidence") or {}).get("spacing"))
    return out
