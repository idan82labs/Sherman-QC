"""Legacy runtime control arm (plan §3 amendment A1, §10 amendments 6c, 6f, 8): the unchanged worker, serially.

The legacy runtime is run exactly as the line would have run it: the unchanged
``backend/bend_analysis_worker.py`` command line (survey contract §6a), one scan
at a time, in its own process group under :func:`..procmon.run_monitored`
(wall time, process-group peak RSS). Nothing in ``backend/`` or
``domains/bend/`` is modified or imported here.

Run conditions (recorded in every record):

- ``--tolerance-angle 1.0 --tolerance-radius 0.5`` (the DEV recipes' default
  angle and flange tolerances, as the contract maps them);
- opaque ``--job-id/--part-id/--part-name`` and opaquely named read-only input
  copies, so the worker sees no file name or part number (its blind mode;
  ``use_part_truth_hints`` stays False inside the worker);
- ``BEND_INSPECTION_CONFIG`` points at a file that does not exist (the
  worker's built-in defaults; survey contract §6 ``config/NONE``): run from the
  repo the worker would otherwise read ``output/improvement_loop/
  best_config.latest.json``, which this chunk is not authorized to read;
  ``SHERMAN_QC_F1_PROGRAM_STATUS_PATH``, ``BEND_EXPECTED_OVERRIDES`` and
  ``BEND_FEATURE_POLICIES`` likewise point at absent files (fail closed; none
  of them changes a per-bend output in blind mode);
- limits: 600 s wall, 12 GiB process-group RSS (immediate kill; the survey
  contract's runner limit for this worker, under the 16 GiB machine rule). A
  killed or failed run is an explicit failure record and every planned bend of
  the case counts as not measured by the legacy arm. The first E4b passes ran
  with 4 GiB; a run of theirs that was not killed and peaked below 12 GiB is
  the run a 12 GiB cap gives (the cap only kills; the worker is deterministic
  under fixed conditions, determinism probe), so :func:`reusable_run` accepts
  it; a run that the lower cap killed is re-run;
- the unary count model: the worker loads ``<repo>/output/unary_models/
  latest.json`` when that file exists (``domains/bend/services/
  structured_count_reporting_service.py``; no environment override, outside
  this chunk's custody, so it is never opened or stat'ed here). Whether it was
  applied is read back from every report (``structured_context`` or per-match
  ``unary_predictions``: :func:`unary_count_model_applied`) and recorded per
  run and in the summary.

**Per-bend extraction** (:func:`parse_legacy_report`). The worker's
``matches`` (one per legacy CAD bend) are mapped to the recipe's compiled
bends by geometry in the CAD frame (the worker's ``reference_mesh_raw`` frame
is the STEP frame, checked per case against E1's sample cloud:
``frame_check``): undirected angle <= 10 deg (the worker's CAD bend lines come
from a mesh detector and are coarse), offset <= max(3 mm, Ri + t + 2 mm) to the
nearest compiled locus, overlap >= 30 % of the shorter line, one-to-one minimum
cost. Per compiled bend the record keeps the legacy angle (``measured_angle`` is
a measurement whenever the bend was detected). The worker folds angles as
``180 - arccos|n1.n2|`` (survey contract §0), i.e. it reports ``max(theta,
180 - theta)`` of the included angle theta, so its nominal-relative deviation
is taken against E1's included nominal folded the same way
(``legacy_delta_vs_e1_deg``, the G2(c) comparison: both arms against the same
nominal); ``legacy_delta_deg`` against the worker's own CAD ``target_angle`` is
kept as a diagnostic. Also kept: the
legacy line (a **measured** line only when it is a scan fit: matches assigned
``CAD_LOCAL_NEIGHBORHOOD``, geometry ``CAD_REFERENCE_PLACEHOLDER`` or a line equal
to the CAD line are placeholders and never count as measured lines), and the
tip-to-tip edge distance (legacy's only flange-like number, a proxy that is
not an OML flange length). Lines are mapped to the raw-scan frame with the
inverse of the worker's ``raw_scan_to_reference_transform`` for the G2a line
metric (:func:`legacy_line_predictions`).

**Visual baseline** (:func:`legacy_render_scene`): a neutral
:class:`..render.RenderScene` (source LEGACY for :class:`..g4.RenderJob`) with the
worker's aligned scan, the final stage model's mesh and the recipe labels; the
legacy line is drawn as the measured line, so the G4 renderer, packets and
LANDS/TAG_OK review treat both arms the same way.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..procmon import run_monitored
from ..runner import RunnerConfig, engine_env
from .common import REPO_ROOT, custody_check, now_utc, read_json, sha256_file

__all__ = [
    "LEGACY_SCHEMA",
    "LEGACY_WORKER_REL",
    "LEGACY_TOLERANCE_ANGLE_DEG",
    "LEGACY_TOLERANCE_RADIUS_MM",
    "LEGACY_RSS_LIMIT_BYTES",
    "LEGACY_TIMEOUT_S",
    "legacy_env",
    "run_legacy_case",
    "reusable_run",
    "unary_count_model_applied",
    "parse_legacy_report",
    "map_legacy_bends",
    "legacy_line_predictions",
    "legacy_render_scene",
    "legacy_render_job",
]

LEGACY_SCHEMA = "sherman_e4b_legacy_control_case_v1"
LEGACY_WORKER_REL = "backend/bend_analysis_worker.py"
LEGACY_TOLERANCE_ANGLE_DEG = 1.0
LEGACY_TOLERANCE_RADIUS_MM = 0.5
LEGACY_TIMEOUT_S = 600.0
LEGACY_RSS_LIMIT_BYTES = 12 * 1024**3
_MAP_ANGLE_DEG = 10.0
_PLACEHOLDER_ASSIGNMENT = "CAD_LOCAL_NEIGHBORHOOD"
_PLACEHOLDER_GEOMETRY = "CAD_REFERENCE_PLACEHOLDER"
_LOCI = ("oml_virtual_sharp", "cylinder_axis", "iml_virtual_sharp", "inner_tangent_parent", "inner_tangent_child",
         "outer_tangent_parent", "outer_tangent_child", "inner_apex", "outer_apex")


def legacy_env(work_case: Path, repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Worker environment (module docstring): thread caps, no bytecode, absent config files."""
    env = engine_env(RunnerConfig(tier_dir=work_case, out_dir=work_case, repo_root=repo_root))
    absent = work_case / "ABSENT"
    env.update({
        "BEND_INSPECTION_CONFIG": str(absent / "bend_inspection_config.json"),
        "SHERMAN_QC_F1_PROGRAM_STATUS_PATH": str(absent / "f1_program_status.json"),
        "BEND_EXPECTED_OVERRIDES": str(absent / "expected_overrides.json"),
        "BEND_FEATURE_POLICIES": str(absent / "feature_policies.json"),
    })
    for key in ("BEND_PHASE_LOG_PATH", "BEND_PHASE_STDOUT", "SHERMAN_QC_FORCE_CAD_LOCAL_REFINEMENT"):
        env.pop(key, None)
    return env


def _stage_readonly(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)
        dst.unlink()
    shutil.copyfile(src, dst)
    os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def run_legacy_case(case: Mapping[str, Any], scan_path: Path, cad_path: Path, work_root: Path, *,
                    repo_root: Path = REPO_ROOT, timeout_s: float = LEGACY_TIMEOUT_S,
                    rss_limit_bytes: int = LEGACY_RSS_LIMIT_BYTES, worker: Path | None = None) -> dict[str, Any]:
    """Run the unchanged worker on one case (module docstring); returns the run record."""
    scan_src = custody_check(scan_path, "legacy control scan")
    cad_src = custody_check(cad_path, "legacy control CAD")
    ws = Path(work_root) / "legacy" / case["case_ref"]
    if ws.exists():
        for p in ws.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
        shutil.rmtree(ws)
    token = "LC" + case["case_ref"][3:]
    scan_in = ws / "inputs" / f"scan_{case['scan_sha256'][:12]}{scan_src.suffix.lower()}"
    cad_in = ws / "inputs" / f"cad_{case['cad_sha256'][:12]}{cad_src.suffix.lower()}"
    _stage_readonly(scan_src, scan_in)
    _stage_readonly(cad_src, cad_in)
    out = ws / "out"
    out.mkdir(parents=True, exist_ok=True)
    script = Path(worker) if worker is not None else repo_root / LEGACY_WORKER_REL
    argv = [sys.executable, "-B", str(script), "--job-id", token, "--cad-path", str(cad_in), "--scan-path", str(scan_in),
            "--part-id", token, "--part-name", token, "--tolerance-angle", f"{LEGACY_TOLERANCE_ANGLE_DEG:g}",
            "--tolerance-radius", f"{LEGACY_TOLERANCE_RADIUS_MM:g}", "--output-dir", str(out)]
    proc = run_monitored(argv, cwd=repo_root, env=legacy_env(ws, repo_root), timeout_s=timeout_s,
                         rss_limit_bytes=rss_limit_bytes, stderr_path=ws / "worker.stderr.log",
                         stdout_path=ws / "worker.stdout.log")
    report = out / "bend_inspection_report.json"
    if proc.timed_out:
        status = "TIMEOUT"
    elif proc.rss_exceeded:
        status = "RSS_LIMIT"
    elif proc.signal is not None:
        status = "CRASH_SIGNAL"
    elif proc.returncode == 0 and report.is_file():
        status = "OK"
    elif proc.returncode == 0:
        status = "NO_REPORT"
    else:
        status = "WORKER_ERROR"
    err = proc.last("ERROR") or {}
    res = proc.last("RESULT") or {}
    inputs_intact = (sha256_file(scan_in) == case["scan_sha256"] and sha256_file(cad_in) == case["cad_sha256"])
    rec: dict[str, Any] = {
        "schema": LEGACY_SCHEMA, "case_ref": case["case_ref"], "status": status,
        "scan_sha256": case["scan_sha256"], "cad_sha256": case["cad_sha256"],
        "worker_error": (str(err.get("error") or err.get("message") or "")[:500] or None),
        "exit_code": proc.returncode, "signal": proc.signal, "wall_s": round(proc.wall_s, 3),
        "peak_rss_bytes": proc.peak_rss_bytes, "timed_out": proc.timed_out, "rss_exceeded": proc.rss_exceeded,
        "processing_time_ms": res.get("processing_time_ms"), "inputs_intact": inputs_intact,
        "conditions": {"worker": LEGACY_WORKER_REL if worker is None else str(script),
                       "tolerance_angle_deg": LEGACY_TOLERANCE_ANGLE_DEG,
                       "tolerance_radius_mm": LEGACY_TOLERANCE_RADIUS_MM, "timeout_s": timeout_s,
                       "rss_limit_bytes": rss_limit_bytes, "runtime_config": "built-in defaults (config absent)",
                       "blind": True, "opaque_inputs": True},
        "work_dir": str(ws), "recorded_at": now_utc(),
        "outputs": {name: (str(out / name) if (out / name).is_file() else None)
                    for name in ("bend_inspection_report.json", "aligned_scan.ply", "deviations.npy",
                                 "reference_mesh.ply")},
    }
    if report.is_file():
        rec["report_sha256"] = sha256_file(report)
    if not inputs_intact and status == "OK":
        rec["status"] = "INPUT_MODIFIED"
    for p in (scan_in, cad_in):  # the staged copies were checked above; the sources stay authoritative
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
        p.unlink()
    return rec


def reusable_run(prev: Mapping[str, Any] | None, case: Mapping[str, Any], legacy_fp: str, *,
                 prev_case: Mapping[str, Any] | None = None, work_case: Path | None = None) -> dict[str, Any]:
    """Whether an earlier legacy run record is a run on identical inputs under identical conditions.

    Identical means: the same case (scan sha256), the same CAD sha256 (from the record, or from the
    earlier freeze's case record for runs that did not store it), the same legacy code fingerprint
    (``backend`` + ``domains/bend``), the same run conditions (tolerances, wall limit, config absent,
    blind, opaque inputs), an RSS cap that never bound (the run was not killed by its cap and its
    recorded peak is below today's :data:`LEGACY_RSS_LIMIT_BYTES`: the cap only kills, so such a run is
    the run today's cap gives), inputs intact, and -- for a successful run -- the worker's report still
    present in the work directory. A failed run is a valid record of a failure."""
    if not prev:
        return {"reusable": False, "why": "NO_EARLIER_RUN"}
    cond = prev.get("conditions") or {}
    scan_sha = prev.get("scan_sha256") or (prev_case or {}).get("scan_sha256")
    cad_sha = prev.get("cad_sha256") or (prev_case or {}).get("cad_sha256")
    peak = prev.get("peak_rss_bytes")
    checks = {
        "case_ref": prev.get("case_ref") == case["case_ref"],
        "scan_sha256": scan_sha == case["scan_sha256"],
        "cad_sha256": cad_sha is not None and cad_sha == case["cad_sha256"],
        "legacy_code_fingerprint": prev.get("legacy_code_fingerprint") == legacy_fp,
        "conditions": (cond.get("tolerance_angle_deg") == LEGACY_TOLERANCE_ANGLE_DEG
                       and cond.get("tolerance_radius_mm") == LEGACY_TOLERANCE_RADIUS_MM
                       and float(cond.get("timeout_s") or 0) == LEGACY_TIMEOUT_S
                       and str(cond.get("runtime_config", "")).startswith("built-in defaults")
                       and bool(cond.get("blind")) and bool(cond.get("opaque_inputs"))
                       and cond.get("worker") == LEGACY_WORKER_REL),
        "rss_cap_never_bound": (peak is not None and not prev.get("rss_exceeded")
                                and prev.get("status") != "RSS_LIMIT" and int(cond.get("rss_limit_bytes") or 0) > 0
                                and int(peak) < LEGACY_RSS_LIMIT_BYTES),
        "inputs_intact": bool(prev.get("inputs_intact")),
    }
    if prev.get("status") == "OK":
        report = None if work_case is None else Path(work_case) / "out" / "bend_inspection_report.json"
        checks["report_present"] = bool(report is not None and report.is_file()
                                        and (prev.get("report_sha256") is None
                                             or sha256_file(report) == prev.get("report_sha256")))
    failed = [k for k, v in checks.items() if not v]
    return {"reusable": not failed, "why": "identical inputs and conditions" if not failed
            else "differs: " + ", ".join(failed), "checks": checks}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _line_pair(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> dict[str, float]:
    """Undirected angle, offset of ``a``'s midpoint from line ``b`` and overlap of ``a`` on ``b``."""
    a0, a1 = np.asarray(a[0], float), np.asarray(a[1], float)
    b0, b1 = np.asarray(b[0], float), np.asarray(b[1], float)
    da, db = _unit(a1 - a0), _unit(b1 - b0)
    ang = float(np.degrees(np.arccos(min(1.0, abs(float(da @ db))))))
    mid = 0.5 * (a0 + a1) - b0
    off = float(np.linalg.norm(mid - db * float(mid @ db)))
    ia = sorted([float((a0 - b0) @ db), float((a1 - b0) @ db)])
    lb = float(np.linalg.norm(b1 - b0))
    ov = max(0.0, min(ia[1], lb) - max(ia[0], 0.0))
    shorter = max(1e-9, min(ia[1] - ia[0], lb))
    return {"angle_deg": ang, "offset_mm": off, "overlap": ov / shorter}


def map_legacy_bends(legacy_lines: Sequence[tuple[str, Sequence[Sequence[float]]]], compiled: Any
                     ) -> dict[str, dict[str, Any]]:
    """One-to-one mapping ``legacy bend id -> compiled bend`` (module docstring)."""
    from scipy.optimize import linear_sum_assignment

    bends = list(compiled.bends)
    if not legacy_lines or not bends:
        return {}
    cost = np.full((len(legacy_lines), len(bends)), 1e9)
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for i, (_, line) in enumerate(legacy_lines):
        for j, b in enumerate(bends):
            limit = max(3.0, float(b.inner_radius_mm) + float(b.thickness_mm) + 2.0)
            for locus in _LOCI:
                lv = getattr(b.loci, locus)
                m = _line_pair(line, (lv.start_mm, lv.end_mm))
                if m["angle_deg"] <= _MAP_ANGLE_DEG and m["offset_mm"] <= limit and m["overlap"] >= 0.3:
                    c = m["angle_deg"] / _MAP_ANGLE_DEG + m["offset_mm"] / limit + (1.0 - min(1.0, m["overlap"]))
                    if c < cost[i, j]:
                        cost[i, j] = c
                        best[(i, j)] = {**{k: round(v, 4) for k, v in m.items()}, "locus": locus,
                                        "limit_mm": round(limit, 3)}
    rows, cols = linear_sum_assignment(cost)
    out = {}
    for i, j in zip(rows, cols):
        if cost[i, j] < 1e9:
            out[legacy_lines[i][0]] = {"bend_uid": bends[j].bend_uid, **best[(i, j)]}
    return out


def _apply(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, float) @ t[:3, :3].T + t[:3, 3]


def _frame_check(reference_mesh: str | None, compiled_dir: Path, compiled: Any) -> dict[str, Any]:
    """Median distance of the worker's reference mesh vertices to E1's final sample cloud."""
    if not reference_mesh or not Path(reference_mesh).is_file():
        return {"status": "NOT_CHECKED", "reason": "no reference_mesh.ply"}
    from scipy.spatial import cKDTree

    from ..plyio import read_ply_vertices

    final = max((s for s in compiled.stages if s.kind == "SEQUENCE_PREFIX"), key=lambda s: s.stage_index)
    with np.load(compiled_dir / final.sample_cloud.rel_path, allow_pickle=False) as z:
        cloud = np.asarray(z["points"], dtype=np.float64)
    verts = read_ply_vertices(reference_mesh).vertices
    sel = np.linspace(0, len(verts) - 1, min(len(verts), 5000)).astype(np.int64)
    d, _ = cKDTree(cloud).query(verts[sel], k=1)
    med, p90 = float(np.median(d)), float(np.percentile(d, 90))
    return {"status": "CONSISTENT" if med <= 0.5 and p90 <= 2.0 else "FRAME_MISMATCH",
            "median_mm": round(med, 4), "p90_mm": round(p90, 4), "samples": int(len(sel))}


def unary_count_model_applied(report: Mapping[str, Any]) -> bool:
    """Whether the worker applied its unary count model to this report (module docstring)."""
    return bool(report.get("structured_context")) or any(
        isinstance(m, Mapping) and "unary_predictions" in m for m in report.get("matches") or [])


def parse_legacy_report(run: Mapping[str, Any], compiled: Any | None, compiled_dir: Path | None
                        ) -> dict[str, Any]:
    """Per-compiled-bend legacy outputs of one run (module docstring)."""
    out: dict[str, Any] = {"case_ref": run["case_ref"], "status": run["status"], "alignment": None,
                           "frame_check": {"status": "NOT_CHECKED"}, "bends": [], "legacy_bends": [],
                           "line_predictions_raw": [], "counts": {}, "unary_count_model_applied": None}
    path = (run.get("outputs") or {}).get("bend_inspection_report.json")
    if run["status"] != "OK" or not path:
        out["counts"] = {"legacy_bends": 0, "mapped": 0, "angle_measured": 0, "line_measured": 0}
        return out
    report = read_json(path)
    out["unary_count_model_applied"] = unary_count_model_applied(report)
    al = report.get("alignment") or {}
    t = al.get("raw_scan_to_reference_transform")
    t_ref = np.asarray(t, dtype=np.float64) if t is not None else None
    out["alignment"] = {"available": t_ref is not None and t_ref.shape == (4, 4),
                        "fitness": al.get("fitness"), "rmse_mm": al.get("rmse_mm"), "scale": al.get("scale"),
                        "raw_scan_to_reference_transform": t,
                        "geometry_frame": al.get("geometry_frame")}
    legacy = []
    for m in report.get("matches") or []:
        cad_line = (m.get("cad_line_start"), m.get("cad_line_end"))
        det = (m.get("detected_line_start"), m.get("detected_line_end"))
        placeholder = (m.get("assignment_source") == _PLACEHOLDER_ASSIGNMENT
                       or m.get("detected_line_geometry_source") == _PLACEHOLDER_GEOMETRY)
        has_det = det[0] is not None and det[1] is not None
        same_as_cad = bool(has_det and cad_line[0] is not None and np.allclose(det[0], cad_line[0], atol=1e-6)
                           and np.allclose(det[1], cad_line[1], atol=1e-6))
        detected = str(m.get("status", "")).upper() != "NOT_DETECTED"
        angle = m.get("measured_angle")
        legacy.append({
            "legacy_bend_id": str(m.get("bend_id")), "status": m.get("status"),
            "observability_state": m.get("observability_state"), "assignment_source": m.get("assignment_source"),
            "geometry_source": m.get("detected_line_geometry_source"), "feature_type": m.get("feature_type"),
            "countable": m.get("countable_in_regression"),
            "target_angle_deg": m.get("target_angle"),
            "measured_angle_deg": angle if (detected and angle is not None) else None,
            "angle_measured": bool(detected and angle is not None),
            "line_measured": bool(detected and has_det and not placeholder and not same_as_cad),
            "line_placeholder": bool(has_det and (placeholder or same_as_cad)),
            "cad_line_ref": [list(map(float, cad_line[0])), list(map(float, cad_line[1]))]
            if cad_line[0] is not None else None,
            "detected_line_ref": [list(map(float, det[0])), list(map(float, det[1]))] if has_det else None,
            "edge_distance_measured_mm": m.get("measured_edge_distance_mm"),
            "edge_distance_expected_mm": m.get("expected_edge_distance_mm"),
        })
    out["legacy_bends"] = legacy
    raw_lines = []
    if t_ref is not None and t_ref.shape == (4, 4):
        inv = np.linalg.inv(t_ref)
        for lb in legacy:
            if lb["line_measured"]:
                pts = _apply(inv, np.asarray(lb["detected_line_ref"], float))
                raw_lines.append({"bend_id": f"legacy_{lb['legacy_bend_id']}", "line_start": pts[0].tolist(),
                                  "line_end": pts[1].tolist(), "source": "LEGACY_SCAN_FIT", "frame": "raw"})
    out["line_predictions_raw"] = raw_lines
    mapping: dict[str, dict[str, Any]] = {}
    if compiled is not None and compiled_dir is not None:
        out["frame_check"] = _frame_check((run.get("outputs") or {}).get("reference_mesh.ply"), compiled_dir, compiled)
        mapping = map_legacy_bends([(lb["legacy_bend_id"], lb["cad_line_ref"]) for lb in legacy
                                    if lb["cad_line_ref"] is not None], compiled)
        by_uid = {v["bend_uid"]: (k, v) for k, v in mapping.items()}
        legacy_by_id = {lb["legacy_bend_id"]: lb for lb in legacy}
        rows = []
        for b in compiled.bends:
            hit = by_uid.get(b.bend_uid)
            lb = legacy_by_id.get(hit[0]) if hit else None
            target = None if lb is None else lb["target_angle_deg"]
            measured = None if lb is None else lb["measured_angle_deg"]
            convention = None
            if target is not None:
                nominal = float(b.nominal_included_deg)
                folded = max(nominal, 180.0 - nominal)
                convention = ("FOLDED_INCLUDED" if abs(float(target) - folded) <= 2.0 else
                              "SWEEP" if abs(float(target) - (180.0 - nominal)) <= 2.0 else "TARGET_DIFFERS_FROM_E1")
            nominal_folded = max(float(b.nominal_included_deg), 180.0 - float(b.nominal_included_deg))
            rows.append({
                "bend_uid": b.bend_uid, "sequence_index": b.sequence_index, "label": b.label,
                "legacy_bend_id": None if hit is None else hit[0], "mapping": None if hit is None else hit[1],
                "angle": {"measured": bool(lb and lb["angle_measured"]), "legacy_measured_deg": measured,
                          "legacy_target_deg": target, "e1_nominal_included_deg": float(b.nominal_included_deg),
                          "e1_nominal_folded_deg": nominal_folded, "convention": convention,
                          "legacy_delta_vs_e1_deg": (float(measured) - nominal_folded) if measured is not None
                          else None,
                          "legacy_delta_deg": (float(measured) - float(target)) if measured is not None
                          and target is not None else None},
                "line": {"measured": bool(lb and lb["line_measured"]), "placeholder": bool(lb and lb["line_placeholder"]),
                         "ref": None if lb is None or not lb["line_measured"] else lb["detected_line_ref"]},
                "edge_distance_proxy": {"measured_mm": None if lb is None else lb["edge_distance_measured_mm"],
                                        "expected_mm": None if lb is None else lb["edge_distance_expected_mm"],
                                        "note": "legacy tip-to-tip distance, not an OML flange length"},
                "legacy_status": None if lb is None else lb["status"],
            })
        out["bends"] = rows
    out["counts"] = {"legacy_bends": len(legacy), "mapped": len(mapping),
                     "unmapped_legacy_bends": len(legacy) - len(mapping),
                     "angle_measured": sum(1 for r in out["bends"] if r["angle"]["measured"]),
                     "line_measured": sum(1 for r in out["bends"] if r["line"]["measured"]),
                     "line_predictions_raw": len(raw_lines)}
    return out


def legacy_line_predictions(parsed: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Legacy scan-fit lines in the raw-scan frame (placeholders never count)."""
    return [dict(r) for r in parsed.get("line_predictions_raw") or []]


def legacy_render_scene(parsed: Mapping[str, Any], run: Mapping[str, Any], compiled: Any, compiled_dir: Path, *,
                        max_points: int = 200_000, title: str = "") -> Any:
    """A neutral :class:`..render.RenderScene` of one legacy run (module docstring)."""
    from ..plyio import read_ply_vertices
    from ..render import RenderScene, _bend_overlay, _nearest_labels

    final = max((s for s in compiled.stages if s.kind == "SEQUENCE_PREFIX"), key=lambda s: s.stage_index)
    rows = {r["bend_uid"]: r for r in parsed.get("bends") or []}
    overlays = []
    for b in compiled.bends:
        ov = _bend_overlay(compiled, final, None, b)
        r = rows.get(b.bend_uid)
        if r is not None and r["line"]["measured"] and r["line"]["ref"] is not None:
            line = np.asarray(r["line"]["ref"], dtype=np.float64)
            ov = replace(ov, state="FORMED", state_reasons=(), measured_oml=line, measured_apex=line,
                         apex_source="LEGACY_DETECTED_LINE", position_reason=None)
        else:
            reason = ("LEGACY_UNMAPPED" if r is None or r["legacy_bend_id"] is None else
                      "LEGACY_PLACEHOLDER_LINE" if r["line"]["placeholder"] else "LEGACY_NOT_DETECTED")
            ov = replace(ov, state="UNKNOWN", state_reasons=(reason,), position_reason=reason)
        overlays.append(ov)
    points = labels = deviations = None
    aligned = (run.get("outputs") or {}).get("aligned_scan.ply")
    spacing = 0.5
    if run.get("status") == "OK" and aligned and Path(aligned).is_file():
        pts = read_ply_vertices(aligned).vertices
        sel = np.unique(np.linspace(0, len(pts) - 1, min(len(pts), max_points)).astype(np.int64))
        points = pts[sel]
        dev_path = (run.get("outputs") or {}).get("deviations.npy")
        if dev_path and Path(dev_path).is_file():
            dev = np.load(dev_path, allow_pickle=False).astype(np.float32).ravel()
            deviations = dev[sel] if len(dev) == len(pts) else None
        with np.load(compiled_dir / final.sample_cloud.rel_path, allow_pickle=False) as z:
            labels = _nearest_labels(points, {k: z[k] for k in z.files})
        from scipy.spatial import cKDTree

        sub = points[np.unique(np.linspace(0, len(points) - 1, min(len(points), 20000)).astype(np.int64))]
        d, _ = cKDTree(points).query(sub, k=2)
        spacing = float(np.median(d[:, 1])) or 0.5
    from ..plyio import read_ply

    mesh = read_ply(compiled_dir / final.display_mesh.rel_path)
    cad_labels = [(b.sequence_index, b.label, b.bend_uid,
                   np.asarray([final.bend_lines[b.bend_uid].start_mm, final.bend_lines[b.bend_uid].end_mm]))
                  for b in compiled.bends]
    valid = run.get("status") == "OK" and bool((parsed.get("alignment") or {}).get("available")) and points is not None
    return RenderScene(
        title=title or f"legacy control {parsed['case_ref']}", points=points, labels=labels,
        label_source="NEAREST_STAGE_MODEL", deviations=deviations, heatmap_band_mm=0.5,
        stage_model_id=final.stage_model_id, stage_vertices=mesh.vertices, stage_faces=mesh.faces, bends=overlays,
        cad_labels=cad_labels, validity_state="VALID" if valid else "INVALID_INSPECTION",
        summary_state="LEGACY_CONTROL", spacing_mm=spacing,
        meta={"engine_version": "legacy-runtime-control", "stage_used": final.stage_model_id,
              "ambiguity_stage": final.stage_model_id, "arm": "LEGACY",
              "frame_check": (parsed.get("frame_check") or {}).get("status")},
        ambiguous_groups=tuple(tuple(g) for g in final.symmetry.ambiguous_bend_groups), pose_ambiguous=False)


def legacy_render_job(case: Mapping[str, Any], case_dir: Path, compiled_dir: Path) -> Any:
    """A G4 :class:`..g4.RenderJob` (source LEGACY) for one recorded legacy run.

    ``case`` is the frozen case record (for ``case_ref`` and the staged flag), ``case_dir`` its
    ``legacy_control/<case_ref>/`` directory, ``compiled_dir`` the frozen recipe's compile output."""
    from ..engine_output import load_compiled
    from ..g4 import RenderJob

    run = read_json(Path(case_dir) / "run.json")
    parsed = read_json(Path(case_dir) / "legacy_case.json")
    compiled = load_compiled(Path(compiled_dir) / "compiled_part.json")
    scene = legacy_render_scene(parsed, run, compiled, Path(compiled_dir))
    return RenderJob(engine_dir=None, compiled_json=None, recipe_json=None, case_ref=case["case_ref"],
                     source="LEGACY", population="PHYSICAL_DEV",
                     incomplete=case.get("stage_class") in ("STAGED_PARTIAL", "BLANK_STAGE0"), scene=scene)
