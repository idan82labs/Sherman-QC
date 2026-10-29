"""Jev A/B on the broad line benchmark (wave 4, research only).

Question: does a Jev "is this a real press-brake bend line?" filter over a lane's
predicted lines raise line-F1 on the broad benchmark (57 scored scans + zero-bend
controls of ``case_list_broad_v3.json``)?

Every predicted line of a lane is one candidate. A candidate's state is built
from the line and the scan only (no identities, no coordinates; the
``state_v1`` deny-list applies):

``candidate``     scan evidence measured directly on the scan around the line,
                  identically for every lane and every line: station support,
                  2-D cross-section "arms" (flanges) and the included angle
                  between them, arm spread, surface distance, point density.
``neighbours``    ``state_v1.neighbour_summary`` over the lane's lines of that scan.
``context``       scan-level facts (lane line count, scan extent, sampling).
``cad_relation``  everything whose value or presence depends on CAD: proposal
                  source (scan-only vs CAD-guided), CAD binding, nominal/measured
                  angle, legacy status/observability, corroboration by the other
                  proposal source. ``--cad-free`` drops this block.

Truth is never in a state. Each candidate is TP/FP by the frozen matcher
(``research/broad_baseline/score.score_case`` -> ``scripts/evaluate_f1_against_
manual_gold.match_lines``: 5 deg, 4 mm, 0.6/0.6 overlap, one-to-one), recomputed
at scoring time from the gold, so a gold override (e.g. the Gold11 relabel) only
needs a re-score, not new Jev calls. A filter only removes lines; the filtered
lane is re-scored with the same matcher, so recall can never rise.

Subcommands::

  export   --case-list CL --runs-root RUNS --lane best_f1:predictions_best_f1.json
           [--lane runtime:predictions_runtime.json] --out NEW_DIR
           [--features-cache DIR] [--calib-fraction 0.33]
  score    --export DIR --answers-normal DIR --answers-cad-free DIR --out results.json
           [--gold-override MAP.json] [--bootstrap 2000]

The live calls in between use ``python -m research.jev.cli live`` (dry-run first);
see docs/production/runlog/jev_ab/JEV_AB.md for the exact command sequence.
Never wired into a runtime decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from research.jev.client_v1 import load_states, request_identity
from research.jev.export_onboarding_v1 import REPO_ROOT, write_export
from research.jev.labels_v1 import make_label, read_jsonl, write_jsonl
from research.jev.splits_v1 import SPLIT_MANIFEST_SCHEMA, load_split_manifest
from research.jev.state_v1 import (
    CandidateRecord,
    build_state,
    enum_token,
    line_from_endpoints,
    line_pair_metrics,
    neighbour_summary,
)

BROAD_DIR = REPO_ROOT / "research" / "broad_baseline"
CANDIDATE_SOURCE = "SCAN_LEGACY_REPORT"
SPLIT_SALT = "jev_ab_v1_calib"
FEATURE_VERSION = "jev_ab_scan_line_features_v1"
MAX_POINTS = 1_500_000
STATIONS = 21
RING_MM = (6.0, 24.0)
HIST_BINS = 36
ARM_MIN_FRACTION = 0.10


def _score_module():
    if str(BROAD_DIR) not in sys.path:
        sys.path.insert(0, str(BROAD_DIR))
    import score as sc  # noqa: WPS433  (research/broad_baseline/score.py, the frozen matcher)

    return sc


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Scan loading and per-line scan evidence
# --------------------------------------------------------------------------


def load_points(path: Path) -> np.ndarray:
    """Scan points (raw frame). Meshes add an area-uniform surface sample; capped."""
    import trimesh

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    pts = np.asarray(loaded.vertices, dtype=np.float64)
    faces = getattr(loaded, "faces", None)
    rng = np.random.default_rng(0)
    if faces is not None and len(faces):
        count = int(min(MAX_POINTS, max(200_000, len(pts))))
        sample, _ = trimesh.sample.sample_surface(loaded, count, seed=0)
        pts = np.vstack([pts, np.asarray(sample, dtype=np.float64)])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > MAX_POINTS:
        pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
    return pts


class ScanIndex:
    def __init__(self, pts: np.ndarray):
        from scipy.spatial import cKDTree

        self.pts = pts
        self.tree = cKDTree(pts)
        rng = np.random.default_rng(1)
        probe = pts[rng.choice(len(pts), min(5000, len(pts)), replace=False)]
        d, _ = self.tree.query(probe, k=2)
        self.spacing = float(np.median(d[:, 1]))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.extent = float(np.linalg.norm(hi - lo))


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _arms(theta: np.ndarray, uv: np.ndarray) -> list[dict[str, float]]:
    """Angular clusters of ring points: contiguous occupied 10-deg bins, >= 10 % of the ring mass."""
    if len(theta) < 6:
        return []
    bins = ((theta + math.pi) / (2 * math.pi) * HIST_BINS).astype(int) % HIST_BINS
    hist = np.bincount(bins, minlength=HIST_BINS).astype(float)
    occupied = hist >= max(2.0, 0.02 * len(theta))
    if occupied.all():
        return [{"mass": 1.0, "dir": 0.0, "spread": float("nan"), "reach": float("nan"), "full": 1.0}]
    start = int(np.argmin(occupied))  # an empty bin: clusters never wrap past it
    clusters: list[list[int]] = []
    current: list[int] = []
    for k in range(HIST_BINS):
        b = (start + k) % HIST_BINS
        if occupied[b]:
            current.append(b)
        elif current:
            clusters.append(current)
            current = []
    if current:
        clusters.append(current)
    out = []
    for cl in clusters:
        mask = np.isin(bins, cl)
        mass = float(mask.sum()) / len(theta)
        if mass < ARM_MIN_FRACTION:
            continue
        sub = uv[mask]
        direction = math.atan2(float(np.sin(theta[mask]).mean()), float(np.cos(theta[mask]).mean()))
        cov = np.cov(sub.T) if len(sub) >= 3 else np.zeros((2, 2))
        ev = np.linalg.eigvalsh(cov)
        out.append({"mass": mass, "dir": direction, "spread": float(2.0 * math.sqrt(max(ev[0], 0.0))),
                    "reach": float(np.linalg.norm(sub, axis=1).max())})
    out.sort(key=lambda a: -a["mass"])
    return out


def line_features(index: ScanIndex, start: Sequence[float], end: Sequence[float]) -> dict[str, Any]:
    """Coordinate-free scan evidence around one line (same procedure for every lane)."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b - a))
    axis = (b - a) / length
    u, w = _basis(axis)
    spacing = index.spacing
    near_r = max(1.5, 2.5 * spacing)
    half_slab = max(2.0, 1.5 * spacing)
    ball = math.hypot(RING_MM[1], half_slab)
    supported, n_arms, included, balance, spread, reach, surface, slab_counts = [], [], [], [], [], [], [], []
    for t in np.linspace(0.05, 0.95, STATIONS):
        c = a + t * (b - a)
        idx = index.tree.query_ball_point(c, ball)
        if not idx:
            supported.append(False)
            n_arms.append(0)
            slab_counts.append(0)
            continue
        d = index.pts[idx] - c
        ax = d @ axis
        d = d[np.abs(ax) <= half_slab]
        uv = np.stack([d @ u, d @ w], axis=1) if len(d) else np.zeros((0, 2))
        r = np.linalg.norm(uv, axis=1) if len(uv) else np.zeros(0)
        slab_counts.append(int(len(uv)))
        supported.append(bool((r <= near_r).sum() >= 2))
        if len(r):
            surface.append(float(r.min()))
        ring = (r >= RING_MM[0]) & (r <= RING_MM[1])
        theta = np.arctan2(uv[ring, 1], uv[ring, 0]) if ring.any() else np.zeros(0)
        arms = _arms(theta, uv[ring]) if ring.any() else []
        if arms and arms[0].get("full"):
            n_arms.append(9)
            continue
        n_arms.append(len(arms))
        if len(arms) >= 2:
            diff = abs(math.degrees(arms[0]["dir"] - arms[1]["dir"])) % 360.0
            included.append(min(diff, 360.0 - diff))
            balance.append(arms[1]["mass"] / arms[0]["mass"])
            reach.append(min(arms[0]["reach"], arms[1]["reach"]))
        for arm in arms[:2]:
            spread.append(arm["spread"])
    sup = np.asarray(supported, bool)
    run = best = 0
    for s in sup:
        run = run + 1 if s else 0
        best = max(best, run)
    arms_arr = np.asarray(n_arms)

    def med(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    inc_med = med(included)
    return {
        "line_length_mm": length,
        "slice_support_fraction": float(sup.mean()),
        "longest_supported_run_fraction": best / STATIONS,
        "empty_ring_slice_fraction": float((arms_arr == 0).mean()),
        "one_arm_slice_fraction": float((arms_arr == 1).mean()),
        "two_arm_slice_fraction": float((arms_arr == 2).mean()),
        "multi_arm_slice_fraction": float(((arms_arr >= 3) & (arms_arr < 9)).mean()),
        "enclosed_slice_fraction": float((arms_arr == 9).mean()),
        "median_included_angle_deg": inc_med,
        "included_angle_iqr_deg": (float(np.subtract(*np.percentile(included, [75, 25]))) if len(included) >= 2 else None),
        "median_bend_sweep_deg": None if inc_med is None else 180.0 - inc_med,
        "median_arm_balance_ratio": med(balance),
        "median_arm_spread_mm": med(spread),
        "median_shorter_arm_reach_mm": med(reach),
        "median_surface_distance_mm": med(surface),
        "slab_points_per_mm": float(np.median(slab_counts)) / (2.0 * half_slab),
    }


# --------------------------------------------------------------------------
# CAD-dependent relation (everything here is dropped by --cad-free)
# --------------------------------------------------------------------------


def _rows(pred: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [r for r in pred.get("owned_bend_regions") or []
            if line_from_endpoints(r.get("line_start"), r.get("line_end")) is not None]


def _corroboration(line, others: Sequence[Any]) -> dict[str, Any]:
    best = None
    for other in others:
        m = line_pair_metrics(line, other)
        key = m["transverse_offset_mm"] + 0.5 * m["axis_angle_deg"]
        if best is None or key < best[0]:
            best = (key, m)
    if best is None:
        return {"cross_source_lines_count": 0, "cross_source_nearest_offset_mm": None,
                "cross_source_nearest_axis_angle_deg": None, "is_cross_source_corroborated": False}
    m = best[1]
    return {"cross_source_lines_count": len(others),
            "cross_source_nearest_offset_mm": m["transverse_offset_mm"],
            "cross_source_nearest_axis_angle_deg": m["axis_angle_deg"],
            "is_cross_source_corroborated": bool(m["axis_angle_deg"] <= 5.0 and m["transverse_offset_mm"] <= 4.0
                                                 and m["reference_overlap_fraction"] >= 0.6
                                                 and m["other_overlap_fraction"] >= 0.6)}


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return None
    return float(v)


def cad_relation(row: Mapping[str, Any], *, has_cad: bool, cad_matches: Mapping[str, Any],
                 cad_planned: int | None, cross_lines: Sequence[Any], line) -> dict[str, Any]:
    proposal = row.get("fusion_source") or row.get("origin") or "UNKNOWN"
    scan_only = proposal == "scan_only_a0"
    target = row.get("cad_target_id") or (row.get("bend_id") if not scan_only and str(row.get("bend_id", "")).startswith("CAD_") else None)
    m = cad_matches.get(str(target)) if target else None
    m = m or {}
    measured = _num(row.get("measured_angle")) if row.get("measured_angle") is not None else _num(m.get("measured_angle"))
    nominal = _num(row.get("target_angle")) if row.get("target_angle") is not None else _num(m.get("target_angle"))
    rel: dict[str, Any] = {
        "has_cad_model": has_cad,
        "proposal_source": "SCAN_ONLY" if scan_only else "CAD_GUIDED",
        "is_bound_to_cad_bend": bool(target),
        "measurement_status": enum_token(row.get("status") or m.get("status")) or "NONE",
        "observability_state": enum_token(row.get("observability_state") or m.get("observability_state")) or "NONE",
        "assignment_source": enum_token(row.get("assignment_source")) or "NONE",
        "detected_geometry_source": enum_token(row.get("detected_line_geometry_source")) or "NONE",
        "bend_form": enum_token(m.get("bend_form")) or "NONE",
        "nominal_angle_deg": nominal,
        "measured_angle_deg": measured,
        "angle_deviation_deg": (None if nominal is None or measured is None else measured - nominal),
        "nominal_radius_mm": _num(m.get("target_radius")),
        "measured_radius_mm": _num(m.get("measured_radius")),
        "line_length_deviation_mm": _num(m.get("line_length_deviation_mm")),
        "detector_confidence_score": _num(m.get("confidence")),
        "cad_planned_bends_count": cad_planned,
    }
    rel.update(_corroboration(line, cross_lines))
    return rel


def _cad_case_for(lane_dir: Path) -> tuple[dict[str, Any], int | None]:
    gated = lane_dir / "predictions_cad_gated.json"
    if not gated.is_file():
        return {}, None
    src = (_read_json(gated).get("source") or {}).get("path")
    if not src or not Path(src).is_file():
        return {}, None
    case = _read_json(src)
    matches = {str(m.get("bend_id")): m for m in case.get("matches") or []}
    return matches, (case.get("summary") or {}).get("total_bends")


def _runtime_planned(pred: Mapping[str, Any]) -> int | None:
    src = (pred.get("source") or {}).get("path")
    if not src or not Path(src).is_file() or not str(src).endswith("bend_inspection_report.json"):
        return None
    report = _read_json(src)
    return (report.get("summary") or {}).get("total_bends")


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def family_splits(families: Sequence[str], calib_fraction: float) -> dict[str, str]:
    order = sorted(families, key=lambda f: hashlib.sha256(f"{SPLIT_SALT}|{f}".encode()).hexdigest())
    n_calib = max(1, round(calib_fraction * len(order)))
    return {f: ("calib" if i < n_calib else "eval") for i, f in enumerate(order)}


def cmd_export(args: argparse.Namespace) -> int:
    sc = _score_module()
    case_list = _read_json(args.case_list)
    runs_root = Path(args.runs_root)
    lanes = [tuple(spec.split(":", 1)) for spec in args.lane]
    cache = Path(args.features_cache) if args.features_cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    frame_records = _read_json(args.frame_records) if args.frame_records else None
    states, labels, sidecar = [], [], []
    families = sorted({c["family_id"] for c in case_list["cases"]})
    split_of = family_splits(families, args.calib_fraction)
    for case in case_list["cases"]:
        cid = case["case_id"]
        per_lane = []
        for lane, pred_file in lanes:
            lane_dir = runs_root / lane / cid
            path = lane_dir / pred_file
            pred = _read_json(path) if path.is_file() else {"owned_bend_regions": []}
            rows = _rows(pred)
            if rows:
                per_lane.append((lane, pred_file, lane_dir, path, pred, rows))
        if not per_lane:
            continue
        cache_file = cache / f"{cid}.json" if cache else None
        cached = _read_json(cache_file) if cache_file and cache_file.is_file() else {}
        index = None
        for lane, pred_file, lane_dir, path, pred, rows in per_lane:
            scored = sc.score_case({"case_id": cid, "tier": case["tier"], "family_id": case["family_id"],
                                    "gold": case["gold"], "label_complete": case.get("label_complete", True),
                                    "predictions": str(path)}, frame_records=frame_records)
            tp_ids = {m["prediction_id"] for m in scored.get("matches") or []}
            complete = scored.get("disposition") in {"SCORED_COMPLETE_LABEL", "ZERO_BEND_CONTROL"}
            cad_matches, cad_planned = _cad_case_for(lane_dir) if lane == "best_f1" else ({}, None)
            if lane != "best_f1":
                cad_planned = _runtime_planned(pred)
            a0_path = runs_root / "best_f1" / cid / "predictions_a0.json"
            a0_lines = [line_from_endpoints(r["line_start"], r["line_end"])
                        for r in (_rows(_read_json(a0_path)) if a0_path.is_file() else [])]
            cad_raw_path = runs_root / "best_f1" / cid / "predictions_cad_raw_ungated.json"
            cad_lines = [line_from_endpoints(r["line_start"], r["line_end"])
                         for r in (_rows(_read_json(cad_raw_path)) if cad_raw_path.is_file() else [])]
            lines = [line_from_endpoints(r["line_start"], r["line_end"]) for r in rows]
            for i, row in enumerate(rows):
                fkey = f"{row['line_start']}|{row['line_end']}"
                feats = cached.get(fkey)
                if feats is None:
                    if index is None:
                        index = ScanIndex(load_points(Path(case["scan_path"])))
                        cached["__scan__"] = {"spacing_mm": index.spacing, "extent_mm": index.extent,
                                              "points": int(len(index.pts))}
                    feats = line_features(index, row["line_start"], row["line_end"])
                    cached[fkey] = feats
                scan_meta = cached["__scan__"]
                proposal = row.get("fusion_source") or row.get("origin")
                cross = cad_lines if proposal == "scan_only_a0" else a0_lines
                relation = cad_relation(row, has_cad=bool(case.get("cad_path")), cad_matches=cad_matches,
                                        cad_planned=cad_planned, cross_lines=[c for c in cross if c is not None],
                                        line=lines[i])
                key = f"jev_ab|{lane}|{cid}|{row['bend_id']}"
                source_sha = (pred.get("scan") or {}).get("sha256") or case["scan_sha256"]
                record = CandidateRecord(
                    source_sha, key, CANDIDATE_SOURCE, feats,
                    neighbours=neighbour_summary(i, lines),
                    context={"lane_lines_count": len(rows), "cloud_extent_mm": scan_meta["extent_mm"],
                             "cloud_spacing_mm": scan_meta["spacing_mm"],
                             "cloud_samples_log1p": math.log1p(scan_meta["points"])},
                    cad_relation=relation)
                rid, state = build_state(record)
                is_tp = row["bend_id"] in tp_ids
                label_source = ("gold_line_matched" if is_tp else
                                "gold_line_unmatched" if complete else "gold_line_unmatched_incomplete")
                states.append((rid, state))
                labels.append(make_label(rid, label_source=label_source, group_sku=case["family_id"],
                                         candidate_key=key, source_sha256=source_sha,
                                         provenance_flags=("consumed",)))
                sidecar.append({"record_id": rid, "lane": lane, "pred_file": pred_file, "case_id": cid,
                                "family_id": case["family_id"], "tier": case["tier"],
                                "in_gold11": bool(case.get("in_gold11")), "bend_id": row["bend_id"],
                                "proposal_source": relation["proposal_source"],
                                "split": split_of[case["family_id"]], "export_truth": int(is_tp)})
        if cache_file is not None:
            cache_file.write_text(json.dumps(cached, sort_keys=True), encoding="utf-8")
        print(f"{cid}: {sum(len(x[5]) for x in per_lane)} lines", flush=True)
    manifest = {"exporter": "broad_ab_v1", "feature_version": FEATURE_VERSION,
                "inputs": [{"name": Path(args.case_list).name, "sha256": _sha_file(Path(args.case_list))}],
                "lanes": [f"{lane}:{pf}" for lane, pf in lanes],
                "split_rule": f"sha256('{SPLIT_SALT}|'+family) order, first {args.calib_fraction:.2f} -> calib",
                "counts": {"candidates": len(states)}, "warnings": []}
    write_export(Path(args.out), states=states, labels=labels, manifest=manifest)
    out = Path(args.out)
    write_jsonl(out / "candidates.jsonl", sidecar)
    splits = {"schema": SPLIT_MANIFEST_SCHEMA,
              "families": {f: {"split": s, "flags": ["consumed"]} for f, s in sorted(split_of.items())},
              "sku_to_family": {f: f for f in families}}
    (out / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "candidates": len(states),
                      "calib_families": sum(s == "calib" for s in split_of.values()),
                      "eval_families": sum(s == "eval" for s in split_of.values())}))
    return 0


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _load_answers(answers_dir: Path, states: Mapping[str, dict], *, cad_free: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in read_jsonl(answers_dir / "answers.jsonl"):
        rid = row["record_id"]
        if rid not in states:
            raise SystemExit(f"answer for unknown record {rid}")
        ident = request_identity(states[rid], model=row["request_model"], variant=row["question_variant"],
                                 question_names=row["question_ids"], cad_free=cad_free)
        if ident["body_sha256"] != row["body_sha256"] or bool(row["ablation"]["cad_free"]) != cad_free:
            raise SystemExit(f"stale or mismatched answer for {rid} in {answers_dir}")
        out[rid] = float(row["answers"]["is_real_bend"]["noul"])
    return out


def _case_rows(case_list: Mapping[str, Any], runs_root: Path, lane: str, pred_file: str,
               frame_records, overrides: Mapping[str, Any], keep: Mapping[str, set] | None) -> list[dict]:
    """Score every case of the lane (optionally with only ``keep[case]`` bend ids kept)."""
    sc = _score_module()
    rows = []
    for case in case_list["cases"]:
        cid = case["case_id"]
        path = runs_root / lane / cid / pred_file
        pred = _read_json(path) if path.is_file() else {"owned_bend_regions": [], "status": "PREDICTIONS_MISSING",
                                                         "scan": {}}
        if keep is not None:
            pred = dict(pred)
            pred["owned_bend_regions"] = [r for r in pred.get("owned_bend_regions") or []
                                          if r.get("bend_id") in keep.get(cid, set())]
        gold = overrides.get(cid, case["gold"])
        row = sc.score_case({"case_id": cid, "tier": case["tier"], "family_id": case["family_id"], "gold": gold,
                             "label_complete": case.get("label_complete", True), "predictions": str(path)},
                            frame_records=frame_records, preloaded_predictions=pred)
        row["in_gold11"] = bool(case.get("in_gold11"))
        rows.append(row)
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if r.get("disposition") == "SCORED_COMPLETE_LABEL"]
    tp = sum(r["true_positive"] for r in complete)
    fp = sum(r["false_positive"] for r in complete)
    fn = sum(r["false_negative"] for r in complete)
    p, r, f = _prf(tp, fp, fn)
    fam: dict[str, list] = defaultdict(list)
    for x in complete:
        fam[str(x["family_id"])].append(x)
    fam_f1 = [_prf(sum(x["true_positive"] for x in v), sum(x["false_positive"] for x in v),
                   sum(x["false_negative"] for x in v))[2] for v in fam.values()]
    controls = [x for x in rows if x.get("disposition") == "ZERO_BEND_CONTROL"]
    return {"scans": len(complete), "families": len(fam), "gold_lines": tp + fn, "predicted_lines": tp + fp,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "micro_precision": round(p, 4), "micro_recall": round(r, 4), "micro_f1": round(f, 4),
            "family_macro_f1": round(float(np.mean(fam_f1)), 4) if fam_f1 else 0.0,
            "control_false_positive_lines": sum(x.get("false_positive_lines", 0) for x in controls)}


def _family_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int, int]]:
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r.get("disposition") == "SCORED_COMPLETE_LABEL":
            s = out[str(r["family_id"])]
            s[0] += r["true_positive"]
            s[1] += r["false_positive"]
            s[2] += r["false_negative"]
    return {k: tuple(v) for k, v in out.items()}


def _bootstrap(base: Mapping[str, tuple], other: Mapping[str, tuple] | None, rounds: int, seed: int) -> dict:
    fams = sorted(base)
    if not fams:
        return {}
    rng = np.random.default_rng(seed)
    b = np.array([base[f] for f in fams], float)
    o = np.array([other[f] for f in fams], float) if other is not None else None

    def metrics(arr, idx):
        s = arr[idx].sum(axis=0)
        micro = _prf(*s)[2]
        fam = np.mean([_prf(*arr[i])[2] for i in idx])
        return micro, fam, _prf(*s)[0], _prf(*s)[1]

    draws = []
    for _ in range(rounds):
        idx = rng.integers(0, len(fams), len(fams))
        mb = metrics(b, idx)
        if o is not None:
            mo = metrics(o, idx)
            draws.append(mo + tuple(x - y for x, y in zip(mo, mb)))
        else:
            draws.append(mb)
    d = np.array(draws)
    names = ["micro_f1", "family_macro_f1", "micro_precision", "micro_recall"]
    if o is not None:
        names += ["delta_micro_f1", "delta_family_macro_f1", "delta_micro_precision", "delta_micro_recall"]
    return {n: [round(float(np.percentile(d[:, i], 2.5)), 4), round(float(np.percentile(d[:, i], 97.5)), 4)]
            for i, n in enumerate(names)}


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def _classification(cands: Sequence[Mapping[str, Any]], probs: Mapping[str, float], threshold: float,
                    rounds: int, seed: int) -> dict[str, Any]:
    rows = [c for c in cands if c["record_id"] in probs]
    if not rows:
        return {"candidates": 0}
    y = np.array([c["truth"] for c in rows])
    s = np.array([probs[c["record_id"]] for c in rows])
    keep = s >= threshold
    tp = int((keep & (y == 1)).sum())
    out = {"candidates": len(rows), "positives": int(y.sum()), "negatives": int((y == 0).sum()),
           "auc": None if _auc(y, s) is None else round(_auc(y, s), 4),
           "keep_threshold": threshold,
           "kept": int(keep.sum()),
           "keep_precision": round(tp / keep.sum(), 4) if keep.sum() else None,
           "keep_recall_of_true_lines": round(tp / y.sum(), 4) if y.sum() else None,
           "removed_false_lines": int((~keep & (y == 0)).sum()),
           "removed_true_lines": int((~keep & (y == 1)).sum()),
           "fp_rejection_rate": round(float((~keep & (y == 0)).sum() / (y == 0).sum()), 4) if (y == 0).sum() else None}
    fams = sorted({c["family_id"] for c in rows})
    by_fam = defaultdict(list)
    for i, c in enumerate(rows):
        by_fam[c["family_id"]].append(i)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(rounds):
        idx = np.concatenate([by_fam[fams[j]] for j in rng.integers(0, len(fams), len(fams))])
        a = _auc(y[idx], s[idx])
        if a is not None:
            aucs.append(a)
    if aucs:
        out["auc_ci95"] = [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]
    return out


def _choose_threshold(case_list, runs_root, lane, pred_file, frame_records, overrides, cands, probs,
                      calib_cases: set[str]) -> tuple[float, list[dict]]:
    """Threshold maximising calib micro line-F1 of lane+filter (ties -> lowest threshold)."""
    sub = {"cases": [c for c in case_list["cases"] if c["case_id"] in calib_cases]}
    grid = sorted({0.0} | {round(probs[c["record_id"]], 6) for c in cands
                          if c["case_id"] in calib_cases and c["record_id"] in probs})
    trace = []
    best = None
    for t in grid:
        keep = defaultdict(set)
        for c in cands:
            if c["record_id"] in probs and probs[c["record_id"]] >= t:
                keep[c["case_id"]].add(c["bend_id"])
        agg = _aggregate(_case_rows(sub, runs_root, lane, pred_file, frame_records, overrides, keep))
        trace.append({"threshold": t, **{k: agg[k] for k in ("micro_f1", "family_macro_f1", "true_positive",
                                                              "false_positive", "false_negative")}})
        if best is None or agg["micro_f1"] > best[0] + 1e-12:
            best = (agg["micro_f1"], t)
    return best[1], trace


def cmd_score(args: argparse.Namespace) -> int:
    export = Path(args.export)
    case_list = _read_json(args.case_list)
    runs_root = Path(args.runs_root)
    frame_records = _read_json(args.frame_records) if args.frame_records else None
    overrides = _read_json(args.gold_override) if args.gold_override else {}
    states = dict(load_states(export / "states.jsonl"))
    cands = list(read_jsonl(export / "candidates.jsonl"))
    split_manifest = load_split_manifest(export / "splits.json")
    split_of = {c["family_id"]: split_manifest.split_of(c["family_id"]) for c in case_list["cases"]}
    variants = {"jev": _load_answers(Path(args.answers_normal), states, cad_free=False),
                "jev_cad_free": _load_answers(Path(args.answers_cad_free), states, cad_free=True)}
    lanes = sorted({(c["lane"], c["pred_file"]) for c in cands})
    cohorts = {
        "ALL": {c["case_id"] for c in case_list["cases"]},
        "EVAL_FAMILIES": {c["case_id"] for c in case_list["cases"] if split_of[c["family_id"]] == "eval"},
        "CALIB_FAMILIES_IN_SAMPLE": {c["case_id"] for c in case_list["cases"] if split_of[c["family_id"]] == "calib"},
        "GOLD11": {c["case_id"] for c in case_list["cases"] if c.get("in_gold11")},
        "NON_GOLD11": {c["case_id"] for c in case_list["cases"] if not c.get("in_gold11")},
    }
    results: dict[str, Any] = {
        "schema": "jev_ab_broad_results_v1",
        "flags": ["exploratory_not_unseen (CONSUMED development scans)", "filter_can_only_remove_lines"],
        "inputs": {"export_manifest_sha256": _sha_file(export / "manifest.json"),
                   "states_sha256": _sha_file(export / "states.jsonl"),
                   "case_list_sha256": _sha_file(Path(args.case_list)),
                   "splits_sha256": split_manifest.sha256,
                   "gold_override": args.gold_override, "gold_override_cases": sorted(overrides)},
        "split": {s: sorted(f for f, v in split_of.items() if v == s) for s in ("calib", "eval")},
        "answers": {k: len(v) for k, v in variants.items()},
        "lanes": {},
    }
    frozen = _read_json(args.operating_points) if args.operating_points else None
    if frozen is not None:
        results["inputs"]["operating_points_sha256"] = _sha_file(Path(args.operating_points))
    frozen_points: dict[str, dict[str, float]] = {}
    for lane, pred_file in lanes:
        base_rows = _case_rows(case_list, runs_root, lane, pred_file, frame_records, overrides, None)
        tp_by_case = {r["case_id"]: {m["prediction_id"] for m in r.get("matches") or []} for r in base_rows}
        lane_cands = []
        for c in cands:
            if c["lane"] != lane:
                continue
            lane_cands.append({**c, "truth": int(c["bend_id"] in tp_by_case.get(c["case_id"], set()))})
        calib_cases = cohorts["CALIB_FAMILIES_IN_SAMPLE"]
        lane_out: dict[str, Any] = {"pred_file": pred_file, "candidates": len(lane_cands), "arms": {}}
        arms = {"lane_alone": (None, None)}
        for vname, probs in variants.items():
            calib_probs = {c["record_id"]: probs[c["record_id"]] for c in lane_cands
                           if c["case_id"] in calib_cases and c["record_id"] in probs}
            t, trace = _choose_threshold(case_list, runs_root, lane, pred_file, frame_records, overrides,
                                         lane_cands, calib_probs, calib_cases)
            lane_out.setdefault("threshold_trace", {})[vname] = trace
            frozen_points.setdefault(lane, {})[vname] = t
            if frozen is not None:
                ft = float(frozen["thresholds"][lane][vname])
                lane_out.setdefault("refit_on_calib_equals_frozen", {})[vname] = abs(ft - t) < 1e-9
                t = ft
            arms[f"lane_plus_{vname}_calibrated"] = (probs, t)
            arms[f"lane_plus_{vname}_t0.5_unfitted"] = (probs, 0.5)
        if args.freeze_out:
            lane_out["arms"] = {}
            results["lanes"][lane] = lane_out
            continue
        base_fam = {}
        for arm, (probs, t) in arms.items():
            if probs is None:
                rows = base_rows
            else:
                keep = defaultdict(set)
                for c in lane_cands:
                    if c["record_id"] in probs and probs[c["record_id"]] >= t:
                        keep[c["case_id"]].add(c["bend_id"])
                rows = _case_rows(case_list, runs_root, lane, pred_file, frame_records, overrides, keep)
            arm_out: dict[str, Any] = {"keep_threshold": t}
            for cname, cases in cohorts.items():
                sub = [r for r in rows if r["case_id"] in cases]
                fam = _family_stats(sub)
                if probs is None:
                    base_fam[cname] = fam
                arm_out[cname] = {**_aggregate(sub),
                                  "ci95": _bootstrap(base_fam[cname] if probs is not None else fam,
                                                     fam if probs is not None else None,
                                                     args.bootstrap, seed=7)}
                if probs is not None:
                    arm_out[cname]["jev_classification"] = _classification(
                        [c for c in lane_cands if c["case_id"] in cases], probs, t, args.bootstrap, seed=11)
            lane_out["arms"][arm] = arm_out
        results["lanes"][lane] = lane_out
    if args.freeze_out:
        doc = {"schema": "jev_ab_operating_points_v1",
               "rule": "per lane and variant: threshold maximising calib-family micro line-F1 of lane+filter "
                       "(grid = calib candidate scores + 0; ties -> lowest); chosen from calib answers only",
               "thresholds": frozen_points, "inputs": results["inputs"], "split": results["split"],
               "calib_answer_sha256": {k: hashlib.sha256(json.dumps(sorted(v.items())).encode()).hexdigest()
                                       for k, v in variants.items()},
               "threshold_trace": {lane: lo.get("threshold_trace") for lane, lo in results["lanes"].items()}}
        Path(args.freeze_out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(frozen_points))
        return 0
    Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for lane, lo in results["lanes"].items():
        for arm, ao in lo["arms"].items():
            for cname in ("ALL", "EVAL_FAMILIES"):
                a = ao[cname]
                print(f"{lane:8s} {arm:38s} {cname:14s} t={ao['keep_threshold']!s:8s} "
                      f"P={a['micro_precision']:.3f} R={a['micro_recall']:.3f} F1={a['micro_f1']:.3f} "
                      f"famF1={a['family_macro_f1']:.3f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m research.jev.broad_ab_v1", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    default_runlog = REPO_ROOT / "docs" / "production" / "runlog" / "broad_baseline"
    for name in ("export", "score"):
        q = sub.add_parser(name)
        q.add_argument("--case-list", default=str(default_runlog / "case_list_broad_v3.json"))
        q.add_argument("--runs-root", default=str(default_runlog / "runs"))
        q.add_argument("--frame-records", default=str(default_runlog / "broad_frame_records.json"))
        if name == "export":
            q.add_argument("--lane", action="append", required=True, help="LANE:PRED_FILE (repeatable)")
            q.add_argument("--out", required=True)
            q.add_argument("--features-cache")
            q.add_argument("--calib-fraction", type=float, default=0.33)
        else:
            q.add_argument("--export", required=True)
            q.add_argument("--answers-normal", required=True)
            q.add_argument("--answers-cad-free", required=True)
            q.add_argument("--out")
            q.add_argument("--freeze-out", help="write calib-only operating points here and stop")
            q.add_argument("--operating-points", help="frozen operating points (from --freeze-out) to apply")
            q.add_argument("--gold-override", help="JSON {case_id: gold spec} (e.g. Gold11 relabel)")
            q.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args(argv)
    return cmd_export(args) if args.command == "export" else cmd_score(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
