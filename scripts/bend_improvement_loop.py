#!/usr/bin/env python3
"""
Iterative bend-inspection improvement loop.

What it does:
1. Discovers part datasets in a source folder (default: ~/Downloads)
2. Separates partial and full scan cohorts using filename heuristics
3. Evaluates multiple bend-detector/matcher configurations
4. Scores each config for progress integrity and full-scan completion quality
5. Persists run history, per-agent scratchpads, and best config for reuse
6. Optionally generates visual diagnostics (projection images)

Usage:
  python3 scripts/bend_improvement_loop.py
  python3 scripts/bend_improvement_loop.py --max-parts 12 --generate-visuals
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from bend_inspection_pipeline import (  # noqa: E402
    BendInspectionRuntimeConfig,
    load_bend_runtime_config,
    load_cad_geometry,
    load_scan_points,
    extract_cad_bend_specs,
    select_cad_bends_for_expected,
    detect_and_match_with_fallback,
)


CAD_EXTS = {".stp", ".step", ".iges", ".igs", ".stl", ".ply", ".obj"}
SCAN_EXTS = {".stl", ".ply", ".obj", ".pcd"}
DRAWING_EXTS = {".pdf"}
SPEC_EXTS = {".xlsx", ".xls"}

CAD_PREFERRED_EXT_ORDER = [".step", ".stp", ".iges", ".igs", ".stl", ".obj", ".ply"]

SCAN_PARTIAL_MARKERS = (
    "scan1",
    "scan2",
    "scan3",
    "second_scan",
    "third_scan",
    "2nd_scan",
    "3rd_scan",
    "before",
    "prel",
    "partial",
    "incomplete",
    "half",
    "mid",
    "wip",
)
SCAN_FULL_MARKERS = ("final", "complete", "finished")
CAD_HINT_MARKERS = ("cad", "orig", "original", "reference")


@dataclass
class ScanFile:
    path: Path
    state: str
    sequence: int
    sequence_confident: bool
    mtime: float


@dataclass
class PartDataset:
    part_key: str
    cad_files: List[Path] = field(default_factory=list)
    scan_files: List[ScanFile] = field(default_factory=list)
    drawing_files: List[Path] = field(default_factory=list)
    spec_files: List[Path] = field(default_factory=list)


@dataclass
class PartPrepared:
    part_key: str
    cad_path: Path
    cad_source: str
    cad_vertices: np.ndarray
    cad_triangles: Optional[np.ndarray]
    cad_bends: Any
    scans: List[ScanFile]
    scan_points_cache: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ScanOutcome:
    config_name: str
    part_key: str
    scan_file: str
    state: str
    sequence: int
    total_bends: int
    cad_total_bends: int
    detected: int
    passed: int
    failed: int
    warnings: int
    progress_ratio: float
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "part_key": self.part_key,
            "scan_file": self.scan_file,
            "state": self.state,
            "sequence": self.sequence,
            "total_bends": self.total_bends,
            "cad_total_bends": self.cad_total_bends,
            "detected": self.detected,
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "progress_ratio": round(self.progress_ratio, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class ConfigScore:
    config_name: str
    score: float
    overdetect_penalty: float
    full_completion_penalty: float
    partial_monotonic_penalty: float
    partial_overcomplete_penalty: float
    completion_recall: float
    in_spec_precision: float
    mean_scan_utility: float
    utility_lcb95: float
    utility_ucb95: float
    bootstrap_samples: int
    promotion_eligible: bool
    gate_reasons: List[str]
    scan_count: int
    outcomes: List[ScanOutcome]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "score": round(self.score, 4),
            "overdetect_penalty": round(self.overdetect_penalty, 6),
            "full_completion_penalty": round(self.full_completion_penalty, 6),
            "partial_monotonic_penalty": round(self.partial_monotonic_penalty, 6),
            "partial_overcomplete_penalty": round(self.partial_overcomplete_penalty, 6),
            "completion_recall": round(self.completion_recall, 6),
            "in_spec_precision": round(self.in_spec_precision, 6),
            "mean_scan_utility": round(self.mean_scan_utility, 6),
            "utility_lcb95": round(self.utility_lcb95, 6),
            "utility_ucb95": round(self.utility_ucb95, 6),
            "bootstrap_samples": int(self.bootstrap_samples),
            "promotion_eligible": bool(self.promotion_eligible),
            "gate_reasons": list(self.gate_reasons),
            "scan_count": self.scan_count,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


@dataclass
class TodoItem:
    key: str
    title: str
    status: str = "pending"  # pending | in_progress | done | failed
    note: str = ""


def _extract_part_key(name: str) -> Optional[str]:
    matches = re.findall(r"\d{6,}", name)
    if not matches:
        return None
    return max(matches, key=len)


def _classify_scan_file(path: Path) -> Tuple[str, int, bool]:
    name = path.stem.lower().replace(" ", "_")

    # Strong sequence hints: explicit staging words.
    if "third" in name or "3rd" in name:
        return "partial", 3, True
    if "second" in name or "2nd" in name:
        return "partial", 2, True
    if "before" in name:
        return "partial", 1, True
    if "prel" in name:
        return "partial", 1, True

    for marker in SCAN_PARTIAL_MARKERS:
        if marker in name:
            # plain "scan1/scan2" often means alternate captures, not process order.
            seq_match = re.search(r"scan[_ ]?(\\d+)", name)
            if seq_match:
                return "partial", int(seq_match.group(1)), False
            return "partial", 10, False

    for marker in SCAN_FULL_MARKERS:
        if marker in name:
            return "full", 99, True

    if path.suffix.lower() in {".ply", ".pcd"}:
        return "full", 90, False

    if "scan" in name:
        return "partial", 10, False

    # Unlabeled mesh scans default to full-state evidence.
    return "full", 95, False


def _pick_primary_cad(cad_files: List[Path]) -> Optional[Path]:
    if not cad_files:
        return None
    ext_rank = {ext: idx for idx, ext in enumerate(CAD_PREFERRED_EXT_ORDER)}
    ranked = sorted(
        cad_files,
        key=lambda p: (
            ext_rank.get(p.suffix.lower(), len(ext_rank)),
            -p.stat().st_mtime,
        ),
    )
    return ranked[0]


def discover_parts(downloads_dir: Path) -> List[PartDataset]:
    grouped: Dict[str, Dict[str, List[Path]]] = {}
    mesh_ambiguous: Dict[str, List[Path]] = {}

    for p in downloads_dir.iterdir():
        if not p.is_file():
            continue
        key = _extract_part_key(p.name)
        if not key:
            continue
        ext = p.suffix.lower()
        grouped.setdefault(key, {"cad": [], "scan": [], "drawing": [], "spec": []})
        mesh_ambiguous.setdefault(key, [])

        if ext in DRAWING_EXTS:
            grouped[key]["drawing"].append(p)
            continue
        if ext in SPEC_EXTS:
            grouped[key]["spec"].append(p)
            continue
        if ext not in CAD_EXTS and ext not in SCAN_EXTS:
            continue

        if ext in {".step", ".stp", ".iges", ".igs"}:
            grouped[key]["cad"].append(p)
            continue

        lower = p.stem.lower()
        if any(marker in lower for marker in CAD_HINT_MARKERS):
            grouped[key]["cad"].append(p)
            continue
        if any(marker in lower for marker in SCAN_PARTIAL_MARKERS) or "scan" in lower:
            grouped[key]["scan"].append(p)
            continue

        mesh_ambiguous[key].append(p)

    datasets: List[PartDataset] = []
    for key, sections in grouped.items():
        cad_files = list(sections["cad"])
        scan_candidates = list(sections["scan"])
        ambiguous = mesh_ambiguous.get(key, [])

        if cad_files:
            scan_candidates.extend(ambiguous)
        else:
            stl_like = [p for p in ambiguous if p.suffix.lower() in {".stl", ".obj"}]
            ply_like = [p for p in ambiguous if p.suffix.lower() in {".ply", ".pcd"}]
            if stl_like and ply_like:
                cad_choice = sorted(stl_like, key=lambda p: -p.stat().st_mtime)[0]
                cad_files.append(cad_choice)
                for item in ambiguous:
                    if item != cad_choice:
                        scan_candidates.append(item)
            elif len(stl_like) == 1 and not ply_like:
                cad_files.append(stl_like[0])
            elif len(stl_like) > 1:
                cad_choice = sorted(stl_like, key=lambda p: p.stat().st_mtime)[0]
                cad_files.append(cad_choice)
                for item in ambiguous:
                    if item != cad_choice:
                        scan_candidates.append(item)
            elif not stl_like and len(ply_like) >= 2:
                cad_choice = sorted(ply_like, key=lambda p: p.stat().st_mtime)[0]
                cad_files.append(cad_choice)
                for item in ambiguous:
                    if item != cad_choice:
                        scan_candidates.append(item)

        if not cad_files or not scan_candidates:
            continue

        scans: List[ScanFile] = []
        for s in scan_candidates:
            state, seq, seq_confident = _classify_scan_file(s)
            scans.append(
                ScanFile(
                    path=s,
                    state=state,
                    sequence=seq,
                    sequence_confident=seq_confident,
                    mtime=s.stat().st_mtime,
                )
            )

        scans.sort(key=lambda sf: (sf.sequence, sf.mtime, sf.path.name))
        datasets.append(
            PartDataset(
                part_key=key,
                cad_files=sorted(cad_files, key=lambda p: -p.stat().st_mtime),
                scan_files=scans,
                drawing_files=sorted(sections["drawing"], key=lambda p: -p.stat().st_mtime),
                spec_files=sorted(sections["spec"], key=lambda p: -p.stat().st_mtime),
            )
        )

    datasets.sort(key=lambda d: d.part_key)
    return datasets


def downsample_points(points: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def prepare_part(
    dataset: PartDataset,
    base_cfg: BendInspectionRuntimeConfig,
    max_cad_points: int,
    max_scan_points: int,
    seed: int,
) -> PartPrepared:
    cad_path = _pick_primary_cad(dataset.cad_files)
    if cad_path is None:
        raise ValueError(f"No CAD selected for part {dataset.part_key}")

    cad_vertices, cad_triangles, cad_source = load_cad_geometry(
        str(cad_path),
        cad_import_deflection=base_cfg.cad_import_deflection,
    )
    cad_vertices_full = cad_vertices
    cad_bends = extract_cad_bend_specs(
        cad_vertices=cad_vertices_full,
        cad_triangles=cad_triangles,
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        cad_extractor_kwargs=base_cfg.cad_extractor_kwargs,
        cad_detector_kwargs=base_cfg.cad_detector_kwargs,
        cad_detect_call_kwargs=base_cfg.cad_detect_call_kwargs,
    )
    cad_vertices = downsample_points(cad_vertices_full, max_cad_points, seed=seed)

    scans: List[ScanFile] = []
    for sf in dataset.scan_files:
        if sf.path.suffix.lower() not in SCAN_EXTS:
            continue
        scans.append(sf)

    prepared = PartPrepared(
        part_key=dataset.part_key,
        cad_path=cad_path,
        cad_source=cad_source,
        cad_vertices=cad_vertices,
        cad_triangles=cad_triangles,
        cad_bends=cad_bends,
        scans=scans,
        scan_points_cache={},
    )

    return prepared


def build_candidate_configs(
    base_cfg: BendInspectionRuntimeConfig,
    include_incumbent: Optional[BendInspectionRuntimeConfig] = None,
) -> List[Tuple[str, BendInspectionRuntimeConfig]]:
    candidates: List[Tuple[str, BendInspectionRuntimeConfig]] = []

    if include_incumbent is not None:
        candidates.append(
            (
                "incumbent_runtime",
                BendInspectionRuntimeConfig.from_dict(include_incumbent.to_dict()),
            )
        )

    def mk(
        name: str,
        det_ctor: Dict[str, Any],
        det_call: Dict[str, Any],
        matcher: Dict[str, Any],
        cad_extractor: Optional[Dict[str, Any]] = None,
        cad_det_ctor: Optional[Dict[str, Any]] = None,
        cad_det_call: Optional[Dict[str, Any]] = None,
    ):
        cfg = BendInspectionRuntimeConfig.from_dict(base_cfg.to_dict())
        # Fast-but-accurate evaluation baseline to keep the loop iterative on large datasets.
        cfg.scan_detector_kwargs.update(
            {
                "ransac_iterations": 180,
                "adaptive_threshold": True,
                "min_plane_points": 40,
                "min_plane_area": 20.0,
                "adjacency_threshold": 5.0,
            }
        )
        cfg.scan_detect_call_kwargs.update(
            {
                "preprocess": True,
                "voxel_size": 1.0,
                "outlier_nb_neighbors": 12,
                "outlier_std_ratio": 2.5,
                "multiscale": False,
            }
        )
        cfg.scan_detector_kwargs.update(det_ctor)
        cfg.scan_detect_call_kwargs.update(det_call)
        cfg.matcher_kwargs.update(matcher)
        if cad_extractor:
            cfg.cad_extractor_kwargs.update(cad_extractor)
        if cad_det_ctor:
            cfg.cad_detector_kwargs.update(cad_det_ctor)
        if cad_det_call:
            cfg.cad_detect_call_kwargs.update(cad_det_call)
        candidates.append((name, cfg))

    mk(
        "baseline",
        det_ctor={},
        det_call={},
        matcher={},
    )
    mk(
        "tight_planes",
        det_ctor={"plane_threshold": 0.4, "adjacency_threshold": 4.0, "min_plane_area": 25.0, "ransac_iterations": 240},
        det_call={},
        matcher={"max_angle_diff": 12.0},
    )
    mk(
        "loose_planes",
        det_ctor={"plane_threshold": 0.8, "adjacency_threshold": 6.0, "min_plane_area": 20.0, "ransac_iterations": 170},
        det_call={},
        matcher={"max_angle_diff": 18.0},
    )
    mk(
        "small_features",
        det_ctor={"min_plane_points": 30, "min_plane_area": 12.0, "min_bend_angle": 8.0, "ransac_iterations": 200},
        det_call={},
        matcher={"max_angle_diff": 16.0},
    )
    mk(
        "strict_features",
        det_ctor={"min_plane_points": 70, "min_plane_area": 45.0, "min_bend_angle": 12.0, "ransac_iterations": 260},
        det_call={},
        matcher={"max_angle_diff": 12.0},
    )
    mk(
        "multiscale_relaxed",
        det_ctor={"min_plane_points": 35, "min_plane_area": 18.0, "ransac_iterations": 180},
        det_call={"multiscale": False, "voxel_size": 1.1},
        matcher={"max_angle_diff": 18.0},
    )
    mk(
        "multiscale_strict",
        det_ctor={"min_plane_points": 45, "min_plane_area": 25.0, "ransac_iterations": 240},
        det_call={"multiscale": False, "voxel_size": 0.8},
        matcher={"max_angle_diff": 12.0},
    )
    mk(
        "coplanar_aggressive",
        det_ctor={
            "min_plane_points": 35,
            "min_plane_area": 18.0,
            "coplanar_angle_threshold": 6.0,
            "coplanar_offset_threshold": 1.6,
            "coplanar_boundary_threshold": 4.5,
            "dedup_merge_threshold": 14.0,
            "dedup_angle_threshold": 14.0,
        },
        det_call={"voxel_size": 0.9},
        matcher={"max_angle_diff": 18.0},
        cad_extractor={
            "detector_kwargs": {
                "coplanar_angle_threshold": 6.0,
                "coplanar_offset_threshold": 1.8,
                "coplanar_boundary_threshold": 4.5,
                "dedup_merge_threshold": 14.0,
                "dedup_angle_threshold": 15.0,
                "min_plane_points": 35,
                "min_plane_area": 15.0,
            },
            "detect_call_kwargs": {"preprocess": True, "voxel_size": 0.8},
        },
    )

    return candidates


def load_expected_bends_map(output_root: Path) -> Dict[str, int]:
    """
    Load drawing cross-check bend expectations if available.
    """
    path = output_root / "manual_cad_validation" / "bend_baseline_crosscheck.json"
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    mapping: Dict[str, int] = {}
    if not isinstance(raw, list):
        return mapping

    for row in raw:
        if not isinstance(row, dict):
            continue
        part = str(row.get("part", "")).strip()
        expected = row.get("drawing_expected_bends", None)
        if not part or expected is None:
            continue
        try:
            expected_int = int(expected)
        except Exception:
            continue
        if expected_int > 0:
            mapping[part] = expected_int

    return mapping


def evaluate_candidate(
    candidate_name: str,
    cfg: BendInspectionRuntimeConfig,
    prepared_parts: List[PartPrepared],
    max_scan_points: int,
    seed: int,
    max_scans_per_part: int = 0,
    expected_bends_by_part: Optional[Dict[str, int]] = None,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[ScanOutcome]:
    outcomes: List[ScanOutcome] = []
    scan_items: List[Tuple[PartPrepared, ScanFile]] = []
    for part in prepared_parts:
        if len(part.cad_bends) == 0:
            continue
        for scan in _select_scans_for_evaluation(part.scans, max_scans_per_part):
            scan_items.append((part, scan))
    total_scans = len(scan_items)

    for scan_idx, (part, scan) in enumerate(scan_items, start=1):
        scan_points = downsample_points(
            load_scan_points(str(scan.path)),
            max_scan_points,
            seed=seed,
        ).astype(np.float64, copy=False)
        if progress_hook:
            progress_hook(
                {
                    "candidate_name": candidate_name,
                    "scan_idx": scan_idx,
                    "scan_count": total_scans,
                    "part_key": part.part_key,
                    "scan_file": scan.path.name,
                }
            )
        scan_start = perf_counter()
        expected_total = (
            max(1, int(expected_bends_by_part[part.part_key]))
            if expected_bends_by_part and part.part_key in expected_bends_by_part
            else len(part.cad_bends)
        )
        effective_cad_bends = select_cad_bends_for_expected(
            part.cad_bends,
            expected_total,
        )
        report, _, _ = detect_and_match_with_fallback(
            scan_points=scan_points,
            cad_bends=effective_cad_bends,
            part_id=part.part_key,
            runtime_config=cfg,
            expected_bend_count=expected_total,
        )
        report.processing_time_ms = (perf_counter() - scan_start) * 1000.0
        report.compute_summary()
        cad_total = report.total_cad_bends
        ratio = (report.detected_count / expected_total) if expected_total > 0 else 0.0
        outcomes.append(
            ScanOutcome(
                config_name=candidate_name,
                part_key=part.part_key,
                scan_file=scan.path.name,
                state=scan.state,
                sequence=scan.sequence,
                total_bends=expected_total,
                cad_total_bends=cad_total,
                detected=report.detected_count,
                passed=report.pass_count,
                failed=report.fail_count,
                warnings=report.warning_count,
                progress_ratio=ratio,
                processing_time_ms=report.processing_time_ms,
            )
        )
        del scan_points

    return outcomes


def _select_scans_for_evaluation(scans: List[ScanFile], max_scans_per_part: int) -> List[ScanFile]:
    """
    Deterministically subsample scans for bounded-cost loop passes.

    Selection policy:
    - max<=0: keep all scans.
    - always prefer including at least one partial and one full scan when possible.
    - fill remaining slots by sequence order.
    """
    if max_scans_per_part <= 0 or len(scans) <= max_scans_per_part:
        return list(scans)

    partial = [s for s in scans if s.state == "partial"]
    full = [s for s in scans if s.state == "full"]
    partial_sorted = sorted(partial, key=lambda s: (s.sequence, s.mtime, s.path.name))
    full_sorted = sorted(full, key=lambda s: (s.sequence, s.mtime, s.path.name))

    selected: List[ScanFile] = []
    used = set()

    def add(sf: ScanFile):
        key = str(sf.path)
        if key in used:
            return
        selected.append(sf)
        used.add(key)

    if max_scans_per_part >= 1:
        if partial_sorted:
            add(partial_sorted[0])
        elif full_sorted:
            add(full_sorted[-1])
        elif scans:
            add(scans[0])

    if max_scans_per_part >= 2 and full_sorted:
        add(full_sorted[-1])

    merged_priority = partial_sorted + full_sorted
    for sf in merged_priority:
        if len(selected) >= max_scans_per_part:
            break
        add(sf)

    if len(selected) < max_scans_per_part:
        for sf in scans:
            if len(selected) >= max_scans_per_part:
                break
            add(sf)

    selected.sort(key=lambda s: (s.sequence, s.mtime, s.path.name))
    return selected


def _scan_completion_recall(outcome: ScanOutcome) -> float:
    expected = max(1, int(outcome.total_bends))
    return float(min(int(outcome.detected), expected) / expected)


def _scan_in_spec_precision(outcome: ScanOutcome) -> float:
    detected = max(1, int(outcome.detected))
    passed = max(0, int(outcome.passed))
    return float(min(passed, detected) / detected)


def _scan_utility(outcome: ScanOutcome) -> float:
    """
    Utility in [0,1] combining completion, in-spec quality, and over-detection risk.

    This is a conservative per-scan utility used for bootstrap confidence bounds.
    """
    expected = max(1, int(outcome.total_bends))
    completion = _scan_completion_recall(outcome)
    in_spec = _scan_in_spec_precision(outcome)
    overdetect = max(0.0, (float(outcome.detected) - expected) / expected)
    fail_ratio = max(0.0, float(outcome.failed) / expected)
    warn_ratio = max(0.0, float(outcome.warnings) / expected)

    utility = (
        0.60 * completion +
        0.25 * in_spec -
        0.08 * overdetect -
        0.05 * fail_ratio -
        0.02 * warn_ratio
    )
    return float(np.clip(utility, 0.0, 1.0))


def _bootstrap_mean_ci(
    values: np.ndarray,
    sample_count: int = 800,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    Returns: (mean, lower_95, upper_95)
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0
    if values.size == 1:
        value = float(values[0])
        return value, value, value

    samples = max(200, int(sample_count))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(samples, values.size))
    sample_means = np.mean(values[idx], axis=1)
    mean = float(np.mean(values))
    lcb = float(np.percentile(sample_means, 2.5))
    ucb = float(np.percentile(sample_means, 97.5))
    return mean, lcb, ucb


def _stable_seed_offset(name: str, modulus: int = 1_000_000) -> int:
    """
    Process-stable hash offset for deterministic score bootstrapping.

    Python's built-in hash(str) is randomized per-process (hash seed), which can
    perturb evaluation rankings across runs. We use SHA-256-derived bytes instead.
    """
    if modulus <= 1:
        return 0
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(value % modulus)


def score_candidate(
    candidate_name: str,
    outcomes: List[ScanOutcome],
    bootstrap_samples: int = 800,
    seed: int = 42,
) -> ConfigScore:
    if not outcomes:
        return ConfigScore(
            config_name=candidate_name,
            score=0.0,
            overdetect_penalty=1.0,
            full_completion_penalty=1.0,
            partial_monotonic_penalty=1.0,
            partial_overcomplete_penalty=1.0,
            completion_recall=0.0,
            in_spec_precision=0.0,
            mean_scan_utility=0.0,
            utility_lcb95=0.0,
            utility_ucb95=0.0,
            bootstrap_samples=max(200, int(bootstrap_samples)),
            promotion_eligible=False,
            gate_reasons=["no_outcomes"],
            scan_count=0,
            outcomes=[],
        )

    overdetect_penalty = 0.0
    full_completion_penalty = 0.0
    partial_overcomplete_penalty = 0.0

    full_count = 0
    partial_count = 0

    for o in outcomes:
        if o.total_bends <= 0:
            continue
        overdetect_penalty += max(0.0, o.progress_ratio - 1.0)
        if o.state == "full":
            full_count += 1
            full_completion_penalty += abs(1.0 - min(1.0, o.progress_ratio))
        elif o.state == "partial":
            partial_count += 1
            partial_overcomplete_penalty += max(0.0, o.progress_ratio - 0.95)

    # Monotonic partial progress penalty by part
    partial_monotonic_penalty = 0.0
    pair_count = 0
    by_part: Dict[str, List[ScanOutcome]] = {}
    for o in outcomes:
        if o.state == "partial":
            by_part.setdefault(o.part_key, []).append(o)
    # If the scoring outcomes include duplicate scan numbers that are likely
    # alternate captures (same sequence), do not force monotonic order.
    for part_key, seq in by_part.items():
        seq_sorted = sorted(seq, key=lambda x: (x.sequence, x.scan_file))
        for i in range(len(seq_sorted) - 1):
            if seq_sorted[i].sequence == seq_sorted[i + 1].sequence:
                continue
            pair_count += 1
            drop = seq_sorted[i].progress_ratio - seq_sorted[i + 1].progress_ratio
            if drop > 0.03:
                partial_monotonic_penalty += drop

    scan_count = max(1, len(outcomes))
    overdetect_penalty /= scan_count
    full_completion_penalty /= max(1, full_count)
    partial_overcomplete_penalty /= max(1, partial_count)
    partial_monotonic_penalty /= max(1, pair_count)

    total_expected = max(1, int(sum(max(1, o.total_bends) for o in outcomes)))
    total_detected_raw = int(sum(max(0, o.detected) for o in outcomes))
    total_detected_capped = int(sum(min(max(0, o.detected), max(1, o.total_bends)) for o in outcomes))
    total_passed = int(sum(max(0, min(o.passed, max(0, o.detected))) for o in outcomes))
    completion_recall = float(total_detected_capped / total_expected)
    in_spec_precision = float(total_passed / max(1, total_detected_raw))

    utilities = np.asarray([_scan_utility(o) for o in outcomes], dtype=np.float64)
    seed_offset = _stable_seed_offset(candidate_name, modulus=1_000_000)
    mean_utility, utility_lcb95, utility_ucb95 = _bootstrap_mean_ci(
        utilities,
        sample_count=bootstrap_samples,
        seed=seed + seed_offset,
    )

    # Weighted quality score [0..100]
    raw_penalty = (
        0.35 * overdetect_penalty +
        0.35 * full_completion_penalty +
        0.20 * partial_monotonic_penalty +
        0.10 * partial_overcomplete_penalty
    )
    penalty_component = float(np.clip(1.0 - raw_penalty, 0.0, 1.0))
    score = float(
        np.clip(
            100.0 * (
                0.55 * utility_lcb95 +
                0.25 * mean_utility +
                0.20 * penalty_component
            ),
            0.0,
            100.0,
        )
    )

    return ConfigScore(
        config_name=candidate_name,
        score=score,
        overdetect_penalty=overdetect_penalty,
        full_completion_penalty=full_completion_penalty,
        partial_monotonic_penalty=partial_monotonic_penalty,
        partial_overcomplete_penalty=partial_overcomplete_penalty,
        completion_recall=completion_recall,
        in_spec_precision=in_spec_precision,
        mean_scan_utility=mean_utility,
        utility_lcb95=utility_lcb95,
        utility_ucb95=utility_ucb95,
        bootstrap_samples=max(200, int(bootstrap_samples)),
        promotion_eligible=True,
        gate_reasons=[],
        scan_count=len(outcomes),
        outcomes=outcomes,
    )


def apply_regression_gate(
    challenger: ConfigScore,
    incumbent: ConfigScore,
    epsilon_lcb: float = 0.005,
    epsilon_overdetect: float = 0.010,
    epsilon_full_completion: float = 0.020,
    epsilon_partial_monotonic: float = 0.015,
    epsilon_partial_overcomplete: float = 0.020,
    epsilon_recall: float = 0.020,
) -> Tuple[bool, List[str]]:
    """
    Non-regression gate against incumbent runtime.

    A challenger may improve score, but must remain non-inferior on critical
    risk metrics to be promotable.
    """
    reasons: List[str] = []

    if challenger.utility_lcb95 < (incumbent.utility_lcb95 - epsilon_lcb):
        reasons.append(
            f"utility_lcb95_regressed ({challenger.utility_lcb95:.4f} < {incumbent.utility_lcb95 - epsilon_lcb:.4f})"
        )
    if challenger.overdetect_penalty > (incumbent.overdetect_penalty + epsilon_overdetect):
        reasons.append(
            f"overdetect_penalty_regressed ({challenger.overdetect_penalty:.4f} > {incumbent.overdetect_penalty + epsilon_overdetect:.4f})"
        )
    if challenger.full_completion_penalty > (incumbent.full_completion_penalty + epsilon_full_completion):
        reasons.append(
            f"full_completion_penalty_regressed ({challenger.full_completion_penalty:.4f} > {incumbent.full_completion_penalty + epsilon_full_completion:.4f})"
        )
    if challenger.partial_monotonic_penalty > (incumbent.partial_monotonic_penalty + epsilon_partial_monotonic):
        reasons.append(
            f"partial_monotonic_penalty_regressed ({challenger.partial_monotonic_penalty:.4f} > {incumbent.partial_monotonic_penalty + epsilon_partial_monotonic:.4f})"
        )
    if challenger.partial_overcomplete_penalty > (incumbent.partial_overcomplete_penalty + epsilon_partial_overcomplete):
        reasons.append(
            f"partial_overcomplete_penalty_regressed ({challenger.partial_overcomplete_penalty:.4f} > {incumbent.partial_overcomplete_penalty + epsilon_partial_overcomplete:.4f})"
        )
    if challenger.completion_recall < (incumbent.completion_recall - epsilon_recall):
        reasons.append(
            f"completion_recall_regressed ({challenger.completion_recall:.4f} < {incumbent.completion_recall - epsilon_recall:.4f})"
        )

    return len(reasons) == 0, reasons


def _append_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _live_mirror_roots(output_root: Path) -> List[Path]:
    """
    Paths where live status artifacts are mirrored for visibility.
    """
    roots = [output_root, ROOT, ROOT.parent]
    unique: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def write_live_log(output_root: Path, line: str):
    text = f"{line.rstrip()}\n"
    for root in _live_mirror_roots(output_root):
        _append_text(root / "LIVE_RUN.log", text)


def init_agent_scratchpads(run_dir: Path):
    agents = {
        "agent_orchestrator.md": "# Agent Orchestrator\n\nTracks run-level plan, stage transitions, and blockers.\n",
        "agent_ingestion.md": "# Agent Ingestion\n\nTracks discovered/validated parts, CAD-scan grouping, and file quality.\n",
        "agent_math_integrity.md": "# Agent Math Integrity\n\nTracks numerical stability and metric correctness issues.\n",
        "agent_algorithm.md": "# Agent Algorithm\n\nTracks configuration evaluations and model-behavior changes.\n",
        "agent_visual_validation.md": "# Agent Visual Validation\n\nTracks projection images and geometric sanity checks.\n",
        "agent_user_flow.md": "# Agent User Flow\n\nTracks API/runtime integration and usability/engineering concerns.\n",
    }
    for filename, header in agents.items():
        path = run_dir / "agents" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(header, encoding="utf-8")


def append_agent_note(run_dir: Path, agent_filename: str, note: str):
    _append_text(run_dir / "agents" / agent_filename, f"\n## {_iso_now()}\n- {note}\n")


def init_todo_board() -> List[TodoItem]:
    return [
        TodoItem("discover", "Discover and group CAD/scan/spec datasets"),
        TodoItem("prepare", "Prepare part bundles and extract CAD bend baselines"),
        TodoItem("evaluate", "Evaluate candidate bend configs across partial/full scans"),
        TodoItem("select", "Select best runtime config and persist artifacts"),
        TodoItem("visualize", "Generate geometric projection visuals and inspect"),
        TodoItem("integrate", "Promote config for API runtime and persist history"),
    ]


def set_todo(todo: List[TodoItem], key: str, status: str, note: str = ""):
    for item in todo:
        if item.key == key:
            item.status = status
            if note:
                item.note = note
            return


def render_todo_markdown(todo: List[TodoItem]) -> str:
    icon = {
        "pending": "[ ]",
        "in_progress": "[~]",
        "done": "[x]",
        "failed": "[!]",
    }
    lines = ["# Improvement Loop TODO Board", "", f"Updated: {_iso_now()}", ""]
    for item in todo:
        suffix = f" - {item.note}" if item.note else ""
        lines.append(f"- {icon.get(item.status, '[ ]')} `{item.key}` {item.title}{suffix}")
    lines.append("")
    return "\n".join(lines)


def write_live_status(
    output_root: Path,
    run_dir: Path,
    todo: List[TodoItem],
    status_payload: Dict[str, Any],
):
    todo_md = render_todo_markdown(todo)
    (run_dir / "TODO.md").write_text(todo_md, encoding="utf-8")
    for root in _live_mirror_roots(output_root):
        (root / "CURRENT_TODO.md").write_text(todo_md, encoding="utf-8")

    payload = dict(status_payload)
    payload["updated_at"] = _iso_now()
    payload["run_dir"] = str(run_dir)
    payload["todo"] = [
        {"key": t.key, "title": t.title, "status": t.status, "note": t.note}
        for t in todo
    ]
    write_json(run_dir / "status.json", payload)
    for root in _live_mirror_roots(output_root):
        write_json(root / "CURRENT_STATUS.json", payload)


def update_agent_scratchpads(
    run_dir: Path,
    all_scores: List[ConfigScore],
    prepared_parts: List[PartPrepared],
    best_name: str,
):
    timestamp = datetime.now().isoformat(timespec="seconds")
    best = next((s for s in all_scores if s.config_name == best_name), None)
    if best is None:
        return

    ingestion_lines = [
        f"\n## {timestamp}\n",
        f"- Parts prepared: {len(prepared_parts)}\n",
        f"- Total scans: {sum(len(p.scans) for p in prepared_parts)}\n",
        "- Parts: " + ", ".join(p.part_key for p in prepared_parts[:40]) + "\n",
    ]
    _append_text(run_dir / "agents" / "agent_ingestion.md", "".join(ingestion_lines))

    math_lines = [
        f"\n## {timestamp}\n",
        f"- Best config: `{best.config_name}`\n",
        f"- Score: {best.score:.3f}\n",
        f"- Mean utility: {best.mean_scan_utility:.4f}\n",
        f"- Utility LCB95: {best.utility_lcb95:.4f}\n",
        f"- Completion recall: {best.completion_recall:.4f}\n",
        f"- In-spec precision: {best.in_spec_precision:.4f}\n",
        f"- Overdetect penalty: {best.overdetect_penalty:.6f}\n",
        f"- Full completion penalty: {best.full_completion_penalty:.6f}\n",
        f"- Partial monotonic penalty: {best.partial_monotonic_penalty:.6f}\n",
        f"- Partial overcomplete penalty: {best.partial_overcomplete_penalty:.6f}\n",
    ]
    _append_text(run_dir / "agents" / "agent_math_integrity.md", "".join(math_lines))

    ranking = sorted(all_scores, key=lambda s: s.score, reverse=True)
    algo_lines = [f"\n## {timestamp}\n", "- Config ranking:\n"]
    for s in ranking:
        gate = "eligible" if s.promotion_eligible else "rejected"
        algo_lines.append(
            f"  - {s.config_name}: score={s.score:.3f}, lcb95={s.utility_lcb95:.4f}, gate={gate}\n"
        )
    _append_text(run_dir / "agents" / "agent_algorithm.md", "".join(algo_lines))


def _projection_points(points: np.ndarray, ax0: int, ax1: int, width: int, height: int):
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)
    proj = points[:, [ax0, ax1]]
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    norm = (proj - mins) / spans
    xs = np.clip((norm[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
    ys = np.clip(((1.0 - norm[:, 1]) * (height - 1)).astype(np.int32), 0, height - 1)
    return np.column_stack([xs, ys])


def generate_projection_image(
    cad_points: np.ndarray,
    scan_points: np.ndarray,
    report,
    out_path: Path,
    sample_size: int = 12000,
):
    rng = np.random.default_rng(42)

    def sample(points: np.ndarray, n: int) -> np.ndarray:
        if len(points) <= n:
            return points
        idx = rng.choice(len(points), size=n, replace=False)
        return points[idx]

    cad = sample(cad_points, sample_size)
    scan = sample(scan_points, sample_size)

    width = 1400
    height = 460
    panel_w = width // 3

    image = Image.new("RGB", (width, height), (16, 18, 24))
    draw = ImageDraw.Draw(image)

    projections = [
        ("XY", 0, 1),
        ("XZ", 0, 2),
        ("YZ", 1, 2),
    ]

    for panel_idx, (label, a0, a1) in enumerate(projections):
        x_offset = panel_idx * panel_w
        draw.rectangle([x_offset, 0, x_offset + panel_w - 1, height - 1], outline=(60, 60, 60), width=1)
        draw.text((x_offset + 8, 8), label, fill=(220, 220, 220))

        # Fit both clouds to same projection range for this panel
        combined = np.vstack([cad[:, [a0, a1]], scan[:, [a0, a1]]])
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-9)

        def map_points(p: np.ndarray) -> np.ndarray:
            if len(p) == 0:
                return np.empty((0, 2), dtype=np.int32)
            proj = p[:, [a0, a1]]
            norm = (proj - mins) / spans
            xs = np.clip((norm[:, 0] * (panel_w - 30) + 12).astype(np.int32), 0, panel_w - 1)
            ys = np.clip(((1.0 - norm[:, 1]) * (height - 24) + 12).astype(np.int32), 0, height - 1)
            xs = xs + x_offset
            return np.column_stack([xs, ys])

        cad_px = map_points(cad)
        scan_px = map_points(scan)

        for x, y in cad_px:
            image.putpixel((int(x), int(y)), (110, 110, 110))
        for x, y in scan_px:
            image.putpixel((int(x), int(y)), (80, 180, 245))

        for m in report.matches:
            if m.detected_bend is None:
                continue
            p1 = m.detected_bend.bend_line_start
            p2 = m.detected_bend.bend_line_end
            mapped = map_points(np.asarray([p1, p2], dtype=np.float64))
            if len(mapped) == 2:
                draw.line(
                    [tuple(mapped[0]), tuple(mapped[1])],
                    fill=(255, 102, 102),
                    width=2,
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative bend-improvement loop")
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "improvement_loop")
    parser.add_argument("--max-parts", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-scans-per-part", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-cad-points", type=int, default=400000)
    parser.add_argument("--max-scan-points", type=int, default=350000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-visuals", action="store_true")
    parser.add_argument(
        "--runtime-config",
        type=str,
        default="",
        help="Optional bend runtime config JSON path",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=800,
        help="Bootstrap resamples for conservative score confidence bounds",
    )
    parser.add_argument(
        "--disable-regression-gate",
        action="store_true",
        help="Allow promotion even when challenger regresses vs incumbent",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = perf_counter()

    # Start each run with a fresh live log to avoid mixing runs.
    for root in _live_mirror_roots(args.output):
        (root / "LIVE_RUN.log").write_text("", encoding="utf-8")

    def log(line: str):
        print(line)
        write_live_log(args.output, line)

    if not args.downloads.exists():
        log(f"[ERROR] Downloads path does not exist: {args.downloads}")
        return 2

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    init_agent_scratchpads(run_dir)
    (args.output / "LIVE_RUN.pid").write_text(str(os.getpid()), encoding="utf-8")
    todo = init_todo_board()
    set_todo(todo, "discover", "in_progress")
    write_live_status(
        args.output,
        run_dir,
        todo,
        {
            "stage": "discover",
            "message": f"Discovering datasets in {args.downloads}",
        },
    )
    append_agent_note(run_dir, "agent_orchestrator.md", f"Run started: {run_id}")

    base_cfg = load_bend_runtime_config(args.runtime_config or None)
    append_agent_note(
        run_dir,
        "agent_user_flow.md",
        "Loaded runtime config baseline for this run."
    )

    log(f"[INFO] Discovering datasets in {args.downloads}")
    discovered = discover_parts(args.downloads)
    if args.max_parts and args.max_parts > 0:
        discovered = discovered[:args.max_parts]
    log(f"[INFO] Parts discovered: {len(discovered)}")
    set_todo(todo, "discover", "done", f"Discovered parts: {len(discovered)}")
    set_todo(todo, "prepare", "in_progress")
    write_live_status(
        args.output,
        run_dir,
        todo,
        {
            "stage": "prepare",
            "message": "Preparing part bundles",
            "discovered_parts": len(discovered),
        },
    )
    append_agent_note(
        run_dir,
        "agent_ingestion.md",
        f"Discovered {len(discovered)} candidate part groups."
    )

    prepared_parts: List[PartPrepared] = []
    prep_errors: List[Dict[str, str]] = []
    for ds in discovered:
        try:
            prepared = prepare_part(
                dataset=ds,
                base_cfg=base_cfg,
                max_cad_points=args.max_cad_points,
                max_scan_points=args.max_scan_points,
                seed=args.seed,
            )
            prepared_parts.append(prepared)
            log(
                f"[PREP] Part {prepared.part_key}: CAD={prepared.cad_path.name} "
                f"bends={len(prepared.cad_bends)} scans={len(prepared.scans)} source={prepared.cad_source}"
            )
            append_agent_note(
                run_dir,
                "agent_ingestion.md",
                f"Prepared part {prepared.part_key}: CAD={prepared.cad_path.name}, "
                f"CAD bends={len(prepared.cad_bends)}, scans={len(prepared.scans)}, source={prepared.cad_source}."
            )
            append_agent_note(
                run_dir,
                "agent_math_integrity.md",
                f"Part {prepared.part_key} baseline CAD bends: {len(prepared.cad_bends)}."
            )
            write_live_status(
                args.output,
                run_dir,
                todo,
                {
                    "stage": "prepare",
                    "message": f"Prepared part {prepared.part_key}",
                    "prepared_parts": len(prepared_parts),
                    "prep_errors": prep_errors,
                },
            )
        except Exception as exc:
            prep_errors.append({"part_key": ds.part_key, "error": str(exc)})
            log(f"[WARN] Failed preparing part {ds.part_key}: {exc}")
            append_agent_note(
                run_dir,
                "agent_ingestion.md",
                f"Failed preparing part {ds.part_key}: {exc}"
            )

    if not prepared_parts:
        log("[ERROR] No parts were prepared for evaluation")
        set_todo(todo, "prepare", "failed", "No parts prepared")
        write_json(
            run_dir / "run_summary.json",
            {
                "status": "failed",
                "reason": "no_prepared_parts",
                "prep_errors": prep_errors,
            },
        )
        write_live_status(
            args.output,
            run_dir,
            todo,
            {
                "stage": "failed",
                "message": "No parts prepared",
                "prep_errors": prep_errors,
            },
        )
        return 3

    set_todo(todo, "prepare", "done", f"Prepared parts: {len(prepared_parts)}")
    set_todo(todo, "evaluate", "in_progress")
    expected_bends_by_part = load_expected_bends_map(args.output)
    if expected_bends_by_part:
        append_agent_note(
            run_dir,
            "agent_math_integrity.md",
            f"Loaded drawing baseline overrides for {len(expected_bends_by_part)} part(s): "
            + ", ".join(f"{k}={v}" for k, v in sorted(expected_bends_by_part.items())[:20]),
        )
        log(
            "[INFO] Using drawing baseline overrides for parts: "
            + ", ".join(f"{k}:{v}" for k, v in sorted(expected_bends_by_part.items()))
        )
    candidates = build_candidate_configs(base_cfg, include_incumbent=base_cfg)
    log(f"[INFO] Evaluating {len(candidates)} candidate configurations")
    append_agent_note(
        run_dir,
        "agent_algorithm.md",
        f"Evaluating {len(candidates)} candidate configurations."
    )
    write_live_status(
        args.output,
        run_dir,
        todo,
        {
            "stage": "evaluate",
            "message": "Starting candidate evaluation",
            "candidate_count": len(candidates),
            "prepared_parts": len(prepared_parts),
        },
    )

    scored: List[ConfigScore] = []
    for name, cfg in candidates:
        c_start = perf_counter()

        def _on_eval_progress(payload: Dict[str, Any]) -> None:
            msg = (
                f"{name}: scan {payload['scan_idx']}/{payload['scan_count']} "
                f"{payload['part_key']}/{payload['scan_file']}"
            )
            log(f"[EVAL] {msg}")
            write_live_status(
                args.output,
                run_dir,
                todo,
                {
                    "stage": "evaluate",
                    "message": msg,
                    "evaluated_candidates": len(scored),
                    "candidate_count": len(candidates),
                    "active_candidate": name,
                    "active_scan_index": payload["scan_idx"],
                    "active_scan_count": payload["scan_count"],
                    "active_part": payload["part_key"],
                    "active_scan_file": payload["scan_file"],
                },
            )

        outcomes = evaluate_candidate(
            candidate_name=name,
            cfg=cfg,
            prepared_parts=prepared_parts,
            max_scan_points=args.max_scan_points,
            seed=args.seed,
            max_scans_per_part=args.max_scans_per_part,
            expected_bends_by_part=expected_bends_by_part,
            progress_hook=_on_eval_progress,
        )
        score = score_candidate(
            name,
            outcomes,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        scored.append(score)
        log(
            f"[EVAL] {name}: score={score.score:.3f} scans={score.scan_count} "
            f"over={score.overdetect_penalty:.4f} full={score.full_completion_penalty:.4f} "
            f"mono={score.partial_monotonic_penalty:.4f} "
            f"lcb95={score.utility_lcb95:.4f} "
            f"time={(perf_counter() - c_start):.1f}s"
        )
        write_json(
            run_dir / "candidates" / f"{name}.json",
            {
                "config_name": name,
                "score": score.to_dict(),
            },
        )
        write_json(
            run_dir / "scores.partial.json",
            {"scores": [s.to_dict() for s in scored]},
        )
        append_agent_note(
            run_dir,
            "agent_algorithm.md",
            f"{name}: score={score.score:.3f}, scans={score.scan_count}, "
            f"over={score.overdetect_penalty:.4f}, full={score.full_completion_penalty:.4f}, "
            f"mono={score.partial_monotonic_penalty:.4f}."
        )
        write_live_status(
            args.output,
            run_dir,
            todo,
            {
                "stage": "evaluate",
                "message": f"Evaluated {name}",
                "evaluated_candidates": len(scored),
                "candidate_count": len(candidates),
                "latest_candidate": name,
                "latest_score": score.score,
                "latest_lcb95": score.utility_lcb95,
            },
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    incumbent_score = next((s for s in scored if s.config_name == "incumbent_runtime"), None)
    if incumbent_score is not None:
        incumbent_score.promotion_eligible = True
        incumbent_score.gate_reasons = []

    rejected: List[ConfigScore] = []
    for s in scored:
        if incumbent_score is None or s.config_name == "incumbent_runtime":
            continue
        if args.disable_regression_gate:
            s.promotion_eligible = True
            s.gate_reasons = []
            continue
        ok, reasons = apply_regression_gate(s, incumbent_score)
        s.promotion_eligible = ok
        s.gate_reasons = reasons
        if not ok:
            rejected.append(s)

    promotable = [s for s in scored if s.promotion_eligible]
    if not promotable:
        promotable = scored[:]

    promotable.sort(key=lambda s: s.score, reverse=True)
    best = promotable[0]
    best_cfg = next(cfg for name, cfg in candidates if name == best.config_name)

    if incumbent_score is not None and rejected:
        for r in rejected:
            log(
                f"[GATE] reject {r.config_name}: " + ", ".join(r.gate_reasons)
            )
            append_agent_note(
                run_dir,
                "agent_math_integrity.md",
                f"Gate rejected {r.config_name}: " + ", ".join(r.gate_reasons),
            )

    if incumbent_score is not None and best.config_name == "incumbent_runtime":
        log(
            f"[BEST] incumbent_runtime survives "
            f"(score={best.score:.3f}, lcb95={best.utility_lcb95:.4f})"
        )
    else:
        log(
            f"[BEST] {best.config_name} score={best.score:.3f} "
            f"(lcb95={best.utility_lcb95:.4f})"
        )
    set_todo(todo, "evaluate", "done", f"Evaluated candidates: {len(candidates)}")
    set_todo(todo, "select", "in_progress")
    append_agent_note(
        run_dir,
        "agent_orchestrator.md",
        f"Best candidate selected: {best.config_name} ({best.score:.3f})."
    )
    write_live_status(
        args.output,
        run_dir,
        todo,
        {
            "stage": "select",
            "message": f"Selected best config {best.config_name}",
            "best_name": best.config_name,
            "best_score": best.score,
            "best_lcb95": best.utility_lcb95,
            "regression_gate_enabled": not args.disable_regression_gate,
            "rejected_candidates": [
                {"name": r.config_name, "reasons": r.gate_reasons}
                for r in rejected
            ],
        },
    )

    write_json(
        run_dir / "scores.json",
        {"scores": [s.to_dict() for s in scored]},
    )
    write_json(
        run_dir / "best_config.json",
        {
            "name": best.config_name,
            "score": best.score,
            "runtime_config": best_cfg.to_dict(),
        },
    )
    write_json(
        args.output / "best_config.latest.json",
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "name": best.config_name,
            "score": best.score,
            "runtime_config": best_cfg.to_dict(),
        },
    )
    set_todo(todo, "select", "done", f"Best: {best.config_name} ({best.score:.3f})")
    set_todo(todo, "visualize", "in_progress")
    write_live_status(
        args.output,
        run_dir,
        todo,
        {
            "stage": "visualize",
            "message": "Generating visuals",
            "best_name": best.config_name,
            "best_score": best.score,
        },
    )

    # Per-scan results for top config
    write_json(
        run_dir / "best_outcomes.json",
        {"outcomes": [o.to_dict() for o in best.outcomes]},
    )

    history_entry = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "downloads": str(args.downloads),
        "prepared_parts": len(prepared_parts),
        "evaluated_configs": len(candidates),
        "best_name": best.config_name,
        "best_score": best.score,
        "best_lcb95": best.utility_lcb95,
        "regression_gate_enabled": not args.disable_regression_gate,
        "incumbent_score": incumbent_score.score if incumbent_score is not None else None,
        "incumbent_lcb95": incumbent_score.utility_lcb95 if incumbent_score is not None else None,
        "rejected_candidates": [
            {"name": r.config_name, "reasons": r.gate_reasons}
            for r in rejected
        ],
        "expected_bends_overrides": expected_bends_by_part,
        "prep_errors": prep_errors,
    }
    history_path = args.output / "history.jsonl"
    _append_text(history_path, json.dumps(history_entry) + "\n")

    update_agent_scratchpads(
        run_dir=run_dir,
        all_scores=scored,
        prepared_parts=prepared_parts,
        best_name=best.config_name,
    )

    if args.generate_visuals:
        log("[INFO] Generating projection diagnostics for best config")
        vis_dir = run_dir / "visuals"
        vis_count = 0
        for part in prepared_parts[:6]:
            for scan in part.scans[:2]:
                try:
                    scan_points = downsample_points(
                        load_scan_points(str(scan.path)),
                        args.max_scan_points,
                        seed=args.seed,
                    ).astype(np.float64, copy=False)
                    expected_total = (
                        max(1, int(expected_bends_by_part[part.part_key]))
                        if part.part_key in expected_bends_by_part
                        else len(part.cad_bends)
                    )
                    effective_cad_bends = select_cad_bends_for_expected(
                        part.cad_bends,
                        expected_total,
                    )
                    report, _, _ = detect_and_match_with_fallback(
                        scan_points=scan_points,
                        cad_bends=effective_cad_bends,
                        part_id=part.part_key,
                        runtime_config=best_cfg,
                        expected_bend_count=expected_total,
                    )
                    out_path = vis_dir / f"{part.part_key}__{scan.path.stem}.png"
                    generate_projection_image(
                        cad_points=part.cad_vertices,
                        scan_points=scan_points,
                        report=report,
                        out_path=out_path,
                    )
                    vis_count += 1
                    append_agent_note(
                        run_dir,
                        "agent_visual_validation.md",
                        f"Generated visual for {part.part_key}/{scan.path.name}"
                    )
                    write_live_status(
                        args.output,
                        run_dir,
                        todo,
                        {
                            "stage": "visualize",
                            "message": f"Generated visual {part.part_key}/{scan.path.name}",
                            "visual_count": vis_count,
                        },
                    )
                except Exception as exc:
                    _append_text(
                        run_dir / "agents" / "agent_visual_validation.md",
                        f"\n- Failed visual for {part.part_key}/{scan.path.name}: {exc}\n",
                    )
        _append_text(
            run_dir / "agents" / "agent_visual_validation.md",
            f"\n- Generated projection images: {vis_count}\n",
        )
    set_todo(todo, "visualize", "done")
    set_todo(todo, "integrate", "in_progress")
    append_agent_note(
        run_dir,
        "agent_user_flow.md",
        "Promoted best runtime config to output/improvement_loop/best_config.latest.json for API consumption."
    )

    elapsed = perf_counter() - t0
    write_json(
        run_dir / "run_summary.json",
        {
            "status": "completed",
            "run_id": run_id,
            "elapsed_seconds": round(elapsed, 2),
            "prepared_parts": len(prepared_parts),
            "best_name": best.config_name,
            "best_score": best.score,
            "best_lcb95": best.utility_lcb95,
            "regression_gate_enabled": not args.disable_regression_gate,
            "incumbent_name": incumbent_score.config_name if incumbent_score is not None else None,
            "incumbent_score": incumbent_score.score if incumbent_score is not None else None,
            "rejected_candidates": [
                {"name": r.config_name, "reasons": r.gate_reasons}
                for r in rejected
            ],
            "expected_bends_overrides": expected_bends_by_part,
            "prep_errors": prep_errors,
            "outputs": {
                "scores": str(run_dir / "scores.json"),
                "best_config": str(run_dir / "best_config.json"),
                "best_outcomes": str(run_dir / "best_outcomes.json"),
            },
        },
    )
    set_todo(todo, "integrate", "done")
    write_live_status(
        args.output,
        run_dir,
        todo,
        {
            "stage": "complete",
            "message": "Run completed",
            "best_name": best.config_name,
            "best_score": best.score,
            "best_lcb95": best.utility_lcb95,
            "regression_gate_enabled": not args.disable_regression_gate,
            "elapsed_seconds": round(elapsed, 2),
            "prepared_parts": len(prepared_parts),
        },
    )

    log(f"[DONE] Run {run_id} completed in {elapsed:.1f}s")
    log(f"[DONE] Best config written to {args.output / 'best_config.latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
