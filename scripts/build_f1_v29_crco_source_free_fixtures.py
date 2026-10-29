#!/usr/bin/env python3
"""Build the frozen, source-free CRCO_V1 fixture suite.

No repository scan, label, CAD, drawing, filename, or benchmark artifact is read.
All range lattices and geometric-core candidate records are constructed from
literal analytic geometry declared in this file and the immutable V29 lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v29_crco_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v29-crco-v1" / "source_free_fixtures"
VISUAL_SPEC_ROOT = ROOT / "output" / "research" / "f1-v29-crco-v1" / "visual_specs"

WIDTH = 320
HEIGHT = 320
PIXEL_PITCH_MM = 0.25
DEPTH_QUANTUM_MM = 0.001
FRAME_ANGLES_DEG = (12.0, 39.0, 73.0, 111.0)
STATION_COUNT = 24
FLANK_MULTIPLIERS = (2, 3, 4, 5, 6, 7, 8)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _rotation_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _pose(angle_deg: float) -> np.ndarray:
    rotation = _rotation_x(angle_deg)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = -100.0 * rotation[:, 2]
    return pose


def _round_half_away(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0.0 else int(math.ceil(value - 0.5))


def _project(point: np.ndarray, pose: np.ndarray) -> tuple[int, int, float] | None:
    camera = pose[:3, :3].T @ (point - pose[:3, 3])
    u = _round_half_away(float(camera[0]) / PIXEL_PITCH_MM + (WIDTH - 1) / 2.0)
    v = _round_half_away(float(camera[1]) / PIXEL_PITCH_MM + (HEIGHT - 1) / 2.0)
    if not (0 <= u < WIDTH and 0 <= v < HEIGHT):
        return None
    return u, v, float(camera[2])


def _set_depth(depth: np.ndarray, valid: np.ndarray, u: int, v: int, value_mm: float) -> None:
    quantized = max(1, _round_half_away(value_mm / DEPTH_QUANTUM_MM))
    if valid[v, u] and int(depth[v, u]) != quantized:
        # Fixtures are constructed to avoid contradictory rays.  A collision is
        # a generator contract failure rather than an implicit overwrite.
        if abs(int(depth[v, u]) - quantized) > 1000:
            raise RuntimeError(f"contradictory synthetic range at {(u, v)}")
        quantized = min(int(depth[v, u]), quantized)
    depth[v, u] = quantized
    valid[v, u] = True


def _draw_path(
    depth: np.ndarray,
    valid: np.ndarray,
    first: tuple[int, int, float],
    second: tuple[int, int, float],
) -> None:
    u0, v0, d0 = first
    u1, v1, d1 = second
    u, v = u0, v0
    steps = abs(u1 - u0) + abs(v1 - v0)
    index = 0
    while True:
        alpha = 0.0 if steps == 0 else index / steps
        _set_depth(depth, valid, u, v, (1.0 - alpha) * d0 + alpha * d1)
        if u == u1 and v == v1:
            break
        if u != u1:
            u += 1 if u1 > u else -1
        else:
            v += 1 if v1 > v else -1
        index += 1


def _candidate(
    *,
    y: float = 0.0,
    z: float = 0.0,
    selected: bool,
    boundary_risk: bool,
    ambiguous: bool,
    rank: str = "0000",
) -> dict[str, Any]:
    return {
        "line_start": [-24.0, y, z],
        "line_end": [24.0, y, z],
        "support_interval": [0.0, 48.0],
        "flank_tangents": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "geometric_core_pass_bits": {"angle": True, "support": True, "span": True},
        "boundary_risk": boundary_risk,
        "rejection_reason": "BOUNDARY_OWNERSHIP_AMBIGUOUS" if ambiguous else "NONE",
        "selected_legacy": selected,
        "legacy_rank_bytes": rank,
        "span": 48.0,
        "quality_summaries": {"core_score": 1.0, "support_density": 1.0},
    }


def _stations(candidate: dict[str, Any]) -> list[np.ndarray]:
    start = np.asarray(candidate["line_start"], dtype=np.float64)
    end = np.asarray(candidate["line_end"], dtype=np.float64)
    return [start + ((index + 0.5) / STATION_COUNT) * (end - start) for index in range(STATION_COUNT)]


def _paint_candidate(
    depth: np.ndarray,
    valid: np.ndarray,
    pose: np.ndarray,
    candidate: dict[str, Any],
    mode: str,
    s_mm: float,
) -> None:
    flanks = [np.asarray(value, dtype=np.float64) for value in candidate["flank_tangents"]]
    for station in _stations(candidate):
        inner: list[tuple[int, int, float]] = []
        for flank_index, flank in enumerate(flanks):
            for multiplier in FLANK_MULTIPLIERS:
                point = station + multiplier * s_mm * flank
                projected = _project(point, pose)
                if projected is None:
                    continue
                u, v, expected = projected
                if mode == "unknown":
                    continue
                delta = 0.0
                if mode == "jump" and flank_index == 1:
                    delta = 6.0 * s_mm
                elif mode == "occluded" and flank_index == 1:
                    delta = -6.0 * s_mm
                _set_depth(depth, valid, u, v, expected + delta)
                if multiplier == FLANK_MULTIPLIERS[0]:
                    inner.append((u, v, expected + delta))
        center = _project(station, pose)
        if mode != "unknown" and center is not None:
            _set_depth(depth, valid, center[0], center[1], center[2])
        if mode == "continuous" and len(inner) == 2:
            _draw_path(depth, valid, inner[0], inner[1])


def _frame(candidate_records: list[dict[str, Any]], modes: list[str], angle: float, s_mm: float, frame_index: int) -> dict[str, Any]:
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    valid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    pose = _pose(angle)
    for candidate, mode in zip(candidate_records, modes, strict=True):
        _paint_candidate(depth, valid, pose, candidate, mode, s_mm)
    # Content-only anchor differentiates otherwise identical lattices without
    # entering any candidate station window.
    _set_depth(depth, valid, 2 + frame_index * 3, 2, 80.0 + frame_index)
    return {
        "schema_version": 1,
        "projection_model": "orthographic_range_lattice",
        "width": WIDTH,
        "height": HEIGHT,
        "depth_quantum_mm": DEPTH_QUANTUM_MM,
        "depth_q": depth.tolist(),
        "valid": valid.tolist(),
        "calibration": {"pixel_pitch_mm": PIXEL_PITCH_MM, "cx": (WIDTH - 1) / 2.0, "cy": (HEIGHT - 1) / 2.0},
        "sensor_to_scan": pose.tolist(),
    }


def _fixture(fixture_id: str, candidates: list[dict[str, Any]], frame_modes: list[list[str]], s_mm: float = 1.0) -> dict[str, Any]:
    if len(frame_modes) != 4 or any(len(row) != len(candidates) for row in frame_modes):
        raise RuntimeError("fixture mode shape")
    frames = [
        _frame(candidates, frame_modes[index], angle, s_mm, index)
        for index, angle in enumerate(FRAME_ANGLES_DEG)
    ]
    return {
        "kind": "v29_crco_source_free_fixture",
        "source_free": True,
        "fixture_id": fixture_id,
        "s_mm": s_mm,
        "candidate_records": candidates,
        "range_frames": frames,
    }


def build_suite() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rescue = _candidate(selected=False, boundary_risk=True, ambiguous=True)
    veto = _candidate(selected=True, boundary_risk=True, ambiguous=False)
    clean = _candidate(selected=True, boundary_risk=False, ambiguous=False)
    rejected = _candidate(selected=False, boundary_risk=False, ambiguous=False)
    fixtures: dict[str, dict[str, Any]] = {
        "SF01_OCCLUDED_90_FOLD": _fixture("SF01_OCCLUDED_90_FOLD", [rescue], [["continuous"]] * 4),
        "SF02_FREE_EDGE_CAUSAL_TWIN": _fixture("SF02_FREE_EDGE_CAUSAL_TWIN", [veto], [["jump"]] * 4),
        "SF03_60_DEG_SCALE_CROP": _fixture("SF03_60_DEG_SCALE_CROP", [rescue], [["continuous"]] * 4, 2.0),
        "SF04_FULLY_OBSERVED_CLEAN_FOLD": _fixture("SF04_FULLY_OBSERVED_CLEAN_FOLD", [clean], [["continuous"]] * 4),
        "HN01_THROUGH_HOLE_RIM": _fixture("HN01_THROUGH_HOLE_RIM", [veto], [["jump"]] * 4),
        "HN02_UNRELATED_PANEL_OCCLUSION": _fixture("HN02_UNRELATED_PANEL_OCCLUSION", [rescue], [["occluded"]] * 4),
        "HN03_FORMED_SEMANTIC_TWIN": _fixture("HN03_FORMED_SEMANTIC_TWIN", [rejected], [["unknown"]] * 4),
        "HN03_WELDED_SEMANTIC_TWIN": _fixture("HN03_WELDED_SEMANTIC_TWIN", [rejected], [["unknown"]] * 4),
        "HN04_REFLECTIVE_INVALID_DROPOUT": _fixture("HN04_REFLECTIVE_INVALID_DROPOUT", [clean], [["unknown"]] * 4),
        "HN06_CONFLICTING_VIEWS": _fixture(
            "HN06_CONFLICTING_VIEWS", [rescue], [["continuous"], ["continuous"], ["jump"], ["jump"]]
        ),
    }
    cap_candidates = [
        _candidate(y=value, selected=True, boundary_risk=True, ambiguous=False, rank=f"{index:04d}")
        for index, value in enumerate((-27.0, -9.0, 9.0, 27.0))
    ]
    fixtures["HN05_CAP_WALL_CYCLE_ALIAS"] = _fixture(
        "HN05_CAP_WALL_CYCLE_ALIAS", cap_candidates, [["jump"] * 4] * 4
    )
    visuals = {
        fixture_id: {
            "fixture_id": fixture_id,
            "analytic_source": "literal_source_free_range_lattice",
            "candidate_modes_by_frame": frame_modes,
            "description": description,
        }
        for fixture_id, frame_modes, description in [
            ("SF01_OCCLUDED_90_FOLD", ["continuous"] * 4, "occluded formed fold ownership rescue"),
            ("SF02_FREE_EDGE_CAUSAL_TWIN", ["jump"] * 4, "free-edge discontinuity veto"),
            ("SF03_60_DEG_SCALE_CROP", ["continuous"] * 4, "scale-aware formed fold rescue"),
            ("SF04_FULLY_OBSERVED_CLEAN_FOLD", ["continuous"] * 4, "legacy clean fold unchanged"),
            ("HN01_THROUGH_HOLE_RIM", ["jump"] * 4, "hole-rim negative control"),
            ("HN02_UNRELATED_PANEL_OCCLUSION", ["occluded"] * 4, "occlusion negative control"),
            ("HN03_FORMED_SEMANTIC_TWIN", ["unknown"] * 4, "formed hidden semantic twin"),
            ("HN03_WELDED_SEMANTIC_TWIN", ["unknown"] * 4, "welded hidden semantic twin"),
            ("HN04_REFLECTIVE_INVALID_DROPOUT", ["unknown"] * 4, "invalid-depth dropout"),
            ("HN05_CAP_WALL_CYCLE_ALIAS", ["jump"] * 4, "four-candidate cap-wall alias"),
            ("HN06_CONFLICTING_VIEWS", ["continuous", "continuous", "jump", "jump"], "cross-view conflict"),
        ]
    }
    return fixtures, visuals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.output_root.resolve() != OUTPUT_ROOT.resolve():
        raise SystemExit("V29 source-free fixture output root is immutable")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    fixtures, visuals = build_suite()
    expected_ids = tuple(lock["source_free_fixture_ids"])
    if set(fixtures) != set(expected_ids):
        raise SystemExit("fixture IDs disagree with immutable lock")
    manifest_rows = []
    for fixture_id in expected_ids:
        fixture_path = OUTPUT_ROOT / f"{fixture_id}.json"
        visual_path = VISUAL_SPEC_ROOT / f"{fixture_id}.json"
        _write(fixture_path, fixtures[fixture_id])
        _write(visual_path, visuals[fixture_id])
        manifest_rows.append(
            {
                "fixture_id": fixture_id,
                "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                "visual_spec_sha256": hashlib.sha256(visual_path.read_bytes()).hexdigest(),
            }
        )
    _write(
        OUTPUT_ROOT / "manifest.json",
        {
            "kind": "v29_crco_source_free_fixture_manifest",
            "source_free": True,
            "lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
            "fixtures": manifest_rows,
        },
    )
    print(json.dumps({"fixture_count": len(manifest_rows), "source_free": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
