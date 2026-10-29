#!/usr/bin/env python3
"""Execute one fixed V30 AMFC source-free fixture/variant.

There is no arbitrary input-path CLI.  The worker can only open the immutable
fixture IDs and preregistered transformations beneath the fixed V30 tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v30_amfc_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v30-amfc-v1"
FIXTURE_ROOT = OUTPUT_ROOT / "source_free_fixtures"
RESULT_ROOT = OUTPUT_ROOT / "source_free_results"
COUNTER_PATH = OUTPUT_ROOT / "invocation_events.jsonl"
PRECOMMIT_PATH = OUTPUT_ROOT / "precommit_manifest.json"
HARNESS_MANIFEST_PATH = OUTPUT_ROOT / "gate1_harness_manifest.json"
VISUAL_MANIFEST_PATH = OUTPUT_ROOT / "gate1_raw_visual_manifest.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_axial_multipath_fold_certificate import (  # noqa: E402
    MAX_WORKERS,
    canonical_result_bytes,
    select_unified_v30,
)
from scripts.render_f1_v30_amfc_source_free import render_diagnostic_fixture  # noqa: E402


VARIANTS = (
    "base",
    "candidate_axis_reversal",
    "parent_plane_swap",
    "point_or_vertex_permutation",
    "face_permutation_and_winding_reversal",
    "rigid_euler_37_-23_61_translation",
    "uniform_scale_0_5",
    "uniform_scale_2_0",
    "composite_parent_swap_axis_reversal_permutation_rigid_scale_2",
)
RUN_TAGS = (
    "base",
    "metamorphic",
    "single_replay_2",
    "single_replay_3",
    "max_replay_1",
    "max_replay_2",
    "max_replay_3",
    "stress_1",
    "stress_2",
    "stress_3",
    "stress_4",
    "stress_5",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_ids() -> tuple[str, ...]:
    return tuple(str(value) for value in _load(LOCK_PATH)["fixture_ids"])


def _require_locks() -> None:
    precommit = _load(PRECOMMIT_PATH)
    if precommit.get("passed") is not True or precommit.get("token") != "V30_AMFC_MECHANISM_LOCKED":
        raise SystemExit("V30 mechanism is not locked")
    for row in precommit["source_hashes"]:
        path = ROOT / row["path"]
        if _sha(path) != row["sha256"]:
            raise SystemExit(f"V30 source hash drift: {row['path']}")
    harness = _load(HARNESS_MANIFEST_PATH)
    visuals = _load(VISUAL_MANIFEST_PATH)
    if harness.get("raw_visuals_signed") is not True or visuals.get("all_signed") is not True:
        raise SystemExit("Gate 1 raw visuals are unsigned")


def _rotation() -> np.ndarray:
    rx, ry, rz = (math.radians(value) for value in (37.0, -23.0, 61.0))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return z @ y @ x


def _permute_points(payload: dict[str, Any]) -> None:
    points = np.asarray(payload["source"]["coordinates"], dtype=np.float64)
    triangles = np.asarray(payload["source"]["triangles"], dtype=np.int64)
    if triangles.size == 0:
        triangles = np.empty((0, 3), dtype=np.int64)
    order = np.arange(len(points) - 1, -1, -1, dtype=np.int64)
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    payload["source"]["coordinates"] = points[order].tolist()
    payload["source"]["triangles"] = inverse[triangles].tolist() if len(triangles) else []


def _axis_reverse(payload: dict[str, Any]) -> None:
    for candidate in payload["candidate_records"]:
        candidate["line_axis"] = (-np.asarray(candidate["line_axis"], dtype=np.float64)).tolist()
        lower, upper = (float(value) for value in candidate["support_interval"])
        candidate["support_interval"] = [-upper, -lower]
        candidate["line_start"], candidate["line_end"] = candidate["line_end"], candidate["line_start"]


def _parent_swap(payload: dict[str, Any]) -> None:
    for candidate in payload["candidate_records"]:
        candidate["plane_normals"].reverse()
        candidate["plane_offsets"].reverse()
        candidate["parent_support_centroids"].reverse()


def _face_reverse(payload: dict[str, Any]) -> None:
    triangles = list(reversed(payload["source"]["triangles"]))
    payload["source"]["triangles"] = [[value[0], value[2], value[1]] for value in triangles]


def _rigid(payload: dict[str, Any]) -> None:
    rotation = _rotation()
    s = float(payload["s_mm"])
    translation = np.asarray([13.0 * s, -7.0 * s, 5.0 * s])
    points = np.asarray(payload["source"]["coordinates"], dtype=np.float64)
    payload["source"]["coordinates"] = (points @ rotation.T + translation).tolist()
    for candidate in payload["candidate_records"]:
        for field in ("line_center", "line_start", "line_end"):
            candidate[field] = (rotation @ np.asarray(candidate[field], dtype=np.float64) + translation).tolist()
        candidate["line_axis"] = (rotation @ np.asarray(candidate["line_axis"], dtype=np.float64)).tolist()
        candidate["parent_support_centroids"] = [(rotation @ np.asarray(value, dtype=np.float64) + translation).tolist() for value in candidate["parent_support_centroids"]]
        old_normals = [np.asarray(value, dtype=np.float64) for value in candidate["plane_normals"]]
        new_normals = [rotation @ value for value in old_normals]
        candidate["plane_normals"] = [value.tolist() for value in new_normals]
        candidate["plane_offsets"] = [float(old_offset) + float(np.dot(new_normal, translation)) for old_offset, new_normal in zip(candidate["plane_offsets"], new_normals)]


def _scale(payload: dict[str, Any], factor: float) -> None:
    points = np.asarray(payload["source"]["coordinates"], dtype=np.float64)
    payload["source"]["coordinates"] = (factor * points).tolist()
    payload["s_mm"] = factor * float(payload["s_mm"])
    for candidate in payload["candidate_records"]:
        for field in ("line_center", "line_start", "line_end"):
            candidate[field] = (factor * np.asarray(candidate[field], dtype=np.float64)).tolist()
        candidate["parent_support_centroids"] = [(factor * np.asarray(value, dtype=np.float64)).tolist() for value in candidate["parent_support_centroids"]]
        candidate["plane_offsets"] = [factor * float(value) for value in candidate["plane_offsets"]]
        candidate["support_interval"] = [factor * float(value) for value in candidate["support_interval"]]


def transform(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    value = json.loads(json.dumps(payload))
    if variant == "base":
        return value
    if variant == "candidate_axis_reversal":
        _axis_reverse(value)
    elif variant == "parent_plane_swap":
        _parent_swap(value)
    elif variant == "point_or_vertex_permutation":
        _permute_points(value)
    elif variant == "face_permutation_and_winding_reversal":
        _face_reverse(value)
    elif variant == "rigid_euler_37_-23_61_translation":
        _rigid(value)
    elif variant == "uniform_scale_0_5":
        _scale(value, 0.5)
    elif variant == "uniform_scale_2_0":
        _scale(value, 2.0)
    elif variant == "composite_parent_swap_axis_reversal_permutation_rigid_scale_2":
        _parent_swap(value)
        _axis_reverse(value)
        _permute_points(value)
        _rigid(value)
        _scale(value, 2.0)
    else:
        raise ValueError(variant)
    return value


def _append_invocation(fixture_id: str, variant: str, run_tag: str, worker_count: int) -> None:
    event = {
        "event": "AMFC_INVOKE",
        "fixture_id": fixture_id,
        "variant": variant,
        "run_tag": run_tag,
        "worker_count": worker_count,
    }
    COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COUNTER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(fixture_id: str, variant: str, run_tag: str, worker_mode: str, render: bool) -> dict[str, Any]:
    _require_locks()
    path = (FIXTURE_ROOT / f"{fixture_id}.json").resolve()
    if FIXTURE_ROOT.resolve() not in path.parents or fixture_id not in _fixture_ids():
        raise SystemExit("source-free fixture path escape")
    payload = _load(path)
    transformed = transform(payload, variant)
    worker_count = 1 if worker_mode == "one" else min(int(os.cpu_count() or 1), MAX_WORKERS)
    _append_invocation(fixture_id, variant, run_tag, worker_count)
    started = time.perf_counter()
    result = select_unified_v30(
        transformed["source"],
        transformed["candidate_records"],
        float(transformed["s_mm"]),
        worker_count=worker_count,
    )
    elapsed = time.perf_counter() - started
    output_dir = RESULT_ROOT / run_tag / fixture_id / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_path = output_dir / "diagnostic.png"
    if render:
        render_diagnostic_fixture(transformed, result, visual_path, "V30 AMFC DIAGNOSTIC")
    serialized = {key: value for key, value in result.items() if key != "_ephemeral"}
    result_path = output_dir / "result.json"
    cache_path = output_dir / "cache.json"
    result_path.write_bytes(canonical_result_bytes(serialized))
    cache_path.write_text(json.dumps(serialized["cache"], sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    metrics = {
        "fixture_id": fixture_id,
        "variant": variant,
        "run_tag": run_tag,
        "worker_count": worker_count,
        "elapsed_seconds": round(elapsed, 9),
        "process_tree_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "result_sha256": _sha(result_path),
        "cache_sha256": _sha(cache_path),
        "visual_sha256": _sha(visual_path) if visual_path.exists() else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-id", required=True, choices=_fixture_ids())
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--run-tag", required=True, choices=RUN_TAGS)
    parser.add_argument("--worker-mode", required=True, choices=("one", "max"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    metrics = run(args.fixture_id, args.variant, args.run_tag, args.worker_mode, args.render)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
