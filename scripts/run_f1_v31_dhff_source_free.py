#!/usr/bin/env python3
"""Execute one immutable V31 DHFF source-free fixture/variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v31_dhff_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v31-dhff-v1"
FIXTURE_ROOT = OUTPUT_ROOT / "source_free_fixtures"
RESULT_ROOT = OUTPUT_ROOT / "source_free_results"
POST_VISUAL_ROOT = OUTPUT_ROOT / "postinvoke_visuals"
COUNTER_PATH = OUTPUT_ROOT / "invocation_events.jsonl"
PRECOMMIT_PATH = OUTPUT_ROOT / "precommit_manifest.json"
HARNESS_MANIFEST_PATH = OUTPUT_ROOT / "gate1_harness_manifest.json"
VISUAL_MANIFEST_PATH = OUTPUT_ROOT / "gate1_raw_visual_manifest.json"
VISUAL_SIGNOFF_PATH = OUTPUT_ROOT / "gate1_manual_visual_signoff.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_distributed_harmonic_fold_flux import (  # noqa: E402
    MAX_WORKERS,
    build_fixed_cache,
    canonical_result_bytes,
    select_unified_v31,
)
from scripts.render_f1_v31_dhff_source_free import render_post_fixture  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_ids() -> tuple[str, ...]:
    return tuple(str(value) for value in _load(LOCK_PATH)["fixture_ids"]) + ("STRESS_200K_128",)


def _require_locks() -> None:
    precommit = _load(PRECOMMIT_PATH)
    if precommit.get("passed") is not True or precommit.get("token") != "V31_DHFF_MECHANISM_LOCKED":
        raise SystemExit("V31 mechanism is not frozen")
    for row in precommit["source_hashes"] + precommit["fixture_hashes"] + precommit["raw_visual_hashes"]:
        path = ROOT / row["path"]
        if _sha(path) != row["sha256"]:
            raise SystemExit(f"V31 frozen hash drift: {row['path']}")
    visuals = _load(VISUAL_MANIFEST_PATH)
    signoff = _load(VISUAL_SIGNOFF_PATH)
    signed = (
        signoff.get("kind") == "v31_dhff_gate1_manual_visual_signoff"
        and signoff.get("all_signed") is True
        and signoff.get("decision") == "ACCEPT_RAW_GEOMETRY"
        and int(signoff.get("artifact_count", -1)) == 42
        and int(signoff.get("fixture_count", -1)) == 21
        and signoff.get("raw_visual_manifest_sha256") == _sha(VISUAL_MANIFEST_PATH)
        and len(visuals.get("rows", [])) == 42
    )
    if not signed:
        raise SystemExit("V31 raw visuals are unsigned")


def _proper_signed_permutations() -> tuple[np.ndarray, ...]:
    matrices: list[np.ndarray] = []
    import itertools

    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if round(float(np.linalg.det(matrix))) == 1:
                matrices.append(matrix)
    matrices.sort(key=lambda value: tuple(int(v) for v in value.ravel()))
    return tuple(matrices)


ROTATIONS = _proper_signed_permutations()
VARIANTS = ("base",) + tuple(f"proper_rotation_{index:02d}" for index in range(24)) + (
    "translation_17_-11_5",
    "quaternion_1_2_3_4",
    "axis_reversal",
    "parent_swap",
    "mesh_winding_and_normal_reversal",
    "input_permutation_1",
    "input_permutation_2",
    "scale_0_25",
    "scale_4",
)


def _rotation_quaternion() -> np.ndarray:
    w, x, y, z = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64) / math.sqrt(30.0)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _apply_rotation(payload: dict[str, Any], rotation: np.ndarray, translation: np.ndarray | None = None) -> None:
    shift = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    source = payload["source"]
    source["positions"] = (np.asarray(source["positions"], dtype=np.float64) @ rotation.T + shift).tolist()
    source["source_normals"] = (np.asarray(source["source_normals"], dtype=np.float64) @ rotation.T).tolist()
    for candidate in payload["candidate_records"]:
        for field in ("center_xyz", "render_line_start", "render_line_end"):
            candidate[field] = (rotation @ np.asarray(candidate[field], dtype=np.float64) + shift).tolist()
        candidate["unit_axis_xyz"] = (rotation @ np.asarray(candidate["unit_axis_xyz"], dtype=np.float64)).tolist()
        for parent_key in ("parent_A", "parent_B"):
            parent = candidate[parent_key]
            normal = rotation @ np.asarray(parent["unit_normal_xyz"], dtype=np.float64)
            parent["unit_normal_xyz"] = normal.tolist()
            parent["centroid_xyz"] = (rotation @ np.asarray(parent["centroid_xyz"], dtype=np.float64) + shift).tolist()
            parent["offset"] = float(parent["offset"] + np.dot(normal, shift))


def _scale(payload: dict[str, Any], factor: float) -> None:
    source = payload["source"]
    source["positions"] = (factor * np.asarray(source["positions"], dtype=np.float64)).tolist()
    for candidate in payload["candidate_records"]:
        for field in ("center_xyz", "render_line_start", "render_line_end"):
            candidate[field] = (factor * np.asarray(candidate[field], dtype=np.float64)).tolist()
        candidate["segment_length_L"] = factor * float(candidate["segment_length_L"])
        candidate["local_scale_s"] = factor * float(candidate["local_scale_s"])
        for parent_key in ("parent_A", "parent_B"):
            parent = candidate[parent_key]
            parent["centroid_xyz"] = (factor * np.asarray(parent["centroid_xyz"], dtype=np.float64)).tolist()
            parent["offset"] = factor * float(parent["offset"])


def _permute(payload: dict[str, Any], mode: int) -> None:
    source = payload["source"]
    count = len(source["positions"])
    if mode == 1:
        order = np.arange(count - 1, -1, -1, dtype=np.int64)
    else:
        order = np.asarray(sorted(range(count), key=lambda index: hashlib.sha256(f"v31-perm-{index}".encode()).digest()), dtype=np.int64)
    inverse = np.empty(count, dtype=np.int64)
    inverse[order] = np.arange(count, dtype=np.int64)
    for field in ("positions", "source_normals", "parent_fragment_ids", "stable_source_keys"):
        source[field] = np.asarray(source[field], dtype=object)[order].tolist()
    source["mesh_adjacencies"] = [[int(inverse[int(a)]), int(inverse[int(b)])] for a, b in source["mesh_adjacencies"]]


def transform(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    value = json.loads(json.dumps(payload))
    scale = float(value["candidate_records"][0]["local_scale_s"])
    if variant == "base":
        return value
    if variant.startswith("proper_rotation_"):
        _apply_rotation(value, ROTATIONS[int(variant.rsplit("_", 1)[1])])
    elif variant == "translation_17_-11_5":
        _apply_rotation(value, np.eye(3), scale * np.asarray([17.0, -11.0, 5.0]))
    elif variant == "quaternion_1_2_3_4":
        _apply_rotation(value, _rotation_quaternion())
    elif variant == "axis_reversal":
        for candidate in value["candidate_records"]:
            candidate["unit_axis_xyz"] = (-np.asarray(candidate["unit_axis_xyz"], dtype=np.float64)).tolist()
            candidate["signed_turn_alpha"] = -float(candidate["signed_turn_alpha"])
            candidate["render_line_start"], candidate["render_line_end"] = candidate["render_line_end"], candidate["render_line_start"]
    elif variant == "parent_swap":
        parent_ids = np.asarray(value["source"]["parent_fragment_ids"], dtype=np.uint32)
        swapped = parent_ids.copy()
        swapped[parent_ids == 1] = 2
        swapped[parent_ids == 2] = 1
        value["source"]["parent_fragment_ids"] = swapped.tolist()
        for candidate in value["candidate_records"]:
            candidate["parent_A"], candidate["parent_B"] = candidate["parent_B"], candidate["parent_A"]
            candidate["signed_turn_alpha"] = -float(candidate["signed_turn_alpha"])
    elif variant == "mesh_winding_and_normal_reversal":
        value["source"]["source_normals"] = (-np.asarray(value["source"]["source_normals"], dtype=np.float64)).tolist()
        value["source"]["mesh_adjacencies"] = [[int(b), int(a)] for a, b in reversed(value["source"]["mesh_adjacencies"])]
        for candidate in value["candidate_records"]:
            for parent_key in ("parent_A", "parent_B"):
                parent = candidate[parent_key]
                parent["unit_normal_xyz"] = (-np.asarray(parent["unit_normal_xyz"], dtype=np.float64)).tolist()
                parent["offset"] = -float(parent["offset"])
    elif variant == "input_permutation_1":
        _permute(value, 1)
    elif variant == "input_permutation_2":
        _permute(value, 2)
    elif variant == "scale_0_25":
        _scale(value, 0.25)
    elif variant == "scale_4":
        _scale(value, 4.0)
    else:
        raise ValueError(variant)
    return value


def _canonicalize_result(result: dict[str, Any], variant: str) -> dict[str, Any]:
    value = json.loads(canonical_result_bytes(result))
    rotation = np.eye(3)
    translation = np.zeros(3)
    scale = 1.0
    if variant.startswith("proper_rotation_"):
        rotation = ROTATIONS[int(variant.rsplit("_", 1)[1])]
    elif variant == "translation_17_-11_5":
        translation = np.asarray([17.0, -11.0, 5.0])
    elif variant == "quaternion_1_2_3_4":
        rotation = _rotation_quaternion()
    elif variant == "scale_0_25":
        scale = 0.25
    elif variant == "scale_4":
        scale = 4.0
    for line in value["final_lines"]:
        for field in ("line_start", "line_end"):
            point = np.asarray(line[field], dtype=np.float64)
            point = rotation.T @ (point / scale - translation)
            line[field] = [round(float(component), 10) for component in point]
    if variant == "axis_reversal":
        for decision in value["decisions"]:
            reasons = list(reversed(decision["station_reasons"]))
            decision["station_reasons"] = reasons
            mask = int(decision["station_mask"])
            decision["station_mask"] = sum(((mask >> index) & 1) << (10 - index) for index in range(11))
        for line in value["final_lines"]:
            line["line_start"], line["line_end"] = line["line_end"], line["line_start"]
    for decision in value["decisions"]:
        profile = {
            "station_mask": int(decision["station_mask"]),
            "station_reasons": decision["station_reasons"],
            "passing_stations": int(decision["passing_stations"]),
            "longest_run": int(decision["longest_run"]),
            "median_normal_backtrack": float(decision["median_normal_backtrack"]),
            "median_off_arc_fraction": float(decision["median_off_arc_fraction"]),
        }
        decision["profile_hash"] = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    fixed_cache = build_fixed_cache(value["decisions"])
    value["cache_sha256"] = hashlib.sha256(fixed_cache).hexdigest()
    value["cache_bytes"] = len(fixed_cache)
    return value


def _append_invocation(fixture_id: str, variant: str, run_tag: str, workers: int) -> None:
    event = {"event": "DHFF_INVOKE", "fixture_id": fixture_id, "variant": variant, "run_tag": run_tag, "worker_count": workers}
    COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COUNTER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(fixture_id: str, variant: str, run_tag: str, worker_mode: str, render: bool) -> dict[str, Any]:
    _require_locks()
    if fixture_id not in _fixture_ids() or variant not in VARIANTS:
        raise SystemExit("unregistered V31 fixture or variant")
    path = (FIXTURE_ROOT / f"{fixture_id}.json").resolve()
    if FIXTURE_ROOT.resolve() not in path.parents:
        raise SystemExit("fixture path escape")
    payload = _load(path)
    transformed = transform(payload, variant)
    workers = 1 if worker_mode == "one" else min(MAX_WORKERS, os.cpu_count() or 1)
    _append_invocation(fixture_id, variant, run_tag, workers)
    result = select_unified_v31(transformed["source"], transformed["candidate_records"], worker_count=workers)
    canonical = _canonicalize_result(result, variant)
    canonical_cache = build_fixed_cache(canonical["decisions"])
    output_dir = RESULT_ROOT / run_tag / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{fixture_id}.json"
    result_path.write_bytes(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n")
    cache_path = output_dir / f"{fixture_id}.cache.bin"
    cache_path.write_bytes(canonical_cache)
    if render:
        render_post_fixture(payload, result, POST_VISUAL_ROOT / fixture_id)
    return {"fixture_id": fixture_id, "variant": variant, "run_tag": run_tag, "worker_count": workers, "result_path": str(result_path.relative_to(ROOT)), "result_sha256": _sha(result_path), "cache_path": str(cache_path.relative_to(ROOT)), "cache_sha256": _sha(cache_path), "result": canonical}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, choices=_fixture_ids())
    parser.add_argument("--variant", default="base", choices=VARIANTS)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--workers", choices=("one", "max"), default="one")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.fixture, args.variant, args.run_tag, args.workers, args.render), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
