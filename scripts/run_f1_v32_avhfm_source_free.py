#!/usr/bin/env python3
"""Execute one frozen V32 AVHFM source-free fixture/variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v32_avhfm_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v32-avhfm-v1"
FIXTURE_ROOT = OUTPUT_ROOT / "source_free_fixtures"
RESULT_ROOT = OUTPUT_ROOT / "source_free_results"
POST_VISUAL_ROOT = OUTPUT_ROOT / "postinvoke_visuals"
COUNTER_PATH = OUTPUT_ROOT / "invocation_events.jsonl"
PRECOMMIT_PATH = OUTPUT_ROOT / "precommit_manifest.json"
VISUAL_MANIFEST_PATH = OUTPUT_ROOT / "preinvoke_visual_manifest.json"
VISUAL_SIGNOFF_PATH = OUTPUT_ROOT / "preinvoke_manual_visual_signoff.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_axial_voronoi_harmonic_fold_measure import (  # noqa: E402
    MAX_WORKERS,
    build_fixed_cache,
    canonical_result_bytes,
    select_unified_v32,
)
from scripts.render_f1_v32_avhfm_source_free import render_post_fixture  # noqa: E402


VARIANTS = (
    "base",
    "source_point_permutation_1",
    "source_point_permutation_2",
    "source_point_permutation_3",
    "mesh_face_permutation",
    "edge_input_permutation",
    "candidate_pool_permutation",
    "axis_reversal",
    "tail_parent_swap",
    "global_normal_reversal",
    "rotation_x_90",
    "rotation_y_90",
    "rotation_z_90",
    "rotation_111_120",
    "translation_7_-11_13",
    "translation_-17_5_3",
    "scale_0_5",
    "scale_2",
    "scale_4",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _fixture_ids() -> tuple[str, ...]:
    return tuple(str(value) for value in _load(LOCK_PATH)["physical_fixture_ids"]) + ("STRESS_200K_128",)


def _require_locks() -> None:
    precommit = _load(PRECOMMIT_PATH)
    if precommit.get("passed") is not True or precommit.get("token") != "V32_AVHFM_MECHANISM_LOCKED":
        raise SystemExit("V32 mechanism is not frozen")
    for row in precommit["source_hashes"] + precommit["fixture_hashes"] + precommit["preinvoke_visual_hashes"]:
        path = ROOT / row["path"]
        if _sha(path) != row["sha256"]:
            raise SystemExit(f"V32 frozen hash drift: {row['path']}")
    visual = _load(VISUAL_MANIFEST_PATH)
    signoff = _load(VISUAL_SIGNOFF_PATH)
    signed = (
        signoff.get("kind") == "v32_avhfm_preinvoke_manual_visual_signoff"
        and signoff.get("all_signed") is True
        and signoff.get("decision") == "ACCEPT_ALL_PREINVOKE_SOURCE_GEOMETRY"
        and int(signoff.get("artifact_count", -1)) == 62
        and int(signoff.get("physical_fixture_count", -1)) == 27
        and int(signoff.get("stress_page_count", -1)) == 8
        and signoff.get("visual_manifest_sha256") == _sha(VISUAL_MANIFEST_PATH)
        and len(visual.get("rows", [])) == 62
        and len(signoff.get("rows", [])) == 62
    )
    if not signed:
        raise SystemExit("V32 pre-invocation visuals are unsigned")


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    angle = math.radians(degrees)
    c, s, q = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray([[c+x*x*q,x*y*q-z*s,x*z*q+y*s],[y*x*q+z*s,c+y*y*q,y*z*q-x*s],[z*x*q-y*s,z*y*q+x*s,c+z*z*q]], dtype=np.float64)


ROTATIONS = {
    "rotation_x_90": _rotation(np.asarray([1.0, 0.0, 0.0]), 90.0),
    "rotation_y_90": _rotation(np.asarray([0.0, 1.0, 0.0]), 90.0),
    "rotation_z_90": _rotation(np.asarray([0.0, 0.0, 1.0]), 90.0),
    "rotation_111_120": _rotation(np.asarray([1.0, 1.0, 1.0]), 120.0),
}


def _apply_rigid(payload: dict[str, Any], rotation: np.ndarray, translation: np.ndarray) -> None:
    source = payload["source"]
    source["positions"] = (np.asarray(source["positions"], dtype=np.float64) @ rotation.T + translation).tolist()
    source["source_normals"] = (np.asarray(source["source_normals"], dtype=np.float64) @ rotation.T).tolist()
    for candidate in payload["candidate_records"]:
        for field in ("center_xyz", "render_line_start", "render_line_end"):
            candidate[field] = (rotation @ np.asarray(candidate[field], dtype=np.float64) + translation).tolist()
        candidate["unit_axis_xyz"] = (rotation @ np.asarray(candidate["unit_axis_xyz"], dtype=np.float64)).tolist()
        for parent_key in ("parent_A", "parent_B"):
            parent = candidate[parent_key]
            normal = rotation @ np.asarray(parent["unit_normal_xyz"], dtype=np.float64)
            parent["unit_normal_xyz"] = normal.tolist()
            parent["centroid_xyz"] = (rotation @ np.asarray(parent["centroid_xyz"], dtype=np.float64) + translation).tolist()
            parent["offset"] = float(parent["offset"] + np.dot(normal, translation))


def _scale(payload: dict[str, Any], factor: float) -> None:
    payload["source"]["positions"] = (factor * np.asarray(payload["source"]["positions"], dtype=np.float64)).tolist()
    for candidate in payload["candidate_records"]:
        for field in ("center_xyz", "render_line_start", "render_line_end"):
            candidate[field] = (factor * np.asarray(candidate[field], dtype=np.float64)).tolist()
        candidate["segment_length_L"] = factor * float(candidate["segment_length_L"])
        candidate["local_scale_s"] = factor * float(candidate["local_scale_s"])
        for parent_key in ("parent_A", "parent_B"):
            parent = candidate[parent_key]
            parent["centroid_xyz"] = (factor * np.asarray(parent["centroid_xyz"], dtype=np.float64)).tolist()
            parent["offset"] = factor * float(parent["offset"])


def _permute_source(payload: dict[str, Any], mode: int) -> None:
    source = payload["source"]
    count = len(source["positions"])
    order = np.asarray(sorted(range(count), key=lambda index: hashlib.sha256(f"v32-source-permutation-{mode}-{index}".encode("ascii")).digest()), dtype=np.int64)
    inverse = np.empty(count, dtype=np.int64)
    inverse[order] = np.arange(count, dtype=np.int64)
    for field in ("positions", "source_normals", "parent_fragment_ids", "stable_source_keys"):
        source[field] = np.asarray(source[field], dtype=object)[order].tolist()
    source["mesh_adjacencies"] = [[int(inverse[int(a)]), int(inverse[int(b)])] for a, b in source["mesh_adjacencies"]]


def transform(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    value = json.loads(json.dumps(payload))
    if variant == "base":
        return value
    if variant.startswith("source_point_permutation_"):
        _permute_source(value, int(variant.rsplit("_", 1)[1]))
    elif variant == "mesh_face_permutation":
        if "mesh_faces" in value["source"]:
            value["source"]["mesh_faces"] = list(reversed(value["source"]["mesh_faces"]))
    elif variant == "edge_input_permutation":
        value["source"]["mesh_adjacencies"] = [[int(b), int(a)] for a, b in reversed(value["source"]["mesh_adjacencies"])]
    elif variant == "candidate_pool_permutation":
        value["candidate_records"] = list(reversed(value["candidate_records"]))
    elif variant == "axis_reversal":
        for candidate in value["candidate_records"]:
            candidate["unit_axis_xyz"] = (-np.asarray(candidate["unit_axis_xyz"], dtype=np.float64)).tolist()
            candidate["signed_turn_alpha"] = -float(candidate["signed_turn_alpha"])
            candidate["render_line_start"], candidate["render_line_end"] = candidate["render_line_end"], candidate["render_line_start"]
    elif variant == "tail_parent_swap":
        pairs = {(int(candidate["parent_A"]["parent_id"]), int(candidate["parent_B"]["parent_id"])) for candidate in value["candidate_records"]}
        if len(pairs) != 1:
            raise ValueError("tail-parent swap requires one frozen parent pair")
        first, second = next(iter(pairs))
        ids = np.asarray(value["source"]["parent_fragment_ids"], dtype=np.uint32)
        swapped = ids.copy(); swapped[ids == first] = second; swapped[ids == second] = first
        value["source"]["parent_fragment_ids"] = swapped.tolist()
        for candidate in value["candidate_records"]:
            candidate["parent_A"], candidate["parent_B"] = candidate["parent_B"], candidate["parent_A"]
            candidate["signed_turn_alpha"] = -float(candidate["signed_turn_alpha"])
    elif variant == "global_normal_reversal":
        value["source"]["source_normals"] = (-np.asarray(value["source"]["source_normals"], dtype=np.float64)).tolist()
        for candidate in value["candidate_records"]:
            for parent_key in ("parent_A", "parent_B"):
                parent = candidate[parent_key]
                parent["unit_normal_xyz"] = (-np.asarray(parent["unit_normal_xyz"], dtype=np.float64)).tolist()
                parent["offset"] = -float(parent["offset"])
    elif variant in ROTATIONS:
        _apply_rigid(value, ROTATIONS[variant], np.zeros(3, dtype=np.float64))
    elif variant == "translation_7_-11_13":
        scale = float(value["candidate_records"][0]["local_scale_s"])
        _apply_rigid(value, np.eye(3), scale * np.asarray([7.0, -11.0, 13.0]))
    elif variant == "translation_-17_5_3":
        scale = float(value["candidate_records"][0]["local_scale_s"])
        _apply_rigid(value, np.eye(3), scale * np.asarray([-17.0, 5.0, 3.0]))
    elif variant.startswith("scale_"):
        _scale(value, {"scale_0_5": 0.5, "scale_2": 2.0, "scale_4": 4.0}[variant])
    else:
        raise ValueError(variant)
    return value


def _canonicalize_result(result: dict[str, Any], base_payload: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(canonical_result_bytes(result))
    base_by_id = {int(candidate["candidate_id"]): candidate for candidate in base_payload["candidate_records"]}
    for decision in value["decisions"]:
        base = base_by_id[int(decision["candidate_id"])]
        decision["candidate_sha256"] = hashlib.sha256(_canonical(base)).hexdigest()
        decision["anchor_xyz"] = base["center_xyz"]
        decision["axis_xyz"] = base["unit_axis_xyz"]
        decision["line_start_xyz"] = base["render_line_start"]
        decision["line_end_xyz"] = base["render_line_end"]
        decision["scale"] = float(base["local_scale_s"])
    for line in value["final_lines"]:
        base = base_by_id[int(line["candidate_id"])]
        line["line_start"] = base["render_line_start"]
        line["line_end"] = base["render_line_end"]
    value.pop("cache_sha256", None)
    value.pop("cache_bytes", None)
    return value


def _append_invocation(fixture_id: str, variant: str, run_tag: str, workers: int) -> None:
    event = {"event": "AVHFM_INVOKE", "fixture_id": fixture_id, "variant": variant, "run_tag": run_tag, "worker_count": workers}
    COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COUNTER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(fixture_id: str, variant: str, run_tag: str, worker_mode: str, render: bool) -> dict[str, Any]:
    _require_locks()
    if fixture_id not in _fixture_ids() or variant not in VARIANTS:
        raise SystemExit("unregistered V32 fixture or variant")
    path = (FIXTURE_ROOT / f"{fixture_id}.json").resolve()
    if FIXTURE_ROOT.resolve() not in path.parents:
        raise SystemExit("fixture path escape")
    base_payload = _load(path)
    transformed = transform(base_payload, variant)
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    workers = 1 if worker_mode == "one" else min(MAX_WORKERS, affinity, max(1, len(transformed["candidate_records"])))
    _append_invocation(fixture_id, variant, run_tag, workers)
    result = select_unified_v32(transformed["source"], transformed["candidate_records"], worker_count=workers)
    canonical = _canonicalize_result(result, base_payload)
    semantic_result_sha = hashlib.sha256(_canonical(canonical) + b"\n").hexdigest()
    parent_sha = _load(LOCK_PATH)["parent_lock_canonical_sha256"]
    cache = build_fixed_cache(
        canonical["decisions"],
        source_count=len(transformed["source"]["positions"]),
        source_sha256=hashlib.sha256(_canonical(transformed["source"])).hexdigest(),
        candidate_sha256=hashlib.sha256(_canonical(transformed["candidate_records"])).hexdigest(),
        parent_sha256=parent_sha,
        lock_sha256=_sha(LOCK_PATH),
        result_sha256=semantic_result_sha,
    )
    canonical["semantic_result_sha256"] = semantic_result_sha
    canonical["cache_bytes"] = len(cache)
    canonical["cache_sha256"] = hashlib.sha256(cache).hexdigest()
    output_dir = RESULT_ROOT / run_tag / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{fixture_id}.json"
    result_path.write_bytes(_canonical(canonical) + b"\n")
    cache_path = output_dir / f"{fixture_id}.cache.bin"
    cache_path.write_bytes(cache)
    if render:
        render_post_fixture(base_payload, result, POST_VISUAL_ROOT / fixture_id)
    return {"fixture_id": fixture_id, "variant": variant, "run_tag": run_tag, "worker_count": workers, "result_path": str(result_path.relative_to(ROOT)), "result_sha256": _sha(result_path), "cache_path": str(cache_path.relative_to(ROOT)), "cache_sha256": _sha(cache_path), "semantic_result_sha256": semantic_result_sha, "result": canonical}


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
