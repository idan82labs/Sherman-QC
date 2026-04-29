#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from backend.bend_inspection_pipeline import load_bend_runtime_config
from backend.zero_shot_bend_profile import (
    PERTURBATION_VOXELS_MM,
    _extract_scan_features,
    _load_scan_geometry,
    _run_candidate_generators,
    _score_policies,
    _select_zero_shot_result,
    _voxel_downsample,
)
from feature_detection.bend_detector import BendDetector


RecipeConfig = Dict[str, Any]

RECIPE_CONFIGS: Dict[str, RecipeConfig] = {
    "current_baseline": {
        "mode": "baseline",
        "description": "Current runtime behavior: external voxel downsample, detector preprocess disabled.",
    },
    "detector_preprocess_default": {
        "mode": "detector_preprocess",
        "description": "After external voxel downsample, run detector preprocessing with default settings.",
        "voxel_size_scale": 1.0,
        "voxel_size_floor": 0.3,
        "outlier_nb_neighbors": 20,
        "outlier_std_ratio": 2.0,
    },
    "detector_preprocess_light": {
        "mode": "detector_preprocess",
        "description": "After external voxel downsample, run lighter detector preprocessing.",
        "voxel_size_scale": 0.5,
        "voxel_size_floor": 0.3,
        "outlier_nb_neighbors": 12,
        "outlier_std_ratio": 3.0,
    },
    "outlier_only_after_voxel": {
        "mode": "custom_points",
        "description": "After external voxel downsample, apply only statistical outlier removal and keep preprocess disabled.",
        "outlier_nb_neighbors": 12,
        "outlier_std_ratio": 3.0,
        "recompute_normals": False,
    },
    "outlier_and_renorm_after_voxel": {
        "mode": "custom_points",
        "description": "After external voxel downsample, apply outlier removal, recompute normals, keep preprocess disabled.",
        "outlier_nb_neighbors": 12,
        "outlier_std_ratio": 3.0,
        "recompute_normals": True,
    },
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _remove_statistical_outliers(
    points: Any,
    normals: Any,
    *,
    nb_neighbors: int,
    std_ratio: float,
) -> Tuple[Any, Any]:
    import open3d as o3d
    import numpy as np

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None and len(normals) == len(points):
        pcd.normals = o3d.utility.Vector3dVector(normals)
    filtered, indices = pcd.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    filtered_points = np.asarray(filtered.points, dtype=np.float64)
    filtered_normals = None
    if normals is not None and len(indices):
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx):
            filtered_normals = np.asarray(normals, dtype=np.float64)[idx]
    if filtered.has_normals():
        filtered_normals = np.asarray(filtered.normals, dtype=np.float64)
    return filtered_points, filtered_normals


def _recompute_normals(points: Any, *, radius_mm: float, max_nn: int = 30) -> Any:
    import open3d as o3d
    import numpy as np

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius_mm), max_nn=int(max_nn))
    )
    return np.asarray(pcd.normals, dtype=np.float64)


def _build_probe_candidate_bundles(
    detector: BendDetector,
    points: Any,
    normals: Any,
    base_detect_kwargs: Dict[str, Any],
    recipe: RecipeConfig,
    original_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    bundles: Dict[str, Dict[str, Any]] = {}
    bundles["original"] = dict(original_bundle) if original_bundle is not None else _run_candidate_generators(
        detector,
        points,
        normals,
        dict(base_detect_kwargs),
    )
    for voxel_size in PERTURBATION_VOXELS_MM:
        ds_points, ds_normals = _voxel_downsample(points, normals, voxel_size)
        mode = str(recipe["mode"])
        if mode == "baseline":
            variant_kwargs = {"preprocess": False}
            bundles[f"voxel_{voxel_size:g}"] = _run_candidate_generators(detector, ds_points, ds_normals, variant_kwargs)
            continue
        if mode == "detector_preprocess":
            variant_kwargs = {
                "preprocess": True,
                "voxel_size": max(float(recipe["voxel_size_floor"]), float(voxel_size) * float(recipe["voxel_size_scale"])),
                "outlier_nb_neighbors": int(recipe["outlier_nb_neighbors"]),
                "outlier_std_ratio": float(recipe["outlier_std_ratio"]),
            }
            bundles[f"voxel_{voxel_size:g}"] = _run_candidate_generators(detector, ds_points, ds_normals, variant_kwargs)
            continue
        if mode == "custom_points":
            filtered_points, filtered_normals = _remove_statistical_outliers(
                ds_points,
                ds_normals,
                nb_neighbors=int(recipe["outlier_nb_neighbors"]),
                std_ratio=float(recipe["outlier_std_ratio"]),
            )
            if bool(recipe.get("recompute_normals")):
                filtered_normals = _recompute_normals(
                    filtered_points,
                    radius_mm=max(float(voxel_size) * 3.0, 2.0),
                )
            bundles[f"voxel_{voxel_size:g}"] = _run_candidate_generators(
                detector,
                filtered_points,
                filtered_normals,
                {"preprocess": False},
            )
            continue
        raise ValueError(f"Unsupported recipe mode: {mode}")
    return bundles


def _policy_snapshot(result: Dict[str, Any], policy_name: str) -> Dict[str, Any]:
    payload = dict((result.get("policy_scores") or {}).get(policy_name) or {})
    support_breakdown = dict(payload.get("support_breakdown") or {})
    perturbation = dict(support_breakdown.get("perturbation") or {})
    return {
        "score": payload.get("score"),
        "estimated_bend_count": payload.get("estimated_bend_count"),
        "estimated_bend_count_range": payload.get("estimated_bend_count_range"),
        "exact_count_eligible": payload.get("exact_count_eligible"),
        "identity_stability": perturbation.get("identity_stability"),
        "matched_reference_fraction_mean": perturbation.get("matched_reference_fraction_mean"),
        "modal_count_support": perturbation.get("modal_count_support"),
        "counts": perturbation.get("counts"),
        "reference_strip_count": perturbation.get("reference_strip_count"),
    }


def score_probe_result(result: Dict[str, Any], expected_bend_count: Optional[int]) -> Dict[str, Any]:
    count_range = result.get("estimated_bend_count_range") or []
    exact_count = result.get("estimated_bend_count")
    abstained = bool(result.get("abstained"))
    width = None
    range_contains_truth = None
    exact_match = None
    if len(count_range) == 2:
        width = int(count_range[1]) - int(count_range[0])
    if expected_bend_count is not None and len(count_range) == 2:
        range_contains_truth = int(int(count_range[0]) <= int(expected_bend_count) <= int(count_range[1]))
    if expected_bend_count is not None and exact_count is not None and not abstained:
        exact_match = int(int(exact_count) == int(expected_bend_count))
    score = 0
    if range_contains_truth is not None:
        score += 100 if range_contains_truth else -200
    if exact_match:
        score += 50
    if width is not None:
        score -= int(width) * 5
    if abstained:
        score -= 10
    return {
        "score": score,
        "exact_match": exact_match,
        "range_contains_truth": range_contains_truth,
        "range_width": width,
    }


def run_probe(
    slice_json: Path,
    *,
    part_keys: Optional[List[str]] = None,
    recipe_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    slice_payload = _load_json(slice_json)
    runtime_config = load_bend_runtime_config(None)
    detector_kwargs = dict(getattr(runtime_config, "scan_detector_kwargs", {}) or {})
    base_detect_kwargs = dict(getattr(runtime_config, "scan_detect_call_kwargs", {}) or {})
    allowed_part_keys = {value.strip() for value in (part_keys or []) if str(value).strip()}
    allowed_recipe_names = {value.strip() for value in (recipe_names or []) if str(value).strip()}

    scans: List[Dict[str, Any]] = []
    aggregate_by_recipe: Dict[str, Dict[str, Any]] = {}

    for entry in list(slice_payload.get("entries") or []):
        scan_path = str(entry["scan_path"])
        part_key = entry.get("part_key")
        if allowed_part_keys and str(part_key) not in allowed_part_keys:
            continue
        expectation = dict(entry.get("expectation") or {})
        expected_count = expectation.get("expected_bend_count")
        points, normals = _load_scan_geometry(scan_path)
        recipe_rows: List[Dict[str, Any]] = []

        for recipe_name, recipe in RECIPE_CONFIGS.items():
            if allowed_recipe_names and recipe_name not in allowed_recipe_names:
                continue
            detector = BendDetector(**detector_kwargs)
            original_bundle = _run_candidate_generators(detector, points, normals, dict(base_detect_kwargs))
            bundles = _build_probe_candidate_bundles(
                detector=detector,
                points=points,
                normals=normals,
                base_detect_kwargs=base_detect_kwargs,
                recipe=recipe,
                original_bundle=original_bundle,
            )
            scan_features = _extract_scan_features(
                points=points,
                candidate_bundles=bundles,
                scan_path=scan_path,
                part_id=part_key,
            )
            policy_scores = _score_policies(scan_features, bundles)
            probe_result = _select_zero_shot_result(
                scan_path=scan_path,
                part_id=part_key,
                policy_scores=policy_scores,
                scan_features=scan_features,
                candidate_bundles=bundles,
            ).to_dict()
            score_summary = score_probe_result(probe_result, expected_count)
            row = {
                "recipe_name": recipe_name,
                "recipe_description": recipe["description"],
                "selected_profile_family": probe_result.get("profile_family"),
                "selected_exact_count": probe_result.get("estimated_bend_count"),
                "selected_count_range": probe_result.get("estimated_bend_count_range"),
                "abstained": probe_result.get("abstained"),
                "abstain_reasons": probe_result.get("abstain_reasons"),
                "hypothesis_confidence": probe_result.get("hypothesis_confidence"),
                "candidate_generator_summary": probe_result.get("candidate_generator_summary"),
                "support_breakdown": probe_result.get("support_breakdown"),
                "score_summary": score_summary,
                "policy_snapshots": {
                    name: _policy_snapshot(probe_result, name)
                    for name in (
                        "conservative_macro",
                        "orthogonal_dense_sheetmetal",
                        "repeated_90_sheetmetal",
                        "rolled_or_mixed_transition",
                    )
                },
            }
            recipe_rows.append(row)

            agg = aggregate_by_recipe.setdefault(
                recipe_name,
                {
                    "score_total": 0,
                    "scan_count": 0,
                    "exact_match_count": 0,
                    "range_contains_truth_count": 0,
                    "abstained_count": 0,
                },
            )
            agg["score_total"] += int(score_summary["score"])
            agg["scan_count"] += 1
            agg["exact_match_count"] += int(score_summary["exact_match"] or 0)
            agg["range_contains_truth_count"] += int(score_summary["range_contains_truth"] or 0)
            agg["abstained_count"] += int(bool(row["abstained"]))

        recipe_rows.sort(key=lambda item: (-int(item["score_summary"]["score"]), item["recipe_name"]))
        scans.append(
            {
                "part_key": part_key,
                "scan_name": Path(scan_path).name,
                "scan_path": scan_path,
                "expected_bend_count": expected_count,
                "recipes": recipe_rows,
                "best_recipe": recipe_rows[0]["recipe_name"] if recipe_rows else None,
            }
        )

    aggregate = {
        name: {
            **payload,
            "mean_score": round(float(payload["score_total"]) / max(1, int(payload["scan_count"])), 3),
        }
        for name, payload in aggregate_by_recipe.items()
    }
    return {
        "slice_json": str(slice_json),
        "recipes": RECIPE_CONFIGS,
        "aggregate_by_recipe": aggregate,
        "scans": scans,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe repeated near-90 detector/preprocess recipes outside the runtime path.")
    parser.add_argument("--slice-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--part-key", action="append", default=None)
    parser.add_argument("--recipe", action="append", default=None)
    args = parser.parse_args()

    payload = run_probe(
        Path(args.slice_json),
        part_keys=args.part_key,
        recipe_names=args.recipe,
    )
    _write_json(Path(args.output_json), payload)
    print(json.dumps({"output_json": args.output_json, "scan_count": len(payload["scans"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
