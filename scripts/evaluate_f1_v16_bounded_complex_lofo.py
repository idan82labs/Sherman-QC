#!/usr/bin/env python3
"""Evaluate V16 with the bounded-complex learner detuned from each part family.

V16's full-resolution and bounded-small routes do not consume V9 scores.  This
script leaves those frozen routes unchanged and retrains only the V9 estimator
for each held-out part family.  It therefore measures the generalization of
the learned bounded-complex lane without pretending that deterministic V16
geometry/route code was learned from the held-out family.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import FEATURE_NAMES, endpoints
from backend.f1_short_mesh_bend_completion import complete_short_scan_mesh_bends
from backend.f1_source_balanced_representative_snap import (
    snap_source_balanced_representatives,
)
from backend.f1_collinear_fragment_stitch import stitch_collinear_selected_fragments
from backend.f1_paired_return_completion import complete_paired_return_axis
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, match_lines, pair_metrics
from scripts.select_f1_mesh_route_v16 import ROUTE_COMPLEX, select_unified
from scripts.train_f1_bounded_mesh_router_v9 import (
    NMS_ANGLE,
    NMS_DISTANCE,
    NMS_OVERLAP,
    PROFILE as V9_PROFILE,
    SCORE_THRESHOLD,
    SMALL_POOL_MAXIMUM,
    _load_cases,
    _model,
)


PROFILE = "F1_V16_BOUNDED_COMPLEX_LEAVE_ONE_FAMILY_OUT_V1"
ESTIMATOR_EXTRA_TREES = "extra_trees"
ESTIMATOR_FAMILY_CLASS_BALANCED_EXTRA_TREES = "family_class_balanced_extra_trees"
ESTIMATOR_REGULARIZED_LOGISTIC = "regularized_logistic"
SHORT_MESH_MAXIMUM_SPAN_MM = 16.0
SHORT_MESH_MINIMUM_NORMAL_TURN_DEG = 60.0
SHORT_MESH_MINIMUM_FINITE_SUPPORT_FRACTION = 0.80
SHORT_MESH_DUPLICATE_OFFSET_MM = 4.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predictions(rows: list[dict]) -> list[dict]:
    return [
        {
            "bend_id": str(index),
            "line_start": endpoints(row)[0].tolist(),
            "line_end": endpoints(row)[1].tolist(),
            "source_hypothesis_id": str(row.get("candidate_id") or ""),
            "representative_source": row.get("representative_source"),
            "source_kind": row.get("source_kind"),
            "span_mm": row.get("span_mm"),
            "finite_support_fraction": row.get("finite_support_fraction"),
            "finite_support_longest_run_fraction": row.get(
                "finite_support_longest_run_fraction"
            ),
            "endpoint_support_fraction": row.get("endpoint_support_fraction"),
            "p90_line_distance_mm": row.get("p90_line_distance_mm"),
            "two_sided_transition_score": row.get("two_sided_transition_score"),
            "two_sided_normal_angle_deg": row.get("two_sided_normal_angle_deg"),
            "two_sided_support_balance": row.get("two_sided_support_balance"),
            "two_sided_min_support_fraction": row.get("two_sided_min_support_fraction"),
            "cluster_source_count": row.get("cluster_source_count"),
        }
        for index, row in enumerate(rows)
    ]


def _gold_candidate_coverage(gold: list[dict], rows: list[dict]) -> list[dict]:
    thresholds = Thresholds()
    coverage = []
    for gold_row in gold:
        gold_line = np.asarray(
            (gold_row["line_start"], gold_row["line_end"]), dtype=np.float64
        )
        matches = []
        for row in rows:
            metrics = pair_metrics(gold_line, endpoints(row))
            if is_match(metrics, thresholds):
                matches.append(
                    {
                        "candidate_id": str(row.get("candidate_id") or ""),
                        "representative_source": row.get("representative_source"),
                        "source_kind": row.get("source_kind"),
                        "axis_angle_deg": round(float(metrics["axis_angle_deg"]), 6),
                        "transverse_offset_mm": round(
                            float(metrics["transverse_offset_mm"]), 6
                        ),
                        "shorter_span_overlap": round(
                            float(metrics["shorter_span_overlap"]), 6
                        ),
                    }
                )
        coverage.append(
            {
                "gold_id": str(gold_row["bend_id"]),
                "covered": bool(matches),
                "matching_candidate_count": len(matches),
                "matching_candidates": sorted(
                    matches,
                    key=lambda item: (
                        item["transverse_offset_mm"],
                        item["axis_angle_deg"],
                        -item["shorter_span_overlap"],
                        item["candidate_id"],
                    ),
                )[:20],
            }
        )
    return coverage


def _aggregate(items: list[dict]) -> dict:
    tp = sum(int(item["true_positive"]) for item in items)
    fp = sum(int(item["false_positive"]) for item in items)
    fn = sum(int(item["false_negative"]) for item in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(bool(item["exact"]) for item in items),
        "case_count": len(items),
    }


def _case_metrics(score: dict) -> dict:
    tp = int(score["true_positive"])
    fp = int(score["false_positive"])
    fn = int(score["false_negative"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact": bool(score["exact"]),
    }


def _estimator(kind: str):
    if kind == ESTIMATOR_EXTRA_TREES:
        return _model().set_params(n_jobs=1)
    if kind == ESTIMATOR_FAMILY_CLASS_BALANCED_EXTRA_TREES:
        return _model().set_params(class_weight=None, n_jobs=1)
    if kind == ESTIMATOR_REGULARIZED_LOGISTIC:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.05,
                class_weight="balanced",
                max_iter=3000,
                random_state=20260814,
                solver="lbfgs",
            ),
        )
    raise ValueError(f"unsupported estimator kind: {kind}")


def _family_class_balanced_weights(groups: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Give every family equal mass and split its mass equally by class."""

    weights = np.zeros(len(target), dtype=np.float64)
    families = sorted(set(groups.tolist()))
    for family in families:
        family_mask = groups == family
        classes = sorted(set(target[family_mask].tolist()))
        for label in classes:
            mask = family_mask & (target == label)
            weights[mask] = 1.0 / (len(families) * len(classes) * int(mask.sum()))
    return weights


def _bundle(estimator, *, score_threshold: float = SCORE_THRESHOLD) -> dict:
    return {
        "profile": V9_PROFILE,
        "feature_names": FEATURE_NAMES,
        "score_threshold": float(score_threshold),
        "small_pool_maximum": SMALL_POOL_MAXIMUM,
        "nms_angle": NMS_ANGLE,
        "nms_distance": NMS_DISTANCE,
        "nms_overlap": NMS_OVERLAP,
        "estimator": estimator,
    }


def _fit_and_score_family(
    family: str,
    *,
    cases: list[dict],
    pools: dict[str, dict],
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    v4: dict,
    estimator_kind: str,
    score_threshold: float,
    short_mesh_recovery: bool,
    source_balanced_snap: bool,
    collinear_fragment_stitch: bool,
    paired_return_completion: bool,
) -> dict:
    started = time.perf_counter()
    train = groups != family
    estimator = _estimator(estimator_kind)
    fit_kwargs = {}
    if estimator_kind == ESTIMATOR_FAMILY_CLASS_BALANCED_EXTRA_TREES:
        fit_kwargs["sample_weight"] = _family_class_balanced_weights(
            groups[train], target[train]
        )
    estimator.fit(features[train], target[train], **fit_kwargs)
    v9 = _bundle(estimator, score_threshold=score_threshold)
    held_out_cases = []
    for case in cases:
        if case["family"] != family:
            continue
        selection = select_unified(pools[case["token"]], v9, v4)
        recovery = {"enabled": False, "added_ids": [], "selected": selection["prediction_rows"]}
        if short_mesh_recovery and selection["route"] == ROUTE_COMPLEX:
            recovery = complete_short_scan_mesh_bends(
                selection["prediction_rows"],
                selection["rows"],
                maximum_span_mm=SHORT_MESH_MAXIMUM_SPAN_MM,
                minimum_normal_turn_deg=SHORT_MESH_MINIMUM_NORMAL_TURN_DEG,
                minimum_finite_support_fraction=(
                    SHORT_MESH_MINIMUM_FINITE_SUPPORT_FRACTION
                ),
                duplicate_offset_mm=SHORT_MESH_DUPLICATE_OFFSET_MM,
            )
        snap = {"selected": recovery["selected"], "changes": []}
        if source_balanced_snap and selection["route"] == ROUTE_COMPLEX:
            snap = snap_source_balanced_representatives(
                recovery["selected"], selection["rows"]
            )
        stitch = {"selected": snap["selected"], "changes": []}
        if collinear_fragment_stitch and selection["route"] == ROUTE_COMPLEX:
            stitch = stitch_collinear_selected_fragments(snap["selected"])
        paired_return = {"selected": stitch["selected"], "changes": []}
        if paired_return_completion and selection["route"] == ROUTE_COMPLEX:
            paired_return = complete_paired_return_axis(
                stitch["selected"], selection["rows"], selection["bbox_span"]
            )
        predictions = _predictions(paired_return["selected"])
        score = match_lines(
            case["gold"], predictions, Thresholds()
        )
        held_out_cases.append(
            {
                "case_token": case["token"],
                "route": selection["route"],
                "line_object_count": len(selection["rows"]),
                "prediction_count": len(selection["prediction_rows"]),
                "post_recovery_prediction_count": len(recovery["selected"]),
                "short_mesh_recovery_added_ids": list(recovery["added_ids"]),
                "source_balanced_snap_changes": list(snap["changes"]),
                "collinear_fragment_stitch_changes": list(stitch["changes"]),
                "paired_return_completion_changes": list(paired_return["changes"]),
                "predictions": predictions,
                "matches": score["matches"],
                "unmatched_gold_ids": score["unmatched_gold_ids"],
                "unmatched_prediction_ids": score["unmatched_prediction_ids"],
                "gold_candidate_coverage": _gold_candidate_coverage(
                    case["gold"], selection["rows"]
                ),
                **_case_metrics(score),
            }
        )
    return {
        "part_family": family,
        "training_family_count": len(set(groups.tolist())) - 1,
        "training_line_object_count": int(train.sum()),
        "training_positive_line_object_count": int(target[train].sum()),
        "fit_and_score_seconds": round(time.perf_counter() - started, 6),
        "metrics": _aggregate(held_out_cases),
        "cases": held_out_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--target-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument(
        "--estimator",
        choices=(
            ESTIMATOR_EXTRA_TREES,
            ESTIMATOR_FAMILY_CLASS_BALANCED_EXTRA_TREES,
            ESTIMATOR_REGULARIZED_LOGISTIC,
        ),
        default=ESTIMATOR_EXTRA_TREES,
    )
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument(
        "--short-mesh-recovery",
        action="store_true",
        help=(
            "Recover nonduplicate <=16 mm scan-mesh-only axes with >=60 degree "
            "normal turn and >=0.80 finite support on the bounded-complex lane."
        ),
    )
    parser.add_argument(
        "--source-balanced-snap",
        action="store_true",
        help=(
            "Translate bounded-complex selections toward a one-vote-per-source "
            "geometric median of nearby fully supported axes."
        ),
    )
    parser.add_argument(
        "--collinear-fragment-stitch",
        action="store_true",
        help=(
            "Merge connected, nearly collinear end-to-end selected fragments "
            "when the fitted union is at least 25 percent longer than every member."
        ),
    )
    parser.add_argument(
        "--paired-return-completion",
        action="store_true",
        help=(
            "Complete one missing stitched lane in a scan-supported paired-return profile."
        ),
    )
    args = parser.parse_args()

    pool_dir = args.pool_dir.resolve()
    gold_manifest = args.gold_manifest.resolve()
    cases = _load_cases(pool_dir, gold_manifest)
    pools = {
        case["token"]: json.loads((pool_dir / f"{case['token']}.json").read_text(encoding="utf-8"))
        for case in cases
    }
    features = np.vstack([case["features"] for case in cases])
    target = np.concatenate([case["target"] for case in cases])
    groups = np.concatenate([[case["family"]] * len(case["rows"]) for case in cases])
    families = sorted(set(groups.tolist()))
    v4 = joblib.load(args.v4_model.resolve())

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _fit_and_score_family,
                family,
                cases=cases,
                pools=pools,
                features=features,
                target=target,
                groups=groups,
                v4=v4,
                estimator_kind=args.estimator,
                score_threshold=args.score_threshold,
                short_mesh_recovery=args.short_mesh_recovery,
                source_balanced_snap=args.source_balanced_snap,
                collinear_fragment_stitch=args.collinear_fragment_stitch,
                paired_return_completion=args.paired_return_completion,
            ): family
            for family in families
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "family": result["part_family"],
                        "metrics": result["metrics"],
                        "seconds": result["fit_and_score_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    results.sort(key=lambda item: item["part_family"])
    all_cases = [case for result in results for case in result["cases"]]
    gold_candidate_coverage = [
        item
        for case in all_cases
        for item in case["gold_candidate_coverage"]
    ]
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v16_bounded_complex_leave_one_family_out",
        "profile": PROFILE,
        "target_contract": str(args.target_contract.resolve()),
        "target_contract_sha256": _sha(args.target_contract.resolve()),
        "pool_manifest_sha256": _sha(pool_dir / "manifest.json"),
        "gold_manifest_sha256": _sha(gold_manifest),
        "v4_model_sha256": _sha(args.v4_model.resolve()),
        "case_count": len(cases),
        "part_family_count": len(families),
        "workers": max(1, args.workers),
        "estimator": args.estimator,
        "score_threshold": float(args.score_threshold),
        "short_mesh_recovery": {
            "enabled": bool(args.short_mesh_recovery),
            "maximum_span_mm": SHORT_MESH_MAXIMUM_SPAN_MM,
            "minimum_normal_turn_deg": SHORT_MESH_MINIMUM_NORMAL_TURN_DEG,
            "minimum_finite_support_fraction": (
                SHORT_MESH_MINIMUM_FINITE_SUPPORT_FRACTION
            ),
            "duplicate_offset_mm": SHORT_MESH_DUPLICATE_OFFSET_MM,
            "added_candidate_count": sum(
                len(case["short_mesh_recovery_added_ids"]) for case in all_cases
            ),
        },
        "source_balanced_snap": {
            "enabled": bool(args.source_balanced_snap),
            "changed_prediction_count": sum(
                len(case["source_balanced_snap_changes"]) for case in all_cases
            ),
        },
        "collinear_fragment_stitch": {
            "enabled": bool(args.collinear_fragment_stitch),
            "changed_component_count": sum(
                len(case["collinear_fragment_stitch_changes"]) for case in all_cases
            ),
        },
        "paired_return_completion": {
            "enabled": bool(args.paired_return_completion),
            "added_candidate_count": sum(
                len(case["paired_return_completion_changes"]) for case in all_cases
            ),
        },
        "wall_seconds": round(time.perf_counter() - started, 6),
        "metrics": _aggregate(all_cases),
        "candidate_coverage": {
            "gold_axis_count": len(gold_candidate_coverage),
            "covered_gold_axis_count": sum(
                bool(item["covered"]) for item in gold_candidate_coverage
            ),
            "uncovered_gold_axis_count": sum(
                not bool(item["covered"]) for item in gold_candidate_coverage
            ),
        },
        "minimum_family_f1": min(
            (float(result["metrics"]["f1"]) for result in results), default=0.0
        ),
        "families": results,
        "method_notes": [
            "Every V9 estimator excludes all candidates and labels from the scored part family.",
            "V16 full-resolution and bounded-small routes are frozen deterministic routes and do not consume V9 scores.",
            "The frozen V4 bundle is retained for full-resolution V11 routing; this report isolates bounded-complex learner generalization.",
            "Part identity is used only to construct held-out groups and is never an estimator feature.",
        ],
        "source_only_features": True,
        "part_identity_feature_used": False,
        "expected_count_feature_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "minimum_family_f1": report["minimum_family_f1"], "wall_seconds": report["wall_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
