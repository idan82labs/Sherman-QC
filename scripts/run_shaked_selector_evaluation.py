#!/usr/bin/env python3
"""Run leakage-bounded F1 selector experiments on the Shaked scan corpus.

Prediction and scoring are deliberately separate commands.  Prediction never
opens labels for its evaluation cases; scoring consumes only already-written
prediction artifacts.  Two claims are supported:

* ``cross_corpus`` tunes on the original 13 scans and predicts all 12 Shaked
  scans without reading any Shaked labels.
* ``adapted_holdout`` tunes on the SHA-frozen Shaked development cases, adds
  those cases to original-corpus training, and predicts only the frozen Shaked
  holdout.

The Shaked annotations remain non-authoritative and user-review-pending.  This
is a development evaluation, not an enterprise promotion benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    CASE_JOBS,
    Thresholds,
    match_lines,
)
from scripts.run_f1_grouped_holdout_selector import (  # noqa: E402
    FEATURE_NAMES,
    THRESHOLD_GRID,
    _features,
    _fit,
    _line_metrics,
    _match_quality,
    _predict,
    _ranked_selection,
    part_group,
)
from backend.f1_dominant_fold_lane_completion import (  # noqa: E402
    complete_dominant_fold_lane,
)
from backend.f1_short_mesh_bend_completion import (  # noqa: E402
    complete_short_scan_mesh_bends,
)
from backend.f1_mesh_consensus_snap import (  # noqa: E402
    snap_selected_mesh_consensus,
)
from backend.f1_mesh_interface_geometry import (  # noqa: E402
    log_minimum_surface_area_span_ratio,
)
from backend.f1_surface_wedge_completion import (  # noqa: E402
    complete_surface_wedges,
)
from backend.f1_parallel_surface_sibling_completion import (  # noqa: E402
    complete_parallel_surface_siblings,
)
from backend.f1_contained_parallel_fragment_pruning import (  # noqa: E402
    prune_contained_parallel_fragments,
)
from backend.f1_scan_supported_short_seed_completion import (  # noqa: E402
    complete_scan_supported_short_seeds,
)
from backend.f1_weak_parallel_outline_pruning import (  # noqa: E402
    prune_weak_parallel_outlines,
)
from backend.f1_surface_graph_outline_pruning import (  # noqa: E402
    prune_surface_graph_outlines,
)


DEFAULT_ORIGINAL_CANDIDATES = (
    ROOT / "output/experiments/f1-10of13/strict-blind-uniform-candidate-pool-mesh-v4"
)
DEFAULT_SHAKED_CANDIDATES = (
    ROOT / "output/manual_labels/shaked-count-backed-12-20260717/candidate_pool_mesh_v4"
)
DEFAULT_SHAKED_LABELS = (
    ROOT / "output/manual_labels/shaked-count-backed-12-20260717/adjudicated_v1"
)
DEFAULT_SPLIT = ROOT / "data/bend_corpus/evaluation_slices/shaked_12_selector_holdout_v1.json"
DEFAULT_OUTPUT = ROOT / "output/experiments/shaked-selector-evaluation-v10"
MODEL_KINDS = ("extra_trees",)
CYLINDRICAL_CAPS = (8.0, 12.0, 1.0e9)
SUPPORT_GATES = (0.0, 0.30, 0.40)
MESH_FEATURE_FIELDS = (
    "bend_angle",
    "angle_between_normals",
    "surface1_area",
    "surface2_area",
    "min_surface_gap",
    "confidence",
    "span_mm",
    "finite_support_fraction",
    "finite_support_longest_run_fraction",
    "endpoint_support_fraction",
    "median_line_distance_mm",
    "p90_line_distance_mm",
    "corridor_points_per_mm",
    "transverse_angular_span_deg",
    "transverse_angular_bin_fraction",
    "normal_second_eigenvalue",
    "normal_spread_deg_scan",
    "two_sided_transition_score",
    "two_sided_normal_angle_deg",
    "two_sided_support_balance",
    "two_sided_min_support_fraction",
    "two_sided_normal_coherence",
    "tight_normal_second_eigenvalue_r2mm",
    "tight_normal_second_eigenvalue_r6mm",
    "tight_to_wide_normal_second_ratio",
)
MESH_FEATURE_NAMES = (
    *(f"mesh_{name}" for name in MESH_FEATURE_FIELDS),
    "mesh_span_over_diagonal",
    *(f"mesh_case_percentile_{name}" for name in MESH_FEATURE_FIELDS),
    "mesh_log_minimum_surface_area_span_ratio",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_paths(root: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(root.glob("[0-9][0-9]_*.json"))}


def validate_split(split: Mapping[str, Any], available_case_ids: Iterable[str]) -> None:
    development = list(split.get("development_case_ids") or ())
    locked = list(split.get("locked_case_ids") or ())
    available = set(available_case_ids)
    if not development or not locked:
        raise ValueError("split must contain non-empty development and locked sets")
    if len(development) != len(set(development)) or len(locked) != len(set(locked)):
        raise ValueError("split case IDs must be unique")
    overlap = sorted(set(development) & set(locked))
    if overlap:
        raise ValueError(f"development and locked sets overlap: {overlap}")
    if set(development) | set(locked) != available:
        raise ValueError("split must exhaustively cover the available Shaked cases")
    rule = dict(split.get("selection_rule") or {})
    if rule.get("kind") != "sha256_rank":
        raise ValueError("split selection rule must be sha256_rank")
    salt = str(rule.get("salt") or "")
    locked_count = int(rule.get("locked_count") or 0)
    ranked = sorted(
        available,
        key=lambda case_id: hashlib.sha256(f"{salt}:{case_id}".encode()).hexdigest(),
    )
    if set(ranked[:locked_count]) != set(locked):
        raise ValueError("locked IDs do not match the declared SHA-256 ranking rule")


def _validate_candidate_payload(payload: Mapping[str, Any], case_id: str) -> None:
    contract = dict(payload.get("inference_contract") or {})
    required_false = (
        "part_identifier_exposed",
        "expected_count_exposed",
        "manual_truth_exposed",
        "cad_exposed",
    )
    leaked = [name for name in required_false if contract.get(name) is not False]
    if leaked:
        raise ValueError(f"{case_id}: blind candidate contract failed: {leaked}")
    if payload.get("source_point_serialization") is not False:
        raise ValueError(f"{case_id}: source points must not be serialized")
    if payload.get("coordinate_frame") != "scan_raw":
        raise ValueError(f"{case_id}: candidate frame must be scan_raw")


def _load_candidate_payloads(
    root: Path,
    case_ids: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    paths = _case_paths(root)
    selected = sorted(paths) if case_ids is None else list(case_ids)
    missing = [case_id for case_id in selected if case_id not in paths]
    if missing:
        raise ValueError(f"candidate payloads missing: {missing}")
    payloads: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for case_id in selected:
        path = paths[case_id]
        payload = _read(path)
        _validate_candidate_payload(payload, case_id)
        payloads[case_id] = payload
        hashes[case_id] = _sha256(path)
    return payloads, hashes


def _load_original_gold() -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    gold: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for case_id, job_id in sorted(CASE_JOBS.items()):
        path = ROOT / "output" / job_id / "bend_instance_labels.json"
        payload = _read(path)
        if payload.get("review_status") != "complete":
            raise ValueError(f"{case_id}: original gold is not complete")
        gold[case_id] = [dict(item) for item in payload.get("manual_bends") or ()]
        hashes[case_id] = _sha256(path)
    return gold, hashes


def _load_shaked_gold(
    root: Path,
    case_ids: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Load exactly the requested Shaked cases; callers control the leakage boundary."""

    gold: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for case_id in case_ids:
        path = root / case_id / "bend_instance_labels.json"
        payload = _read(path)
        if payload.get("case_id") != case_id:
            raise ValueError(f"{case_id}: label case ID mismatch")
        if payload.get("authoritative_ground_truth") is not False:
            raise ValueError(f"{case_id}: Shaked labels must remain non-authoritative")
        if payload.get("review_status") != "assistant_multiview_complete_user_review_pending":
            raise ValueError(f"{case_id}: unexpected Shaked review status")
        accepted = [
            dict(item)
            for item in payload.get("manual_bends") or ()
            if item.get("review_decision") == "accepted_user_review_pending"
        ]
        if len(accepted) != len(payload.get("manual_bends") or ()) or not accepted:
            raise ValueError(f"{case_id}: expected only accepted pending-review bend axes")
        gold[case_id] = accepted
        hashes[case_id] = _sha256(path)
    return gold, hashes


def _build_rows(
    payloads: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[tuple[dict[str, Any], list[float], int, float]]]:
    thresholds = Thresholds()
    rows: dict[str, list[tuple[dict[str, Any], list[float], int, float]]] = {}
    for case_id, payload in payloads.items():
        candidates = [dict(item) for item in payload.get("line_objects") or ()]
        case_rows = []
        for candidate in candidates:
            quality = _match_quality(candidate, gold[case_id], thresholds)
            case_rows.append(
                (
                    candidate,
                    _features(candidate, payload, include_arrangement=False),
                    int(quality > 0.0),
                    quality,
                )
            )
        if not case_rows:
            raise ValueError(f"{case_id}: no line objects")
        rows[case_id] = case_rows
    return rows


def _mesh_case_features(
    candidates: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> list[list[float]]:
    values = {
        name: np.asarray([float(item.get(name) or 0.0) for item in candidates], dtype=np.float64)
        for name in MESH_FEATURE_FIELDS
    }
    percentiles = {
        name: rankdata(field_values, method="average") / max(len(candidates), 1)
        for name, field_values in values.items()
    }
    diagonal = max(float(np.linalg.norm(payload.get("scan_bbox_mm") or (1, 1, 1))), 1.0)
    rows = []
    for index in range(len(candidates)):
        raw = []
        for name in MESH_FEATURE_FIELDS:
            value = float(values[name][index])
            if name in ("surface1_area", "surface2_area"):
                value = float(np.log1p(max(value, 0.0)))
            raw.append(value)
        span = float(values["span_mm"][index])
        rows.append(
            raw
            + [span / diagonal]
            + [float(percentiles[name][index]) for name in MESH_FEATURE_FIELDS]
            + [log_minimum_surface_area_span_ratio(candidates[index])]
        )
    return rows


def _build_mesh_rows(
    payloads: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[tuple[dict[str, Any], list[float], int, float]]]:
    thresholds = Thresholds()
    rows = {}
    for case_id, payload in payloads.items():
        candidates = [
            dict(item)
            for item in payload.get("candidates") or ()
            if item.get("candidate_source") == "scan_mesh_adjacency"
        ]
        if not candidates:
            raise ValueError(f"{case_id}: no scan-mesh adjacency candidates")
        features = _mesh_case_features(candidates, payload)
        rows[case_id] = []
        for candidate, feature_values in zip(candidates, features):
            quality = _match_quality(candidate, gold[case_id], thresholds)
            rows[case_id].append(
                (candidate, feature_values, int(quality > 0.0), quality)
            )
    return rows


def _fit_mesh_rows(rows, case_ids: Sequence[str], *, logistic_c: float = 0.20):
    flattened = [row for case_id in case_ids for row in rows[case_id]]
    features = np.asarray([row[1] for row in flattened], dtype=np.float64)
    labels = np.asarray([row[2] for row in flattened], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("mesh selector training requires both positive and negative candidates")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=float(logistic_c),
            class_weight="balanced",
            max_iter=2000,
            random_state=17,
        ),
    )
    model.fit(features, labels)
    return model


def _predict_mesh_rows(model, case_rows):
    features = np.asarray([row[1] for row in case_rows], dtype=np.float64)
    probabilities = model.predict_proba(features)[:, 1]
    return [row[0] for row in case_rows], probabilities, probabilities


def _tune_mesh(
    rows,
    gold,
    development_case_ids: Sequence[str],
    *,
    line_objects_by_case: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
):
    logistic_c_grid = (0.20,)
    out_of_fold = {value: {} for value in logistic_c_grid}
    folds = []
    for held_out in development_case_ids:
        training = [case_id for case_id in development_case_ids if case_id != held_out]
        for logistic_c in logistic_c_grid:
            model = _fit_mesh_rows(rows, training, logistic_c=logistic_c)
            out_of_fold[logistic_c][held_out] = _predict_mesh_rows(model, rows[held_out])
        folds.append({"held_out_case_ids": [held_out], "training_case_ids": training})
    scored = []
    for logistic_c in logistic_c_grid:
        for threshold in np.arange(0.05, 0.951, 0.05):
            for duplicate_offset in (4.0,):
                for mesh_gap_cap in (6.0, 8.0, 10.0, 12.0, 1.0e9):
                    config = {
                        "model_kind": "logistic_mesh_adjacency",
                        "logistic_c": logistic_c,
                        "probability_threshold": float(round(threshold, 8)),
                        "cylindrical_offset_cap_mm": 1.0e9,
                        "duplicate_offset_tolerance_mm": duplicate_offset,
                        "two_sided_min_support_gate": 0.0,
                        "mesh_max_surface_gap_mm": mesh_gap_cap,
                    }
                    predictions = {
                        case_id: _select(out_of_fold[logistic_c][case_id], config)
                        for case_id in development_case_ids
                    }
                    scored.append({"config": config, "metrics": _line_metrics(predictions, gold)})
    selected = max(scored, key=_score_key)
    config = dict(selected["config"])
    metrics = dict(selected["metrics"])
    if line_objects_by_case is not None:
        missing = set(development_case_ids) - set(line_objects_by_case)
        if missing:
            raise ValueError(f"missing line objects for completion tuning: {sorted(missing)}")
        base_predictions = {
            case_id: _select(out_of_fold[float(config["logistic_c"])][case_id], config)
            for case_id in development_case_ids
        }
        completion_scored = []
        completion_grid = [
            {
                "short_mesh_max_span_mm": 0.0,
                "short_mesh_min_normal_turn_deg": 999.0,
                "short_mesh_min_finite_support_fraction": 1.0,
            }
        ]
        completion_grid.extend(
            {
                "short_mesh_max_span_mm": maximum_span,
                "short_mesh_min_normal_turn_deg": minimum_turn,
                "short_mesh_min_finite_support_fraction": minimum_support,
            }
            for maximum_span in (12.0, 16.0, 20.0, 25.0)
            for minimum_turn in (60.0, 70.0, 80.0)
            for minimum_support in (0.8, 0.9)
        )
        for completion_config in completion_grid:
            predictions = {
                case_id: _snap_mesh_consensus(
                    _complete_short_mesh(
                        base_predictions[case_id],
                        line_objects_by_case[case_id],
                        completion_config,
                    )["selected"],
                    line_objects_by_case[case_id],
                    completion_config,
                )["selected"]
                for case_id in development_case_ids
            }
            completion_scored.append(
                {
                    "config": completion_config,
                    "metrics": _line_metrics(predictions, gold),
                }
            )
        completion_selected = max(
            completion_scored,
            key=lambda item: (
                float(item["metrics"]["f1"]),
                float(item["metrics"]["exact"]),
                -float(item["metrics"]["fp"]),
                -float(item["metrics"]["fn"]),
                -float(item["config"]["short_mesh_max_span_mm"]),
                float(item["config"]["short_mesh_min_normal_turn_deg"]),
                float(item["config"]["short_mesh_min_finite_support_fraction"]),
            ),
        )
        config.update(completion_selected["config"])
        metrics = dict(completion_selected["metrics"])
    return config, metrics, folds


def _complete_short_mesh(selected, line_objects, config: Mapping[str, Any]):
    return complete_short_scan_mesh_bends(
        selected,
        line_objects,
        maximum_span_mm=float(config.get("short_mesh_max_span_mm") or 0.0),
        minimum_normal_turn_deg=float(
            config.get("short_mesh_min_normal_turn_deg") or 999.0
        ),
        minimum_finite_support_fraction=float(
            config.get("short_mesh_min_finite_support_fraction") or 1.0
        ),
        duplicate_offset_mm=float(config.get("duplicate_offset_tolerance_mm") or 4.0),
    )


def _snap_mesh_consensus(selected, line_objects, config: Mapping[str, Any]):
    return snap_selected_mesh_consensus(
        selected,
        line_objects,
        minimum_source_count=3,
        duplicate_offset_mm=float(config.get("duplicate_offset_tolerance_mm") or 4.0),
    )


def _complete_surface_wedges(selected, line_objects, config: Mapping[str, Any]):
    return complete_surface_wedges(
        selected,
        line_objects,
        minimum_source_count=2,
        maximum_single_source_span_mm=10.0,
        duplicate_offset_mm=float(config.get("duplicate_offset_tolerance_mm") or 4.0),
    )


def _complete_parallel_siblings(selected, candidates):
    return complete_parallel_surface_siblings(selected, candidates)


def _prune_parallel_fragments(selected):
    return prune_contained_parallel_fragments(selected)


def _complete_scan_supported_short_seeds(selected, line_objects, candidates):
    return complete_scan_supported_short_seeds(selected, line_objects, candidates)


def _prune_weak_parallel_outlines(selected):
    return prune_weak_parallel_outlines(selected)


def _prune_surface_graph_outlines(selected):
    return prune_surface_graph_outlines(selected)


def _fit_rows(
    rows: Mapping[str, Sequence[tuple[dict[str, Any], list[float], int, float]]],
    case_ids: Sequence[str],
    model_kind: str,
):
    flattened = [row for case_id in case_ids for row in rows[case_id]]
    features = np.asarray([row[1] for row in flattened], dtype=np.float64)
    labels = np.asarray([row[2] for row in flattened], dtype=np.int64)
    qualities = np.asarray([row[3] for row in flattened], dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("selector training requires both positive and negative candidates")
    group_sizes: dict[str, int] = {}
    for case_id in case_ids:
        group = part_group(case_id)
        group_sizes[group] = group_sizes.get(group, 0) + len(rows[case_id])
    sample_weights = np.asarray(
        [
            1.0 / group_sizes[part_group(case_id)]
            for case_id in case_ids
            for _row in rows[case_id]
        ],
        dtype=np.float64,
    )
    sample_weights *= len(sample_weights) / max(float(np.sum(sample_weights)), 1e-12)
    return _fit(
        features,
        labels,
        qualities,
        model_kind=model_kind,
        sample_weights=sample_weights,
    )


def _predict_rows(model: tuple[Any, Any], case_rows):
    features = np.asarray([row[1] for row in case_rows], dtype=np.float64)
    probabilities, qualities = _predict(model[0], model[1], features)
    return [row[0] for row in case_rows], probabilities, qualities


def _select(prediction, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates, probabilities, qualities = prediction
    maximum_mesh_gap = float(config.get("mesh_max_surface_gap_mm") or 1.0e9)
    retained = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.get("candidate_source") != "scan_mesh_adjacency"
        or float(candidate.get("min_surface_gap") or 0.0) <= maximum_mesh_gap
    ]
    return _ranked_selection(
        [candidates[index] for index in retained],
        [probabilities[index] for index in retained],
        [qualities[index] for index in retained],
        probability_threshold=float(config["probability_threshold"]),
        cylindrical_offset_cap_mm=float(config["cylindrical_offset_cap_mm"]),
        duplicate_offset_tolerance_mm=float(config["duplicate_offset_tolerance_mm"]),
        two_sided_min_support_gate=float(config.get("two_sided_min_support_gate") or 0.0),
        topology_completion_probability_delta=0.0,
    )


def _score_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = item["metrics"]
    config = item["config"]
    return (
        float(metrics["f1"]),
        float(metrics["exact"]),
        -float(metrics["fp"]),
        -float(metrics["fn"]),
        float(config["probability_threshold"]),
        -float(config["duplicate_offset_tolerance_mm"]),
        -float(config["cylindrical_offset_cap_mm"]),
        str(config["model_kind"]),
    )


def _tune(
    rows: Mapping[str, Sequence[tuple[dict[str, Any], list[float], int, float]]],
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    base_training_case_ids: Sequence[str],
    validation_case_ids: Sequence[str],
    group_by_physical_part: bool,
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    if group_by_physical_part:
        validation_groups = sorted({part_group(case_id) for case_id in validation_case_ids})
        fold_cases = [
            [case_id for case_id in validation_case_ids if part_group(case_id) == group]
            for group in validation_groups
        ]
    else:
        fold_cases = [[case_id] for case_id in validation_case_ids]

    out_of_fold: dict[str, dict[str, tuple[Any, Any, Any]]] = {
        model_kind: {} for model_kind in MODEL_KINDS
    }
    fold_manifest = []
    for held_cases in fold_cases:
        training_cases = list(base_training_case_ids) + [
            case_id for case_id in validation_case_ids if case_id not in held_cases
        ]
        training_cases = list(dict.fromkeys(training_cases))
        for model_kind in MODEL_KINDS:
            model = _fit_rows(rows, training_cases, model_kind)
            for case_id in held_cases:
                out_of_fold[model_kind][case_id] = _predict_rows(model, rows[case_id])
        fold_manifest.append(
            {"held_out_case_ids": held_cases, "training_case_ids": training_cases}
        )

    baseline_scores = []
    for model_kind in MODEL_KINDS:
        for threshold in THRESHOLD_GRID:
            for offset_cap in CYLINDRICAL_CAPS:
                for duplicate_offset in (5.0,):
                    config = {
                        "model_kind": model_kind,
                        "probability_threshold": threshold,
                        "cylindrical_offset_cap_mm": offset_cap,
                        "duplicate_offset_tolerance_mm": duplicate_offset,
                        "two_sided_min_support_gate": 0.0,
                    }
                    predictions = {
                        case_id: _select(out_of_fold[model_kind][case_id], config)
                        for case_id in validation_case_ids
                    }
                    baseline_scores.append(
                        {"config": config, "metrics": _line_metrics(predictions, gold)}
                    )
    baseline = max(baseline_scores, key=_score_key)

    gate_scores = []
    for support_gate in SUPPORT_GATES:
        config = dict(baseline["config"])
        config["two_sided_min_support_gate"] = support_gate
        predictions = {
            case_id: _select(out_of_fold[config["model_kind"]][case_id], config)
            for case_id in validation_case_ids
        }
        gate_scores.append({"config": config, "metrics": _line_metrics(predictions, gold)})
    selected = max(gate_scores, key=_score_key)
    return dict(selected["config"]), dict(selected["metrics"]), fold_manifest


def _prediction_case_payload(
    case_id: str,
    selected: Sequence[Mapping[str, Any]],
    experiment: str,
    config: Mapping[str, Any],
    training_case_ids: Sequence[str],
) -> dict[str, Any]:
    owned = []
    for index, candidate in enumerate(selected, start=1):
        item = dict(candidate)
        item["bend_id"] = f"F1S{index}"
        owned.append(item)
    return {
        "schema_version": 1,
        "algorithm": "f1_shaked_selector_evaluation_v3",
        "experiment": experiment,
        "case_id": case_id,
        "evaluation_labels_read": False,
        "training_case_ids": list(training_case_ids),
        "selection_config": dict(config),
        "source_point_serialization": False,
        "coordinate_frame": "scan_raw",
        "owned_bend_regions": owned,
    }


def predict(args: argparse.Namespace) -> None:
    split = _read(args.split)
    shaked_paths = _case_paths(args.shaked_candidate_root)
    validate_split(split, shaked_paths)
    original_payloads, original_candidate_hashes = _load_candidate_payloads(
        args.original_candidate_root
    )
    original_gold, original_label_hashes = _load_original_gold()
    if set(original_payloads) != set(original_gold):
        raise ValueError("original candidate/gold case IDs differ")

    shaked_evaluation_ids: list[str]
    shaked_training_ids: list[str]
    shaked_training_gold: dict[str, list[dict[str, Any]]]
    shaked_training_label_hashes: dict[str, str]
    if args.experiment == "cross_corpus":
        shaked_training_ids = []
        shaked_evaluation_ids = sorted(shaked_paths)
        shaked_training_gold = {}
        shaked_training_label_hashes = {}
    else:
        shaked_training_ids = list(split["development_case_ids"])
        shaked_evaluation_ids = list(split["locked_case_ids"])
        shaked_training_gold, shaked_training_label_hashes = _load_shaked_gold(
            args.shaked_label_root,
            shaked_training_ids,
        )

    needed_shaked = shaked_training_ids + shaked_evaluation_ids
    shaked_payloads, shaked_candidate_hashes = _load_candidate_payloads(
        args.shaked_candidate_root,
        needed_shaked,
    )
    training_payloads = dict(original_payloads)
    training_payloads.update(
        {case_id: shaked_payloads[case_id] for case_id in shaked_training_ids}
    )
    training_gold = dict(original_gold)
    training_gold.update(shaked_training_gold)
    rows = _build_rows(training_payloads, training_gold)

    if args.experiment == "cross_corpus":
        config, tuning_metrics, folds = _tune(
            rows,
            training_gold,
            base_training_case_ids=[],
            validation_case_ids=sorted(original_payloads),
            group_by_physical_part=True,
        )
    else:
        config, tuning_metrics, folds = _tune(
            rows,
            training_gold,
            base_training_case_ids=sorted(original_payloads),
            validation_case_ids=shaked_training_ids,
            group_by_physical_part=False,
        )

    final_training_ids = sorted(original_payloads) + shaked_training_ids
    final_model = _fit_rows(rows, final_training_ids, str(config["model_kind"]))
    evaluation_payloads = {
        case_id: shaked_payloads[case_id] for case_id in shaked_evaluation_ids
    }
    evaluation_rows_without_labels = {}
    for case_id, payload in evaluation_payloads.items():
        candidates = [dict(item) for item in payload.get("line_objects") or ()]
        evaluation_rows_without_labels[case_id] = [
            (
                candidate,
                _features(candidate, payload, include_arrangement=False),
                0,
                0.0,
            )
            for candidate in candidates
        ]

    run_dir = args.output_root / args.experiment
    prediction_paths: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    for case_id in shaked_evaluation_ids:
        prediction = _predict_rows(final_model, evaluation_rows_without_labels[case_id])
        selected = _select(prediction, config)
        path = run_dir / "predictions" / case_id / "decomposition_result.json"
        _write(
            path,
            _prediction_case_payload(
                case_id,
                selected,
                args.experiment,
                config,
                final_training_ids,
            ),
        )
        prediction_paths[case_id] = str(path)
        prediction_hashes[case_id] = _sha256(path)

    manifest = {
        "schema_version": 1,
        "algorithm": "f1_shaked_selector_evaluation_v3",
        "experiment": args.experiment,
        "phase": "predictions_frozen_before_evaluation_labels",
        "evaluation_case_ids": shaked_evaluation_ids,
        "training_case_ids": final_training_ids,
        "selection_config": config,
        "tuning_metrics": tuning_metrics,
        "tuning_folds": folds,
        "feature_names": list(FEATURE_NAMES),
        "input_hashes": {
            "split": _sha256(args.split),
            "original_candidates": original_candidate_hashes,
            "original_training_labels": original_label_hashes,
            "shaked_candidates": shaked_candidate_hashes,
            "shaked_training_labels": shaked_training_label_hashes,
        },
        "prediction_paths": prediction_paths,
        "prediction_hashes": prediction_hashes,
        "leakage_contract": {
            "evaluation_labels_read_during_prediction": False,
            "candidate_generation_uses_part_identifier": False,
            "candidate_generation_uses_expected_count": False,
            "candidate_generation_uses_manual_truth": False,
            "candidate_generation_uses_cad": False,
            "selector_features_use_part_identifier_or_count": False,
            "source_points_serialized": False,
        },
        "claim": (
            "cross_corpus_zero_shot_development_evaluation"
            if args.experiment == "cross_corpus"
            else "label_isolated_adapted_development_holdout"
        ),
        "limitations": [
            "Shaked labels are assistant-reviewed, user-review-pending, and non-authoritative.",
            "Most Shaked axes were selected or adjusted from candidate lines, so endpoint truth is not fully independent.",
            "This result is not an enterprise promotion benchmark.",
        ],
    }
    _write(run_dir / "prediction_manifest.json", manifest)
    print(json.dumps({"experiment": args.experiment, "config": config, "tuning": tuning_metrics, "prediction_counts": {case_id: len(_read(Path(path))["owned_bend_regions"]) for case_id, path in prediction_paths.items()}}, indent=2))


def predict_mesh(args: argparse.Namespace) -> None:
    """Fit only on development labels and freeze raw mesh-adjacency predictions."""

    split = _read(args.split)
    shaked_paths = _case_paths(args.shaked_candidate_root)
    validate_split(split, shaked_paths)
    development_ids = list(split["development_case_ids"])
    evaluation_ids = list(split["locked_case_ids"])
    payloads, candidate_hashes = _load_candidate_payloads(
        args.shaked_candidate_root,
        development_ids + evaluation_ids,
    )
    development_gold, training_label_hashes = _load_shaked_gold(
        args.shaked_label_root,
        development_ids,
    )
    development_rows = _build_mesh_rows(
        {case_id: payloads[case_id] for case_id in development_ids},
        development_gold,
    )
    config, tuning_metrics, folds = _tune_mesh(
        development_rows,
        development_gold,
        development_ids,
        line_objects_by_case={
            case_id: payloads[case_id].get("line_objects") or ()
            for case_id in development_ids
        },
    )
    model = _fit_mesh_rows(
        development_rows,
        development_ids,
        logistic_c=float(config["logistic_c"]),
    )

    evaluation_rows = {}
    for case_id in evaluation_ids:
        candidates = [
            dict(item)
            for item in payloads[case_id].get("candidates") or ()
            if item.get("candidate_source") == "scan_mesh_adjacency"
        ]
        features = _mesh_case_features(candidates, payloads[case_id])
        evaluation_rows[case_id] = [
            (candidate, feature_values, 0, 0.0)
            for candidate, feature_values in zip(candidates, features)
        ]

    run_dir = args.output_root / args.experiment
    prediction_paths = {}
    prediction_hashes = {}
    for case_id in evaluation_ids:
        prediction = _predict_mesh_rows(model, evaluation_rows[case_id])
        selected = _select(prediction, config)
        short_completion = _complete_short_mesh(
            selected,
            payloads[case_id].get("line_objects") or (),
            config,
        )
        selected = list(short_completion["selected"])
        consensus_snap = _snap_mesh_consensus(
            selected,
            payloads[case_id].get("line_objects") or (),
            config,
        )
        selected = list(consensus_snap["selected"])
        wedge_completion = _complete_surface_wedges(
            selected,
            payloads[case_id].get("line_objects") or (),
            config,
        )
        selected = list(wedge_completion["selected"])
        lane = complete_dominant_fold_lane(payloads[case_id].get("line_objects") or ())
        if lane["eligible"]:
            selected = list(lane["selected"])
        sibling_completion = _complete_parallel_siblings(
            selected,
            payloads[case_id].get("candidates") or (),
        )
        selected = list(sibling_completion["selected"])
        fragment_pruning = _prune_parallel_fragments(selected)
        selected = list(fragment_pruning["selected"])
        seed_completion = _complete_scan_supported_short_seeds(
            selected,
            payloads[case_id].get("line_objects") or (),
            payloads[case_id].get("candidates") or (),
        )
        selected = list(seed_completion["selected"])
        outline_pruning = _prune_weak_parallel_outlines(selected)
        selected = list(outline_pruning["selected"])
        surface_graph_pruning = _prune_surface_graph_outlines(selected)
        selected = list(surface_graph_pruning["selected"])
        path = run_dir / "predictions" / case_id / "decomposition_result.json"
        prediction_payload = _prediction_case_payload(
            case_id,
            selected,
            args.experiment,
            config,
            development_ids,
        )
        prediction_payload["dominant_fold_lane_completion"] = {
            key: value for key, value in lane.items() if key != "selected"
        }
        prediction_payload["short_scan_mesh_bend_completion"] = {
            key: value for key, value in short_completion.items() if key != "selected"
        }
        prediction_payload["mesh_cross_source_consensus_snap"] = {
            key: value for key, value in consensus_snap.items() if key != "selected"
        }
        prediction_payload["surface_wedge_completion"] = {
            key: value for key, value in wedge_completion.items() if key != "selected"
        }
        prediction_payload["parallel_surface_sibling_completion"] = {
            key: value for key, value in sibling_completion.items() if key != "selected"
        }
        prediction_payload["contained_parallel_fragment_pruning"] = {
            key: value for key, value in fragment_pruning.items() if key != "selected"
        }
        prediction_payload["scan_supported_short_seed_completion"] = {
            key: value for key, value in seed_completion.items() if key != "selected"
        }
        prediction_payload["weak_parallel_outline_pruning"] = {
            key: value for key, value in outline_pruning.items() if key != "selected"
        }
        prediction_payload["surface_graph_outline_pruning"] = {
            key: value
            for key, value in surface_graph_pruning.items()
            if key != "selected"
        }
        _write(
            path,
            prediction_payload,
        )
        prediction_paths[case_id] = str(path)
        prediction_hashes[case_id] = _sha256(path)

    manifest = {
        "schema_version": 1,
        "algorithm": "f1_shaked_mesh_adjacency_selector_v8",
        "experiment": args.experiment,
        "phase": "predictions_frozen_before_evaluation_labels",
        "evaluation_case_ids": evaluation_ids,
        "training_case_ids": development_ids,
        "selection_config": config,
        "tuning_metrics": tuning_metrics,
        "tuning_folds": folds,
        "feature_names": list(MESH_FEATURE_NAMES),
        "input_hashes": {
            "split": _sha256(args.split),
            "shaked_candidates": candidate_hashes,
            "shaked_training_labels": training_label_hashes,
        },
        "prediction_paths": prediction_paths,
        "prediction_hashes": prediction_hashes,
        "leakage_contract": {
            "evaluation_labels_read_during_prediction": False,
            "candidate_generation_uses_part_identifier": False,
            "candidate_generation_uses_expected_count": False,
            "candidate_generation_uses_manual_truth": False,
            "candidate_generation_uses_cad": False,
            "selector_features_use_part_identifier_or_count": False,
            "source_points_serialized": False,
        },
        "claim": "post_development_label_isolated_mesh_adjacency_holdout",
        "limitations": [
            "The corpus and holdout were visually analyzed before this model family was finalized; this is not a fresh blind result.",
            "Shaked labels are assistant-reviewed, user-review-pending, and non-authoritative.",
            "Most Shaked axes were selected or adjusted from candidate lines, so endpoint truth is not fully independent.",
            "This result is not an enterprise promotion benchmark.",
        ],
    }
    _write(run_dir / "prediction_manifest.json", manifest)
    print(
        json.dumps(
            {
                "experiment": args.experiment,
                "config": config,
                "tuning": tuning_metrics,
                "prediction_counts": {
                    case_id: len(_read(Path(path))["owned_bend_regions"])
                    for case_id, path in prediction_paths.items()
                },
            },
            indent=2,
        )
    )


def score(args: argparse.Namespace) -> None:
    run_dir = args.output_root / args.experiment
    manifest_path = run_dir / "prediction_manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("phase") != "predictions_frozen_before_evaluation_labels":
        raise ValueError("predictions are not in the frozen pre-score state")
    case_ids = list(manifest.get("evaluation_case_ids") or ())
    predictions: dict[str, list[dict[str, Any]]] = {}
    for case_id in case_ids:
        path = Path(manifest["prediction_paths"][case_id])
        if _sha256(path) != manifest["prediction_hashes"][case_id]:
            raise ValueError(f"{case_id}: frozen prediction hash changed")
        predictions[case_id] = list(_read(path).get("owned_bend_regions") or ())
    gold, label_hashes = _load_shaked_gold(args.shaked_label_root, case_ids)

    thresholds = Thresholds()
    case_rows = []
    total_tp = total_fp = total_fn = exact_count = 0
    for case_id in case_ids:
        result = match_lines(gold[case_id], predictions[case_id], thresholds)
        total_tp += int(result["true_positive"])
        total_fp += int(result["false_positive"])
        total_fn += int(result["false_negative"])
        exact_count += int(result["exact"])
        case_rows.append(
            {
                "case_id": case_id,
                "gold_count": len(gold[case_id]),
                "prediction_count": len(predictions[case_id]),
                **result,
            }
        )
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "schema_version": 1,
        "experiment": args.experiment,
        "prediction_manifest_path": str(manifest_path),
        "prediction_manifest_sha256": _sha256(manifest_path),
        "score_phase_only": True,
        "evaluation_label_hashes": label_hashes,
        "thresholds": thresholds.__dict__,
        "metrics": {
            "case_count": len(case_ids),
            "gold_count": sum(len(gold[case_id]) for case_id in case_ids),
            "prediction_count": sum(len(predictions[case_id]) for case_id in case_ids),
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_geometry_case_count": exact_count,
            "exact_geometry_case_rate": exact_count / len(case_ids) if case_ids else 0.0,
            "exact_count_case_count": sum(
                int(row["gold_count"] == row["prediction_count"]) for row in case_rows
            ),
        },
        "cases": case_rows,
        "claim": manifest["claim"],
        "limitations": manifest["limitations"],
    }
    _write(run_dir / "locked_score.json", report)
    print(json.dumps(report["metrics"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "score"))
    parser.add_argument(
        "--experiment",
        choices=("cross_corpus", "adapted_holdout", "mesh_adjacency_holdout"),
        required=True,
    )
    parser.add_argument("--original-candidate-root", type=Path, default=DEFAULT_ORIGINAL_CANDIDATES)
    parser.add_argument("--shaked-candidate-root", type=Path, default=DEFAULT_SHAKED_CANDIDATES)
    parser.add_argument("--shaked-label-root", type=Path, default=DEFAULT_SHAKED_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.original_candidate_root = args.original_candidate_root.resolve()
    args.shaked_candidate_root = args.shaked_candidate_root.resolve()
    args.shaked_label_root = args.shaked_label_root.resolve()
    args.split = args.split.resolve()
    args.output_root = args.output_root.resolve()
    if args.phase == "predict":
        if args.experiment == "mesh_adjacency_holdout":
            predict_mesh(args)
        else:
            predict(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
