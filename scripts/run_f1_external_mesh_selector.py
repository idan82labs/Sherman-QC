#!/usr/bin/env python3
"""Train on the prior Shaked corpus and freeze mesh-selector predictions externally."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.run_shaked_selector_evaluation import (
    DEFAULT_SHAKED_CANDIDATES,
    DEFAULT_SHAKED_LABELS,
    MESH_FEATURE_NAMES,
    _build_mesh_rows,
    _complete_parallel_siblings,
    _complete_scan_supported_short_seeds,
    _complete_short_mesh,
    _complete_surface_wedges,
    _fit_mesh_rows,
    _load_candidate_payloads,
    _load_shaked_gold,
    _mesh_case_features,
    _predict_mesh_rows,
    _prune_parallel_fragments,
    _prune_surface_graph_outlines,
    _prune_weak_parallel_outlines,
    _select,
    _snap_mesh_consensus,
    _tune_mesh,
)
from backend.f1_dominant_fold_lane_completion import complete_dominant_fold_lane


DEFAULT_EXTERNAL_CANDIDATES = (
    ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/candidate_pool"
)
DEFAULT_EXTERNAL_LABELS = (
    ROOT
    / "output/research/f1-new-shaked-pairs-20260730-v1/manual_adjudication/labels"
)
DEFAULT_OUTPUT = (
    ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/mesh_selector_prior12"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_payloads(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    payloads = {}
    hashes = {}
    for path in sorted(root.glob("[0-9][0-9]_C[0-9][0-9].json")):
        payload = _read(path)
        token = str(payload.get("case_id") or "").split("_", 1)[-1]
        if not token.startswith("C") or token in payloads:
            raise ValueError(f"invalid or duplicate external token in {path}: {token}")
        payloads[token] = payload
        hashes[token] = _sha256(path)
    if len(payloads) != 12:
        raise ValueError(f"expected 12 external candidate pools, found {len(payloads)}")
    return payloads, hashes


def _prediction_payload(token: str, selected: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "algorithm": "f1_prior12_mesh_adjacency_external_v1",
        "case_token": token,
        "part_id": None,
        "inference_contract": {
            "uses_part_identifier": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
            "uses_cad": False,
        },
        "selection_config": config,
        "owned_bend_regions": selected,
    }


def predict(args: argparse.Namespace) -> None:
    training_ids = sorted(path.parent.name for path in args.training_label_root.glob("*/bend_instance_labels.json"))
    if len(training_ids) != 12:
        raise ValueError(f"expected 12 prior training cases, found {len(training_ids)}")
    training_payloads, training_candidate_hashes = _load_candidate_payloads(
        args.training_candidate_root, training_ids
    )
    training_gold, training_label_hashes = _load_shaked_gold(
        args.training_label_root, training_ids
    )
    rows = _build_mesh_rows(training_payloads, training_gold)
    config, tuning_metrics, folds = _tune_mesh(
        rows,
        training_gold,
        training_ids,
        line_objects_by_case={
            case_id: training_payloads[case_id].get("line_objects") or []
            for case_id in training_ids
        },
    )
    model = _fit_mesh_rows(rows, training_ids, logistic_c=float(config["logistic_c"]))

    external_payloads, external_hashes = _external_payloads(args.external_candidate_root)
    prediction_paths = {}
    prediction_hashes = {}
    counts = {}
    diagnostics = {}
    for token, payload in sorted(external_payloads.items()):
        candidates = [
            dict(item)
            for item in payload.get("candidates") or []
            if item.get("candidate_source") == "scan_mesh_adjacency"
        ]
        features = _mesh_case_features(candidates, payload)
        prediction = _predict_mesh_rows(
            model,
            [(candidate, values, 0, 0.0) for candidate, values in zip(candidates, features)],
        )
        selected = _select(prediction, config)
        short_completion = _complete_short_mesh(selected, payload.get("line_objects") or [], config)
        selected = list(short_completion["selected"])
        consensus_snap = _snap_mesh_consensus(selected, payload.get("line_objects") or [], config)
        selected = list(consensus_snap["selected"])
        wedge = _complete_surface_wedges(selected, payload.get("line_objects") or [], config)
        selected = list(wedge["selected"])
        lane = complete_dominant_fold_lane(payload.get("line_objects") or [])
        if lane["eligible"]:
            selected = list(lane["selected"])
        siblings = _complete_parallel_siblings(selected, payload.get("candidates") or [])
        selected = list(siblings["selected"])
        fragments = _prune_parallel_fragments(selected)
        selected = list(fragments["selected"])
        short_seeds = _complete_scan_supported_short_seeds(
            selected,
            payload.get("line_objects") or [],
            payload.get("candidates") or [],
        )
        selected = list(short_seeds["selected"])
        weak_outlines = _prune_weak_parallel_outlines(selected)
        selected = list(weak_outlines["selected"])
        graph_outlines = _prune_surface_graph_outlines(selected)
        selected = list(graph_outlines["selected"])

        output_path = args.output_root / "predictions" / token / "decomposition_result.json"
        result = _prediction_payload(token, selected, config)
        result["diagnostics"] = {
            "raw_mesh_candidate_count": len(candidates),
            "short_completion": {k: v for k, v in short_completion.items() if k != "selected"},
            "consensus_snap": {k: v for k, v in consensus_snap.items() if k != "selected"},
            "surface_wedge_completion": {k: v for k, v in wedge.items() if k != "selected"},
            "dominant_fold_lane": {k: v for k, v in lane.items() if k != "selected"},
            "parallel_sibling_completion": {k: v for k, v in siblings.items() if k != "selected"},
            "fragment_pruning": {k: v for k, v in fragments.items() if k != "selected"},
            "short_seed_completion": {k: v for k, v in short_seeds.items() if k != "selected"},
            "weak_outline_pruning": {k: v for k, v in weak_outlines.items() if k != "selected"},
            "surface_graph_outline_pruning": {k: v for k, v in graph_outlines.items() if k != "selected"},
        }
        _write(output_path, result)
        prediction_paths[token] = str(output_path)
        prediction_hashes[token] = _sha256(output_path)
        counts[token] = len(selected)
        diagnostics[token] = result["diagnostics"]

    manifest = {
        "schema_version": 1,
        "algorithm": "f1_prior12_mesh_adjacency_external_v1",
        "phase": "predictions_frozen_before_external_labels",
        "training_case_ids": training_ids,
        "evaluation_case_tokens": sorted(external_payloads),
        "selection_config": config,
        "training_tuning_metrics": tuning_metrics,
        "training_tuning_folds": folds,
        "feature_names": list(MESH_FEATURE_NAMES),
        "input_hashes": {
            "training_candidates": training_candidate_hashes,
            "training_labels": training_label_hashes,
            "external_candidates": external_hashes,
        },
        "prediction_paths": prediction_paths,
        "prediction_hashes": prediction_hashes,
        "prediction_counts": counts,
        "leakage_contract": {
            "external_labels_read_during_prediction": False,
            "external_sender_counts_read_during_prediction": False,
            "candidate_generation_uses_part_identifier": False,
            "selector_features_use_part_identifier_or_count": False,
            "source_points_serialized": False,
        },
    }
    _write(args.output_root / "prediction_manifest.json", manifest)
    print(json.dumps({"selection_config": config, "prediction_counts": counts}, indent=2))


def score(args: argparse.Namespace) -> None:
    manifest_path = args.output_root / "prediction_manifest.json"
    manifest = _read(manifest_path)
    if manifest.get("phase") != "predictions_frozen_before_external_labels":
        raise ValueError("predictions are not frozen in the expected phase")
    thresholds = Thresholds()
    cases = []
    total_tp = total_fp = total_fn = exact_geometry = exact_count = 0
    label_hashes = {}
    for token in manifest["evaluation_case_tokens"]:
        prediction_path = Path(manifest["prediction_paths"][token])
        if _sha256(prediction_path) != manifest["prediction_hashes"][token]:
            raise ValueError(f"{token}: frozen prediction hash changed")
        label_path = args.external_label_root / token / "bend_instance_labels.json"
        label_hashes[token] = _sha256(label_path)
        labels = _read(label_path)
        gold = list(labels.get("manual_bends") or [])
        predictions = list(_read(prediction_path).get("owned_bend_regions") or [])
        result = match_lines(gold, predictions, thresholds)
        total_tp += int(result["true_positive"])
        total_fp += int(result["false_positive"])
        total_fn += int(result["false_negative"])
        exact_geometry += int(result["exact"])
        exact_count += int(len(gold) == len(predictions))
        cases.append(
            {
                "case_token": token,
                "part_id": labels.get("part_id"),
                "role": labels.get("role"),
                "gold_count": len(gold),
                "prediction_count": len(predictions),
                **result,
            }
        )
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "schema_version": 1,
        "prediction_manifest_path": str(manifest_path),
        "prediction_manifest_sha256": _sha256(manifest_path),
        "score_phase_only": True,
        "external_label_hashes": label_hashes,
        "thresholds": thresholds.__dict__,
        "metrics": {
            "case_count": len(cases),
            "gold_count": sum(row["gold_count"] for row in cases),
            "prediction_count": sum(row["prediction_count"] for row in cases),
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_geometry_case_count": exact_geometry,
            "exact_geometry_case_rate": exact_geometry / len(cases),
            "exact_count_case_count": exact_count,
            "exact_count_case_rate": exact_count / len(cases),
        },
        "cases": cases,
        "limitations": [
            "External labels are assistant-reviewed and user-review-pending.",
            "The 38172000 pair was previously analyzed and corrected before this external selector run.",
            "This is research evidence, not an enterprise promotion benchmark.",
        ],
    }
    _write(args.output_root / "locked_score.json", report)
    print(json.dumps(report["metrics"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "score"))
    parser.add_argument("--training-candidate-root", type=Path, default=DEFAULT_SHAKED_CANDIDATES)
    parser.add_argument("--training-label-root", type=Path, default=DEFAULT_SHAKED_LABELS)
    parser.add_argument("--external-candidate-root", type=Path, default=DEFAULT_EXTERNAL_CANDIDATES)
    parser.add_argument("--external-label-root", type=Path, default=DEFAULT_EXTERNAL_LABELS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for name in (
        "training_candidate_root",
        "training_label_root",
        "external_candidate_root",
        "external_label_root",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.phase == "predict":
        predict(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
