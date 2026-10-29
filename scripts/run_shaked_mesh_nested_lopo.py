#!/usr/bin/env python3
"""Nested leave-one-part-out development evaluation for Shaked mesh F1.

Every outer case is predicted by a model that excludes that case's labels.
The probability threshold is chosen by inner leave-one-case-out evaluation on
the remaining cases.  This is a grouped development checkpoint, not a fresh
blind or enterprise-release benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines  # noqa: E402
from scripts.run_shaked_selector_evaluation import (  # noqa: E402
    DEFAULT_SHAKED_CANDIDATES,
    DEFAULT_SHAKED_LABELS,
    DEFAULT_SPLIT,
    MESH_FEATURE_NAMES,
    _build_mesh_rows,
    _complete_short_mesh,
    _complete_surface_wedges,
    _complete_parallel_siblings,
    _complete_scan_supported_short_seeds,
    _prune_weak_parallel_outlines,
    _prune_surface_graph_outlines,
    _prune_parallel_fragments,
    _fit_mesh_rows,
    _line_metrics,
    _load_candidate_payloads,
    _load_shaked_gold,
    _predict_mesh_rows,
    _prediction_case_payload,
    _read,
    _select,
    _sha256,
    _snap_mesh_consensus,
    _tune_mesh,
    _write,
    validate_split,
)
from backend.f1_dominant_fold_lane_completion import (  # noqa: E402
    complete_dominant_fold_lane,
)

DEFAULT_OUTPUT = ROOT / "output/experiments/shaked-mesh-nested-lopo-v6"


def outer_training_case_ids(case_ids: list[str], held_out: str) -> list[str]:
    return [case_id for case_id in case_ids if case_id != held_out]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_SHAKED_CANDIDATES)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_SHAKED_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    candidate_root = args.candidate_root.resolve()
    label_root = args.label_root.resolve()
    split_path = args.split.resolve()
    output_dir = args.output_dir.resolve()

    split = _read(split_path)
    case_ids = list(split["development_case_ids"]) + list(split["locked_case_ids"])
    candidate_paths = {
        path.stem: path for path in sorted(candidate_root.glob("[0-9][0-9]_*.json"))
    }
    validate_split(split, candidate_paths)
    payloads, candidate_hashes = _load_candidate_payloads(candidate_root, case_ids)
    gold, label_hashes = _load_shaked_gold(label_root, case_ids)
    rows = _build_mesh_rows(payloads, gold)

    predictions = {}
    prediction_paths = {}
    folds = []
    for held_out in case_ids:
        training_ids = outer_training_case_ids(case_ids, held_out)
        config, inner_metrics, inner_folds = _tune_mesh(
            rows,
            gold,
            training_ids,
            line_objects_by_case={
                case_id: payloads[case_id].get("line_objects") or ()
                for case_id in training_ids
            },
        )
        model = _fit_mesh_rows(
            rows,
            training_ids,
            logistic_c=float(config["logistic_c"]),
        )
        selected = _select(_predict_mesh_rows(model, rows[held_out]), config)
        short_completion = _complete_short_mesh(
            selected,
            payloads[held_out].get("line_objects") or (),
            config,
        )
        selected = list(short_completion["selected"])
        consensus_snap = _snap_mesh_consensus(
            selected,
            payloads[held_out].get("line_objects") or (),
            config,
        )
        selected = list(consensus_snap["selected"])
        wedge_completion = _complete_surface_wedges(
            selected,
            payloads[held_out].get("line_objects") or (),
            config,
        )
        selected = list(wedge_completion["selected"])
        lane = complete_dominant_fold_lane(payloads[held_out].get("line_objects") or ())
        if lane["eligible"]:
            selected = list(lane["selected"])
        sibling_completion = _complete_parallel_siblings(
            selected,
            payloads[held_out].get("candidates") or (),
        )
        selected = list(sibling_completion["selected"])
        fragment_pruning = _prune_parallel_fragments(selected)
        selected = list(fragment_pruning["selected"])
        seed_completion = _complete_scan_supported_short_seeds(
            selected,
            payloads[held_out].get("line_objects") or (),
            payloads[held_out].get("candidates") or (),
        )
        selected = list(seed_completion["selected"])
        outline_pruning = _prune_weak_parallel_outlines(selected)
        selected = list(outline_pruning["selected"])
        surface_graph_pruning = _prune_surface_graph_outlines(selected)
        selected = list(surface_graph_pruning["selected"])
        predictions[held_out] = selected
        path = output_dir / "predictions" / held_out / "decomposition_result.json"
        prediction_payload = _prediction_case_payload(
            held_out,
            selected,
            "mesh_nested_leave_one_part_out",
            config,
            training_ids,
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
        prediction_paths[held_out] = str(path)
        folds.append(
            {
                "held_out_case_id": held_out,
                "training_case_ids": training_ids,
                "selection_config": config,
                "inner_metrics": inner_metrics,
                "inner_folds": inner_folds,
                "same_case_labels_in_training": False,
                "same_case_labels_used_for_threshold_selection": False,
                "dominant_fold_lane_completion": {
                    key: value for key, value in lane.items() if key != "selected"
                },
                "short_scan_mesh_bend_completion": {
                    key: value for key, value in short_completion.items() if key != "selected"
                },
                "mesh_cross_source_consensus_snap": {
                    key: value for key, value in consensus_snap.items() if key != "selected"
                },
                "surface_wedge_completion": {
                    key: value for key, value in wedge_completion.items() if key != "selected"
                },
                "parallel_surface_sibling_completion": {
                    key: value for key, value in sibling_completion.items() if key != "selected"
                },
                "contained_parallel_fragment_pruning": {
                    key: value for key, value in fragment_pruning.items() if key != "selected"
                },
                "scan_supported_short_seed_completion": {
                    key: value for key, value in seed_completion.items() if key != "selected"
                },
                "weak_parallel_outline_pruning": {
                    key: value for key, value in outline_pruning.items() if key != "selected"
                },
                "surface_graph_outline_pruning": {
                    key: value
                    for key, value in surface_graph_pruning.items()
                    if key != "selected"
                },
            }
        )

    aggregate = _line_metrics(predictions, gold)
    case_rows = []
    for case_id in case_ids:
        result = match_lines(gold[case_id], predictions[case_id], Thresholds())
        case_rows.append(
            {
                "case_id": case_id,
                "gold_count": len(gold[case_id]),
                "prediction_count": len(predictions[case_id]),
                **result,
            }
        )
    report = {
        "schema_version": 1,
        "algorithm": "f1_shaked_mesh_nested_lopo_v8",
        "evaluation_mode": "nested_leave_one_physical_part_out_development",
        "case_count": len(case_ids),
        "metrics": {
            **aggregate,
            "precision": (
                aggregate["tp"] / (aggregate["tp"] + aggregate["fp"])
                if aggregate["tp"] + aggregate["fp"]
                else 0.0
            ),
            "recall": (
                aggregate["tp"] / (aggregate["tp"] + aggregate["fn"])
                if aggregate["tp"] + aggregate["fn"]
                else 0.0
            ),
            "exact_geometry_case_rate": aggregate["exact"] / len(case_ids),
            "exact_count_case_count": sum(
                int(row["gold_count"] == row["prediction_count"]) for row in case_rows
            ),
        },
        "cases": case_rows,
        "folds": folds,
        "feature_names": list(MESH_FEATURE_NAMES),
        "prediction_paths": prediction_paths,
        "input_hashes": {
            "split": _sha256(split_path),
            "candidates": candidate_hashes,
            "labels": label_hashes,
        },
        "leakage_contract": {
            "same_case_labels_in_training": False,
            "same_case_labels_used_for_threshold_selection": False,
            "candidate_generation_uses_part_identifier": False,
            "candidate_generation_uses_expected_count": False,
            "candidate_generation_uses_manual_truth": False,
            "candidate_generation_uses_cad": False,
            "source_points_serialized": False,
            "all_labels_are_used_in_other_outer_folds": True,
        },
        "limitations": [
            "This is development cross-validation after the corpus was visually analyzed, not a fresh blind benchmark.",
            "Shaked labels are assistant-reviewed, user-review-pending, and non-authoritative.",
            "Candidate-coupled axes are not fully independent endpoint ground truth.",
        ],
    }
    _write(output_dir / "nested_lopo_line_accuracy.json", report)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
