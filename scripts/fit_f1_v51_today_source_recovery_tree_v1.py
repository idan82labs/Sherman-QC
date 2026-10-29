#!/usr/bin/env python3
"""Fit the consumed-development V51 source-only recovery tree.

This is the only truth-aware component in the recovery experiment.  Its output
contains a frozen feature-only decision tree and operating constants; it does
not contain case IDs, bend IDs, expected counts, endpoints, part identities, or
per-case decisions.  Runtime inference is implemented separately in
``apply_f1_v51_source_only_recovery_v1.py`` and cannot open the catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_f1_v51_source_only_recovery_v1 import (  # noqa: E402
    MODEL_KIND,
    candidate_endpoints,
)
from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    Thresholds,
    is_match,
    pair_metrics,
)


FEATURE_NAMES = [
    "case_percentile_score",
    "case_percentile_consensus_count",
    "case_percentile_cluster_source_count",
    "case_percentile_support_count",
    "case_percentile_finite_support_fraction",
    "case_percentile_endpoint_support_fraction",
    "case_percentile_normal_second_eigenvalue",
    "case_percentile_normal_spread_deg_scan",
    "case_percentile_tight_normal_second_eigenvalue_r2mm",
    "case_percentile_tight_to_wide_normal_second_ratio",
    "case_percentile_corridor_points_per_mm",
    "case_percentile_transverse_angular_span_deg",
    "case_percentile_cluster_member_count",
    "case_percentile_two_sided_transition_score",
    "case_percentile_two_sided_normal_angle_deg",
    "case_percentile_two_sided_support_balance",
    "case_percentile_two_sided_min_support_fraction",
    "case_percentile_two_sided_normal_coherence",
    "finite_support_fraction",
    "endpoint_support_fraction",
    "normal_second_eigenvalue",
    "two_sided_transition_score",
    "two_sided_normal_angle_deg",
    "two_sided_support_balance",
    "two_sided_min_support_fraction",
    "two_sided_normal_coherence",
    "cluster_source_count",
    "cluster_member_count",
    "median_line_distance_mm",
    "p90_line_distance_mm",
    "corridor_points_per_mm",
    "transverse_angular_span_deg",
    "tight_normal_second_eigenvalue_r2mm",
    "tight_to_wide_normal_second_ratio",
]

DEFAULT_CATALOG = (
    ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
)
DEFAULT_CACHE_ROOT = (
    ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2/primary"
)
DEFAULT_OUTPUT = (
    ROOT / "data/bend_corpus/models/f1_v51_today_source_recovery_tree_v1_r3.json"
)
TODAY_SCAN_ROLE = "2026-08-30_ai_multiview_consensus_development"
MATCH_THRESHOLDS = Thresholds(5.0, 4.0, 0.60, 0.60)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _features(row: dict[str, Any]) -> list[float]:
    values = []
    for name in FEATURE_NAMES:
        value = row.get(name)
        number = 0.0 if value is None else float(value)
        if not np.isfinite(number):
            raise ValueError(f"non-finite feature {name}")
        values.append(number)
    return values


def _gold_endpoints(row: dict[str, Any]) -> np.ndarray:
    endpoints = np.asarray([row.get("line_start"), row.get("line_end")], dtype=np.float64)
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("gold endpoint record is invalid")
    return endpoints


def _today_cases(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [
        dict(row)
        for row in catalog.get("cases") or []
        if row.get("scan_role") == TODAY_SCAN_ROLE
    ]
    if [row.get("case_id") for row in cases] != [f"T{index:03d}" for index in range(1, 29)]:
        raise ValueError("training catalog must contain ordered T001..T028 daily cases")
    if any(row.get("truth_tier") != "codex_manual_multiview_scan_only" for row in cases):
        raise ValueError("unexpected daily truth tier")
    return cases


def fit_model(catalog_path: Path, cache_root: Path) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    cache_root = cache_root.resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    cases = _today_cases(catalog)
    feature_rows: list[list[float]] = []
    labels: list[bool] = []
    pool_hashes: list[str] = []
    positive_gold_ids: set[str] = set()
    for case in cases:
        case_id = str(case["case_id"])
        pool_path = cache_root / case_id / "pool" / f"{case_id}.json"
        if not pool_path.is_file():
            raise FileNotFoundError(f"{case_id}: cached candidate pool is missing")
        pool_hashes.append(_sha(pool_path))
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        if int(pool.get("working_point_budget") or 0) != 200000:
            raise ValueError(f"{case_id}: cached pool point budget mismatch")
        gold = [(str(row["bend_id"]), _gold_endpoints(row)) for row in case["bends"]]
        for candidate in pool.get("line_objects") or []:
            endpoints = candidate_endpoints(candidate)
            hits = [
                bend_id
                for bend_id, truth_endpoints in gold
                if is_match(pair_metrics(truth_endpoints, endpoints), MATCH_THRESHOLDS)
            ]
            feature_rows.append(_features(candidate))
            labels.append(bool(hits))
            positive_gold_ids.update(hits)
    matrix = np.asarray(feature_rows, dtype=np.float64)
    target = np.asarray(labels, dtype=np.bool_)
    if matrix.shape != (10115, len(FEATURE_NAMES)):
        raise ValueError(f"unexpected training matrix shape: {matrix.shape}")
    if int(target.sum()) != 131:
        raise ValueError(f"unexpected positive candidate row count: {int(target.sum())}")

    classifier = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=0,
        criterion="entropy",
    )
    classifier.fit(matrix, target)
    tree = classifier.tree_
    probabilities = []
    for node in range(tree.node_count):
        values = np.asarray(tree.value[node]).reshape(-1)
        probabilities.append(float(values[1] / values.sum()))
    payload = {
        "schema_version": 1,
        "kind": MODEL_KIND,
        "model_name": "V51_SOURCE_ONLY_RECOVERY_TREE_V1",
        "development_tuned_model": True,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "training_evidence_class": "CONSUMED_AI_ADJUDICATED_PHYSICAL_DEVELOPMENT",
        "training_catalog_sha256": _sha(catalog_path),
        "training_pool_set_sha256": hashlib.sha256("\n".join(pool_hashes).encode()).hexdigest(),
        "training_candidate_row_count": int(matrix.shape[0]),
        "training_positive_candidate_row_count": int(target.sum()),
        "training_unique_gold_coverage_count": len(positive_gold_ids),
        "feature_names": FEATURE_NAMES,
        "tree_parameters": {
            "criterion": "entropy",
            "max_depth": 12,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "random_state": 0,
        },
        "tree": {
            "children_left": [int(value) for value in tree.children_left],
            "children_right": [int(value) for value in tree.children_right],
            "feature_index": [int(value) for value in tree.feature],
            "threshold": [float(value) for value in tree.threshold],
            "positive_probability": probabilities,
        },
        "profiles": {
            "balanced": {
                "tree_probability_threshold": 0.993482262799,
                "tree_recovery_cap": 3,
                "duplicate_angle_deg": 5.0,
                "duplicate_offset_mm": 4.0,
                "duplicate_span_overlap": 0.60,
                "adjacency_supplement": {
                    "candidate_source": "scan_mesh_adjacency",
                    "min_confidence": 0.98,
                    "min_transition_score": 0.30,
                    "max_transition_score": 0.50,
                    "min_normal_second_eigenvalue": 0.31,
                    "min_corridor_points_per_mm": 14.0,
                    "emission_score": 1.001,
                },
                "narrow_replacement": {
                    "min_tree_probability": 1.0,
                    "recovery_source_prefixes": ["cylindrical_standard", "cylindrical_wide"],
                    "incumbent_source_kinds": ["scan_mesh_adjacency"],
                    "max_incumbent_selector_score": 0.70,
                },
            },
            "recall_max": {
                "tree_probability_threshold": 0.974429045481,
                "tree_recovery_cap": 3,
                "duplicate_angle_deg": 5.0,
                "duplicate_offset_mm": 4.0,
                "duplicate_span_overlap": 0.60,
                "adjacency_supplement": {
                    "candidate_source": "scan_mesh_adjacency",
                    "min_confidence": 0.98,
                    "min_transition_score": 0.30,
                    "max_transition_score": 0.50,
                    "min_normal_second_eigenvalue": 0.31,
                    "min_corridor_points_per_mm": 14.0,
                    "emission_score": 1.001,
                },
                "narrow_replacement": {
                    "min_tree_probability": 0.974429045481,
                    "recovery_source_prefixes": ["cylindrical_standard", "cylindrical_wide"],
                    "incumbent_source_kinds": [
                        "scan_mesh_adjacency",
                        "cylindrical_standard",
                        "cylindrical_wide"
                    ],
                    "max_incumbent_selector_score": 0.70,
                },
            },
        },
        "inference_contract": {
            "catalog_opened_at_inference": False,
            "labels_opened_at_inference": False,
            "expected_count_used_at_inference": False,
            "cad_used_at_inference": False,
            "drawing_used_at_inference": False,
            "part_identity_used_at_inference": False,
            "runtime_inputs": ["retained_v51_semantic_result", "scan_only_candidate_pool", "this_model"],
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = fit_model(args.catalog, args.cache_root)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a frozen recovery model")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(payload))
    print(
        json.dumps(
            {
                "model": str(output),
                "model_sha256": _sha(output),
                "node_count": len(payload["tree"]["children_left"]),
                "training_candidate_rows": payload["training_candidate_row_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
