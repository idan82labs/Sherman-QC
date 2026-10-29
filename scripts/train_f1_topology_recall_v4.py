#!/usr/bin/env python3
"""Train V4: the V3 precision core plus a narrow adjacency recall route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import FEATURE_NAMES, endpoints, select_indices_configurable
from scripts.train_f1_topology_ranker_v3 import ALL_CASES, CASES_V2, load_case, metrics, model as core_model


PROFILE = "F1_TOPOLOGY_ADJACENCY_RECALL_V4"
ADJACENCY_SOURCE = "scan_mesh_adjacency"
ADJACENCY_THRESHOLD = 0.70
RANDOM_STATE = 20260811


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjacency_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _similar(first: dict, second: dict) -> bool:
    first_line = endpoints(first)
    second_line = endpoints(second)
    first_axis = first_line[1] - first_line[0]
    second_axis = second_line[1] - second_line[0]
    first_axis /= np.linalg.norm(first_axis)
    second_axis /= np.linalg.norm(second_axis)
    angle = np.degrees(np.arccos(np.clip(abs(float(first_axis @ second_axis)), -1.0, 1.0)))
    if angle > 5.0:
        return False
    first_mid = first_line.mean(axis=0)
    second_mid = second_line.mean(axis=0)
    delta = second_mid - first_mid
    first_distance = np.linalg.norm(delta - float(delta @ first_axis) * first_axis)
    second_distance = np.linalg.norm(delta - float(delta @ second_axis) * second_axis)
    if max(first_distance, second_distance) > 4.0:
        return False
    first_extent = np.sort((first_line - first_mid) @ first_axis)
    second_extent = np.sort((second_line - first_mid) @ first_axis)
    overlap = max(0.0, min(first_extent[1], second_extent[1]) - max(first_extent[0], second_extent[0]))
    return overlap / min(float(np.ptp(first_extent)), float(np.ptp(second_extent))) >= 0.60


def select(rows: list[dict], core_scores: np.ndarray, adjacency_scores: np.ndarray) -> list[int]:
    accepted = select_indices_configurable(
        rows,
        core_scores,
        threshold=0.50,
        max_angle=12.0,
        max_distance=10.0,
        min_overlap=0.30,
    )
    additions = [
        index
        for index, row in enumerate(rows)
        if row.get("representative_source") == ADJACENCY_SOURCE
        and float(adjacency_scores[index]) >= ADJACENCY_THRESHOLD
        and index not in accepted
    ]
    additions.sort(key=lambda index: (-float(adjacency_scores[index]), str(rows[index].get("candidate_id") or "")))
    for index in additions:
        if all(not _similar(rows[index], rows[prior]) for prior in accepted):
            accepted.append(index)
    return accepted


def fit_adjacency(data: dict, cases: list[str]) -> ExtraTreesClassifier:
    pairs = [
        (case, index)
        for case in cases
        for index, row in enumerate(data[case]["rows"])
        if row.get("representative_source") == ADJACENCY_SOURCE
    ]
    features = np.vstack([data[case]["features"][index] for case, index in pairs])
    labels = np.asarray([data[case]["object_ids"][index] is not None for case, index in pairs], dtype=np.uint8)
    if len(set(labels.tolist())) != 2:
        raise RuntimeError("adjacency route training requires both ownership classes")
    return adjacency_model().fit(features, labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--v2-development-dir", type=Path, required=True)
    parser.add_argument("--v3-development-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    data = {
        case: load_case(args.pool_dir, args.v2_development_dir if case in CASES_V2 else args.v3_development_dir, case)
        for case in ALL_CASES
    }
    held_out = []
    by_case = {}
    for case in ALL_CASES:
        train = [value for value in ALL_CASES if value != case]
        core = core_model().fit(np.vstack([data[value]["features"] for value in train]), np.concatenate([data[value]["target"] for value in train]))
        adjacency = fit_adjacency(data, train)
        core_scores = core.predict_proba(data[case]["features"])[:, 1]
        adjacency_scores = np.zeros(len(data[case]["rows"]), dtype=float)
        indices = [index for index, row in enumerate(data[case]["rows"]) if row.get("representative_source") == ADJACENCY_SOURCE]
        if indices:
            adjacency_scores[indices] = adjacency.predict_proba(data[case]["features"][indices])[:, 1]
        selected = select(data[case]["rows"], core_scores, adjacency_scores)
        item = (selected, data[case]["object_ids"], data[case]["expected"])
        held_out.append(item)
        by_case[case] = {**metrics([item]), "selected_count": len(selected), "candidate_pool_sha256": data[case]["pool_sha256"]}
    core = core_model().fit(np.vstack([data[value]["features"] for value in ALL_CASES]), np.concatenate([data[value]["target"] for value in ALL_CASES]))
    adjacency = fit_adjacency(data, list(ALL_CASES))
    bundle = {
        "profile": PROFILE,
        "feature_names": FEATURE_NAMES,
        "development_cases": ALL_CASES,
        "lock_sha256": sha(args.lock),
        "core_estimator": core,
        "adjacency_estimator": adjacency,
        "adjacency_threshold": ADJACENCY_THRESHOLD,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model, compress=3)
    report = {
        "schema_version": 1,
        "profile": PROFILE,
        "validation": "leave_one_family_out",
        "development_cases": list(ALL_CASES),
        "held_out_micro_metrics": metrics(held_out),
        "held_out_by_case": by_case,
        "model_sha256": sha(args.model),
        "lock_sha256": sha(args.lock),
        "terminal_transfer_feedback_used": False,
        "new_unseen_transfer_evidence": False,
        "c006_opened": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["held_out_micro_metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
