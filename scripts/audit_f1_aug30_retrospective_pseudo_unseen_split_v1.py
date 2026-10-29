#!/usr/bin/env python3
"""Source-free audit for the Aug-30 retrospective pseudo-unseen split.

This module deliberately knows only the sealed research artifacts.  It does
not discover or open scans, labels, predictions, evaluator products, CAD,
drawings, sender metadata, original filenames, or part identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "research"

PREREG_NAME = "f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
DESCRIPTORS_NAME = "f1_aug30_retrospective_pseudo_unseen_descriptors_v1.json"
SPLIT_NAME = "f1_aug30_retrospective_pseudo_unseen_split_v1.json"
TRAIN_ALLOWLIST_NAME = "f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
HASHES_NAME = "f1_aug30_retrospective_pseudo_unseen_hashes_v1.json"
SEAL_NAME = "f1_aug30_retrospective_pseudo_unseen_hashes_v1.sha256"

TIERS = ("SIMILAR_FAMILY", "ADJACENT_FAMILY", "DISTANT_FAMILY")


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distance(left: Iterable[float], right: Iterable[float]) -> float:
    a = tuple(float(value) for value in left)
    b = tuple(float(value) for value in right)
    if len(a) != len(b) or not a:
        raise ValueError("descriptor vectors must have the same positive width")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def select_tiers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the deterministic geometry-only tier selection.

    The only inputs consumed are opaque case IDs and normalized geometry
    vectors.  Opaque identity is used solely as a stable tie breaker.
    """

    ordered = sorted(rows, key=lambda row: str(row["case_id"]))
    case_ids = [str(row["case_id"]) for row in ordered]
    if len(case_ids) != 28 or len(set(case_ids)) != 28:
        raise ValueError("tier selection requires 28 unique opaque cases")
    vectors = {
        str(row["case_id"]): tuple(float(value) for value in row["normalized_vector"])
        for row in ordered
    }
    widths = {len(vector) for vector in vectors.values()}
    if len(widths) != 1 or next(iter(widths)) == 0:
        raise ValueError("normalized vectors have inconsistent widths")
    if not all(math.isfinite(value) for vector in vectors.values() for value in vector):
        raise ValueError("normalized vectors must be finite")

    pair_distance: dict[tuple[str, str], float] = {}
    neighbors: dict[str, list[tuple[float, str]]] = {}
    for case_id in case_ids:
        ranked = []
        for other_id in case_ids:
            if other_id == case_id:
                continue
            key = tuple(sorted((case_id, other_id)))
            if key not in pair_distance:
                pair_distance[key] = _distance(vectors[case_id], vectors[other_id])
            ranked.append((pair_distance[key], other_id))
        neighbors[case_id] = sorted(ranked, key=lambda item: (item[0], item[1]))

    novelty = {
        case_id: sum(distance for distance, _ in neighbors[case_id][:3]) / 3.0
        for case_id in case_ids
    }
    sparse_order = sorted(case_ids, key=lambda case_id: (-novelty[case_id], case_id))
    dense_order = sorted(case_ids, key=lambda case_id: (novelty[case_id], case_id))
    dense_cutoff = sorted(novelty.values())[(len(novelty) - 1) // 2]

    distant = sparse_order[:3]
    holdout = set(distant)
    reserved_train: set[str] = set()
    anchors: dict[str, list[str]] = {case_id: [] for case_id in distant}

    adjacent: list[str] = []
    for case_id in sparse_order[3:]:
        if case_id in holdout or case_id in reserved_train:
            continue
        retained = next(
            (
                other_id
                for _, other_id in neighbors[case_id]
                if other_id not in holdout and other_id not in adjacent
            ),
            None,
        )
        if retained is None:
            continue
        adjacent.append(case_id)
        holdout.add(case_id)
        reserved_train.add(retained)
        anchors[case_id] = [retained]
        if len(adjacent) == 3:
            break
    if len(adjacent) != 3:
        raise ValueError("could not select three adjacent cases")

    similar: list[str] = []
    for case_id in dense_order:
        if novelty[case_id] > dense_cutoff + 1e-15:
            break
        if case_id in holdout or case_id in reserved_train:
            continue
        retained = [
            other_id
            for _, other_id in neighbors[case_id]
            if other_id not in holdout and other_id not in similar
        ][:2]
        if len(retained) != 2:
            continue
        similar.append(case_id)
        holdout.add(case_id)
        reserved_train.update(retained)
        anchors[case_id] = retained
        if len(similar) == 4:
            break
    if len(similar) != 4:
        raise ValueError("could not select four dense interpolation cases")

    train = sorted(set(case_ids) - holdout)
    if len(train) != 18 or len(holdout) != 10:
        raise AssertionError("unexpected split cardinality")
    if not reserved_train.issubset(train):
        raise AssertionError("a retained neighbor escaped the training partition")

    rank_desc = {case_id: index + 1 for index, case_id in enumerate(sparse_order)}
    nearest_train: dict[str, list[dict[str, Any]]] = {}
    for case_id in sorted(holdout):
        nearest_train[case_id] = [
            {"case_id": other_id, "distance": round(distance, 12)}
            for distance, other_id in neighbors[case_id]
            if other_id in train
        ][:3]

    return {
        "train": train,
        "tiers": {
            "SIMILAR_FAMILY": sorted(similar),
            "ADJACENT_FAMILY": sorted(adjacent),
            "DISTANT_FAMILY": sorted(distant),
        },
        "retained_neighbor_anchors": {
            case_id: sorted(case_anchors) for case_id, case_anchors in sorted(anchors.items())
        },
        "nearest_train": nearest_train,
        "novelty": {case_id: round(novelty[case_id], 12) for case_id in case_ids},
        "novelty_rank_desc": rank_desc,
        "dense_cutoff": round(dense_cutoff, 12),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object: {path.name}")
    return payload


def _assert_case_records_are_outcome_free(records: list[dict[str, Any]]) -> None:
    forbidden = {
        "label",
        "labels",
        "truth",
        "prediction",
        "predictions",
        "score",
        "scores",
        "tp",
        "fp",
        "fn",
        "sender_count",
        "expected_count",
        "cad",
        "drawing",
        "filename",
        "original_filename",
        "part_id",
    }
    for record in records:
        overlap = forbidden.intersection(str(key).lower() for key in record)
        if overlap:
            raise AssertionError(f"outcome/identity field leaked into case record: {sorted(overlap)}")


def audit_artifacts(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    *,
    verify_code_bindings: bool = True,
) -> dict[str, Any]:
    prereg_path = artifact_dir / PREREG_NAME
    descriptors_path = artifact_dir / DESCRIPTORS_NAME
    split_path = artifact_dir / SPLIT_NAME
    allowlist_path = artifact_dir / TRAIN_ALLOWLIST_NAME
    hashes_path = artifact_dir / HASHES_NAME
    seal_path = artifact_dir / SEAL_NAME

    prereg = _read_json(prereg_path)
    descriptors = _read_json(descriptors_path)
    split = _read_json(split_path)
    allowlist = _read_json(allowlist_path)
    hashes = _read_json(hashes_path)

    if prereg.get("status") != "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN":
        raise AssertionError("prereg is not explicitly retrospective pseudo-unseen")
    if prereg.get("historical_cohort_consumed") is not True:
        raise AssertionError("historical consumption disclosure is missing")
    if prereg.get("prospective_or_unseen_claim_authorized") is not False:
        raise AssertionError("prospective/unseen claim must be forbidden")
    if prereg.get("model_fit_performed") is not False or prereg.get("model_score_performed") is not False:
        raise AssertionError("split freeze must not fit or score a model")

    gates = prereg.get("retention_gates") or {}
    expected_tier_floor = {
        "SIMILAR_FAMILY": -0.02,
        "ADJACENT_FAMILY": -0.02,
        "DISTANT_FAMILY": 0.0,
    }
    if gates.get("overall_holdout_micro_f1_min_delta") != 0.05:
        raise AssertionError("overall micro-F1 gate drifted")
    if gates.get("holdout_tp_min_delta") != 0:
        raise AssertionError("TP nondecrease gate drifted")
    if gates.get("holdout_fp_max_increase") != {
        "formula": "max(2, 0.10 * baseline_fp)",
        "absolute_floor": 2,
        "baseline_fraction": 0.1,
    }:
        raise AssertionError("FP cap drifted")
    if gates.get("tier_micro_f1_min_delta") != expected_tier_floor:
        raise AssertionError("tier regression gates drifted")
    if gates.get("exact_case_count_min_delta") != 0:
        raise AssertionError("exact-case gate drifted")
    if gates.get("primary_replay_bytes") != "EXACT_SHA256_IDENTITY_REQUIRED":
        raise AssertionError("primary/replay byte gate drifted")
    if gates.get("all_gates_conjunctive") is not True:
        raise AssertionError("retention gates must be conjunctive")

    descriptor_rows = descriptors.get("cases") or []
    assignment_rows = split.get("assignments") or []
    allowlist_rows = allowlist.get("cases") or []
    if len(descriptor_rows) != 28 or len(assignment_rows) != 28 or len(allowlist_rows) != 18:
        raise AssertionError("unexpected artifact case counts")
    _assert_case_records_are_outcome_free(descriptor_rows)
    _assert_case_records_are_outcome_free(assignment_rows)
    _assert_case_records_are_outcome_free(allowlist_rows)

    ids_by_artifact = [
        {str(row["case_id"]) for row in records}
        for records in (descriptor_rows, assignment_rows)
    ]
    if ids_by_artifact[0] != ids_by_artifact[1] or len(ids_by_artifact[0]) != 28:
        raise AssertionError("case identity sets differ")
    sha_by_descriptor = {str(row["case_id"]): str(row["scan_sha256"]) for row in descriptor_rows}
    sha_by_assignment = {str(row["case_id"]): str(row["scan_sha256"]) for row in assignment_rows}
    if sha_by_descriptor != sha_by_assignment or len(set(sha_by_descriptor.values())) != 28:
        raise AssertionError("scan hash binding mismatch")

    recomputed = select_tiers(descriptor_rows)
    expected_tiers = {
        tier: sorted(str(case_id) for case_id in split["holdout_tiers"][tier])
        for tier in TIERS
    }
    if recomputed["tiers"] != expected_tiers:
        raise AssertionError("stored tier membership is not reproducible from descriptors")
    if recomputed["train"] != sorted(str(case_id) for case_id in split["train_case_ids"]):
        raise AssertionError("stored training membership is not reproducible")

    train_rows = [row for row in assignment_rows if row.get("split") == "TRAIN"]
    holdout_rows = [row for row in assignment_rows if row.get("split") == "HOLDOUT"]
    if len(train_rows) != 18 or len(holdout_rows) != 10:
        raise AssertionError("split is not 18/10")
    if any(row.get("holdout_locked") is not True for row in holdout_rows):
        raise AssertionError("every holdout case must be locked")
    if any(row.get("model_tuning_allowed") is not False for row in holdout_rows):
        raise AssertionError("holdout model tuning must be forbidden")
    if any(row.get("tier") not in TIERS for row in holdout_rows):
        raise AssertionError("holdout tier is missing")
    if any(row.get("tier") is not None for row in train_rows):
        raise AssertionError("training cases cannot carry holdout tiers")

    train_ids = sorted(str(row["case_id"]) for row in train_rows)
    if train_ids != sorted(str(row["case_id"]) for row in allowlist_rows):
        raise AssertionError("train allowlist does not match split")
    if {str(row["case_id"]): str(row["scan_sha256"]) for row in allowlist_rows} != {
        case_id: sha_by_assignment[case_id] for case_id in train_ids
    }:
        raise AssertionError("train allowlist scan hashes do not match split")

    for case_id in recomputed["tiers"]["SIMILAR_FAMILY"]:
        anchors = recomputed["retained_neighbor_anchors"][case_id]
        if len(anchors) != 2 or not set(anchors).issubset(train_ids):
            raise AssertionError("similar holdout lacks two retained train neighbors")
    for case_id in recomputed["tiers"]["ADJACENT_FAMILY"]:
        anchors = recomputed["retained_neighbor_anchors"][case_id]
        if len(anchors) != 1 or anchors[0] not in train_ids:
            raise AssertionError("adjacent holdout lacks a retained train anchor")

    artifact_bindings = hashes.get("artifact_bindings") or []
    binding_by_path = {str(row["path"]): row for row in artifact_bindings}
    required_sealed = {
        f"research/{PREREG_NAME}": prereg_path,
        f"research/{DESCRIPTORS_NAME}": descriptors_path,
        f"research/{SPLIT_NAME}": split_path,
        f"research/{TRAIN_ALLOWLIST_NAME}": allowlist_path,
    }
    for relative, local_path in required_sealed.items():
        binding = binding_by_path.get(relative)
        if binding is None:
            raise AssertionError(f"missing artifact binding: {relative}")
        if _sha256(local_path) != binding.get("sha256"):
            raise AssertionError(f"artifact hash mismatch: {relative}")
        if local_path.stat().st_size != binding.get("size_bytes"):
            raise AssertionError(f"artifact size mismatch: {relative}")

    if verify_code_bindings:
        for relative, binding in binding_by_path.items():
            if not relative.startswith(("scripts/", "tests/")):
                continue
            path = ROOT / relative
            if _sha256(path) != binding.get("sha256") or path.stat().st_size != binding.get("size_bytes"):
                raise AssertionError(f"code binding mismatch: {relative}")

    source_bindings = hashes.get("source_bindings") or {}
    raw_bindings = source_bindings.get("raw_scans") or []
    if len(raw_bindings) != 28:
        raise AssertionError("raw scan hash inventory must contain 28 cases")
    if {str(row["case_id"]): str(row["sha256"]) for row in raw_bindings} != sha_by_descriptor:
        raise AssertionError("source hash inventory does not match descriptor evidence")
    if any("path" in row or "filename" in row or "part_id" in row for row in raw_bindings):
        raise AssertionError("source inventory leaked a path or source identity")

    expected_seal = f"{_sha256(hashes_path)}  {HASHES_NAME}\n"
    if seal_path.read_text(encoding="ascii") != expected_seal:
        raise AssertionError("hash manifest seal mismatch")

    return {
        "status": "PASS",
        "source_free": True,
        "case_count": 28,
        "train_count": 18,
        "holdout_count": 10,
        "tier_counts": {tier: len(recomputed["tiers"][tier]) for tier in TIERS},
        "hash_manifest_sha256": _sha256(hashes_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--skip-code-bindings", action="store_true")
    args = parser.parse_args()
    result = audit_artifacts(
        args.artifact_dir,
        verify_code_bindings=not args.skip_code_bindings,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
