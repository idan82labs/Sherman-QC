#!/usr/bin/env python3
"""Build and seal the Aug-30 retrospective pseudo-unseen geometry split.

The sole cohort input is the fixed opaque render manifest named below and the
raw PLY bytes it binds.  The builder never discovers or reads labels, model
predictions, evaluator outputs, sender counts, CAD, drawings, or source
identity metadata.  It derives descriptors directly from geometry and does
not fit or score a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_f1_aug30_retrospective_pseudo_unseen_split_v1 import (  # noqa: E402
    DESCRIPTORS_NAME,
    HASHES_NAME,
    PREREG_NAME,
    SEAL_NAME,
    SPLIT_NAME,
    TRAIN_ALLOWLIST_NAME,
    _canonical_bytes,
    select_tiers,
)


INPUT_MANIFEST = ROOT / "data/bend_corpus/incoming/2026-08-30_shaked_daily_scans/opaque_render_manifest.json"
OUTPUT_DIR = ROOT / "research"
LOCKED_AT_UTC = "2026-08-31T16:17:42Z"
EXPECTED_MANIFEST_SCHEMA = "shermanqc.opaque_inference_manifest.v1"
EXPECTED_COHORT_DATE = "2026-08-30"
MAX_VERTEX_SAMPLE = 200_000
MAX_FACE_SAMPLE = 200_000

BUILDER_PATH = ROOT / "scripts/build_f1_aug30_retrospective_pseudo_unseen_split_v1.py"
AUDITOR_PATH = ROOT / "scripts/audit_f1_aug30_retrospective_pseudo_unseen_split_v1.py"
TEST_PATH = ROOT / "tests/test_build_f1_aug30_retrospective_pseudo_unseen_split_v1.py"

FEATURE_NAMES = (
    "log1p_vertex_count",
    "log1p_triangle_count",
    "log1p_rms_radius",
    "log1p_bbox_diagonal",
    "pca_variance_share_1",
    "pca_variance_share_2",
    "pca_variance_share_3",
    "pca_span_share_1",
    "pca_span_share_2",
    "pca_span_share_3",
    "radial_q25_over_rms",
    "radial_q50_over_rms",
    "radial_q90_over_rms",
    "edge_q25_over_rms",
    "edge_q50_over_rms",
    "edge_q90_over_rms",
    "log1p_surface_area_over_rms_sq",
    "normal_moment_share_1",
    "normal_moment_share_2",
    "normal_moment_share_3",
    "face_vertex_ratio",
)

VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
    ]
)
FACE_DTYPE = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        lines = []
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("PLY header ended before end_header")
            line = raw.decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
        header_bytes = handle.tell()
    if not lines or lines[0] != "ply" or "format binary_little_endian 1.0" not in lines:
        raise ValueError("only binary little-endian PLY input is accepted")
    elements = [line.split() for line in lines if line.startswith("element ")]
    element_counts = {parts[1]: int(parts[2]) for parts in elements}
    vertex_count = element_counts.get("vertex")
    face_count = element_counts.get("face")
    expected_properties = [
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property list uchar int vertex_indices",
    ]
    properties = [line for line in lines if line.startswith("property ")]
    if not vertex_count or not face_count or properties != expected_properties:
        raise ValueError("unexpected PLY element/property schema")
    expected_size = header_bytes + vertex_count * VERTEX_DTYPE.itemsize + face_count * FACE_DTYPE.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError("PLY byte size does not match the fixed triangle record schema")
    return header_bytes, vertex_count, face_count


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    sample_count = min(count, maximum)
    if sample_count == count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, sample_count, dtype=np.int64)


def _round(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded


def _geometry_descriptor(path: Path) -> dict[str, Any]:
    header_bytes, vertex_count, face_count = _parse_header(path)
    vertices = np.memmap(
        path,
        dtype=VERTEX_DTYPE,
        mode="r",
        offset=header_bytes,
        shape=(vertex_count,),
    )
    face_offset = header_bytes + vertex_count * VERTEX_DTYPE.itemsize
    faces = np.memmap(path, dtype=FACE_DTYPE, mode="r", offset=face_offset, shape=(face_count,))
    if not bool(np.all(faces["count"] == 3)):
        raise ValueError("non-triangle face found in fixed-width PLY input")

    vertex_indices = _sample_indices(vertex_count, MAX_VERTEX_SAMPLE)
    points = np.column_stack(
        (vertices["x"][vertex_indices], vertices["y"][vertex_indices], vertices["z"][vertex_indices])
    ).astype(np.float64, copy=False)
    vertex_normals = np.column_stack(
        (vertices["nx"][vertex_indices], vertices["ny"][vertex_indices], vertices["nz"][vertex_indices])
    ).astype(np.float64, copy=False)
    if not bool(np.isfinite(points).all()) or not bool(np.isfinite(vertex_normals).all()):
        raise ValueError("non-finite vertex geometry found")

    bbox_min = np.array(
        [float(vertices[axis].min()) for axis in ("x", "y", "z")], dtype=np.float64
    )
    bbox_max = np.array(
        [float(vertices[axis].max()) for axis in ("x", "y", "z")], dtype=np.float64
    )
    bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))

    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    variance_total = float(eigenvalues.sum())
    if variance_total <= 0:
        raise ValueError("degenerate zero-scale scan")
    rms_radius = math.sqrt(variance_total)
    variance_share = eigenvalues / variance_total

    projections = centered @ eigenvectors
    spans = np.quantile(projections, 0.99, axis=0) - np.quantile(projections, 0.01, axis=0)
    spans = np.sort(np.maximum(spans, 0.0))[::-1]
    span_share = spans / max(float(spans.sum()), 1e-15)
    radii = np.linalg.norm(centered, axis=1) / rms_radius
    radial_quantiles = np.quantile(radii, [0.25, 0.50, 0.90])

    face_indices = _sample_indices(face_count, MAX_FACE_SAMPLE)
    sampled_faces = np.asarray(faces["indices"][face_indices], dtype=np.int64)
    if sampled_faces.min() < 0 or sampled_faces.max() >= vertex_count:
        raise ValueError("face index outside vertex range")
    triangle_points = np.empty((len(sampled_faces), 3, 3), dtype=np.float64)
    for column, axis in enumerate(("x", "y", "z")):
        triangle_points[:, :, column] = vertices[axis][sampled_faces]
    edges_a = triangle_points[:, 1] - triangle_points[:, 0]
    edges_b = triangle_points[:, 2] - triangle_points[:, 0]
    cross = np.cross(edges_a, edges_b)
    double_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * double_area
    surface_area_estimate = float(areas.mean()) * face_count
    all_edges = np.concatenate(
        (
            np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 0], axis=1),
            np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 1], axis=1),
            np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 2], axis=1),
        )
    )
    edge_quantiles = np.quantile(all_edges / rms_radius, [0.25, 0.50, 0.90])

    normal_lengths = np.linalg.norm(vertex_normals, axis=1)
    valid_normals = normal_lengths > 1e-12
    if int(valid_normals.sum()) < 3:
        raise ValueError("insufficient finite nonzero vertex normals")
    unit_normals = vertex_normals[valid_normals] / normal_lengths[valid_normals, None]
    normal_moment = unit_normals.T @ unit_normals / len(unit_normals)
    normal_eigenvalues = np.linalg.eigvalsh(normal_moment)[::-1]
    normal_share = normal_eigenvalues / max(float(normal_eigenvalues.sum()), 1e-15)

    feature_values = (
        math.log1p(vertex_count),
        math.log1p(face_count),
        math.log1p(rms_radius),
        math.log1p(bbox_diagonal),
        *variance_share.tolist(),
        *span_share.tolist(),
        *radial_quantiles.tolist(),
        *edge_quantiles.tolist(),
        math.log1p(surface_area_estimate / (rms_radius * rms_radius)),
        *normal_share.tolist(),
        face_count / vertex_count,
    )
    if len(feature_values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in feature_values):
        raise AssertionError("invalid feature vector")
    return {
        "geometry_summary": {
            "vertex_count": vertex_count,
            "triangle_count": face_count,
            "sampled_vertex_count": len(vertex_indices),
            "sampled_triangle_count": len(face_indices),
            "bbox_diagonal": _round(bbox_diagonal),
            "rms_radius": _round(rms_radius),
            "surface_area_estimate": _round(surface_area_estimate),
            "zero_area_sample_fraction": _round(float(np.mean(double_area <= 1e-15))),
        },
        "feature_vector": {
            name: _round(value) for name, value in zip(FEATURE_NAMES, feature_values)
        },
    }


def _load_cases() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    if set(manifest) != {"schema", "cohort_date", "cases"}:
        raise ValueError("opaque input manifest top-level schema drift")
    if manifest["schema"] != EXPECTED_MANIFEST_SCHEMA or manifest["cohort_date"] != EXPECTED_COHORT_DATE:
        raise ValueError("unexpected opaque input manifest identity")
    rows = manifest["cases"]
    if not isinstance(rows, list) or len(rows) != 28:
        raise ValueError("expected exactly 28 opaque scan records")
    if any(set(row) != {"scan_path", "sha256"} for row in rows):
        raise ValueError("opaque scan record schema drift")
    hashes = [str(row["sha256"]) for row in rows]
    if len(set(hashes)) != 28 or any(len(value) != 64 for value in hashes):
        raise ValueError("scan hashes must be 28 unique SHA-256 values")

    permitted_root = INPUT_MANIFEST.parent.resolve()
    ordered_rows = sorted(rows, key=lambda row: str(row["sha256"]))
    cases = []
    for ordinal, row in enumerate(ordered_rows, start=1):
        scan_path = (ROOT / str(row["scan_path"])).resolve()
        if permitted_root not in scan_path.parents or not scan_path.is_file():
            raise ValueError("opaque manifest scan path escaped its permitted cohort root")
        actual_sha = _sha256(scan_path)
        if actual_sha != str(row["sha256"]):
            raise ValueError("raw scan bytes do not match opaque manifest SHA-256")
        descriptor = _geometry_descriptor(scan_path)
        cases.append(
            {
                "case_id": f"A30-{ordinal:03d}",
                "scan_sha256": actual_sha,
                "size_bytes": scan_path.stat().st_size,
                **descriptor,
            }
        )
    return manifest, cases


def _normalize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = np.asarray(
        [[case["feature_vector"][name] for name in FEATURE_NAMES] for case in cases],
        dtype=np.float64,
    )
    centers = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    scales = q75 - q25
    std = matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, np.where(std > 1e-12, std, 1.0))
    normalized = np.clip((matrix - centers) / scales, -8.0, 8.0)
    for case, vector in zip(cases, normalized):
        case["normalized_vector"] = [_round(value) for value in vector]
    return {
        "method": "cohort_median_and_iqr_then_rms_euclidean",
        "unsupervised_split_construction_only": True,
        "feature_names": list(FEATURE_NAMES),
        "centers": [_round(value) for value in centers],
        "scales": [_round(value) for value in scales],
        "clip": [-8.0, 8.0],
        "distance": "sqrt(mean((z_i-z_j)^2))",
    }


def _binding(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "role": role,
    }


def _payloads() -> tuple[dict[str, bytes], dict[str, Any]]:
    manifest, cases = _load_cases()
    normalization = _normalize(cases)
    selection = select_tiers(cases)

    prereg = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_retrospective_pseudo_unseen_prereg_v1",
        "status": "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN",
        "locked_at_utc": LOCKED_AT_UTC,
        "cohort_date": EXPECTED_COHORT_DATE,
        "cohort_case_count": 28,
        "historical_cohort_consumed": True,
        "retrospective_pseudo_unseen_reason": "the complete 2026-08-30 cohort was historically consumed before this split was created",
        "prospective_or_unseen_claim_authorized": False,
        "research_only": True,
        "promotion_eligible_from_this_split_alone": False,
        "model_fit_performed": False,
        "model_score_performed": False,
        "input_boundary": {
            "allowed": [
                "the fixed 28-row opaque scan-only manifest",
                "raw PLY geometry bytes bound by that manifest",
                "geometry-derived descriptors",
                "opaque case identity only as a deterministic tie breaker",
            ],
            "forbidden": [
                "labels or truth",
                "model predictions",
                "evaluator outputs",
                "sender or expected counts",
                "CAD",
                "drawings",
                "original filenames",
                "part IDs",
                "family IDs",
            ],
        },
        "split_design": {
            "train_count": 18,
            "locked_holdout_count": 10,
            "holdout_tier_counts": {
                "SIMILAR_FAMILY": 4,
                "ADJACENT_FAMILY": 3,
                "DISTANT_FAMILY": 3,
            },
            "geometry_only": True,
            "selection_rule": [
                "robust-normalize the frozen geometry descriptor vector over all 28 scans",
                "rank each case by mean distance to its three nearest descriptor neighbors",
                "assign the three highest-novelty cases to DISTANT_FAMILY",
                "assign the next eligible high-novelty frontier cases to ADJACENT_FAMILY while reserving each closest neighbor for train",
                "assign four cases at or below the median novelty to SIMILAR_FAMILY only when two closest eligible neighbors can be reserved for train",
                "assign all remaining cases to train",
            ],
            "tie_breaker": "ascending opaque case_id only",
        },
        "holdout_lock": {
            "all_tiers_locked": True,
            "model_tuning_on_any_holdout_tier": "FORBIDDEN",
            "feature_or_threshold_selection_from_holdout_results": "FORBIDDEN",
            "case_removal_or_tier_reassignment_after_seal": "FORBIDDEN",
            "holdout_labels_may_open_for_final_frozen_comparison_only": True,
        },
        "retention_gates": {
            "reference": "FROZEN_V51_ON_THE_SAME_LOCKED_10_CASES",
            "candidate_minus_reference_convention": True,
            "overall_holdout_micro_f1_min_delta": 0.05,
            "holdout_tp_min_delta": 0,
            "holdout_fp_max_increase": {
                "formula": "max(2, 0.10 * baseline_fp)",
                "absolute_floor": 2,
                "baseline_fraction": 0.1,
            },
            "tier_micro_f1_min_delta": {
                "SIMILAR_FAMILY": -0.02,
                "ADJACENT_FAMILY": -0.02,
                "DISTANT_FAMILY": 0.0,
            },
            "exact_case_count_min_delta": 0,
            "primary_replay_bytes": "EXACT_SHA256_IDENTITY_REQUIRED",
            "all_gates_conjunctive": True,
            "failure_policy": "do not retain or promote the candidate",
        },
        "evaluation_deferred": True,
        "evaluation_note": "this preregistration freezes decision rules only; this builder neither fits nor scores a model",
    }
    prereg_bytes = _canonical_bytes(prereg)
    prereg_sha = hashlib.sha256(prereg_bytes).hexdigest()

    descriptor_payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_geometry_descriptor_evidence_v1",
        "created_at_utc": LOCKED_AT_UTC,
        "source_isolation": "OPAQUE_SCAN_GEOMETRY_ONLY",
        "model_fit_performed": False,
        "model_score_performed": False,
        "input_manifest": {
            "path": str(INPUT_MANIFEST.relative_to(ROOT)),
            "sha256": _sha256(INPUT_MANIFEST),
            "schema": manifest["schema"],
            "cohort_date": manifest["cohort_date"],
        },
        "descriptor_method": {
            "geometry_source": "PLY vertex positions, vertex normals, and triangle topology only",
            "orientation_handling": "rotation-invariant PCA, radial, edge-length, area, and normal-moment summaries",
            "vertex_sampling": f"deterministic evenly spaced indices, capped at {MAX_VERTEX_SAMPLE}",
            "triangle_sampling": f"deterministic evenly spaced indices, capped at {MAX_FACE_SAMPLE}",
            "raw_scan_hash_verified_before_descriptor": True,
        },
        "normalization": normalization,
        "cases": cases,
    }
    descriptor_bytes = _canonical_bytes(descriptor_payload)
    descriptor_sha = hashlib.sha256(descriptor_bytes).hexdigest()

    tier_by_case = {
        case_id: tier
        for tier, case_ids in selection["tiers"].items()
        for case_id in case_ids
    }
    sha_by_case = {case["case_id"]: case["scan_sha256"] for case in cases}
    assignments = []
    for case_id in sorted(sha_by_case):
        is_holdout = case_id in tier_by_case
        assignments.append(
            {
                "case_id": case_id,
                "scan_sha256": sha_by_case[case_id],
                "split": "HOLDOUT" if is_holdout else "TRAIN",
                "tier": tier_by_case.get(case_id),
                "assignment_locked": True,
                "holdout_locked": True if is_holdout else False,
                "model_tuning_allowed": False if is_holdout else True,
            }
        )
    holdout_evidence = []
    for case_id in sorted(tier_by_case):
        holdout_evidence.append(
            {
                "case_id": case_id,
                "tier": tier_by_case[case_id],
                "mean_three_neighbor_distance": selection["novelty"][case_id],
                "novelty_rank_desc": selection["novelty_rank_desc"][case_id],
                "retained_train_neighbor_anchors": selection["retained_neighbor_anchors"][case_id],
                "nearest_train_neighbors": selection["nearest_train"][case_id],
            }
        )
    split_payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1",
        "status": "LOCKED_RETROSPECTIVE_PSEUDO_UNSEEN",
        "locked_at_utc": LOCKED_AT_UTC,
        "historical_cohort_consumed": True,
        "prospective_or_unseen_claim_authorized": False,
        "prereg_binding": {"path": f"research/{PREREG_NAME}", "sha256": prereg_sha},
        "descriptor_binding": {
            "path": f"research/{DESCRIPTORS_NAME}",
            "sha256": descriptor_sha,
        },
        "counts": {"total": 28, "train": 18, "holdout": 10},
        "holdout_tier_counts": {tier: len(selection["tiers"][tier]) for tier in selection["tiers"]},
        "all_holdout_locked": True,
        "model_tuning_on_holdout": "FORBIDDEN",
        "train_case_ids": selection["train"],
        "holdout_tiers": selection["tiers"],
        "assignments": assignments,
        "geometry_tier_evidence": {
            "local_novelty": "mean robust-normalized descriptor distance to three nearest cohort neighbors",
            "dense_cutoff": selection["dense_cutoff"],
            "rows": holdout_evidence,
        },
        "label_or_outcome_fields_present": False,
        "model_fit_performed": False,
        "model_score_performed": False,
    }
    split_bytes = _canonical_bytes(split_payload)
    split_sha = hashlib.sha256(split_bytes).hexdigest()

    train_allowlist = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train_only_scan_allowlist_v1",
        "status": "LOCKED_TRAIN_ONLY",
        "split_binding": {"path": f"research/{SPLIT_NAME}", "sha256": split_sha},
        "prereg_binding": {"path": f"research/{PREREG_NAME}", "sha256": prereg_sha},
        "case_count": 18,
        "scope": "opaque scan identity allowlist only; contains no labels or outcomes",
        "reject_any_case_not_listed": True,
        "cases": [
            {"case_id": case_id, "scan_sha256": sha_by_case[case_id]}
            for case_id in selection["train"]
        ],
    }
    train_bytes = _canonical_bytes(train_allowlist)

    encoded = {
        PREREG_NAME: prereg_bytes,
        DESCRIPTORS_NAME: descriptor_bytes,
        SPLIT_NAME: split_bytes,
        TRAIN_ALLOWLIST_NAME: train_bytes,
    }
    artifact_bindings = []
    for name, payload in encoded.items():
        artifact_bindings.append(
            {
                "path": f"research/{name}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "role": {
                    PREREG_NAME: "PREREGISTRATION",
                    DESCRIPTORS_NAME: "GEOMETRY_DESCRIPTOR_EVIDENCE",
                    SPLIT_NAME: "LOCKED_SPLIT",
                    TRAIN_ALLOWLIST_NAME: "TRAIN_ONLY_SCAN_ALLOWLIST",
                }[name],
            }
        )
    artifact_bindings.extend(
        [
            _binding(BUILDER_PATH, "GEOMETRY_ONLY_SPLIT_BUILDER"),
            _binding(AUDITOR_PATH, "SOURCE_FREE_AUDITOR"),
            _binding(TEST_PATH, "SOURCE_FREE_TESTS"),
        ]
    )
    artifact_bindings.sort(key=lambda row: row["path"])
    raw_bindings = [
        {
            "case_id": case["case_id"],
            "sha256": case["scan_sha256"],
            "size_bytes": case["size_bytes"],
        }
        for case in cases
    ]
    hashes_payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_retrospective_pseudo_unseen_hash_manifest_v1",
        "sealed_at_utc": LOCKED_AT_UTC,
        "hash_algorithm": "SHA-256",
        "artifact_bindings": artifact_bindings,
        "artifact_binding_digest_sha256": hashlib.sha256(_canonical_bytes(artifact_bindings)).hexdigest(),
        "source_bindings": {
            "input_manifest": {
                "path": str(INPUT_MANIFEST.relative_to(ROOT)),
                "sha256": _sha256(INPUT_MANIFEST),
                "size_bytes": INPUT_MANIFEST.stat().st_size,
            },
            "raw_scans": raw_bindings,
            "raw_scan_binding_digest_sha256": hashlib.sha256(_canonical_bytes(raw_bindings)).hexdigest(),
            "source_paths_or_original_identities_disclosed": False,
        },
        "model_fit_performed": False,
        "model_score_performed": False,
    }
    hashes_bytes = _canonical_bytes(hashes_payload)
    encoded[HASHES_NAME] = hashes_bytes
    encoded[SEAL_NAME] = f"{hashlib.sha256(hashes_bytes).hexdigest()}  {HASHES_NAME}\n".encode()
    return encoded, selection


def _write_once(encoded: dict[str, bytes]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collisions = [name for name in encoded if (OUTPUT_DIR / name).exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite sealed artifacts: {sorted(collisions)}")
    written = []
    try:
        for name, payload in encoded.items():
            path = OUTPUT_DIR / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            written.append(path)
    except Exception:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the sealed artifacts exactly once")
    args = parser.parse_args()
    encoded, selection = _payloads()
    if args.write:
        _write_once(encoded)
    result = {
        "status": "SEALED" if args.write else "PREVIEW_ONLY",
        "retrospective_pseudo_unseen": True,
        "model_fit_performed": False,
        "model_score_performed": False,
        "train_count": len(selection["train"]),
        "holdout_tier_counts": {tier: len(case_ids) for tier, case_ids in selection["tiers"].items()},
        "memberships": selection["tiers"],
        "hash_manifest_sha256": hashlib.sha256(encoded[HASHES_NAME]).hexdigest(),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
