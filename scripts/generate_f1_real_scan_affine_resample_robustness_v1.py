#!/usr/bin/env python3
"""Build a development-only robustness lane from real scan morphology.

The rejected synthetic prospective lanes matched mesh spacing but not the
information content of real manufactured scans.  This generator therefore
starts from four already-exposed real scans and creates two deterministic
near-neighbour observations per scan using only global affine changes,
bounded mesh decimation, and small normal-direction measurement noise.

This is *not* new-family or locked benchmark evidence.  The hidden reference
lines are transformed, previously visually audited V51 outputs and are named
development pseudo-truth throughout.  They are useful for detecting geometric
instability and new false positives, but never for claiming absolute F1 or
promotion readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = 1
KIND = "sherman_qc_f1_real_scan_affine_resample_robustness_v1"
PSEUDO_TRUTH_KIND = "development_pseudo_truth_from_visually_audited_v51_outputs"
SOURCE_ROOT = ROOT / "data/bend_corpus/incoming/2026-08-13_shaked_new_parts_blind/opaque"
REFERENCE_ROOT = ROOT / "output/research/f1-v51-single-axis-parallel-return-recovery-v1"
MAX_VERTICES = 195_000

PARENTS: tuple[str, ...] = ("U01", "U02", "U03", "U04")
VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "A",
        "target_vertices": 190_000,
        "scale": (1.025, 0.985, 1.010),
        "shear": ((1.0, 0.010, -0.004), (0.0, 1.0, 0.007), (0.003, 0.0, 1.0)),
        "rotation_deg_xyz": (7.0, -11.0, 19.0),
        "normal_noise_sigma_mm": 0.035,
    },
    {
        "variant": "B",
        "target_vertices": 145_000,
        "scale": (0.970, 1.035, 0.990),
        "shear": ((1.0, -0.008, 0.006), (0.004, 1.0, -0.006), (0.0, 0.005, 1.0)),
        "rotation_deg_xyz": (-13.0, 8.0, -23.0),
        "normal_noise_sigma_mm": 0.070,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temp.replace(path)


def _rotation_xyz(degrees: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(degrees, dtype=np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    Ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    Rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return Rz @ Ry @ Rx


def affine_matrix(spec: dict[str, Any]) -> np.ndarray:
    scale = np.diag(np.asarray(spec["scale"], dtype=np.float64))
    shear = np.asarray(spec["shear"], dtype=np.float64)
    return _rotation_xyz(tuple(float(v) for v in spec["rotation_deg_xyz"])) @ shear @ scale


def transform_points(points: np.ndarray, matrix: np.ndarray, center: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (points - center) @ matrix.T + center + translation


def transform_lines(
    predictions: list[dict[str, Any]], matrix: np.ndarray, center: np.ndarray, translation: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ordinal, prediction in enumerate(predictions):
        endpoints = np.asarray((prediction["line_start"], prediction["line_end"]), dtype=np.float64)
        mapped = transform_points(endpoints, matrix, center, translation)
        rows.append({
            "line_id": f"PT{ordinal + 1:02d}",
            "line_start": mapped[0].round(9).tolist(),
            "line_end": mapped[1].round(9).tolist(),
            "source_reference_id": prediction.get("bend_id"),
        })
    return rows


def _load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if not len(mesh.vertices) or not len(mesh.triangles):
        raise RuntimeError(f"triangle mesh required: {path}")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _bounded_mesh(mesh: o3d.geometry.TriangleMesh, target_vertices: int) -> o3d.geometry.TriangleMesh:
    target_vertices = min(int(target_vertices), MAX_VERTICES)
    current = mesh
    while len(current.vertices) > target_vertices:
        target_triangles = max(2_000, int(len(current.triangles) * target_vertices / len(current.vertices) * 0.94))
        reduced = current.simplify_quadric_decimation(target_triangles)
        reduced.remove_degenerate_triangles()
        reduced.remove_unreferenced_vertices()
        if len(reduced.vertices) >= len(current.vertices):
            raise RuntimeError("quadric decimation did not reduce the mesh")
        current = reduced
    current.compute_vertex_normals()
    return current


def _apply_variant(
    mesh: o3d.geometry.TriangleMesh, spec: dict[str, Any], *, seed: int
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray, np.ndarray]:
    mesh = _bounded_mesh(mesh, int(spec["target_vertices"]))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    matrix = affine_matrix(spec)
    rng = np.random.default_rng(seed)
    translation = rng.uniform(-30.0, 30.0, size=3)
    mapped = transform_points(vertices, matrix, center, translation)
    mesh.vertices = o3d.utility.Vector3dVector(mapped)
    mesh.compute_vertex_normals()
    sigma = float(spec["normal_noise_sigma_mm"])
    if sigma > 0:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        # Zero-mean, bounded measurement perturbation.  Clipping avoids isolated
        # spikes that would be synthetic line evidence rather than scan noise.
        amplitude = np.clip(rng.normal(0.0, sigma, size=len(mapped)), -3.0 * sigma, 3.0 * sigma)
        mapped = mapped + normals * amplitude[:, None]
        mesh.vertices = o3d.utility.Vector3dVector(mapped)
        mesh.compute_vertex_normals()
    return mesh, matrix, center, translation


def _edge_quantiles(mesh: o3d.geometry.TriangleMesh) -> list[float]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles) > 400_000:
        triangles = triangles[np.linspace(0, len(triangles) - 1, 400_000, dtype=np.int64)]
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 1e-8)]
    return np.percentile(lengths, (10, 50, 90)).round(6).tolist()


def _pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    _, _, vt = np.linalg.svd(points[:: max(1, len(points) // 50_000)] - center, full_matrices=False)
    return center, vt


def _render_views(mesh: o3d.geometry.TriangleMesh, path: Path, title: str) -> None:
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if len(points) > 80_000:
        points = points[np.linspace(0, len(points) - 1, 80_000, dtype=np.int64)]
    center, basis = _pca_basis(points)
    local = (points - center) @ basis.T
    views = ((0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (1, 2))
    signs = ((1, 1), (1, 1), (1, 1), (-1, 1), (-1, 1), (-1, 1))
    labels = ("PC1 / PC2", "PC1 / PC3", "PC2 / PC3", "reverse PC1 / PC2", "reverse PC1 / PC3", "reverse PC2 / PC3")
    fig, axes = plt.subplots(2, 3, figsize=(22, 13), facecolor="#f4f6f8")
    for ax, dims, sign, label in zip(axes.flat, views, signs, labels):
        x = sign[0] * local[:, dims[0]]
        y = sign[1] * local[:, dims[1]]
        ax.scatter(x, y, s=0.16, c="#4f5965", alpha=0.55, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, fontsize=13, weight="bold")
    fig.suptitle(title, fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_parent_comparison(
    parent: o3d.geometry.TriangleMesh, variant: o3d.geometry.TriangleMesh, path: Path, title: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor="#f4f6f8")
    for ax, mesh, label in ((axes[0], parent, "real parent scan"), (axes[1], variant, "affine/resampled variant")):
        points = np.asarray(mesh.vertices, dtype=np.float64)
        if len(points) > 90_000:
            points = points[np.linspace(0, len(points) - 1, 90_000, dtype=np.int64)]
        center, basis = _pca_basis(points)
        local = (points - center) @ basis.T
        ax.scatter(local[:, 0], local[:, 1], s=0.18, c="#46515d", alpha=0.55, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, fontsize=15, weight="bold")
    fig.suptitle(title, fontsize=19, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate(output_root: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    public_cases: list[dict[str, Any]] = []
    evaluator_cases: list[dict[str, Any]] = []
    ordinal = 0
    for parent_index, parent_id in enumerate(PARENTS):
        source_path = SOURCE_ROOT / f"{parent_id}.ply"
        reference_path = REFERENCE_ROOT / f"{parent_id}.json"
        parent_mesh = _load_mesh(source_path)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        predictions = reference["predictions"]
        for variant_index, spec in enumerate(VARIANTS):
            ordinal += 1
            opaque_id = f"R{ordinal:03d}"
            seed = 7_700_000 + parent_index * 101 + variant_index * 17
            variant, matrix, center, translation = _apply_variant(parent_mesh, spec, seed=seed)
            case_root = output_root / "public" / opaque_id
            case_root.mkdir(parents=True, exist_ok=True)
            output_path = case_root / "source.ply"
            if not o3d.io.write_triangle_mesh(str(output_path), variant, write_ascii=False, compressed=False, write_vertex_normals=True):
                raise RuntimeError(f"failed to write {output_path}")
            source_sha = _sha256(output_path)
            audit_path = output_root / "source-only-audits" / f"{opaque_id}.png"
            _render_views(variant, audit_path, f"{opaque_id} REAL-SCAN-DERIVED SOURCE-ONLY AUDIT")
            comparison_path = output_root / "evaluator_only" / "parent-comparisons" / f"{opaque_id}.png"
            _render_parent_comparison(parent_mesh, variant, comparison_path, f"{opaque_id} morphology preservation check")
            public_cases.append({
                "opaque_case_id": opaque_id,
                "source_file": f"public/{opaque_id}/source.ply",
                "source_sha256": source_sha,
                "source_vertex_count": len(variant.vertices),
                "source_triangle_count": len(variant.triangles),
                "source_edge_length_mm_p10_p50_p90": _edge_quantiles(variant),
                "source_only_audit_file": f"source-only-audits/{opaque_id}.png",
                "source_only_audit_sha256": _sha256(audit_path),
            })
            evaluator_cases.append({
                "opaque_case_id": opaque_id,
                "parent_case_id": parent_id,
                "parent_scan_file": str(source_path.relative_to(ROOT)),
                "parent_scan_sha256": _sha256(source_path),
                "reference_report_file": str(reference_path.relative_to(ROOT)),
                "reference_report_sha256": _sha256(reference_path),
                "reference_line_count": len(predictions),
                "pseudo_truth_kind": PSEUDO_TRUTH_KIND,
                "pseudo_truth_lines": transform_lines(predictions, matrix, center, translation),
                "variant": spec["variant"],
                "seed": seed,
                "affine_matrix": matrix.round(12).tolist(),
                "affine_center": center.round(12).tolist(),
                "translation_mm": translation.round(12).tolist(),
                "normal_noise_sigma_mm": spec["normal_noise_sigma_mm"],
                "target_vertices": spec["target_vertices"],
                "parent_comparison_file": str(comparison_path.relative_to(output_root)),
                "parent_comparison_sha256": _sha256(comparison_path),
            })
    public = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "case_count": len(public_cases),
        "source_only": True,
        "research_only": True,
        "locked_benchmark_eligible": False,
        "development_pseudo_truth_used_by_generator": False,
        "detector_execution_authorized": False,
        "visual_preacceptance_required": True,
        "max_source_vertices": MAX_VERTICES,
        "cases": public_cases,
    }
    evaluator = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_evaluator_only",
        "warning": "Development pseudo-truth only; do not report as human ground truth, independent F1, or locked-family evidence.",
        "pseudo_truth_kind": PSEUDO_TRUTH_KIND,
        "cases": evaluator_cases,
    }
    _atomic_json(output_root / "public_manifest.json", public)
    _atomic_json(output_root / "evaluator_only" / "manifest.json", evaluator)
    return public


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.output_root.resolve())
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "case_count": report["case_count"],
        "locked_benchmark_eligible": False,
        "detector_execution_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
