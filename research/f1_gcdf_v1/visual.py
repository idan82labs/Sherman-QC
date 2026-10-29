"""Deterministic mandatory GCDF-V1 visual evidence package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .config import ATTRITION_TOKENS, CONFIG, sha_json
from .core import CanonicalNodes, SurfaceGraph


CAMERA_COMMITMENT = sha_json({
    "projection": "ORTHOGRAPHIC", "views": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
    "mosaic_pixels": [6144, 4096], "source_bounds": "COMPLETE_UNCLIPPED",
})
RENDERER_COMMITMENT = sha_json({
    "backend": "MATPLOTLIB_AGG", "lossless": "PNG", "global_mosaic_pixels": [6144, 4096],
    "atlas_panel_pixels": [1600, 1000], "zero_results_show_highest_rejected": True,
})

_VIEWS = {
    "+X": (1, 2, 1, 1), "-X": (1, 2, -1, 1),
    "+Y": (0, 2, 1, 1), "-Y": (0, 2, -1, 1),
    "+Z": (0, 1, 1, 1), "-Z": (0, 1, 1, -1),
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project(points: np.ndarray, view: tuple[int, int, int, int]) -> np.ndarray:
    first, second, sign_first, sign_second = view
    return np.stack([sign_first * points[:, first], sign_second * points[:, second]], axis=1)


def _equal(ax: Any, projected: np.ndarray) -> None:
    minimum = np.min(projected, axis=0)
    maximum = np.max(projected, axis=0)
    span = max(float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1]), np.finfo(float).eps)
    center = 0.5 * (minimum + maximum)
    ax.set_xlim(center[0] - 0.525 * span, center[0] + 0.525 * span)
    ax.set_ylim(center[1] - 0.525 * span, center[1] + 0.525 * span)
    ax.set_aspect("equal", adjustable="box")


def _global_mosaic(
    path: Path, *, title: str, nodes: CanonicalNodes,
    point_overlay: np.ndarray | None = None,
    path_overlays: Sequence[np.ndarray] = (),
    graph_edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    source = nodes.metric_xyz
    stride = max(1, len(source) // 100_000)
    figure, axes = plt.subplots(2, 3, figsize=(38.4, 25.6), dpi=160, constrained_layout=True)
    for ax, (label, view) in zip(axes.ravel(), _VIEWS.items(), strict=True):
        projected = _project(source[::stride], view)
        ax.scatter(projected[:, 0], projected[:, 1], s=0.16, c="#7e8996", alpha=0.28, linewidths=0, rasterized=True)
        if graph_edges is not None:
            left, right = graph_edges
            edge_stride = max(1, len(left) // 250_000)
            segments3 = np.stack([source[left[::edge_stride]], source[right[::edge_stride]]], axis=1)
            segments2 = np.stack([_project(segment, view) for segment in segments3])
            ax.add_collection(LineCollection(segments2, colors="#2b6f9f", linewidths=0.18, alpha=0.18, rasterized=True))
        if path_overlays:
            segments = []
            for path_nodes in path_overlays:
                if len(path_nodes) > 1:
                    projected_path = _project(source[path_nodes], view)
                    segments.extend(np.stack([projected_path[:-1], projected_path[1:]], axis=1))
            if segments:
                ax.add_collection(LineCollection(segments, colors="#e76f51", linewidths=0.35, alpha=0.22, rasterized=True))
        if point_overlay is not None and len(point_overlay):
            overlay = _project(source[point_overlay], view)
            ax.scatter(overlay[:, 0], overlay[:, 1], s=5.0, c="#d62828", alpha=0.85, linewidths=0, rasterized=True)
        _equal(ax, projected)
        ax.set_title(label, fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
    figure.suptitle(title, fontsize=28)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, metadata={"Software": "ShermanQC GCDF-V1"})
    plt.close(figure)


def _atlas_panel(path: Path, *, token: str, status: str, detail: Mapping[str, Any] | None, nodes: CanonicalNodes) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 6.25), dpi=160, constrained_layout=True)
    source = nodes.metric_xyz
    stride = max(1, len(source) // 60_000)
    axes[0].scatter(source[::stride, 0], source[::stride, 2], s=0.18, c="#8b95a1", alpha=0.25, linewidths=0)
    if detail and detail.get("visual_path_nodes"):
        ids = np.asarray(detail["visual_path_nodes"], dtype=np.uint32)
        points = source[ids]
        axes[0].plot(points[:, 0], points[:, 2], color="#d62828", linewidth=1.5)
        if detail.get("peak"):
            peak = source[int(detail["peak"]["center_node"])]
            axes[0].scatter([peak[0]], [peak[2]], s=45, c="#ffb703", edgecolors="#111111")
    elif detail and detail.get("peak_nodes"):
        points = source[np.asarray(detail["peak_nodes"], dtype=np.uint32)]
        axes[0].scatter(points[:, 0], points[:, 2], s=12, c="#d62828", alpha=0.9)
    _equal(axes[0], source[::stride][:, [0, 2]])
    axes[0].set_title("source-space evidence")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Z")
    axes[1].axis("off")
    lines = [token, f"status: {status}"]
    if detail:
        for key in (
            "id", "component_id", "gate", "margin", "distance", "chord", "global_deficit_ppm",
            "concentration_ppm", "vote_count", "unique_peak_count", "endpoint_diversity",
            "diameter", "branch_ppm", "spine_chord_ppm", "causally_evaluable_vote_count",
        ):
            if key in detail:
                lines.append(f"{key}: {detail[key]}")
        profile = detail.get("profile")
        if profile:
            inset = axes[1].inset_axes([0.05, 0.05, 0.9, 0.42])
            inset.plot([row["s"] for row in profile], [row["d"] for row in profile], color="#33658a")
            inset.axhline(CONFIG.local_peak_minimum_ppm, color="#d62828", linestyle="--")
            inset.set_title("cumulative arclength vs local deficit ppm")
            inset.set_xlabel("Q16 arclength")
            inset.set_ylabel("ppm")
    axes[1].text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10, wrap=True)
    figure.suptitle(f"GCDF-V1 deterministic attrition atlas — {token}", fontsize=14)
    figure.savefig(path, dpi=160, metadata={"Software": "ShermanQC GCDF-V1"})
    plt.close(figure)


def _role_comparison(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, ax = plt.subplots(figsize=(10, 6.25), dpi=160, constrained_layout=True)
    ax.axis("off")
    lines = [title, ""]
    keys = (
        "role", "canonical_unique_node_table_sha256", "graph_edge_set_sha256",
        "landmark_sha256", "vote_table_sha256", "candidate_axis_sha256", "decision",
    )
    for row in rows:
        lines.append(str(row.get("role", "")))
        lines.extend(f"  {key}: {row.get(key, '')}" for key in keys[1:])
        lines.append("")
    roots = [{key: row.get(key) for key in keys[1:]} for row in rows]
    lines.append(f"exact_semantic_root_agreement: {all(root == roots[0] for root in roots[1:]) if roots else False}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)
    figure.savefig(path, dpi=160, metadata={"Software": "ShermanQC GCDF-V1"})
    plt.close(figure)


def build_visual_package(
    output: Path, *, nodes: CanonicalNodes, graph: SurfaceGraph,
    scratch: Mapping[str, Any], roles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    visual = output / "visual"
    visual.mkdir(parents=True, exist_ok=False)
    path_result = scratch["path_result"]
    candidate_result = scratch["candidate_result"]
    causal_result = scratch["causal_result"]
    votes = path_result.get("votes", [])
    all_paths = [np.asarray(row["nodes"], dtype=np.uint32) for row in path_result.get("visual_global_paths", [])]
    vote_nodes = np.asarray(sorted({int(row["peak_node"]) for row in votes}), dtype=np.uint32)
    spine_paths = [np.asarray(row["spine_peak_nodes"], dtype=np.uint32) for row in candidate_result.get("candidates", [])]
    passing_ids = set(causal_result.get("passing_candidate_ids", []))
    causal_paths = [np.asarray(row["spine_peak_nodes"], dtype=np.uint32) for row in candidate_result.get("candidates", []) if row["candidate_id"] in passing_ids]
    global_specs = (
        ("global-complete-source.png", "Complete unclipped source", None, (), None),
        ("global-surface-graph-audit.png", "Surface graph and shortcut audit", None, (), (graph.edges_left, graph.edges_right)),
        ("global-all-global-deficit-paths.png", "All globally passing paths", None, all_paths, None),
        ("global-accepted-peak-votes.png", "All accepted peak votes", vote_nodes, (), None),
        ("global-geometric-spines.png", "All geometric spines; highest rejected evidence remains visible", vote_nodes, spine_paths, None),
        ("global-causal-candidates.png", "All causally passing candidates; highest rejected evidence remains visible", vote_nodes, causal_paths or spine_paths, None),
    )
    manifest: list[dict[str, Any]] = []
    for filename, title, points, paths, edges in global_specs:
        image = visual / filename
        _global_mosaic(image, title=title, nodes=nodes, point_overlay=points, path_overlays=paths, graph_edges=edges)
        manifest.append({"subject": filename[:-4], "kind": "GLOBAL", "path": str(image.relative_to(output)), "sha256": _sha_file(image), "camera_sha256": CAMERA_COMMITMENT})
    controls = path_result.get("controls", {})
    components = candidate_result.get("components", [])
    causal_rows = causal_result.get("candidate_results", [])
    for token in ATTRITION_TOKENS:
        detail: Mapping[str, Any] | None = None
        status = "NOT_PRESENT"
        if token in graph.gate_census and graph.gate_census[token]:
            detail, status = {"margin": graph.gate_census[token], "gate": token}, "OBSERVED"
        elif controls.get(token):
            detail, status = controls[token][0], "OBSERVED_NEAREST_THRESHOLD"
        else:
            matched = [row for row in components if row.get("gate") == token]
            if matched:
                detail, status = sorted(matched, key=lambda row: (abs(int(row.get("margin", 0))), row["component_id"]))[0], "OBSERVED_NEAREST_THRESHOLD"
            else:
                matched_causal = [row for row in causal_rows if row.get("gate") == token]
                if matched_causal:
                    detail, status = matched_causal[0], "OBSERVED_NEAREST_THRESHOLD"
        if token.startswith("CAUSAL_") or token == "WRONG_SIGN_CONTROL_HIGH":
            if candidate_result.get("decision") != "GCDF_V1_GEOMETRIC_SPINES_SURVIVE":
                status = f"NOT_EVALUABLE: {candidate_result.get('decision')}"
        image = visual / f"atlas-{token.lower()}.png"
        _atlas_panel(image, token=token, status=status, detail=detail, nodes=nodes)
        manifest.append({"subject": token, "kind": "ATTRITION_ATLAS", "path": str(image.relative_to(output)), "sha256": _sha_file(image), "camera_sha256": CAMERA_COMMITMENT})
    comparison = visual / "role-semantic-comparison.png"
    _role_comparison(comparison, "Ordinary / permuted / independent semantic comparison", roles)
    manifest.append({"subject": "ROLE_SEMANTIC_COMPARISON", "kind": "ROLE_COMPARISON", "path": str(comparison.relative_to(output)), "sha256": _sha_file(comparison), "camera_sha256": CAMERA_COMMITMENT})
    xor = visual / "role-raster-xor.png"
    _role_comparison(xor, "Raster XOR: zero-difference iff semantic roots agree", roles)
    manifest.append({"subject": "ROLE_RASTER_XOR", "kind": "ROLE_COMPARISON", "path": str(xor.relative_to(output)), "sha256": _sha_file(xor), "camera_sha256": CAMERA_COMMITMENT})
    return manifest


def visual_manifest_sha256(manifest: Sequence[Mapping[str, Any]]) -> str:
    return sha_json(list(manifest))

