"""Read-only deterministic visual package for PGRR evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from .config import CONFIG, canonical_json_bytes, sha_json
from .geometry import NormalField, Ribbon, Sample, pole_basis


CAMERA_COMMITMENT = sha_json({
    "projection": "ORTHOGRAPHIC",
    "global_views": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
    "equal_data_aspect": True,
    "autoscale_from_complete_source_bounds": True,
    "source_decimation": "CANONICAL_STRIDE_TO_AT_MOST_50000_POINTS_FOR_RASTER_ONLY",
})

_VIEWS = {
    "+X": (1, 2, 1, 1),
    "-X": (1, 2, -1, 1),
    "+Y": (0, 2, 1, 1),
    "-Y": (0, 2, -1, 1),
    "+Z": (0, 1, 1, 1),
    "-Z": (0, 1, 1, -1),
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raster_source(xyz: np.ndarray) -> np.ndarray:
    stride = max(1, int(math.ceil(len(xyz) / 50_000)))
    return xyz[::stride]


def _set_equal(ax: Any, x: np.ndarray, y: np.ndarray) -> None:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    span = max(xmax - xmin, ymax - ymin, np.finfo(float).eps)
    xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    ax.set_xlim(xmid - 0.525 * span, xmid + 0.525 * span)
    ax.set_ylim(ymid - 0.525 * span, ymid + 0.525 * span)
    ax.set_aspect("equal", adjustable="box")


def _project(points: np.ndarray, view: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    first, second, sign_first, sign_second = view
    return sign_first * points[:, first], sign_second * points[:, second]


def render_global_contact_sheet(
    path: Path, sample: Sample, ribbons: Sequence[Ribbon]
) -> None:
    source = _raster_source(sample.xyz)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=160, constrained_layout=True)
    palette = plt.get_cmap("tab20")
    for ax, (label, view) in zip(axes.ravel(), _VIEWS.items(), strict=True):
        x, y = _project(source, view)
        ax.scatter(x, y, s=0.08, c="#8d99a8", alpha=0.30, linewidths=0, rasterized=True)
        for index, ribbon in enumerate(ribbons):
            points = sample.xyz[ribbon.member_indices]
            rx, ry = _project(points, view)
            ax.scatter(
                rx, ry, s=1.5, color=palette(index % 20), alpha=0.85,
                linewidths=0, rasterized=True,
            )
        ax.set_title(label)
        ax.set_xlabel("projected axis 1 (mm)")
        ax.set_ylabel("projected axis 2 (mm)")
        _set_equal(ax, x, y)
        ax.grid(False)
    figure.suptitle(
        f"PGRR global canonical overlay — {len(ribbons)} deduplicated ribbon(s)",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"Software": "ShermanQC PGRR-V1"})
    plt.close(figure)


def render_negative_control(
    path: Path,
    *,
    code: str,
    row: Mapping[str, Any],
    member_indices: np.ndarray,
    sample: Sample,
    field: NormalField,
) -> None:
    axis = np.asarray(row.get("axis") or [0.0, 0.0, 1.0], dtype=np.float64)
    if np.linalg.norm(axis) == 0.0:
        axis = np.asarray([0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    u, v = pole_basis(axis)
    points = sample.xyz[member_indices]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=160, constrained_layout=True)
    source = _raster_source(sample.xyz)
    axes[0].scatter(source[:, 0], source[:, 2], s=0.08, c="#9aa5b1", alpha=0.24, linewidths=0)
    axes[0].scatter(points[:, 0], points[:, 2], s=2.0, c="#d1495b", alpha=0.9, linewidths=0)
    _set_equal(axes[0], source[:, 0], source[:, 2])
    axes[0].set_title("whole source / candidate")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Z (mm)")
    axes[1].scatter(points @ u, points @ v, s=2.0, c=points @ axis, cmap="viridis", linewidths=0)
    _set_equal(axes[1], points @ u, points @ v)
    axes[1].set_title("pole-aligned source preimage")
    axes[1].set_xlabel("u (mm)")
    axes[1].set_ylabel("v (mm)")
    normals = field.mean_normal[member_indices]
    theta = np.mod(np.arctan2(normals @ v, normals @ u), math.pi)
    band = np.degrees(np.arcsin(np.clip(np.abs(normals @ axis), 0.0, 1.0)))
    axes[2].scatter(np.degrees(theta), band, s=3.0, c=points @ axis, cmap="viridis", linewidths=0)
    axes[2].axhline(12.0, color="#d1495b", linewidth=1.0, linestyle="--")
    axes[2].set_xlim(0.0, 180.0)
    axes[2].set_ylim(0.0, max(13.0, float(np.max(band)) * 1.05))
    axes[2].set_title("projective Gauss band")
    axes[2].set_xlabel("projective angle (deg)")
    axes[2].set_ylabel("great-circle band distance (deg)")
    figure.suptitle(f"Negative control: {code}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"Software": "ShermanQC PGRR-V1"})
    plt.close(figure)


def render_ribbon_subject(
    path: Path,
    *,
    ribbon: Ribbon,
    sample: Sample,
    field: NormalField,
    subject_title: str,
) -> None:
    points = sample.xyz[ribbon.member_indices]
    u, v = pole_basis(ribbon.axis)
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=160, constrained_layout=True)
    source = _raster_source(sample.xyz)
    axes[0, 0].scatter(source[:, 0], source[:, 2], s=0.08, c="#9aa5b1", alpha=0.25, linewidths=0)
    axes[0, 0].scatter(points[:, 0], points[:, 2], s=2.0, c=ribbon.bin_indices, cmap="turbo", linewidths=0)
    _set_equal(axes[0, 0], source[:, 0], source[:, 2])
    axes[0, 0].set_title("whole source")
    axes[0, 0].set_xlabel("X (mm)")
    axes[0, 0].set_ylabel("Z (mm)")
    axes[0, 1].scatter(points @ u, points @ v, s=2.5, c=ribbon.bin_indices, cmap="turbo", linewidths=0)
    _set_equal(axes[0, 1], points @ u, points @ v)
    axes[0, 1].set_title("pole-aligned close-up")
    axes[0, 1].set_xlabel("u (mm)")
    axes[0, 1].set_ylabel("v (mm)")
    normals = field.mean_normal[ribbon.member_indices]
    theta = np.mod(np.arctan2(normals @ v, normals @ u) - ribbon.theta_start, math.pi)
    band = np.degrees(np.arcsin(np.clip(np.abs(normals @ ribbon.axis), 0.0, 1.0)))
    axes[1, 0].scatter(np.degrees(theta), band, s=3.0, c=ribbon.bin_indices, cmap="turbo", linewidths=0)
    for boundary in np.linspace(0.0, math.degrees(ribbon.theta_span), ribbon.bin_count + 1):
        axes[1, 0].axvline(boundary, color="#6c757d", linewidth=0.35, alpha=0.5)
    cut = (ribbon.bin_count - 1) // 2
    axes[1, 0].axvline(
        math.degrees(ribbon.theta_span) * (cut + 1) / ribbon.bin_count,
        color="#d1495b", linewidth=1.5,
    )
    axes[1, 0].axhline(12.0, color="#d1495b", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("projective Gauss arc / canonical cut")
    axes[1, 0].set_xlabel("unwrapped angle (deg)")
    axes[1, 0].set_ylabel("band distance (deg)")
    center = np.asarray([float(index) for index in range(ribbon.bin_count)])
    axes[1, 1].plot(center, np.ones_like(center), "o-", color="#2878b5", linewidth=1.8)
    axes[1, 1].plot(center[: cut + 1], np.zeros(cut + 1), "o-", color="#2878b5", linewidth=1.8)
    axes[1, 1].plot(center[cut + 1 :], np.zeros(ribbon.bin_count - cut - 1), "o-", color="#d1495b", linewidth=1.8)
    axes[1, 1].set_ylim(-0.4, 1.4)
    axes[1, 1].set_yticks([0, 1], ["mid-cut", "full"])
    axes[1, 1].set_xlabel("angular-bin node")
    axes[1, 1].set_title("full chain vs central cut")
    figure.suptitle(subject_title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"Software": "ShermanQC PGRR-V1"})
    plt.close(figure)


def build_visual_package(
    output_dir: Path,
    *,
    sample: Sample,
    field: NormalField,
    ribbons: Sequence[Ribbon],
    controls: Mapping[str, Mapping[str, Any]],
    handoff: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    visual_dir = output_dir / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    global_path = visual_dir / "global-six-view.png"
    render_global_contact_sheet(global_path, sample, ribbons)
    manifest.append({
        "visual_subject_id": "GLOBAL_CANONICAL_OVERLAY",
        "subject_type": "GLOBAL",
        "ribbon_id": "",
        "target_id": "",
        "image_path": str(global_path.relative_to(output_dir)),
        "image_sha256": _sha_file(global_path),
        "camera_commitment_sha256": CAMERA_COMMITMENT,
        "counter_claim": f"deduplicated_ribbon_count={len(ribbons)}",
        "review_reconciliation": "PENDING_MANUAL_VISUAL_REVIEW",
    })
    control_map = {
        "ARC_SPAN_LOW": "insufficient_arc_span",
        "ANGULAR_DISCONNECTION": "angular_disconnection",
        "LONGITUDINAL_OCCUPANCY": "longitudinal_occupancy_failure",
        "RULING_COHERENCE": "ruling_coherence_failure",
    }
    for code, label in control_map.items():
        row = controls.get(code)
        if row is None:
            manifest.append({
                "visual_subject_id": f"CONTROL_{code}",
                "subject_type": "NEGATIVE_CONTROL",
                "ribbon_id": "",
                "target_id": "",
                "image_path": "NOT_PRESENT",
                "image_sha256": "",
                "camera_commitment_sha256": CAMERA_COMMITMENT,
                "counter_claim": f"{label}=NOT_PRESENT",
                "review_reconciliation": "NOT_PRESENT",
            })
            continue
        member_indices = np.asarray(row["member_indices"], dtype=np.uint32)
        path = visual_dir / f"control-{label}.png"
        enriched = dict(row)
        pole_id = int(enriched.get("pole_id", 0))
        # Axis is deterministically reconstructed from the same fixed table.
        from .geometry import projective_poles

        enriched["axis"] = projective_poles()[pole_id].tolist()
        render_negative_control(
            path,
            code=code,
            row=enriched,
            member_indices=member_indices,
            sample=sample,
            field=field,
        )
        manifest.append({
            "visual_subject_id": f"CONTROL_{code}",
            "subject_type": "NEGATIVE_CONTROL",
            "ribbon_id": "",
            "target_id": "",
            "image_path": str(path.relative_to(output_dir)),
            "image_sha256": _sha_file(path),
            "camera_commitment_sha256": CAMERA_COMMITMENT,
            "counter_claim": label,
            "review_reconciliation": "PENDING_MANUAL_VISUAL_REVIEW",
        })
    if handoff is not None:
        subjects = [
            ("DECISIVE_PASS", row) for row in handoff.get("complete_subjects", [])
        ] + [
            ("FORMAL_NEAR_MISS", row) for row in handoff.get("formal_near_misses", [])
        ]
        by_id = {
            f"PGRR_{index:04d}_{row.membership_sha256[:16]}": row
            for index, row in enumerate(ribbons)
        }
        for subject_type, subject in subjects:
            ribbon = by_id[subject["ribbon_id"]]
            identifier = sha_json(subject)[:16]
            path = visual_dir / f"subject-{identifier}.png"
            render_ribbon_subject(
                path,
                ribbon=ribbon,
                sample=sample,
                field=field,
                subject_title=f"{subject_type}: {subject['target_id']} / {subject['ribbon_id']}",
            )
            manifest.append({
                "visual_subject_id": f"SUBJECT_{identifier}",
                "subject_type": subject_type,
                "ribbon_id": subject["ribbon_id"],
                "target_id": subject["target_id"],
                "image_path": str(path.relative_to(output_dir)),
                "image_sha256": _sha_file(path),
                "camera_commitment_sha256": CAMERA_COMMITMENT,
                "counter_claim": json.dumps(subject, sort_keys=True, separators=(",", ":")),
                "review_reconciliation": "PENDING_MANUAL_VISUAL_REVIEW",
            })
    critique = visual_dir / "visual_critique.md"
    critique.write_text(
        "# PGRR-V1 visual critique\n\n"
        "Manual read-only reconciliation is pending. The numerical decision is not final until every present image is inspected.\n\n"
        "- clipping_present: PENDING\n"
        "- voxel_or_grid_phase_artifact: PENDING\n"
        "- angular_partition_boundary_artifact: PENDING\n"
        "- misregistration: PENDING\n"
        "- duplicate_spatial_witness: PENDING\n"
        "- boundary_condition_accident: PENDING\n"
        "- aggregate_misleading: PENDING\n"
        "- counter_image_disagreement: PENDING\n"
        "- review_unresolved: true\n",
        encoding="utf-8",
    )
    return manifest
