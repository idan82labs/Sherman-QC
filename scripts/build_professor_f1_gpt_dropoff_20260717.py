#!/usr/bin/env python3
"""Build the decision-ready Sherman QC professor GPT dropoff for 2026-07-17."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_f1_scan_only_candidate_pool import (
    _attach_finite_scan_support,
    _attach_local_bend_support,
)
from scripts.render_manual_bend_review import render as render_manual_review


PACKAGE_NAME = "Sherman_QC_F1_GPT_Dropoff_20260717"
PACKAGE_DIR = ROOT / "output/professor_package" / PACKAGE_NAME
ARCHIVE_PATH = ROOT / "output/professor_package" / f"{PACKAGE_NAME}.zip"
PDF_PATH = ROOT / "output/pdf/Sherman_QC_F1_Professor_Evidence_20260717.pdf"
PREVIOUS_RESPONSE = Path(
    "/Users/idantal/.codex/attachments/51d7e80c-aa16-442f-961e-5bbbf1f769d2/pasted-text.txt"
)
CHECKPOINT_REPORT = (
    ROOT
    / "output/experiments/f1-10of13/"
    "grouped-holdout-selector-v8-reproduced-after-rejected-feature-cleanup/"
    "grouped_holdout_line_accuracy.json"
)
EXPANSION_LABELS = (
    ROOT
    / "output/experiments/f1-training-expansion/cad-guided-provisional-v1/"
    "18_48991006/bend_instance_labels.json"
)

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#2563EB")
TEAL = colors.HexColor("#0F766E")
AMBER = colors.HexColor("#B45309")
RED = colors.HexColor("#B91C1C")
SLATE = colors.HexColor("#475569")
LINE = colors.HexColor("#CBD5E1")
PALE_BLUE = colors.HexColor("#EFF6FF")
PALE_AMBER = colors.HexColor("#FFF7ED")
PALE_GRAY = colors.HexColor("#F8FAFC")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line(record: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    endpoints = record.get("span_endpoints")
    if endpoints is not None:
        return list(endpoints[0]), list(endpoints[1])
    return list(record["line_start"]), list(record["line_end"])


def _case_inputs() -> list[dict[str, Any]]:
    case06_manifest = _read(
        ROOT / "output/experiments/f1-10of13/plane-lattice-probe/06_38172000/plane_lattice_manifest.json"
    )
    case08_manifest = _read(
        ROOT / "output/experiments/f1-10of13/plane-lattice-probe/08_40500000/plane_lattice_manifest.json"
    )
    case07_decomposition_path = (
        ROOT
        / "output/experiments/f1-11of13/case07-same-pair-span-alias/38172000/"
        "38172000_01_SCAN_2_1.ply--9f27b7c6991e/decomposition_result.json"
    )
    case07_decomposition = _read(case07_decomposition_path)
    case07_ids = {"OB1", "OB3", "OB4", "OB8"}
    case07_axes = [
        record
        for record in case07_decomposition.get("owned_bend_regions") or ()
        if record.get("bend_id") in case07_ids
    ]
    expansion = _read(EXPANSION_LABELS)
    case18_axes = []
    for bend in expansion["manual_bends"]:
        if bend["bend_id"] not in {"P06", "P07", "P08", "P09"}:
            continue
        support = bend["scan_support"]
        case18_axes.append(
            {
                "bend_id": bend["bend_id"],
                "span_endpoints": [support["trimmed_line_start"], support["trimmed_line_end"]],
                "engineering_hypothesis": bend["review_decision"],
                "engineering_reason": bend.get("review_reason"),
                "source_bend_id": bend.get("source_bend_id"),
            }
        )
    return [
        {
            "case_id": "06_38172000",
            "title": "Long transition candidates versus face-interior ridges",
            "scan": Path(case06_manifest["scan_path"]),
            "axes": list(case06_manifest["owned_bend_regions"]),
            "question": (
                "Classify each rendered axis as a countable physical bend or a non-bend. "
                "Then state what raw cross-section evidence would justify any additional axis without using a target count."
            ),
            "source": "scan-supported plane lattice",
        },
        {
            "case_id": "07_38172000",
            "title": "Sub-spacing aliases and maximal physical instances",
            "scan": Path(case07_decomposition["scan_path"]),
            "axes": case07_axes,
            "question": (
                "Decide whether OB1/OB3 are one physical bend or two bends separated by a finite flange, "
                "and independently verify whether OB4/OB8 are one maximal instance."
            ),
            "source": "F1 owned bend regions",
        },
        {
            "case_id": "08_40500000",
            "title": "Six plane intersections and support-gap semantics",
            "scan": Path(case08_manifest["scan_path"]),
            "axes": list(case08_manifest["owned_bend_regions"]),
            "question": (
                "Classify the six rendered axes and define when missing scan support splits one maximal bend into two. "
                "Do not infer an additional bend from any known count."
            ),
            "source": "scan-supported plane lattice",
        },
        {
            "case_id": "18_48991006",
            "title": "Vertical tab folds versus cut/free edges and a face crossing",
            "scan": Path(expansion["scan_path"]),
            "axes": case18_axes,
            "question": (
                "Critique the engineering hypothesis that P06, P07, and P08 are countable tab folds while P09 is a "
                "scan-supported face crossing. Give an axis-local rule that separates these classes."
            ),
            "source": "CAD-guided provisional axes, cropped by raw-scan support",
        },
    ]


def _load_scan(scan_path: Path, max_points: int = 200_000) -> tuple[np.ndarray, np.ndarray, int]:
    cloud = o3d.io.read_point_cloud(str(scan_path))
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    if not len(points):
        mesh = o3d.io.read_triangle_mesh(str(scan_path))
        points = np.asarray(mesh.vertices, dtype=np.float64)
        if mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        else:
            normals = np.empty((0, 3), dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    if len(normals) == len(points):
        finite &= np.isfinite(normals).all(axis=1)
    points = points[finite]
    if len(normals) == len(finite):
        normals = normals[finite]
    raw_count = len(points)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points = points[order]
    if len(normals) == raw_count:
        normals = normals[order]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
        if len(normals) == raw_count:
            normals = normals[indices]
    if len(normals) != len(points):
        sampled = o3d.geometry.PointCloud()
        sampled.points = o3d.utility.Vector3dVector(points)
        sampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4.0, max_nn=30)
        )
        normals = np.asarray(sampled.normals, dtype=np.float64)
    return points, normals, raw_count


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    references = np.eye(3)
    reference = references[int(np.argmin(np.abs(references @ axis)))]
    first = np.cross(axis, reference)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(axis, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return first, second


def _prepare_case(case: dict[str, Any], generated_dir: Path) -> dict[str, Any]:
    points, normals, raw_count = _load_scan(case["scan"])
    candidates: list[dict[str, Any]] = []
    labels = []
    for record in case["axes"]:
        start, end = _line(record)
        span = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
        candidate = {
            "bend_id": str(record.get("bend_id") or record.get("candidate_id")),
            "span_endpoints": [start, end],
            "span_mm": span,
            "engineering_hypothesis": record.get("engineering_hypothesis"),
            "engineering_reason": record.get("engineering_reason"),
        }
        candidates.append(candidate)
        labels.append(
            {
                "bend_id": candidate["bend_id"],
                "line_start": start,
                "line_end": end,
                "engineering_hypothesis": record.get("engineering_hypothesis"),
                "engineering_reason": record.get("engineering_reason"),
            }
        )
    _attach_finite_scan_support(candidates, points, distance_mm=3.0, sample_count=201)
    _attach_local_bend_support(candidates, points, normals, corridor_radius_mm=12.0, sample_count=25)

    labels_path = generated_dir / f"{case['case_id']}_diagnostic_axes.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "professor_diagnostic_axes",
                "coordinate_frame": "scan_raw",
                "manual_bends": labels,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    overview_path = generated_dir / f"{case['case_id']}_six_view_overview.png"
    render_manual_review(
        case["scan"],
        labels_path,
        overview_path,
        200_000,
    )
    cross_path = generated_dir / f"{case['case_id']}_axis_cross_sections.png"
    _render_cross_sections(case, candidates, points, normals, cross_path)
    return {
        **case,
        "raw_point_count": raw_count,
        "sampled_point_count": len(points),
        "candidates": candidates,
        "overview_path": overview_path,
        "cross_section_path": cross_path,
    }


def _render_cross_sections(
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    normals: np.ndarray,
    output_path: Path,
) -> None:
    count = len(candidates)
    figure, axes = plt.subplots(count, 3, figsize=(18, max(3.0, 2.75 * count)), dpi=180)
    axes = np.asarray(axes).reshape(count, 3)
    tree = cKDTree(points)
    for row, candidate in enumerate(candidates):
        endpoints = np.asarray(candidate["span_endpoints"], dtype=np.float64)
        delta = endpoints[1] - endpoints[0]
        span = float(np.linalg.norm(delta))
        axis = delta / max(span, 1e-12)
        first, second = _basis(axis)
        relative = points - endpoints[0]
        longitudinal = relative @ axis
        first_coordinate = relative @ first
        second_coordinate = relative @ second
        radial = np.sqrt(first_coordinate**2 + second_coordinate**2)
        corridor = (
            (longitudinal >= -2.0)
            & (longitudinal <= span + 2.0)
            & (radial <= 12.0)
        )
        middle = 0.5 * span
        slice_half_width = max(2.0, min(8.0, 0.08 * span))
        central = corridor & (np.abs(longitudinal - middle) <= slice_half_width)
        if int(np.sum(central)) < 40:
            central = corridor & (np.abs(longitudinal - middle) <= 2.0 * slice_half_width)
        normal_angle = np.mod(
            np.arctan2(normals[:, :] @ second, normals[:, :] @ first),
            2.0 * np.pi,
        )
        ax = axes[row, 0]
        ax.scatter(
            first_coordinate[central],
            second_coordinate[central],
            c=normal_angle[central],
            cmap="twilight",
            s=2.0,
            alpha=0.58,
            linewidths=0,
        )
        ax.scatter([0], [0], marker="+", s=90, color="#dc2626", linewidths=1.8)
        ax.axhline(0, color="#94a3b8", linewidth=0.5)
        ax.axvline(0, color="#94a3b8", linewidth=0.5)
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{candidate['bend_id']} transverse slice", fontsize=9, fontweight="bold")
        ax.set_xlabel("local v (mm)", fontsize=7)
        ax.set_ylabel("local w (mm)", fontsize=7)

        ax = axes[row, 1]
        shown = np.flatnonzero(corridor)
        if len(shown) > 20_000:
            shown = shown[np.linspace(0, len(shown) - 1, 20_000, dtype=np.int64)]
        ax.scatter(
            longitudinal[shown],
            radial[shown],
            c=normal_angle[shown],
            cmap="twilight",
            s=1.4,
            alpha=0.46,
            linewidths=0,
        )
        ax.set_xlim(-2, span + 2)
        ax.set_ylim(0, 12)
        ax.set_title("longitudinal corridor support", fontsize=9, fontweight="bold")
        ax.set_xlabel("axis position (mm)", fontsize=7)
        ax.set_ylabel("distance from axis (mm)", fontsize=7)

        parameters = np.linspace(0.0, 1.0, 201)
        samples = endpoints[0] + parameters[:, None] * delta
        distances, _ = tree.query(samples, k=1, workers=1)
        ax = axes[row, 2]
        ax.plot(parameters * span, distances, color="#2563eb", linewidth=1.4)
        ax.axhline(3.0, color="#dc2626", linewidth=1.0, linestyle="--", label="3 mm support gate")
        ax.set_xlim(0, span)
        ax.set_ylim(0, max(6.0, min(30.0, float(np.quantile(distances, 0.98)) * 1.15)))
        ax.set_title(
            "nearest-scan support | "
            f"two-sided angle {float(candidate.get('two_sided_normal_angle_deg') or 0):.1f} deg",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel("axis position (mm)", fontsize=7)
        ax.set_ylabel("nearest point (mm)", fontsize=7)
        ax.legend(fontsize=6, loc="upper right")
        for column in range(3):
            axes[row, column].tick_params(labelsize=6)
            axes[row, column].grid(alpha=0.15)
    figure.suptitle(
        f"{case['case_id']} - raw-scan axis-local evidence (normal angle color)",
        fontsize=15,
        fontweight="bold",
        y=0.997,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.25, w_pad=1.2)
    figure.savefig(output_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25,
            leading=29, textColor=NAVY, spaceAfter=6 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["BodyText"], fontSize=11, leading=15,
            textColor=SLATE, spaceAfter=4 * mm,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18,
            leading=22, textColor=NAVY, spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12,
            leading=15, textColor=TEAL, spaceBefore=2 * mm, spaceAfter=1.5 * mm,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=9.2, leading=13,
            textColor=colors.HexColor("#1E293B"), spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontSize=7.5, leading=9.5, textColor=SLATE,
        ),
        "center": ParagraphStyle(
            "center", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9,
            leading=11, textColor=NAVY, alignment=TA_CENTER,
        ),
    }


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setStrokeColor(LINE)
    canvas.line(15 * mm, height - 12 * mm, width - 15 * mm, height - 12 * mm)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(15 * mm, height - 9 * mm, "SHERMAN QC - PROFESSOR GPT EVIDENCE")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(SLATE)
    canvas.drawRightString(width - 15 * mm, height - 9 * mm, "17 July 2026")
    canvas.line(15 * mm, 11 * mm, width - 15 * mm, 11 * mm)
    canvas.drawString(15 * mm, 7 * mm, "Research-only semantic adjudication - no runtime promotion authority")
    canvas.drawRightString(width - 15 * mm, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _fit_image(path: Path, max_width_mm: float, max_height_mm: float) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width_mm * mm / width, max_height_mm * mm / height)
    result = Image(str(path), width=width * scale, height=height * scale)
    result.hAlign = "CENTER"
    return result


def _metric_table(styles: Mapping[str, ParagraphStyle], metrics: Mapping[str, Any]) -> Table:
    values = [
        ("Development scans", str(metrics["case_count"])),
        ("Gold axes", str(metrics["gold_bend_count"])),
        ("TP / FP / FN", f"{metrics['true_positive']} / {metrics['false_positive']} / {metrics['false_negative']}"),
        ("Grouped-holdout F1", f"{metrics['f1']:.6f}"),
        ("Exact parts", f"{metrics['exact_part_count']} / {metrics['case_count']}"),
    ]
    table = Table(
        [[Paragraph(label, styles["small"]) for label, _ in values],
         [Paragraph(value, styles["center"]) for _, value in values]],
        colWidths=[52 * mm] * 5,
        rowHeights=[8 * mm, 12 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE_GRAY),
                ("BACKGROUND", (0, 1), (-1, 1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _axis_table(styles: Mapping[str, ParagraphStyle], candidates: Sequence[Mapping[str, Any]]) -> Table:
    rows = [[
        Paragraph("Axis", styles["small"]),
        Paragraph("Span mm", styles["small"]),
        Paragraph("Finite support", styles["small"]),
        Paragraph("Longest run", styles["small"]),
        Paragraph("Two-sided angle", styles["small"]),
        Paragraph("Two-sided score", styles["small"]),
        Paragraph("Min-side support", styles["small"]),
    ]]
    for candidate in candidates:
        rows.append(
            [
                Paragraph(str(candidate["bend_id"]), styles["center"]),
                Paragraph(f"{float(candidate['span_mm']):.1f}", styles["center"]),
                Paragraph(f"{float(candidate.get('finite_support_fraction') or 0):.2f}", styles["center"]),
                Paragraph(f"{float(candidate.get('finite_support_longest_run_fraction') or 0):.2f}", styles["center"]),
                Paragraph(f"{float(candidate.get('two_sided_normal_angle_deg') or 0):.1f} deg", styles["center"]),
                Paragraph(f"{float(candidate.get('two_sided_transition_score') or 0):.2f}", styles["center"]),
                Paragraph(f"{float(candidate.get('two_sided_min_support_fraction') or 0):.2f}", styles["center"]),
            ]
        )
    table = Table(rows, colWidths=[30 * mm, 29 * mm, 36 * mm, 34 * mm, 38 * mm, 36 * mm, 40 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (0, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _build_pdf(cases: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    styles = _styles()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(A4),
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=17 * mm,
        bottomMargin=15 * mm,
        title="Sherman QC F1 Professor GPT Evidence - 17 July 2026",
        author="Sherman QC engineering",
    )
    story = [
        Spacer(1, 4 * mm),
        Paragraph("Sherman QC F1 semantic blocker", styles["title"]),
        Paragraph(
            "Decision-ready GPT evidence book: raw-scan cross-sections, axis-local diagnostics, and falsifiable algorithm questions",
            styles["subtitle"],
        ),
        _metric_table(styles, metrics),
        Spacer(1, 5 * mm),
        Table(
            [[Paragraph(
                "The honest retained result is grouped leave-one-physical-part-out F1 0.487805 and 0/13 exact parts. "
                "Same-corpus fitting above 0.80 is not zero-shot evidence. Do not choose axes to reproduce any known count.",
                styles["body"],
            )]],
            colWidths=[263 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PALE_AMBER),
                ("BOX", (0, 0), (-1, -1), 0.8, AMBER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]),
        ),
        Spacer(1, 5 * mm),
        Paragraph("What this follow-up adds", styles["h2"]),
        Paragraph(
            "The previous reviewer correctly abstained because sparse orthographic screenshots lacked raw geometry, normals, and axis-local sections. "
            "This package supplies four raw PLY scans, consistent six-view overlays, cross-sections perpendicular to every disputed axis, "
            "longitudinal support corridors, nearest-point profiles, and machine-readable scan_raw endpoints.",
            styles["body"],
        ),
        Paragraph("Requested outcome", styles["h2"]),
        Paragraph(
            "Return a measurable semantic definition and deterministic decision procedure that distinguishes a two-sheet-region fold from a free edge, "
            "face ridge/seam, scan-density artifact, and duplicate alias. Classify the supplied axes, state uncertainty, and propose a falsifiable validation experiment. "
            "Do not provide generic advice or a code patch without the underlying geometric rule.",
            styles["body"],
        ),
        PageBreak(),
        Paragraph("How to read the cross-section pages", styles["h1"]),
        Paragraph(
            "Each row corresponds to one candidate axis. Left: a central transverse slice in a deterministic local frame; color encodes projected normal angle. "
            "Middle: points within a 12 mm corridor along the full finite axis. Right: nearest raw-scan distance sampled at 201 positions; the dashed line is the 3 mm support gate. "
            "These plots are scan-only diagnostics. They are not labels and they do not use expected bend counts.",
            styles["body"],
        ),
        Paragraph("Interpretation boundary", styles["h2"]),
        Paragraph(
            "Point proximity is necessary but not sufficient. A line crossing one planar face may have perfect support, and a real bend may be one-sided under occlusion. "
            "The reviewer must explicitly state what second-region, normal-field, continuity, and maximal-instance evidence is required.",
            styles["body"],
        ),
        Table(
            [
                [Paragraph("Direct observation", styles["small"]), Paragraph("Visible or measurable in an attached raw scan, cross-section, or endpoint record.", styles["body"])],
                [Paragraph("Inference", styles["small"]), Paragraph("A physical interpretation supported by observations but not directly measured.", styles["body"])],
                [Paragraph("Unknown", styles["small"]), Paragraph("Evidence is insufficient; request a precise additional section or annotation rather than guessing.", styles["body"])],
            ],
            colWidths=[45 * mm, 218 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), PALE_GRAY),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]),
        ),
    ]
    for case in cases:
        story.extend(
            [
                PageBreak(),
                Paragraph(f"{case['case_id']} - {case['title']}", styles["h1"]),
                Paragraph(case["question"], styles["body"]),
                Paragraph(
                    f"Source: {case['source']}. Raw points: {case['raw_point_count']:,}; deterministic diagnostic sample: {case['sampled_point_count']:,}.",
                    styles["small"],
                ),
                Spacer(1, 2 * mm),
                _fit_image(case["overview_path"], 263, 145),
                PageBreak(),
                Paragraph(f"{case['case_id']} - axis-local raw-scan evidence", styles["h1"]),
                _axis_table(styles, case["candidates"]),
                Spacer(1, 2 * mm),
                _fit_image(
                    case["cross_section_path"],
                    263,
                    100 if len(case["candidates"]) > 4 else 118,
                ),
            ]
        )
    story.extend(
        [
            PageBreak(),
            Paragraph("Response requirements", styles["h1"]),
            Paragraph(
                "Use the attached JSON schema. Return JSON only. For every axis, separate direct observations, inference, and unknowns; do not infer count from filenames or context. "
                "The most important deliverable is an implementable geometric rule with formulas or pseudocode, threshold provenance, failure modes, and a validation plan that could falsify it.",
                styles["body"],
            ),
            Table(
                [
                    [Paragraph("Required", styles["small"]), Paragraph("Per-axis semantic classification and confidence for all four cases.", styles["body"])],
                    [Paragraph("Required", styles["small"]), Paragraph("Necessary conditions for a countable bend and explicit rejection conditions for outlines/ridges/seams.", styles["body"])],
                    [Paragraph("Required", styles["small"]), Paragraph("Alias/maximal-span rule and occlusion rule.", styles["body"])],
                    [Paragraph("Required", styles["small"]), Paragraph("Deterministic decision procedure with measurable features, units, and threshold provenance.", styles["body"])],
                    [Paragraph("Required", styles["small"]), Paragraph("One next experiment, its success criterion, and what outcome would refute the proposed rule.", styles["body"])],
                    [Paragraph("Forbidden", styles["small"]), Paragraph("Count forcing, invented coordinates, CAD assumptions, or claiming release readiness.", styles["body"])],
                ],
                colWidths=[38 * mm, 225 * mm],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (0, 4), PALE_BLUE),
                    ("BACKGROUND", (0, 5), (0, 5), colors.HexColor("#FEF2F2")),
                    ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                    ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]),
            ),
        ]
    )
    document.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


def _response_schema(case_axes: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    case_properties = {}
    for case_id, axes in case_axes.items():
        axis_assessment = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "classification", "countable", "direct_observations",
                "inference", "unknowns", "confidence", "needed_followup"
            ],
            "properties": {
                "classification": {
                    "enum": [
                        "countable_bend", "free_or_cut_edge", "face_ridge_or_seam",
                        "scan_artifact", "duplicate_alias", "insufficient_evidence"
                    ]
                },
                "countable": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                "direct_observations": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "inference": {"type": "string"},
                "unknowns": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "needed_followup": {"type": "array", "items": {"type": "string"}},
            },
        }
        case_properties[case_id] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["evidence_sufficient", "axis_assessments", "case_conclusion", "additional_evidence_needed"],
            "properties": {
                "evidence_sufficient": {"type": "boolean"},
                "axis_assessments": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(axes),
                    "properties": {axis_id: axis_assessment for axis_id in axes},
                },
                "case_conclusion": {"type": "string"},
                "additional_evidence_needed": {"type": "array", "items": {"type": "string"}},
            },
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "sherman-qc-f1-professor-dropoff-response-20260717-v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "submission_id", "global_assessment", "cases", "semantic_definition",
            "algorithmic_rule", "validation_plan", "final_recommendation"
        ],
        "properties": {
            "submission_id": {"const": "sherman-qc-f1-semantic-dropoff-20260717-v1"},
            "global_assessment": {
                "type": "object",
                "additionalProperties": False,
                "required": ["raw_evidence_sufficient", "count_forcing_avoided", "limitations"],
                "properties": {
                    "raw_evidence_sufficient": {"type": "boolean"},
                    "count_forcing_avoided": {"const": True},
                    "limitations": {"type": "array", "items": {"type": "string"}},
                },
            },
            "cases": {
                "type": "object",
                "additionalProperties": False,
                "required": list(case_properties),
                "properties": case_properties,
            },
            "semantic_definition": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "countable_bend_necessary_conditions", "rejection_conditions",
                    "alias_merge_rule", "maximal_instance_rule", "occlusion_rule"
                ],
                "properties": {
                    "countable_bend_necessary_conditions": {"type": "array", "minItems": 3, "items": {"type": "string"}},
                    "rejection_conditions": {"type": "array", "minItems": 3, "items": {"type": "string"}},
                    "alias_merge_rule": {"type": "string"},
                    "maximal_instance_rule": {"type": "string"},
                    "occlusion_rule": {"type": "string"},
                },
            },
            "algorithmic_rule": {
                "type": "object",
                "additionalProperties": False,
                "required": ["local_coordinate_system", "features", "decision_procedure", "thresholds", "failure_modes"],
                "properties": {
                    "local_coordinate_system": {"type": "string"},
                    "features": {
                        "type": "array", "minItems": 3,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["name", "formula_or_measurement", "units", "semantic_role"],
                            "properties": {
                                "name": {"type": "string"},
                                "formula_or_measurement": {"type": "string"},
                                "units": {"type": "string"},
                                "semantic_role": {"type": "string"},
                            },
                        },
                    },
                    "decision_procedure": {"type": "string", "minLength": 100},
                    "thresholds": {
                        "type": "array",
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["name", "value_or_range", "units", "provenance", "learn_or_fix"],
                            "properties": {
                                "name": {"type": "string"},
                                "value_or_range": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "units": {"type": "string"},
                                "provenance": {"type": "string"},
                                "learn_or_fix": {"enum": ["learn", "fix", "unknown"]},
                            },
                        },
                    },
                    "failure_modes": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                },
            },
            "validation_plan": {
                "type": "object",
                "additionalProperties": False,
                "required": ["next_experiment", "success_criterion", "falsification_criterion", "minimum_new_labels"],
                "properties": {
                    "next_experiment": {"type": "string"},
                    "success_criterion": {"type": "string"},
                    "falsification_criterion": {"type": "string"},
                    "minimum_new_labels": {"type": "integer", "minimum": 0},
                },
            },
            "final_recommendation": {
                "type": "object",
                "additionalProperties": False,
                "required": ["safe_to_implement_research_prototype", "safe_to_promote_runtime", "reason", "next_action"],
                "properties": {
                    "safe_to_implement_research_prototype": {"type": "boolean"},
                    "safe_to_promote_runtime": {"const": False},
                    "reason": {"type": "string"},
                    "next_action": {"type": "string"},
                },
            },
        },
    }


def _write_submission_files(cases: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    case_axes = {case["case_id"]: [str(item["bend_id"]) for item in case["candidates"]] for case in cases}
    schema = _response_schema(case_axes)
    (PACKAGE_DIR / "03_response_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )
    context = {
        "schema_version": 1,
        "submission_id": "sherman-qc-f1-semantic-dropoff-20260717-v1",
        "coordinate_frame": "scan_raw",
        "inference_contract": {
            "part_identity_used": False,
            "expected_counts_used": False,
            "cad_used_for_case_18_source_axes_only": True,
            "cad_used_for_scan_support_or_cross_sections": False,
            "manual_truth_used_for_candidate_generation": False,
        },
        "retained_grouped_holdout_metrics": dict(metrics),
        "same_corpus_fit_f1_not_zero_shot": 0.821918,
        "runtime_promotion_authorized": False,
        "cases": [],
    }
    for case in cases:
        context["cases"].append(
            {
                "case_id": case["case_id"],
                "title": case["title"],
                "question": case["question"],
                "source": case["source"],
                "scan_file": f"raw_scans/{case['case_id']}_{case['scan'].name}",
                "scan_sha256": _sha256(case["scan"]),
                "raw_point_count": case["raw_point_count"],
                "diagnostic_sample_point_count": case["sampled_point_count"],
                "axes": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key
                        in {
                            "bend_id", "span_endpoints", "span_mm", "finite_support_fraction",
                            "finite_support_longest_run_fraction", "endpoint_support_fraction",
                            "median_line_distance_mm", "p90_line_distance_mm",
                            "two_sided_transition_score", "two_sided_normal_angle_deg",
                            "two_sided_support_balance", "two_sided_min_support_fraction",
                            "two_sided_normal_coherence", "engineering_hypothesis",
                            "engineering_reason",
                        }
                    }
                    for candidate in case["candidates"]
                ],
            }
        )
    (PACKAGE_DIR / "02_machine_context.json").write_text(
        json.dumps(context, indent=2) + "\n", encoding="utf-8"
    )
    prompt = """# Sherman QC F1 - professor semantic adjudication

You are reviewing a zero-shot sheet-metal bend inspection system. Your task is not to improve a score by matching known counts. Your task is to provide an implementable physical-semantic rule for deciding whether a scan-supported line is one countable bend, a free/cut edge, a face ridge or seam, a scan artifact, or a duplicate alias.

## Uploads

1. `01_evidence_book.pdf` - indexed six-view overlays, raw-scan cross-sections, support profiles, and current honest metrics.
2. `02_machine_context.json` - exact scan_raw endpoints and numeric diagnostics.
3. `03_response_schema.json` - mandatory JSON response contract.
4. `04_previous_professor_response.json` - prior review; treat it as an earlier opinion, not ground truth.
5. Four raw PLY scans in `raw_scans/` - primary evidence when a plot is ambiguous.

## Required decisions

- `06_38172000`: classify all four rendered plane-intersection axes. State what raw evidence would justify any additional axis, without using a target count.
- `07_38172000`: decide whether OB1/OB3 are one alias pair or two bends separated by a finite flange. Independently verify OB4/OB8 under the same maximal-instance rule.
- `08_40500000`: classify all six plane intersections and define when a support gap splits one maximal physical bend.
- `18_48991006`: critique the current engineering hypothesis that P06/P07/P08 are vertical tab folds while P09 is a face crossing. This is the key calibration case.

## Method

1. Inspect the evidence book first, then consult the raw PLY and machine context where necessary.
2. For every axis, separate direct observation, inference, and unknowns.
3. Define necessary conditions for a countable bend. Point proximity alone is not sufficient.
4. Provide formulas or precise measurements for the proposed local features, including units.
5. State which thresholds are physical constants, fixed engineering tolerances, learned parameters, or currently unknown.
6. Give a deterministic decision procedure or pseudocode suitable for a scan-only research prototype.
7. Give one falsifiable next experiment with success and refutation criteria.

## Hard constraints

- Do not use expected bend counts, filenames, or part identity to choose classifications.
- Do not assume CAD geometry except that case-18 source axes originated from an old CAD/scan match; all support and cross-section evidence is raw-scan derived.
- Do not invent coordinates; exact endpoints are already in `02_machine_context.json`.
- Do not call the system release-ready. Runtime promotion is prohibited.
- Prefer `insufficient_evidence` and a precise evidence request over speculation.
- Avoid generic advice such as "train on more data" unless you specify the exact new annotation and why it identifies the disputed factor.

## Response

Return exactly one JSON object valid against `03_response_schema.json`. Do not use Markdown fences and do not add prose outside the JSON.
"""
    (PACKAGE_DIR / "00_PASTE_THIS_PROMPT.md").write_text(prompt, encoding="utf-8")
    readme = """# Sherman QC F1 professor GPT dropoff - 17 July 2026

This is a direct GPT submission bundle, not an email package.

1. Start a fresh GPT conversation with vision/file analysis enabled.
2. Upload the files listed in `gpt_submission.json` in the stated order.
3. Paste `00_PASTE_THIS_PROMPT.md` verbatim.
4. Require JSON-only output and validate it against `03_response_schema.json`.

The four PLY files are primary evidence. The PDF is an indexed visual guide. The prior response is included only for continuity and is not ground truth.
"""
    (PACKAGE_DIR / "README.md").write_text(readme, encoding="utf-8")


def _build_package() -> dict[str, Any]:
    required = [CHECKPOINT_REPORT, EXPANSION_LABELS, PREVIOUS_RESPONSE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing package inputs: " + ", ".join(missing))
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True)
    generated_dir = PACKAGE_DIR / "generated_evidence"
    raw_dir = PACKAGE_DIR / "raw_scans"
    generated_dir.mkdir()
    raw_dir.mkdir()

    cases = [_prepare_case(case, generated_dir) for case in _case_inputs()]
    metrics = _read(CHECKPOINT_REPORT)["metrics"]
    _build_pdf(cases, metrics)
    shutil.copy2(PDF_PATH, PACKAGE_DIR / "01_evidence_book.pdf")
    previous = json.loads(PREVIOUS_RESPONSE.read_text(encoding="utf-8"))
    (PACKAGE_DIR / "04_previous_professor_response.json").write_text(
        json.dumps(previous, indent=2) + "\n", encoding="utf-8"
    )
    for case in cases:
        shutil.copy2(case["scan"], raw_dir / f"{case['case_id']}_{case['scan'].name}")
    _write_submission_files(cases, metrics)

    upload_order = [
        "01_evidence_book.pdf",
        "02_machine_context.json",
        "03_response_schema.json",
        "04_previous_professor_response.json",
        *[f"raw_scans/{case['case_id']}_{case['scan'].name}" for case in cases],
    ]
    submission = {
        "schema_version": 1,
        "submission_id": "sherman-qc-f1-semantic-dropoff-20260717-v1",
        "mode": "direct_multimodal_gpt_submission",
        "paste_file": "00_PASTE_THIS_PROMPT.md",
        "upload_order": upload_order,
        "expected_response_mime_type": "application/json",
        "response_schema": "03_response_schema.json",
        "runtime_promotion_authorized": False,
    }
    (PACKAGE_DIR / "gpt_submission.json").write_text(
        json.dumps(submission, indent=2) + "\n", encoding="utf-8"
    )

    manifest_files = sorted(path for path in PACKAGE_DIR.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "package_name": PACKAGE_NAME,
        "generated_at": "2026-07-17",
        "purpose": "Direct multimodal GPT adjudication of the Sherman QC F1 semantic blocker",
        "runtime_promotion_authorized": False,
        "files": [
            {
                "path": str(path.relative_to(PACKAGE_DIR)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in manifest_files
        ],
    }
    (PACKAGE_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    checksum_files = sorted(path for path in PACKAGE_DIR.rglob("*") if path.is_file())
    (PACKAGE_DIR / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(PACKAGE_DIR)}\n" for path in checksum_files),
        encoding="utf-8",
    )

    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{PACKAGE_NAME}/{path.relative_to(PACKAGE_DIR)}")
    return {
        "package_dir": str(PACKAGE_DIR),
        "archive": str(ARCHIVE_PATH),
        "pdf": str(PDF_PATH),
        "upload_count": len(upload_order),
        "archive_bytes": ARCHIVE_PATH.stat().st_size,
        "archive_sha256": _sha256(ARCHIVE_PATH),
    }


def main() -> None:
    print(json.dumps(_build_package(), indent=2))


if __name__ == "__main__":
    main()
