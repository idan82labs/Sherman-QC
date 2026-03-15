"""
Dedicated report and artifact generation for bend inspection jobs.

This module turns a bend-inspection report into:
- canonical display geometry for the UI and exported artifacts
- a bend overlay manifest JSON
- static bend overlay images rendered from the backend
- a dedicated bend-inspection PDF report
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ai_analyzer import Model3DSnapshotRenderer

logger = logging.getLogger(__name__)


STATUS_COLORS = {
    "PASS": {"rgb": [0.21, 0.79, 0.46], "hex": "#34d399"},
    "WARNING": {"rgb": [0.98, 0.75, 0.14], "hex": "#fbbf24"},
    "FAIL": {"rgb": [0.97, 0.44, 0.44], "hex": "#f87171"},
    "NOT_DETECTED": {"rgb": [0.39, 0.45, 0.55], "hex": "#64748b"},
}
CAD_LINE_RGB = [0.73, 0.78, 0.85]
CAD_LINE_HEX = "#cbd5e1"
DETECTED_LINE_HEX = "#38bdf8"


@dataclass
class BendArtifactBundle:
    manifest: Dict[str, Any]
    artifacts: Dict[str, Any]
    pdf_path: Optional[str] = None


def _safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("_") or "bend"


def _as_point(raw: Any) -> Optional[np.ndarray]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        arr = np.asarray(raw[:3], dtype=np.float64)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _round_point(point: Optional[np.ndarray]) -> Optional[List[float]]:
    if point is None:
        return None
    return [round(float(point[0]), 4), round(float(point[1]), 4), round(float(point[2]), 4)]


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    idx = min(len(sorted_vals) - 1, max(0.0, (len(sorted_vals) - 1) * q))
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    t = idx - lo
    return sorted_vals[lo] * (1 - t) + sorted_vals[hi] * t


def _compute_model_max_dim(points: np.ndarray) -> float:
    if points.size < 3:
        return 4.0
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return float(max(1.0, *(maxs - mins)))


def _clip_line_to_reference(
    midpoint: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    reference_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    direction = end - start
    original_length = float(np.linalg.norm(direction))
    if original_length < 1e-6:
        return start.copy(), end.copy()

    dir_vec = direction / original_length
    model_max_dim = _compute_model_max_dim(reference_positions)
    max_half_span = max(0.25, model_max_dim * 0.42)

    if reference_positions.size < 12:
        half = min(original_length / 2.0, max_half_span)
        return midpoint - dir_vec * half, midpoint + dir_vec * half

    near_distance = max(0.08, model_max_dim * 0.045)
    samples: List[float] = []

    for ref_pt in reference_positions:
        offset = ref_pt - midpoint
        axial = float(np.dot(offset, dir_vec))
        if abs(axial) > max_half_span * 1.35:
            continue
        perp = offset - dir_vec * axial
        if float(np.linalg.norm(perp)) <= near_distance:
            samples.append(axial)

    if len(samples) < 12:
        half = min(original_length / 2.0, max_half_span)
        return midpoint - dir_vec * half, midpoint + dir_vec * half

    lo = _percentile(samples, 0.08)
    hi = _percentile(samples, 0.92)
    pad = max(0.04, model_max_dim * 0.02)
    half = min(max_half_span, max(0.18, ((hi - lo) / 2.0) + pad))
    return midpoint - dir_vec * half, midpoint + dir_vec * half


def _load_reference_positions(reference_mesh_path: str | Path) -> np.ndarray:
    mesh_path = Path(reference_mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.has_vertices():
        return np.asarray(mesh.vertices, dtype=np.float64)
    pcd = o3d.io.read_point_cloud(str(mesh_path))
    if pcd.has_points():
        return np.asarray(pcd.points, dtype=np.float64)
    return np.zeros((0, 3), dtype=np.float64)


def augment_report_with_overlay_geometry(
    report_dict: Dict[str, Any],
    reference_mesh_path: str | Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Add canonical display geometry to bend matches and create a manifest.

    The output report fields are consumed by:
    - the browser 3D viewer
    - backend-rendered bend overlay images
    - the bend PDF report
    """
    reference_positions = _load_reference_positions(reference_mesh_path)
    matches = report_dict.get("matches", []) if isinstance(report_dict, dict) else []

    issue_tiles: List[Dict[str, Any]] = []
    bends: List[Dict[str, Any]] = []

    for idx, match in enumerate(matches):
        if not isinstance(match, dict):
            continue

        status = str(match.get("status", "NOT_DETECTED")).upper()
        bend_id = str(match.get("bend_id", f"B{idx+1}"))
        cad_start = _as_point(match.get("cad_line_start"))
        cad_end = _as_point(match.get("cad_line_end"))
        detected_start = _as_point(match.get("detected_line_start"))
        detected_end = _as_point(match.get("detected_line_end"))

        display_cad_start = display_cad_end = None
        display_detected_start = display_detected_end = None
        anchor = None

        if cad_start is not None and cad_end is not None:
            cad_mid = (cad_start + cad_end) / 2.0
            display_cad_start, display_cad_end = _clip_line_to_reference(
                cad_mid, cad_start, cad_end, reference_positions
            )
            match["display_cad_line_start"] = _round_point(display_cad_start)
            match["display_cad_line_end"] = _round_point(display_cad_end)
            anchor = (display_cad_start + display_cad_end) / 2.0

        if detected_start is not None and detected_end is not None:
            detected_mid = (detected_start + detected_end) / 2.0
            display_detected_start, display_detected_end = _clip_line_to_reference(
                detected_mid, detected_start, detected_end, reference_positions
            )
            match["display_detected_line_start"] = _round_point(display_detected_start)
            match["display_detected_line_end"] = _round_point(display_detected_end)
            if status != "NOT_DETECTED":
                anchor = (display_detected_start + display_detected_end) / 2.0

        if anchor is not None:
            match["callout_anchor"] = _round_point(anchor)

        include_in_overview = status in {"FAIL", "WARNING"}
        issue_filename = None
        if include_in_overview:
            issue_filename = f"bend_overlay_issue_{_safe_slug(bend_id)}.png"
            issue_tiles.append(
                {
                    "bend_id": bend_id,
                    "status": status,
                    "image": issue_filename,
                    "title": f"{bend_id} {status.title()}",
                }
            )

        delta = {
            "angle_deg": match.get("angle_deviation"),
            "radius_mm": match.get("radius_deviation"),
            "center_mm": match.get("line_center_deviation_mm"),
            "tolerance_angle_deg": match.get("tolerance_angle"),
            "tolerance_radius_mm": match.get("tolerance_radius"),
        }

        bends.append(
            {
                "bend_id": bend_id,
                "status": status,
                "cad_line": {
                    "raw_start": match.get("cad_line_start"),
                    "raw_end": match.get("cad_line_end"),
                    "display_start": match.get("display_cad_line_start"),
                    "display_end": match.get("display_cad_line_end"),
                },
                "detected_line": {
                    "raw_start": match.get("detected_line_start"),
                    "raw_end": match.get("detected_line_end"),
                    "display_start": match.get("display_detected_line_start"),
                    "display_end": match.get("display_detected_line_end"),
                },
                "delta": delta,
                "anchor": match.get("callout_anchor"),
                "include_in_overview": include_in_overview,
                "issue_tile": issue_filename,
                "action_item": match.get("action_item"),
                "issue_location": match.get("issue_location"),
            }
        )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "overview_image": "bend_overlay_overview.png",
        "issue_tiles": issue_tiles,
        "bends": bends,
    }
    return report_dict, manifest


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_bend_overlay_artifacts(
    manifest: Dict[str, Any],
    reference_mesh_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overview_path = out_dir / "bend_overlay_overview.png"
    manifest_path = out_dir / "bend_overlay_manifest.json"
    _write_json(manifest_path, manifest)

    renderer = Model3DSnapshotRenderer()
    bends = manifest.get("bends", [])
    issue_tiles = manifest.get("issue_tiles", [])

    overview_bytes = renderer.render_bend_status_overlay(
        bends=bends,
        reference_mesh_path=str(reference_mesh_path),
        view="iso",
        width=1600,
        height=1200,
        focused_bend_ids=None,
        show_issue_callouts=True,
    )
    overview_path.write_bytes(overview_bytes)

    issue_paths: List[str] = []
    for tile in issue_tiles:
        bend_id = str(tile.get("bend_id"))
        filename = str(tile.get("image"))
        if not bend_id or not filename:
            continue
        issue_path = out_dir / filename
        issue_bytes = renderer.render_bend_status_overlay(
            bends=bends,
            reference_mesh_path=str(reference_mesh_path),
            view="iso",
            width=1200,
            height=900,
            focused_bend_ids=[bend_id],
            show_issue_callouts=False,
        )
        issue_path.write_bytes(issue_bytes)
        issue_paths.append(str(issue_path))

    return {
        "overview_path": str(overview_path),
        "issue_paths": issue_paths,
        "manifest_path": str(manifest_path),
    }


class _BendPDF:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title = ParagraphStyle(
            "BendTitle",
            parent=self.styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0B1320"),
        )
        self.h2 = ParagraphStyle(
            "BendH2",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=6,
        )
        self.body = ParagraphStyle(
            "BendBody",
            parent=self.styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#334155"),
        )
        self.small = ParagraphStyle(
            "BendSmall",
            parent=self.styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10,
            textColor=colors.HexColor("#64748B"),
        )

    def build(
        self,
        report_dict: Dict[str, Any],
        manifest: Dict[str, Any],
        output_path: str | Path,
        output_dir: str | Path,
    ) -> str:
        out_path = str(output_path)
        out_dir = Path(output_dir)
        doc = SimpleDocTemplate(
            out_path,
            pagesize=A4,
            leftMargin=16 * mm,
            rightMargin=16 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
            title=f"Bend Report {report_dict.get('part_id', '')}",
        )
        story: List[Any] = []
        content_width = A4[0] - 32 * mm

        summary = report_dict.get("summary", {})
        operator_brief = report_dict.get("operator_brief", {})
        scan_quality = report_dict.get("scan_quality", {})
        matches = report_dict.get("matches", [])

        story.append(Paragraph("Bend Inspection Report", self.title))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            f"Part: <b>{report_dict.get('part_id', '-')}</b> &nbsp;&nbsp; "
            f"Completed: <b>{summary.get('completed_bends', summary.get('detected', 0))}/"
            f"{summary.get('expected_bends', summary.get('total_bends', 0))}</b>",
            self.body,
        ))
        story.append(Spacer(1, 5 * mm))

        kpi_cells = [
            self._kpi("In Spec", str(summary.get("completed_in_spec", summary.get("passed", 0))), "#059669"),
            self._kpi("Out of Spec", str(summary.get("completed_out_of_spec", 0)), "#DC2626"),
            self._kpi("Remaining", str(summary.get("remaining_bends", 0)), "#F59E0B" if summary.get("remaining_bends", 0) else "#059669"),
            self._kpi("Scan Quality", str(summary.get("scan_quality_status", "-")), "#2563EB"),
        ]
        kpi_table = Table([kpi_cells], colWidths=[content_width / 4.0] * 4)
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 4 * mm))

        if operator_brief:
            story.append(Paragraph("Operator Brief", self.h2))
            story.append(self._box(operator_brief.get("headline", "-"), bg="#EEF2FF", border="#93C5FD"))
            story.append(Spacer(1, 3 * mm))

        if scan_quality:
            quality_line = (
                f"Coverage {scan_quality.get('coverage_pct', '-')}"
                f"% | Density {scan_quality.get('density_pts_per_cm2', '-')}"
                f" pts/cm² | Median spacing {scan_quality.get('median_spacing_mm', '-')}"
                f" mm"
            )
            story.append(Paragraph("Scan Quality", self.h2))
            story.append(self._box(quality_line, bg="#F8FAFC", border="#CBD5E1"))
            story.append(Spacer(1, 3 * mm))

        overview_path = out_dir / str(manifest.get("overview_image", ""))
        if overview_path.exists():
            story.append(Paragraph("3D Bend Overlay Overview", self.h2))
            story.append(Image(str(overview_path), width=content_width, height=content_width * 0.72))
            story.append(Spacer(1, 4 * mm))

        story.append(Paragraph("Bend-by-Bend Status", self.h2))
        story.append(self._bend_table(matches, content_width))
        story.append(Spacer(1, 4 * mm))

        issue_tiles = manifest.get("issue_tiles", [])
        if issue_tiles:
            story.append(PageBreak())
            story.append(Paragraph("Out-of-Tolerance Bend Views", self.h2))
            for i in range(0, len(issue_tiles), 4):
                chunk = issue_tiles[i:i + 4]
                rows: List[List[Any]] = []
                row: List[Any] = []
                for tile in chunk:
                    image_path = out_dir / str(tile.get("image", ""))
                    bend_id = str(tile.get("bend_id", "-"))
                    status = str(tile.get("status", "-"))
                    if image_path.exists():
                        img = Image(str(image_path), width=(content_width / 2.0) - 8, height=((content_width / 2.0) - 8) * 0.72)
                        cell = [img, Spacer(1, 1 * mm), Paragraph(f"<b>{bend_id}</b> - {status}", self.small)]
                    else:
                        cell = [Paragraph(f"<b>{bend_id}</b> - {status}", self.small)]
                    row.append(cell)
                    if len(row) == 2:
                        rows.append(row)
                        row = []
                if row:
                    row.append("")
                    rows.append(row)
                tile_table = Table(rows, colWidths=[content_width / 2.0, content_width / 2.0])
                tile_table.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                    ("INNERGRID", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(tile_table)
                if i + 4 < len(issue_tiles):
                    story.append(PageBreak())
                    story.append(Paragraph("Out-of-Tolerance Bend Views", self.h2))

        doc.build(story)
        return out_path

    def _kpi(self, label: str, value: str, color_hex: str):
        return Paragraph(
            f"<font size='16' color='{color_hex}'><b>{value}</b></font><br/>"
            f"<font size='9' color='#64748B'>{label}</font>",
            self.body,
        )

    def _box(self, text: str, *, bg: str, border: str):
        tbl = Table([[Paragraph(text, self.body)]], colWidths=[A4[0] - 32 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(border)),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return tbl

    def _bend_table(self, matches: Iterable[Dict[str, Any]], content_width: float):
        rows: List[List[str]] = [[
            "Bend", "Status", "Exp/Act Angle", "ΔA", "ΔR", "ΔC", "Action",
        ]]
        for match in matches:
            exp = match.get("target_angle")
            act = match.get("measured_angle")
            action = str(match.get("action_item") or "-")
            rows.append([
                str(match.get("bend_id", "-")),
                str(match.get("status", "-")),
                f"{exp:.1f} / {act:.1f}" if isinstance(exp, (int, float)) and isinstance(act, (int, float)) else f"{exp if exp is not None else '-'} / -",
                self._fmt_num(match.get("angle_deviation"), "°", 1),
                self._fmt_num(match.get("radius_deviation"), " mm", 2),
                self._fmt_num(match.get("line_center_deviation_mm"), " mm", 2),
                action,
            ])

        widths = [18 * mm, 24 * mm, 36 * mm, 22 * mm, 22 * mm, 22 * mm, content_width - 144 * mm]
        table = Table(rows, repeatRows=1, colWidths=widths)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        for r in range(1, len(rows)):
            status = rows[r][1]
            palette = STATUS_COLORS.get(status, STATUS_COLORS["NOT_DETECTED"])
            style.append(("BACKGROUND", (1, r), (1, r), colors.HexColor(palette["hex"])))
            style.append(("TEXTCOLOR", (1, r), (1, r), colors.white))
        table.setStyle(TableStyle(style))
        return table

    @staticmethod
    def _fmt_num(value: Any, suffix: str, digits: int) -> str:
        if not isinstance(value, (int, float)):
            return "-"
        return f"{value:+.{digits}f}{suffix}"


def generate_bend_inspection_pdf(
    report_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    output_path: str | Path,
    output_dir: str | Path,
) -> str:
    return _BendPDF().build(report_dict=report_dict, manifest=manifest, output_path=output_path, output_dir=output_dir)


def build_bend_artifact_bundle(
    report_dict: Dict[str, Any],
    reference_mesh_path: str | Path,
    output_dir: str | Path,
) -> BendArtifactBundle:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_dict, manifest = augment_report_with_overlay_geometry(report_dict, reference_mesh_path)
    render_info = render_bend_overlay_artifacts(
        manifest=manifest,
        reference_mesh_path=reference_mesh_path,
        output_dir=out_dir,
    )
    pdf_path = out_dir / "bend_inspection_report.pdf"
    generate_bend_inspection_pdf(
        report_dict=report_dict,
        manifest=manifest,
        output_path=pdf_path,
        output_dir=out_dir,
    )
    issue_urls = [
        f"/api/bend-inspection/{{job_id}}/overlay/issues/{Path(p).name}"
        for p in render_info["issue_paths"]
    ]
    artifacts = {
        "reference_mesh_url": f"/api/reference-mesh/{{job_id}}.ply",
        "bend_overlay_overview_url": f"/api/bend-inspection/{{job_id}}/overlay/overview.png",
        "bend_overlay_issue_urls": issue_urls,
        "bend_overlay_manifest_url": f"/api/bend-inspection/{{job_id}}/overlay/manifest.json",
        "bend_report_pdf_url": f"/api/bend-inspection/{{job_id}}/pdf",
    }
    return BendArtifactBundle(manifest=manifest, artifacts=artifacts, pdf_path=str(pdf_path))
