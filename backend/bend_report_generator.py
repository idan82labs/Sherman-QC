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
import io
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

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
from PIL import Image as PILImage, ImageDraw, ImageFont

from ai_analyzer import Model3DSnapshotRenderer
from contracts.ui.bend_results import BendDecisionContract
from domains.bend.application.finalize_bend_report import finalize_bend_report
from domains.bend.services.inspection_frame import validate_alignment

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
RESEARCH_WATERMARK = "RESEARCH ONLY - NOT PRODUCTION QC"


def _export_decision_evidence(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate typed V1 evidence and its exact correspondence to raw targets.

    No coercion, repaired authorization claim, missing row, duplicate identity,
    or legacy count creates an observation. Countability follows the UI's
    top-level-or-nested nullish fallback; all raw rows require exact coverage.
    """
    try:
        raw = report.get("decision_contract")
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("reasons"), list)
            or not isinstance(raw.get("bends"), list)
            or any(not isinstance(row, dict) or not isinstance(row.get("reasons"), list) for row in raw["bends"])
        ):
            raise ValueError("explicit decision evidence fields are required")
        decision = BendDecisionContract.model_validate(raw, strict=True)
        alignment = report.get("alignment")
        if alignment is not None and not isinstance(alignment, Mapping):
            raise ValueError("alignment evidence must be a mapping or unavailable")
        abstained = alignment.get("alignment_abstained") if alignment is not None else None
        if abstained is not None and type(abstained) is not bool:
            raise ValueError("alignment abstention evidence must be boolean or unavailable")
        if abstained is True and decision.registration != "ABSTAINED":
            raise ValueError("decision evidence contradicts raw registration abstention")
        matches = report.get("matches")
        if not isinstance(matches, list):
            raise ValueError("raw target rows are unavailable")
        rows = {row.bend_id: row for row in decision.bends}
        identities = set()
        expected = {"formed": 0, "unformed": 0, "unknown": 0, "total": 0, "partially_observed": 0}
        for match in matches:
            if not isinstance(match, dict):
                raise ValueError("raw target row is malformed")
            identity = match.get("bend_id")
            if not isinstance(identity, str) or not identity.strip() or identity in identities or identity not in rows:
                raise ValueError("raw targets require unique, covered identities")
            identities.add(identity)
            cad = match.get("cad_bend")
            cad = cad if isinstance(cad, Mapping) else {}
            countable = match.get("countable_in_regression")
            if countable is None:
                countable = cad.get("countable_in_regression")
            if countable is not None and type(countable) is not bool:
                raise ValueError("effective countability evidence must be boolean or unavailable")
            if countable is False:
                continue
            row = rows[identity]
            expected["total"] += 1
            expected[row.state.lower()] += 1
            if row.state == "UNKNOWN" and row.observation_detail == "PARTIALLY_OBSERVED":
                expected["partially_observed"] += 1
        if identities != set(rows) or expected != decision.observed_counts.model_dump():
            raise ValueError("decision counts and coverage must match countable raw targets")
    except (TypeError, ValueError):
        return {
            "schema_version": 1,
            "kind": "BEND_DECISION_EVIDENCE_UNAVAILABLE",
            "observed_counts": None,
            "registration": "UNAVAILABLE",
            "render_evidence": "UNAVAILABLE",
            "production_disposition": "NOT_AUTHORIZED",
            "reasons": ["DECISION_CONTRACT_UNAVAILABLE_OR_MALFORMED", "NO_PRODUCTION_DISPOSITION"],
            "bends": [],
        }
    return decision.model_dump()


def _export_qualification(report: Dict[str, Any]) -> Dict[str, Any]:
    qualification = report.get("qualification_context")
    return {
        **(qualification if isinstance(qualification, dict) else {}),
        "operating_mode": "RESEARCH_ONLY",
        "production_use_allowed": False,
        "promotion_status": "DECISION_EVIDENCE_NONAUTHORIZING",
    }


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
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _round_point(point: Optional[np.ndarray]) -> Optional[List[float]]:
    if point is None:
        return None
    return [round(float(point[0]), 4), round(float(point[1]), 4), round(float(point[2]), 4)]


# Display validation only, not a manufacturing tolerance or bend measurement.
DISPLAY_SUPPORT_RADIUS_MM = 3.0
DISPLAY_SAMPLE_SPACING_MM = 0.25
DISPLAY_MIN_SUPPORT_FRACTION = 0.6
DISPLAY_MIN_SPAN_MM = 8.0
DISPLAY_MAX_LINE_SAMPLES = 8193


def _has_reference_geometry_frame(report: Dict[str, Any]) -> bool:
    try:
        validate_alignment(report.get("alignment"))
        return True
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return False


def _load_reference_surface(reference_mesh_path: str | Path):
    """Distance queries must use faces: CAD tessellations have sparse vertices."""
    mesh = o3d.io.read_triangle_mesh(str(reference_mesh_path))
    if not mesh.has_triangles() or not mesh.has_vertices():
        return None
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not np.all(np.isfinite(vertices)) or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        return None
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _supported_display_line(start: Optional[np.ndarray], end: Optional[np.ndarray], scene):
    """Trim to supported extent on the original axis; never snap to a surface.

    These are reference-supported axis markers, not measured surface locations.
    A conservative half-spacing margin bounds distances between query samples.
    """
    diagnostics = {"support_fraction": 0.0, "maximum_sample_distance_mm": None}
    if start is None or end is None:
        return None, "MISSING_LINE", diagnostics
    if scene is None:
        return None, "REFERENCE_TRIANGLES_UNAVAILABLE", diagnostics
    length = float(np.linalg.norm(end - start))
    if length <= 1e-6:
        return None, "DEGENERATE_LINE", diagnostics
    intervals = max(1, int(math.ceil(length / DISPLAY_SAMPLE_SPACING_MM)))
    if intervals + 1 > DISPLAY_MAX_LINE_SAMPLES:
        return None, "DISPLAY_VALIDATION_BUDGET_EXCEEDED", diagnostics
    points = np.linspace(start, end, intervals + 1)
    distances = scene.compute_distance(o3d.core.Tensor(points.astype(np.float32)), nthreads=1).numpy()
    if not np.all(np.isfinite(distances)):
        return None, "NONFINITE_SURFACE_DISTANCE", diagnostics
    supported = distances + length / intervals / 2.0 <= DISPLAY_SUPPORT_RADIUS_MM
    diagnostics["maximum_sample_distance_mm"] = float(np.max(distances))
    # Find one contiguous interval, so a line cannot bridge unsupported space.
    boundaries = np.diff(np.r_[False, supported, False].astype(np.int8))
    runs = list(zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1) - 1))
    if not runs:
        return None, "LINE_NOT_SUPPORTED_BY_REFERENCE_SURFACE", diagnostics
    first, last = max(runs, key=lambda pair: pair[1] - pair[0])
    fraction = float(last - first) / intervals
    diagnostics.update({
        "support_fraction": fraction,
        "supported_span_mm": fraction * length,
        "supported_maximum_sample_distance_mm": float(np.max(distances[first:last + 1])),
        "retained_parameter_interval": [float(first / intervals), float(last / intervals)],
    })
    if fraction < DISPLAY_MIN_SUPPORT_FRACTION or fraction * length < DISPLAY_MIN_SPAN_MM:
        return None, "INSUFFICIENT_CONTIGUOUS_REFERENCE_SUPPORT", diagnostics
    return (points[first], points[last]), None, diagnostics


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
    frame_valid = _has_reference_geometry_frame(report_dict)
    reference_surface = _load_reference_surface(reference_mesh_path) if frame_valid else None
    matches = report_dict.get("matches", []) if isinstance(report_dict, dict) else []
    matches = matches if isinstance(matches, list) else []

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

        anchor = None
        match.pop("display_cad_line_start", None)
        match.pop("display_cad_line_end", None)
        match.pop("display_detected_line_start", None)
        match.pop("display_detected_line_end", None)
        match.pop("callout_anchor", None)
        match["display_geometry_frame"] = "reference_mesh_raw" if frame_valid else None
        match["display_geometry_provenance"] = "reference_triangle_surface_validation_v2" if frame_valid else None
        match["display_geometry_valid"] = False
        reasons = {}
        support_diagnostics = {}

        for kind, start, end in (
            ("cad", cad_start, cad_end),
            ("detected", detected_start, detected_end),
        ):
            if kind == "detected" and match.get("detected_line_geometry_source") == "CAD_REFERENCE_PLACEHOLDER":
                # Refinement can measure scan planes while borrowing CAD axis
                # endpoints. Those endpoints are not a detected positional line.
                segment, reason, diagnostics = None, "CAD_REFERENCE_PLACEHOLDER_NOT_SCAN_LINE", {}
            else:
                segment, reason, diagnostics = (
                    _supported_display_line(start, end, reference_surface)
                    if frame_valid else (None, "REFERENCE_FRAME_UNPROVEN", {})
                )
            reasons[kind] = reason
            support_diagnostics[kind] = diagnostics
            if segment is None:
                continue
            supported_start, supported_end = segment
            match[f"display_{kind}_line_start"] = _round_point(supported_start)
            match[f"display_{kind}_line_end"] = _round_point(supported_end)
            if kind == "cad" or status != "NOT_DETECTED":
                anchor = (supported_start + supported_end) / 2.0

        match["display_geometry_validation"] = {
            "support_radius_mm": DISPLAY_SUPPORT_RADIUS_MM,
            "minimum_support_fraction": DISPLAY_MIN_SUPPORT_FRACTION,
            "minimum_span_mm": DISPLAY_MIN_SPAN_MM,
            "semantics": "reference_supported_axis_not_measured_surface_location",
            "lines": support_diagnostics,
            "surface": "reference_mesh_triangles",
            "rejections": reasons,
        }
        if anchor is not None:
            match["callout_anchor"] = _round_point(anchor)
            match["display_geometry_valid"] = True

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
                    "geometry_source": match.get("detected_line_geometry_source"),
                    "raw_start": match.get("detected_line_start"),
                    "raw_end": match.get("detected_line_end"),
                    "display_start": match.get("display_detected_line_start"),
                    "display_end": match.get("display_detected_line_end"),
                },
                "delta": delta,
                "anchor": match.get("callout_anchor"),
                "display_geometry_frame": match.get("display_geometry_frame"),
                "display_geometry_provenance": match.get("display_geometry_provenance"),
                "display_geometry_valid": bool(match.get("display_geometry_valid")),
                "display_geometry_validation": match.get("display_geometry_validation"),
                "include_in_overview": include_in_overview,
                "issue_tile": issue_filename,
                "action_item": match.get("action_item"),
                "issue_location": match.get("issue_location"),
                # Edge-to-edge flange distance for Bender's View dimension arrows
                "measured_edge_distance_mm": match.get("measured_edge_distance_mm"),
                "expected_edge_distance_mm": match.get("expected_edge_distance_mm"),
                "measured_edge_tip1": match.get("measured_edge_tip1"),
                "measured_edge_tip2": match.get("measured_edge_tip2"),
            }
        )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "qualification_context": _export_qualification(report_dict),
        "decision_contract": _export_decision_evidence(report_dict),
        "overview_image": "bend_overlay_overview.png",
        "issue_tiles": issue_tiles,
        "bends": bends,
    }
    return report_dict, manifest


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _production_use_allowed(manifest: Dict[str, Any]) -> bool:
    """V1 and missing/unknown contracts cannot authorize detached artifacts.

    A stale qualification flag is not an independent production release gate.
    Any future authorizing schema requires a separately implemented verifier.
    """
    return False


def _stamp_research_only(image_bytes: bytes) -> bytes:
    """Add an inseparable warning to a detached visual artifact."""
    with PILImage.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    banner_height = max(38, int(round(image.height * 0.065)))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, banner_height), fill=(136, 19, 55))
    font_size = max(20, int(round(image.height * 0.026)))
    font = None
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), RESEARCH_WATERMARK, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        ((image.width - text_width) / 2, (banner_height - text_height) / 2),
        RESEARCH_WATERMARK,
        fill=(255, 255, 255),
        font=font,
    )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


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
    if not _production_use_allowed(manifest):
        overview_bytes = _stamp_research_only(overview_bytes)
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
        if not _production_use_allowed(manifest):
            issue_bytes = _stamp_research_only(issue_bytes)
        issue_path.write_bytes(issue_bytes)
        issue_paths.append(str(issue_path))

    # Generate 4 orthographic bender view PNGs
    BENDER_VIEWS = [
        ("front", "bender_view_front.png"),
        ("side", "bender_view_profile.png"),
        ("top", "bender_view_top.png"),
        ("iso", "bender_view_overall.png"),
    ]
    bender_view_paths: Dict[str, str] = {}
    for view_name, filename in BENDER_VIEWS:
        view_path = out_dir / filename
        try:
            view_bytes = renderer.render_bend_status_overlay(
                bends=bends,
                reference_mesh_path=str(reference_mesh_path),
                view=view_name,
                width=1200,
                height=900,
                focused_bend_ids=None,
                show_issue_callouts=True,
                show_detected_lines=False,
                show_3d_markers=False,
                show_cad_lines=False,  # Clean bender view — mesh + 2D labels only
            )
            if not _production_use_allowed(manifest):
                view_bytes = _stamp_research_only(view_bytes)
            view_path.write_bytes(view_bytes)
            bender_view_paths[view_name] = str(view_path)
        except Exception:
            pass

    return {
        "overview_path": str(overview_path),
        "issue_paths": issue_paths,
        "manifest_path": str(manifest_path),
        "bender_view_paths": bender_view_paths,
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
        summary = summary if isinstance(summary, dict) else {}
        scan_quality = report_dict.get("scan_quality", {})
        scan_quality = scan_quality if isinstance(scan_quality, dict) else {}
        matches = report_dict.get("matches", [])
        matches = matches if isinstance(matches, list) else []
        decision = _export_decision_evidence(report_dict)
        observed = decision["observed_counts"]

        story.append(Paragraph("Bend Inspection Report", self.title))
        story.append(Spacer(1, 3 * mm))
        story.append(self._box(
            "<b>RESEARCH ONLY - NOT QUALIFIED FOR PRODUCTION ACCEPT/REJECT.</b> "
            "Observation states and tolerance comparisons are experimental evidence, not part disposition. "
            "A missing observation is not an unfinished or defective bend.",
            bg="#FFF1F2",
            border="#E11D48",
        ))
        story.append(Spacer(1, 3 * mm))
        observed_text = f"{observed['formed']}/{observed['total']}" if observed is not None else "Unavailable"
        story.append(Paragraph(
            f"Part: <b>{escape(str(report_dict.get('part_id', '-')))}</b> &nbsp;&nbsp; "
            f"Observed formed: <b>{observed_text}</b>",
            self.body,
        ))
        story.append(Spacer(1, 5 * mm))

        kpi_cells = [
            self._kpi("Observed formed", str(observed["formed"]) if observed is not None else "Unavailable", "#2563EB"),
            self._kpi("Observed unformed", str(observed["unformed"]) if observed is not None else "Unavailable", "#2563EB"),
            self._kpi("Unknown", str(observed["unknown"]) if observed is not None else "Unavailable", "#64748B"),
            self._kpi("Disposition", "Not authorized", "#BE123C"),
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

        story.append(Paragraph("Evidence limitations", self.h2))
        story.append(self._box(
            f"Registration: <b>{decision['registration']}</b>; "
            f"render evidence: <b>{decision['render_evidence']}</b>. "
            "Image availability does not establish correct registration, visibility or measurement validity.",
            bg="#F8FAFC", border="#CBD5E1",
        ))
        story.append(Spacer(1, 3 * mm))
        if summary:
            estimated = summary.get("completed_bends", summary.get("detected", "Unavailable"))
            expected = summary.get("expected_bends", summary.get("total_bends", "Unavailable"))
            remaining = summary.get("remaining_bends", "Unavailable")
            story.append(Paragraph("Model estimate - not observed completion", self.h2))
            story.append(self._box(
                f"Legacy estimate: {escape(str(estimated))}/{escape(str(expected))}; "
                f"estimated remaining: {escape(str(remaining))}. "
                "These model estimates are not observed formed/unformed counts or manufacturing instructions.",
                bg="#F8FAFC", border="#CBD5E1",
            ))
            story.append(Spacer(1, 3 * mm))

        if scan_quality:
            quality_line = (
                f"Coverage {scan_quality.get('coverage_pct', '-')}"
                f"% | Density {scan_quality.get('density_pts_per_cm2', '-')}"
                f" pts/cm² | Median spacing {scan_quality.get('median_spacing_mm', '-')}"
                f" mm"
            )
            story.append(Paragraph("Experimental scan-quality measurements", self.h2))
            story.append(self._box(escape(quality_line), bg="#F8FAFC", border="#CBD5E1"))
            story.append(Spacer(1, 3 * mm))

        overview_path = out_dir / str(manifest.get("overview_image", ""))
        if overview_path.exists():
            story.append(Paragraph("Experimental 3D overlay - geometry unverified", self.h2))
            story.append(Image(str(overview_path), width=content_width, height=content_width * 0.72))
            story.append(Spacer(1, 4 * mm))

        story.append(Paragraph("Observation states and experimental measurements", self.h2))
        story.append(self._bend_table(matches, content_width, decision))
        story.append(Spacer(1, 4 * mm))

        issue_tiles = manifest.get("issue_tiles", [])
        if issue_tiles:
            story.append(PageBreak())
            story.append(Paragraph("Experimental flagged-bend views - not defect dispositions", self.h2))
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
                        cell = [img, Spacer(1, 1 * mm), Paragraph(f"<b>{escape(bend_id)}</b> - experimental {escape(status)}", self.small)]
                    else:
                        cell = [Paragraph(f"<b>{escape(bend_id)}</b> - experimental {escape(status)}", self.small)]
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
                    story.append(Paragraph("Experimental flagged-bend views - not defect dispositions", self.h2))

        doc.build(story, onFirstPage=self._research_footer, onLaterPages=self._research_footer)
        return out_path

    @staticmethod
    def _research_footer(canvas, _doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#BE123C"))
        canvas.drawString(16 * mm, 7 * mm, RESEARCH_WATERMARK + " - DISPOSITION NOT AUTHORIZED")
        canvas.restoreState()

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

    def _bend_table(self, matches: Iterable[Dict[str, Any]], content_width: float, decision: Optional[Dict[str, Any]] = None):
        observations = {row["bend_id"]: row["state"] for row in (decision or {}).get("bends", [])}
        rows: List[List[str]] = [[
            "Bend", "Observation", "Target/Meas Angle", "dAngle", "dRadius", "dCenter", "Experimental\ncomparison",
        ]]
        for match in matches:
            if not isinstance(match, dict):
                continue
            exp = match.get("target_angle")
            act = match.get("measured_angle")
            cad = match.get("cad_bend")
            cad = cad if isinstance(cad, dict) else {}
            bend_id = str(match.get("bend_id", cad.get("bend_id", "-")))
            rows.append([
                bend_id,
                observations.get(bend_id, "Unavailable"),
                f"{exp:.1f} / {act:.1f}" if isinstance(exp, (int, float)) and isinstance(act, (int, float)) else f"{exp if exp is not None else '-'} / -",
                self._fmt_num(match.get("angle_deviation"), "°", 1),
                self._fmt_num(match.get("radius_deviation"), " mm", 2),
                self._fmt_num(match.get("line_center_deviation_mm"), " mm", 2),
                str(match.get("status", "-")),
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
            style.append(("BACKGROUND", (1, r), (1, r), colors.HexColor("#E2E8F0")))
            style.append(("TEXTCOLOR", (1, r), (1, r), colors.HexColor("#334155")))
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
    # Availability is a generation receipt, not native visual qualification.
    # Legacy reports without V1 stay unavailable rather than gaining inferred
    # observations as a side effect of exporting them.
    if _export_decision_evidence(report_dict)["kind"] == "BEND_DECISION_CONTRACT_V1":
        report_dict = finalize_bend_report(
            report_dict, render_artifacts_available=bool(render_info.get("overview_path")),
        )
    manifest["decision_contract"] = _export_decision_evidence(report_dict)
    manifest["qualification_context"] = _export_qualification(report_dict)
    _write_json(out_dir / "bend_overlay_manifest.json", manifest)
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
        # Bender's View orthographic snapshots
        "bender_view_front_url": f"/api/bend-inspection/{{job_id}}/overlay/bender_view_front.png",
        "bender_view_profile_url": f"/api/bend-inspection/{{job_id}}/overlay/bender_view_profile.png",
        "bender_view_top_url": f"/api/bend-inspection/{{job_id}}/overlay/bender_view_top.png",
        "bender_view_overall_url": f"/api/bend-inspection/{{job_id}}/overlay/bender_view_overall.png",
    }
    return BendArtifactBundle(manifest=manifest, artifacts=artifacts, pdf_path=str(pdf_path))
