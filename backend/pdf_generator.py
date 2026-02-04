"""
Professional PDF Report Generator for Sherman QC System.

Generates industrial-grade PDF reports from QCReport data using ReportLab.
Design follows the reference QC_Analysis_Report format with:
- Dark header band with white text
- Color-coded verdict block
- KPI cards in 2x2 grid
- Regional Analysis table with status badges
- Identified Issues table with severity shading
- Root Cause Analysis blocks
- Recommended Actions table with priority shading
- Professional footer with page numbers
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Table, TableStyle, KeepTogether, Image, PageBreak,
    NextPageTemplate
)


# -----------------------------
# Text sanitization
# -----------------------------
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE
)

_EXTRA_STRIP = str.maketrans({
    "\u2713": "",  # checkmark
    "\u2714": "",  # heavy checkmark
    "\u2715": "",  # multiplication x
    "\u2716": "",  # heavy multiplication x
    "\u2717": "",  # ballot x
    "\u2718": "",  # heavy ballot x
    "\u26a0": "",  # warning sign
    "\u274c": "",  # cross mark
    "\u274e": "",  # cross mark button
    "\u2705": "",  # white heavy check mark
    "\u25bc": "",  # down triangle
    "\u25b2": "",  # up triangle
    "\u25b6": "",  # right triangle
    "\u25c0": "",  # left triangle
})


def sanitize_text(s: str) -> str:
    """Remove emojis and decorative symbols from text."""
    if not s:
        return ""
    s = _EMOJI_PATTERN.sub("", s)
    s = s.translate(_EXTRA_STRIP)
    s = s.replace("\u200d", "")
    s = s.replace("\ufe0f", "")
    s = " ".join(s.split())
    return s.strip()


# -----------------------------
# Color scheme (industrial palette)
# -----------------------------
class Colors:
    HEADER_BG = colors.HexColor("#0B1320")
    TEXT_DARK = colors.HexColor("#0B1320")
    TEXT_GRAY = colors.HexColor("#64748B")
    TEXT_LIGHT_GRAY = colors.HexColor("#94A3B8")
    BORDER = colors.HexColor("#CBD5E1")
    BORDER_LIGHT = colors.HexColor("#E2E8F0")
    BG_LIGHT = colors.HexColor("#F8FAFC")
    BG_HEADER = colors.HexColor("#F1F5F9")

    FAIL_BG = colors.HexColor("#FDE2E2")
    FAIL_TEXT = colors.HexColor("#B91C1C")
    FAIL_DARK = colors.HexColor("#991B1B")

    PASS_BG = colors.HexColor("#DCFCE7")
    PASS_TEXT = colors.HexColor("#166534")

    WARNING_BG = colors.HexColor("#FEF3C7")
    WARNING_TEXT = colors.HexColor("#92400E")

    MEDIUM_BG = colors.HexColor("#FFEDD5")
    MEDIUM_TEXT = colors.HexColor("#9A3412")


class PDFReportGenerator:
    """Generate professional PDF reports matching reference design."""

    def __init__(self):
        self.page_w, self.page_h = A4
        self.margin = 18 * mm
        self._setup_styles()

    def _setup_styles(self):
        """Setup paragraph styles."""
        base = getSampleStyleSheet()

        self.H1 = ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontName="Helvetica-Bold", fontSize=22, leading=26,
            textColor=colors.white, spaceAfter=6
        )
        self.SUB = ParagraphStyle(
            "SUB", parent=base["Normal"],
            fontName="Helvetica", fontSize=11, leading=14,
            textColor=colors.white
        )
        self.H2 = ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontName="Helvetica-Bold", fontSize=16, leading=20,
            textColor=Colors.TEXT_DARK, spaceBefore=14, spaceAfter=8
        )
        self.BODY = ParagraphStyle(
            "BODY", parent=base["Normal"],
            fontName="Helvetica", fontSize=10.5, leading=14,
            textColor=Colors.TEXT_DARK
        )
        self.SMALL = ParagraphStyle(
            "SMALL", parent=base["Normal"],
            fontName="Helvetica", fontSize=9.5, leading=12,
            textColor=Colors.TEXT_GRAY
        )
        self.base_styles = base

    def generate(self, report_data: Dict[str, Any], output_path: str) -> str:
        """Generate PDF report from QCReport data dict."""
        # Extract data
        part_id = sanitize_text(report_data.get("part_id", "Unknown"))
        part_name = sanitize_text(report_data.get("part_name", "Unknown Part"))
        material = sanitize_text(report_data.get("material", "N/A"))
        tolerance = report_data.get("tolerance", 0.5)
        timestamp = report_data.get("timestamp", datetime.now().isoformat())

        verdict = report_data.get("overall_result", "FAIL").upper()
        quality_score = report_data.get("quality_score", 0)
        confidence = report_data.get("confidence", 0)

        stats = report_data.get("statistics", {})
        total_points = stats.get("total_points", 0)
        pass_rate = stats.get("pass_rate", 0)
        max_dev = stats.get("max_deviation_mm", 0)
        mean_dev = abs(stats.get("mean_deviation_mm", 0))

        regions = report_data.get("regions", [])
        root_causes = report_data.get("root_causes", [])
        recommendations = report_data.get("recommendations", [])
        ai_summary = sanitize_text(report_data.get("ai_summary", ""))

        # 3D snapshots
        snapshots_3d = report_data.get("snapshots_3d", {})

        # Drawing analysis data
        enhanced_analysis = report_data.get("enhanced_analysis") or {}
        drawing_extraction = report_data.get("drawing_extraction") or enhanced_analysis.get("drawing_extraction")
        drawing_context = sanitize_text(report_data.get("drawing_context", ""))

        # Store for page callbacks
        self._part_name = part_name
        self._part_id = part_id

        # Create document
        doc = BaseDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin,
            title=f"QC Report - {part_name}",
            author="Sherman Scan QC System",
        )

        # Setup page templates
        frame_first = Frame(self.margin, self.margin,
                            self.page_w - 2*self.margin, self.page_h - 2*self.margin, id="F1")
        frame_later = Frame(self.margin, self.margin,
                            self.page_w - 2*self.margin, self.page_h - 2*self.margin, id="F2")

        doc.addPageTemplates([
            PageTemplate(id="First", frames=[frame_first], onPage=self._first_page),
            PageTemplate(id="Later", frames=[frame_later], onPage=self._later_pages),
        ])

        # Build flowables
        flow: List[Any] = []
        content_width = self.page_w - 2*self.margin

        # === HEADER SECTION ===
        flow.append(Spacer(1, 8*mm))
        flow.append(Paragraph("Quality Control Analysis Report", self.H1))
        flow.append(Paragraph("AI-Powered Dimensional Inspection System", self.SUB))
        flow.append(Spacer(1, 10*mm))

        # Part info row - use two-row table for labels and values
        part_info = [
            ("Part Number", part_id),
            ("Part Name", part_name),
            ("Material", material),
            ("Tolerance", f"+/-{tolerance}mm"),
        ]

        # Create two rows: labels and values
        labels_row = [k for k, v in part_info]
        values_row = [v for k, v in part_info]

        info_tbl = Table([labels_row, values_row], colWidths=[content_width/len(part_info)] * len(part_info))
        info_tbl.setStyle(TableStyle([
            # Labels row - small gray text
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#94A3B8")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,0), 9),
            # Values row - white bold text
            ("TEXTCOLOR", (0,1), (-1,1), colors.white),
            ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
            ("FONTSIZE", (0,1), (-1,1), 11),
            # General
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,0), 8),
            ("BOTTOMPADDING", (0,0), (-1,0), 2),
            ("TOPPADDING", (0,1), (-1,1), 2),
            ("BOTTOMPADDING", (0,1), (-1,1), 8),
        ]))
        flow.append(info_tbl)
        flow.append(Spacer(1, 10*mm))

        # === VERDICT BLOCK ===
        verdict_color = Colors.FAIL_TEXT if verdict == "FAIL" else (
            Colors.WARNING_TEXT if verdict == "WARNING" else Colors.PASS_TEXT
        )
        verdict_bg = Colors.FAIL_BG if verdict == "FAIL" else (
            Colors.WARNING_BG if verdict == "WARNING" else Colors.PASS_BG
        )

        summary_text = ai_summary if ai_summary else (
            f"Part {'FAILS' if verdict == 'FAIL' else 'PASSES'} quality inspection. "
            f"{pass_rate:.1f}% of points meet the +/-{tolerance}mm tolerance. "
            f"Maximum deviation of {max_dev:.3f}mm {'exceeds acceptable limits.' if verdict == 'FAIL' else 'within limits.'}"
        )

        verdict_style = ParagraphStyle(
            "VERDICT", parent=self.base_styles["Heading1"],
            fontName="Helvetica-Bold", fontSize=28, textColor=verdict_color
        )

        # Style for inspection label
        label_style = ParagraphStyle(
            "LABEL", parent=self.base_styles["Normal"],
            fontName="Helvetica", fontSize=9, leading=11,
            textColor=Colors.TEXT_GRAY
        )

        verdict_tbl = Table([
            [
                Paragraph("INSPECTION<br/>VERDICT", label_style),
                Paragraph(verdict, verdict_style),
                Paragraph(sanitize_text(summary_text), self.BODY)
            ]
        ], colWidths=[35*mm, 35*mm, content_width - 70*mm])

        verdict_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (1,0), verdict_bg),
            ("BOX", (0,0), (-1,-1), 1, Colors.BORDER),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ]))
        flow.append(verdict_tbl)
        flow.append(Spacer(1, 4*mm))
        flow.append(Paragraph(f"Quality Score: {quality_score:.1f}/100 | Confidence: {confidence:.0f}%", self.SMALL))
        flow.append(Spacer(1, 10*mm))

        # === KPI CARDS ===
        pass_note = f"<font color='#B91C1C'>Below 99% threshold</font>" if pass_rate < 99 else ""
        max_dev_note = f"<font color='#B91C1C'>{max_dev/tolerance:.1f}x tolerance</font>" if max_dev > tolerance else ""

        kpi_data = [
            {"value": f"{total_points:,}", "label": "Points Analyzed", "note": ""},
            {"value": f"{pass_rate:.1f}%", "label": "Pass Rate", "note": pass_note},
            {"value": f"{max_dev:.3f}mm", "label": "Max Deviation", "note": max_dev_note},
            {"value": f"{mean_dev:.3f}mm", "label": "Mean |Deviation|", "note": ""},
        ]

        kpi_cells = []
        for kpi in kpi_data:
            note_html = f"<br/>{kpi['note']}" if kpi.get('note') else ""
            kpi_cells.append(Paragraph(
                f"<font size='18'><b>{kpi['value']}</b></font><br/>"
                f"<font size='10' color='#64748B'>{kpi['label']}</font>{note_html}",
                self.BODY
            ))

        kpi_tbl = Table([kpi_cells[:2], kpi_cells[2:4]], colWidths=[content_width/2]*2)
        kpi_tbl.setStyle(TableStyle([
            ("BOX", (0,0), (-1,-1), 1, Colors.BORDER),
            ("INNERGRID", (0,0), (-1,-1), 1, Colors.BORDER_LIGHT),
            ("BACKGROUND", (0,0), (-1,-1), Colors.BG_LIGHT),
            ("LEFTPADDING", (0,0), (-1,-1), 14),
            ("RIGHTPADDING", (0,0), (-1,-1), 14),
            ("TOPPADDING", (0,0), (-1,-1), 14),
            ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ]))
        flow.append(kpi_tbl)

        # Switch to "Later" page template for subsequent pages (no dark header)
        flow.append(NextPageTemplate("Later"))

        # === DIMENSION ANALYSIS (Specification-Based - Non-AI) ===
        dimension_analysis = report_data.get("dimension_analysis")
        if dimension_analysis:
            flow.append(PageBreak())

            # Section header with badge
            dim_header_tbl = Table([
                [
                    Paragraph("Dimension Analysis", self.H2),
                    Paragraph(
                        "<font color='#059669' size='9'><b>SPECIFICATION-BASED</b></font>",
                        ParagraphStyle("BADGE", parent=self.base_styles["Normal"], alignment=2)  # Right align
                    )
                ]
            ], colWidths=[content_width * 0.7, content_width * 0.3])
            dim_header_tbl.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            flow.append(dim_header_tbl)

            flow.append(Paragraph(
                "Comparison of measured dimensions against specifications from the dimension sheet (XLSX). "
                "This analysis uses deterministic tolerance checking without AI interpretation.",
                self.SMALL
            ))
            flow.append(Spacer(1, 4*mm))

            # Dimension Summary KPI Cards
            dim_summary = dimension_analysis.get("summary", {})
            bend_summary = dimension_analysis.get("bend_summary", {})

            total_dims = dim_summary.get("total_dimensions", 0)
            measured = dim_summary.get("measured", 0)
            passed = dim_summary.get("passed", 0)
            failed = dim_summary.get("failed", 0)
            pass_rate = dim_summary.get("pass_rate", 0)

            # Summary cards row
            dim_kpi_data = [
                {"value": str(total_dims), "label": "Total Dimensions"},
                {"value": str(measured), "label": "Measured"},
                {"value": str(passed), "label": "Passed", "color": "#059669"},
                {"value": str(failed), "label": "Failed", "color": "#DC2626" if failed > 0 else "#059669"},
            ]

            dim_kpi_cells = []
            for kpi in dim_kpi_data:
                color = kpi.get("color", "#0B1320")
                dim_kpi_cells.append(Paragraph(
                    f"<font size='16' color='{color}'><b>{kpi['value']}</b></font><br/>"
                    f"<font size='9' color='#64748B'>{kpi['label']}</font>",
                    self.BODY
                ))

            dim_kpi_tbl = Table([dim_kpi_cells], colWidths=[content_width/4]*4)
            dim_kpi_tbl.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 1, Colors.BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 1, Colors.BORDER_LIGHT),
                ("BACKGROUND", (0, 0), (-1, -1), Colors.BG_LIGHT),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            flow.append(dim_kpi_tbl)
            flow.append(Spacer(1, 3*mm))

            # Pass Rate Bar (visual indicator)
            pass_color = "#059669" if pass_rate >= 90 else ("#F59E0B" if pass_rate >= 70 else "#DC2626")
            flow.append(Paragraph(
                f"<font size='10'>Pass Rate: <b><font color='{pass_color}'>{pass_rate:.1f}%</font></b></font>",
                self.BODY
            ))
            flow.append(Spacer(1, 5*mm))

            # Dimension Comparison Table
            dimensions = dimension_analysis.get("dimensions", [])
            if dimensions:
                flow.append(Paragraph("Dimension Comparison", ParagraphStyle(
                    "H3", parent=self.base_styles["Heading3"],
                    fontName="Helvetica-Bold", fontSize=12, leading=14,
                    textColor=Colors.TEXT_DARK, spaceBefore=4, spaceAfter=4
                )))

                dim_comp_header = ["Dim #", "Type", "Expected (Ref)", "Measured", "Deviation", "Status"]
                dim_comp_data = [dim_comp_header]

                for dim in dimensions:
                    dim_id = str(dim.get("dim_id", "-"))
                    dim_type = dim.get("type", "-").capitalize()
                    expected = dim.get("expected", 0)
                    ref_source = dim.get("reference_source", "XLSX")
                    unit = "°" if dim_type.lower() == "angle" else "mm"
                    scan_value = dim.get("scan_value")
                    scan_deviation = dim.get("scan_deviation")
                    status = dim.get("status", "pending").upper()

                    # Show reference source (CAD or XLSX) alongside expected value
                    expected_str = f"{expected:.2f}{unit} ({ref_source})"
                    measured_str = f"{scan_value:.2f}{unit}" if scan_value is not None else "N/A"
                    deviation_str = f"{scan_deviation:+.2f}{unit}" if scan_deviation is not None else "-"

                    dim_comp_data.append([dim_id, dim_type, expected_str, measured_str, deviation_str, status])

                dim_comp_col_widths = [20*mm, 30*mm, 35*mm, 35*mm, 35*mm, 25*mm]
                dim_comp_tbl = Table(dim_comp_data, repeatRows=1, colWidths=dim_comp_col_widths)

                dim_comp_style = [
                    ("BACKGROUND", (0, 0), (-1, 0), Colors.BG_HEADER),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("TEXTCOLOR", (0, 0), (-1, 0), Colors.TEXT_GRAY),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("GRID", (0, 0), (-1, -1), 0.75, Colors.BORDER_LIGHT),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]

                for r in range(1, len(dim_comp_data)):
                    status = dim_comp_data[r][5]
                    if status == "FAIL":
                        dim_comp_style.append(("BACKGROUND", (5, r), (5, r), Colors.FAIL_BG))
                        dim_comp_style.append(("TEXTCOLOR", (5, r), (5, r), Colors.FAIL_DARK))
                        dim_comp_style.append(("FONTNAME", (5, r), (5, r), "Helvetica-Bold"))
                    elif status == "WARNING":
                        dim_comp_style.append(("BACKGROUND", (5, r), (5, r), Colors.WARNING_BG))
                        dim_comp_style.append(("TEXTCOLOR", (5, r), (5, r), Colors.WARNING_TEXT))
                        dim_comp_style.append(("FONTNAME", (5, r), (5, r), "Helvetica-Bold"))
                    elif status == "PASS":
                        dim_comp_style.append(("BACKGROUND", (5, r), (5, r), Colors.PASS_BG))
                        dim_comp_style.append(("TEXTCOLOR", (5, r), (5, r), Colors.PASS_TEXT))
                        dim_comp_style.append(("FONTNAME", (5, r), (5, r), "Helvetica-Bold"))
                    else:
                        dim_comp_style.append(("BACKGROUND", (5, r), (5, r), Colors.BG_LIGHT))
                        dim_comp_style.append(("TEXTCOLOR", (5, r), (5, r), Colors.TEXT_GRAY))

                dim_comp_tbl.setStyle(TableStyle(dim_comp_style))
                flow.append(dim_comp_tbl)
                flow.append(Spacer(1, 5*mm))

            # Bend-by-Bend Analysis Table
            bend_analysis = dimension_analysis.get("bend_analysis")
            if bend_analysis and bend_analysis.get("matches"):
                total_bends = bend_summary.get("total_bends", 0)
                bends_passed = bend_summary.get("passed", 0)

                flow.append(Paragraph(
                    f"Bend-by-Bend Analysis <font size='9' color='#64748B'>({bends_passed} of {total_bends} passed)</font>",
                    ParagraphStyle(
                        "H3", parent=self.base_styles["Heading3"],
                        fontName="Helvetica-Bold", fontSize=12, leading=14,
                        textColor=Colors.TEXT_DARK, spaceBefore=4, spaceAfter=4
                    )
                ))

                bend_match_header = ["Dim #", "Expected (CAD)", "XLSX Spec", "Measured", "Deviation", "Status"]
                bend_match_data = [bend_match_header]

                for match in bend_analysis.get("matches", []):
                    dim_id = str(match.get("dim_id", "-"))
                    cad_info = match.get("cad", {})
                    scan_info = match.get("scan", {})
                    cad_angle = cad_info.get("angle") if cad_info else None
                    xlsx_expected = match.get("expected", 0)
                    measured = scan_info.get("angle")
                    deviation = scan_info.get("deviation")
                    status = match.get("status", "pending").upper()

                    # CAD is the reference; show XLSX separately as spec guide
                    cad_str = f"{cad_angle:.1f}°" if cad_angle is not None else "N/A"
                    xlsx_str = f"{xlsx_expected:.1f}°"
                    measured_str = f"{measured:.1f}°" if measured is not None else "N/A"
                    deviation_str = f"{deviation:+.2f}°" if deviation is not None else "-"

                    bend_match_data.append([dim_id, cad_str, xlsx_str, measured_str, deviation_str, status])

                bend_match_col_widths = [20*mm, 35*mm, 30*mm, 35*mm, 30*mm, 30*mm]
                bend_match_tbl = Table(bend_match_data, repeatRows=1, colWidths=bend_match_col_widths)

                bend_match_style = [
                    ("BACKGROUND", (0, 0), (-1, 0), Colors.BG_HEADER),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("TEXTCOLOR", (0, 0), (-1, 0), Colors.TEXT_GRAY),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("GRID", (0, 0), (-1, -1), 0.75, Colors.BORDER_LIGHT),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]

                for r in range(1, len(bend_match_data)):
                    status = bend_match_data[r][4]
                    if status == "FAIL":
                        bend_match_style.append(("BACKGROUND", (4, r), (4, r), Colors.FAIL_BG))
                        bend_match_style.append(("TEXTCOLOR", (4, r), (4, r), Colors.FAIL_DARK))
                        bend_match_style.append(("FONTNAME", (4, r), (4, r), "Helvetica-Bold"))
                    elif status == "WARNING":
                        bend_match_style.append(("BACKGROUND", (4, r), (4, r), Colors.WARNING_BG))
                        bend_match_style.append(("TEXTCOLOR", (4, r), (4, r), Colors.WARNING_TEXT))
                        bend_match_style.append(("FONTNAME", (4, r), (4, r), "Helvetica-Bold"))
                    elif status == "PASS":
                        bend_match_style.append(("BACKGROUND", (4, r), (4, r), Colors.PASS_BG))
                        bend_match_style.append(("TEXTCOLOR", (4, r), (4, r), Colors.PASS_TEXT))
                        bend_match_style.append(("FONTNAME", (4, r), (4, r), "Helvetica-Bold"))
                    else:
                        bend_match_style.append(("BACKGROUND", (4, r), (4, r), Colors.BG_LIGHT))
                        bend_match_style.append(("TEXTCOLOR", (4, r), (4, r), Colors.TEXT_GRAY))

                bend_match_tbl.setStyle(TableStyle(bend_match_style))
                flow.append(bend_match_tbl)
                flow.append(Spacer(1, 5*mm))

            # Failed Dimensions Detail
            failed_dimensions = dimension_analysis.get("failed_dimensions", [])
            if failed_dimensions:
                flow.append(Paragraph("Failed Dimensions Detail", ParagraphStyle(
                    "H3", parent=self.base_styles["Heading3"],
                    fontName="Helvetica-Bold", fontSize=12, leading=14,
                    textColor=Colors.FAIL_DARK, spaceBefore=4, spaceAfter=4
                )))

                for dim in failed_dimensions:
                    dim_id = dim.get("dim_id", "-")
                    dim_type = dim.get("type", "-").capitalize()
                    expected = dim.get("expected", 0)
                    ref_source = dim.get("reference_source", "XLSX")
                    cad_value = dim.get("cad_value")
                    xlsx_value = dim.get("xlsx_value")
                    scan_value = dim.get("scan_value")
                    scan_deviation = dim.get("scan_deviation")
                    unit = "°" if dim_type.lower() == "angle" else "mm"

                    expected_str = f"{expected:.2f}{unit}"
                    measured_str = f"{scan_value:.2f}{unit}" if scan_value is not None else "N/A"
                    deviation_str = f"{scan_deviation:+.2f}{unit}" if scan_deviation is not None else "N/A"

                    # Show reference source (CAD or XLSX)
                    ref_label = f"Reference ({ref_source})"
                    fail_content = (
                        f"<b>Dimension {dim_id}</b> ({dim_type})<br/>"
                        f"<font color='#64748B'>{ref_label}:</font> {expected_str}"
                    )
                    # Show XLSX spec alongside if CAD is reference and XLSX is available
                    if ref_source == "CAD" and xlsx_value is not None:
                        fail_content += f"  |  <font color='#94A3B8'>XLSX Spec: {xlsx_value:.2f}{unit}</font>"
                    fail_content += (
                        f"  |  <font color='#64748B'>Measured:</font> {measured_str}  |  "
                        f"<font color='#DC2626'>Deviation: {deviation_str}</font>"
                    )

                    fail_box = Table([[Paragraph(fail_content, self.BODY)]], colWidths=[content_width])
                    fail_box.setStyle(TableStyle([
                        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FCA5A5")),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]))
                    flow.append(fail_box)
                    flow.append(Spacer(1, 3*mm))

            flow.append(Spacer(1, 6*mm))

        # === 3D MODEL VISUALIZATION ===
        snapshot_path = snapshots_3d.get("multi_view", "") or snapshots_3d.get("iso", "")
        if snapshot_path and Path(snapshot_path).exists():
            flow.append(PageBreak())
            flow.append(Paragraph("3D Model Analysis", self.H2))
            flow.append(Paragraph(
                "Deviation heatmap overlay on 3D model. Green indicates points within tolerance, "
                "yellow shows borderline areas, and red highlights out-of-tolerance regions.",
                self.SMALL
            ))
            flow.append(Spacer(1, 5*mm))

            try:
                # Add the 3D snapshot image
                img = Image(snapshot_path, width=content_width, height=content_width * 0.75)
                img.hAlign = 'CENTER'
                flow.append(img)
            except Exception as e:
                flow.append(Paragraph(f"[3D visualization could not be loaded]", self.SMALL))

            flow.append(Spacer(1, 5*mm))

        # === AI DRAWING ANALYSIS ===
        if drawing_extraction or drawing_context:
            flow.append(Paragraph("AI Drawing Analysis", self.H2))
            flow.append(Paragraph(
                "Analysis of technical drawing specifications extracted using AI vision. "
                "Dimensions and tolerances are compared against measured point cloud data.",
                self.SMALL
            ))
            flow.append(Spacer(1, 4*mm))

            # AI Analysis text from drawing
            if drawing_context:
                context_box = Table([[Paragraph(drawing_context, self.BODY)]], colWidths=[content_width])
                context_box.setStyle(TableStyle([
                    ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#A855F7")),
                    ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FAF5FF")),
                    ("LEFTPADDING", (0,0), (-1,-1), 12),
                    ("RIGHTPADDING", (0,0), (-1,-1), 12),
                    ("TOPPADDING", (0,0), (-1,-1), 10),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
                ]))
                flow.append(context_box)
                flow.append(Spacer(1, 5*mm))

            if drawing_extraction:
                # Extracted Dimensions Table
                dimensions = drawing_extraction.get("dimensions", [])
                if dimensions:
                    flow.append(Paragraph("Extracted Dimensions from Drawing", ParagraphStyle(
                        "H3", parent=self.base_styles["Heading3"],
                        fontName="Helvetica-Bold", fontSize=12, leading=14,
                        textColor=Colors.TEXT_DARK, spaceBefore=4, spaceAfter=4
                    )))

                    dim_header = ["ID", "Type", "Nominal", "Tolerance (+/-)", "Location"]
                    dim_data = [dim_header]

                    for dim in dimensions:
                        dim_id = dim.get("id", "-")
                        dim_type = dim.get("dimension_type", dim.get("type", "-")).capitalize()
                        nominal = dim.get("nominal", 0)
                        unit = dim.get("unit", "mm")
                        tol_plus = dim.get("tolerance_plus", 0)
                        tol_minus = dim.get("tolerance_minus", tol_plus)
                        location = dim.get("location", "-")

                        dim_data.append([
                            dim_id,
                            dim_type,
                            f"{nominal}{unit}",
                            f"+{tol_plus} / -{tol_minus}{unit}",
                            location if location else "-"
                        ])

                    dim_col_widths = [25*mm, 35*mm, 35*mm, 45*mm, content_width-140*mm]
                    dim_tbl = Table(dim_data, repeatRows=1, colWidths=dim_col_widths)
                    dim_tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0,0), (-1,0), Colors.BG_HEADER),
                        ("BACKGROUND", (0,1), (-1,-1), colors.white),
                        ("TEXTCOLOR", (0,0), (-1,0), Colors.TEXT_GRAY),
                        ("FONTNAME", (0,0), (-1,0), "Helvetica"),
                        ("FONTSIZE", (0,0), (-1,-1), 9.5),
                        ("GRID", (0,0), (-1,-1), 0.75, Colors.BORDER_LIGHT),
                        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                        ("LEFTPADDING", (0,0), (-1,-1), 6),
                        ("RIGHTPADDING", (0,0), (-1,-1), 6),
                        ("TOPPADDING", (0,0), (-1,-1), 6),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                    ]))
                    flow.append(dim_tbl)
                    flow.append(Spacer(1, 5*mm))

                # Extracted Bends Table
                bends = drawing_extraction.get("bends", [])
                if bends:
                    flow.append(Paragraph("Bend Specifications from Drawing", ParagraphStyle(
                        "H3", parent=self.base_styles["Heading3"],
                        fontName="Helvetica-Bold", fontSize=12, leading=14,
                        textColor=Colors.TEXT_DARK, spaceBefore=4, spaceAfter=4
                    )))

                    bend_header = ["Bend", "Sequence", "Angle", "Tolerance", "Radius", "Direction"]
                    bend_data = [bend_header]

                    for bend in bends:
                        bend_id = bend.get("id", "-")
                        seq = bend.get("sequence", "-")
                        angle = f"{bend.get('angle_nominal', 0)}°"
                        tol = f"±{bend.get('angle_tolerance', 0)}°" if bend.get("angle_tolerance") else "-"
                        radius = f"{bend.get('radius', 0)}mm" if bend.get("radius") else "-"
                        direction = bend.get("direction", "-")

                        bend_data.append([bend_id, str(seq), angle, tol, radius, direction if direction else "-"])

                    bend_col_widths = [25*mm, 25*mm, 30*mm, 30*mm, 30*mm, content_width-140*mm]
                    bend_tbl = Table(bend_data, repeatRows=1, colWidths=bend_col_widths)
                    bend_tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0,0), (-1,0), Colors.BG_HEADER),
                        ("BACKGROUND", (0,1), (-1,-1), colors.white),
                        ("TEXTCOLOR", (0,0), (-1,0), Colors.TEXT_GRAY),
                        ("FONTNAME", (0,0), (-1,0), "Helvetica"),
                        ("FONTSIZE", (0,0), (-1,-1), 9.5),
                        ("GRID", (0,0), (-1,-1), 0.75, Colors.BORDER_LIGHT),
                        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                        ("LEFTPADDING", (0,0), (-1,-1), 6),
                        ("RIGHTPADDING", (0,0), (-1,-1), 6),
                        ("TOPPADDING", (0,0), (-1,-1), 6),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                    ]))
                    flow.append(bend_tbl)
                    flow.append(Spacer(1, 5*mm))

                # Material & Thickness from Drawing
                material_from_drawing = drawing_extraction.get("material")
                thickness_from_drawing = drawing_extraction.get("thickness")
                if material_from_drawing or thickness_from_drawing:
                    flow.append(Paragraph("Material Specifications from Drawing", ParagraphStyle(
                        "H3", parent=self.base_styles["Heading3"],
                        fontName="Helvetica-Bold", fontSize=12, leading=14,
                        textColor=Colors.TEXT_DARK, spaceBefore=4, spaceAfter=4
                    )))

                    mat_items = []
                    if material_from_drawing:
                        mat_items.append(f"<b>Material:</b> {sanitize_text(material_from_drawing)}")
                    if thickness_from_drawing:
                        mat_items.append(f"<b>Thickness:</b> {thickness_from_drawing}mm")

                    mat_box = Table([[Paragraph("  |  ".join(mat_items), self.BODY)]], colWidths=[content_width])
                    mat_box.setStyle(TableStyle([
                        ("BOX", (0,0), (-1,-1), 1, Colors.BORDER),
                        ("BACKGROUND", (0,0), (-1,-1), Colors.BG_LIGHT),
                        ("LEFTPADDING", (0,0), (-1,-1), 12),
                        ("RIGHTPADDING", (0,0), (-1,-1), 12),
                        ("TOPPADDING", (0,0), (-1,-1), 8),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                    ]))
                    flow.append(mat_box)
                    flow.append(Spacer(1, 5*mm))

                # Extraction confidence
                extraction_confidence = drawing_extraction.get("extraction_confidence")
                if extraction_confidence:
                    flow.append(Paragraph(
                        f"Drawing extraction confidence: {extraction_confidence*100:.0f}%",
                        self.SMALL
                    ))

            flow.append(Spacer(1, 6*mm))

        # === REGIONAL ANALYSIS ===
        flow.append(Paragraph("Regional Analysis", self.H2))

        reg_header = ["Region", "Points", "Mean Dev.", "Max Dev.", "Pass Rate", "Status"]
        reg_data = [reg_header]

        for region in regions:
            status = region.get("status", "OK")
            reg_data.append([
                region.get("name", "Unknown"),
                f"{region.get('point_count', 0):,}",
                f"{region.get('mean_deviation_mm', 0):.4f}mm",
                f"{region.get('max_deviation_mm', 0):.4f}mm",
                f"{region.get('pass_rate', 0):.1f}%",
                status
            ])

        col_widths = [50*mm, 25*mm, 28*mm, 28*mm, 25*mm, 25*mm]
        reg_tbl = Table(reg_data, repeatRows=1, colWidths=col_widths)

        reg_style = [
            ("BACKGROUND", (0,0), (-1,0), Colors.BG_HEADER),
            ("TEXTCOLOR", (0,0), (-1,0), Colors.TEXT_GRAY),
            ("FONTNAME", (0,0), (-1,0), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,0), 9.5),
            ("GRID", (0,0), (-1,-1), 0.75, Colors.BORDER_LIGHT),
            ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,1), (-1,-1), 10),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]

        for r in range(1, len(reg_data)):
            status = reg_data[r][5]
            if status == "ATTENTION":
                reg_style.append(("BACKGROUND", (5,r), (5,r), Colors.FAIL_BG))
                reg_style.append(("TEXTCOLOR", (5,r), (5,r), Colors.FAIL_DARK))
                reg_style.append(("FONTNAME", (5,r), (5,r), "Helvetica-Bold"))
            else:
                reg_style.append(("BACKGROUND", (5,r), (5,r), Colors.PASS_BG))
                reg_style.append(("TEXTCOLOR", (5,r), (5,r), Colors.PASS_TEXT))
                reg_style.append(("FONTNAME", (5,r), (5,r), "Helvetica-Bold"))

        reg_tbl.setStyle(TableStyle(reg_style))
        flow.append(reg_tbl)

        # === BEND ANALYSIS (Enhanced) ===
        bend_results = report_data.get("bend_results", [])
        if bend_results:
            # Compute summary counts
            total_bends = len(bend_results)
            measured_bends = sum(1 for b in bend_results if b.get("scan_measured", False) or b.get("bend", {}).get("scan_measured", False))
            pass_bends = sum(1 for b in bend_results if (b.get("scan_measured", False) or b.get("bend", {}).get("scan_measured", False)) and abs(b.get("scan_deviation", 0) or b.get("angle_deviation_deg", 0) or 0) <= 3.0)

            flow.append(Paragraph(
                f"Bend Analysis "
                f"<font size='9' color='#64748B'>({measured_bends} measured, {pass_bends} of {measured_bends} within ±3°)</font>",
                self.H2
            ))
            flow.append(Paragraph(
                "Detailed analysis of each bend detected in the part. "
                "Includes angle deviation, measurement method, and confidence level.",
                self.SMALL
            ))
            flow.append(Spacer(1, 3*mm))

            bend_header = ["Bend", "Measured", "Nominal", "Deviation", "Status", "Issue"]
            bend_data = [bend_header]

            for br in bend_results:
                # Support both nested format (bend_info inside "bend" key)
                # and flat format (fields directly on br dict)
                bend_info = br.get("bend", {})
                if bend_info:
                    bend_name = bend_info.get("bend_name", f"Bend {bend_info.get('bend_id', '?')}")
                else:
                    bend_name = br.get("bend_name", f"Bend {br.get('bend_id', '?')}")

                # Scan measurement (flat: scan_angle / scan_measured, nested: angle_degrees)
                scan_measured = br.get("scan_measured", False)
                scan_angle = br.get("scan_angle", 0) or bend_info.get("angle_degrees", 0)
                if scan_measured and scan_angle:
                    measured = f"{scan_angle:.1f} deg"
                elif not scan_measured:
                    measured = "N/M"
                else:
                    measured = f"{scan_angle:.1f} deg"

                # Nominal angle (flat: bend_angle, nested: nominal_angle)
                nominal = br.get("bend_angle") or br.get("nominal_angle")
                nominal_str = f"{nominal:.1f} deg" if nominal else "N/A"

                # Deviation (flat: scan_deviation, nested: angle_deviation_deg)
                deviation = br.get("scan_deviation") or br.get("angle_deviation_deg", 0) or 0
                deviation_str = f"{deviation:+.2f} deg" if deviation != 0 else "0.00 deg"

                # Status: derive from scan_measured + deviation magnitude
                explicit_status = br.get("status")
                if explicit_status:
                    status = explicit_status.upper()
                elif not scan_measured:
                    status = "N/M"
                elif abs(deviation) <= 1.0:
                    status = "PASS"
                elif abs(deviation) <= 3.0:
                    status = "WARNING"
                else:
                    status = "FAIL"

                # Determine primary issue
                springback = br.get("springback_indicator", 0)
                overbend = br.get("over_bend_indicator", 0)
                springback_diagnosis = br.get("springback_diagnosis")

                if not scan_measured:
                    issue = "No scan coverage"
                elif springback and springback > 0.3:
                    if springback_diagnosis:
                        issue = f"Springback: {springback_diagnosis}"
                    else:
                        issue = f"Springback ({springback*100:.0f}%)"
                elif overbend and overbend > 0.3:
                    issue = f"Overbend ({overbend*100:.0f}%)"
                elif deviation > 3.0:
                    issue = "Under-bend"
                elif deviation < -3.0:
                    issue = "Over-bend"
                else:
                    issue = "None"

                # Add method + confidence info
                method = br.get("measurement_method", "")
                conf = br.get("measurement_confidence", "")
                if method and scan_measured:
                    method_short = method.replace("signed_distance_gradient", "gradient").replace("surface_classification", "surface")
                    issue = f"{issue} [{method_short}/{conf}]" if issue != "None" else f"[{method_short}/{conf}]"

                bend_data.append([bend_name, measured, nominal_str, deviation_str, status, issue])

            bend_col_widths = [28*mm, 30*mm, 30*mm, 30*mm, 25*mm, content_width-143*mm]
            bend_tbl = Table(bend_data, repeatRows=1, colWidths=bend_col_widths)

            bend_style = [
                ("BACKGROUND", (0,0), (-1,0), Colors.BG_HEADER),
                ("BACKGROUND", (0,1), (-1,-1), colors.white),
                ("TEXTCOLOR", (0,0), (-1,0), Colors.TEXT_GRAY),
                ("FONTNAME", (0,0), (-1,0), "Helvetica"),
                ("FONTSIZE", (0,0), (-1,-1), 10),
                ("GRID", (0,0), (-1,-1), 0.75, Colors.BORDER_LIGHT),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ]

            for r in range(1, len(bend_data)):
                status = bend_data[r][4]
                if status == "FAIL":
                    bend_style.append(("BACKGROUND", (4,r), (4,r), Colors.FAIL_BG))
                    bend_style.append(("TEXTCOLOR", (4,r), (4,r), Colors.FAIL_DARK))
                    bend_style.append(("FONTNAME", (4,r), (4,r), "Helvetica-Bold"))
                elif status == "WARNING":
                    bend_style.append(("BACKGROUND", (4,r), (4,r), Colors.WARNING_BG))
                    bend_style.append(("TEXTCOLOR", (4,r), (4,r), Colors.WARNING_TEXT))
                    bend_style.append(("FONTNAME", (4,r), (4,r), "Helvetica-Bold"))
                elif status == "N/M":
                    bend_style.append(("BACKGROUND", (4,r), (4,r), colors.HexColor("#F0F0F0")))
                    bend_style.append(("TEXTCOLOR", (4,r), (4,r), colors.HexColor("#888888")))
                    bend_style.append(("FONTNAME", (4,r), (4,r), "Helvetica-Bold"))
                else:
                    bend_style.append(("BACKGROUND", (4,r), (4,r), Colors.PASS_BG))
                    bend_style.append(("TEXTCOLOR", (4,r), (4,r), Colors.PASS_TEXT))
                    bend_style.append(("FONTNAME", (4,r), (4,r), "Helvetica-Bold"))

            bend_tbl.setStyle(TableStyle(bend_style))
            flow.append(bend_tbl)

            # Add material-specific springback reference note if available
            material = report_data.get("material", "")
            if material and bend_results:
                # Check if any bend has springback info
                any_springback = any(br.get("springback_indicator", 0) > 0 for br in bend_results)
                expected_range = bend_results[0].get("expected_springback_range") if bend_results else None
                if any_springback and expected_range:
                    springback_note = f"<i>Note: Expected springback for {material}: {expected_range}</i>"
                    flow.append(Paragraph(springback_note, self.SMALL))

            flow.append(Spacer(1, 6*mm))

        # === 2D-3D CORRELATION (Enhanced) ===
        correlation = report_data.get("correlation_2d_3d")
        if correlation:
            flow.append(Paragraph("Drawing Correlation", self.H2))
            flow.append(Paragraph(
                f"Mapping of technical drawing specifications to measured 3D features. "
                f"Overall correlation score: {correlation.get('overall_correlation_score', 0)*100:.0f}%",
                self.SMALL
            ))
            flow.append(Spacer(1, 3*mm))

            # Critical deviations
            critical_devs = correlation.get("critical_deviations", [])
            if critical_devs:
                flow.append(Paragraph("Critical Deviations", ParagraphStyle(
                    "H3", parent=self.base_styles["Heading3"],
                    fontName="Helvetica-Bold", fontSize=12, leading=14,
                    textColor=Colors.FAIL_DARK, spaceBefore=4, spaceAfter=4
                )))

                for dev in critical_devs[:5]:  # Limit to 5
                    severity = dev.get("severity", "minor").upper()
                    desc = sanitize_text(dev.get("description", ""))
                    deviation_mm = dev.get("deviation_mm", 0)
                    loc = dev.get("location", [0, 0, 0])

                    bg_color = Colors.FAIL_BG if severity == "CRITICAL" else (
                        Colors.WARNING_BG if severity == "MAJOR" else Colors.PASS_BG
                    )

                    dev_content = (
                        f"<b>{severity}</b> | Deviation: {deviation_mm:.3f}mm<br/>"
                        f"{desc}<br/>"
                        f"<font color='#64748B'>Location: ({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f})</font>"
                    )

                    dev_box = Table([[Paragraph(dev_content, self.BODY)]], colWidths=[content_width])
                    dev_box.setStyle(TableStyle([
                        ("BOX", (0,0), (-1,-1), 1, Colors.BORDER),
                        ("BACKGROUND", (0,0), (-1,-1), bg_color),
                        ("LEFTPADDING", (0,0), (-1,-1), 10),
                        ("RIGHTPADDING", (0,0), (-1,-1), 10),
                        ("TOPPADDING", (0,0), (-1,-1), 8),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                    ]))
                    flow.append(dev_box)
                    flow.append(Spacer(1, 3*mm))

                if len(critical_devs) > 5:
                    flow.append(Paragraph(
                        f"+ {len(critical_devs) - 5} additional deviations (see full analysis)",
                        self.SMALL
                    ))

            flow.append(Spacer(1, 6*mm))

        # === IDENTIFIED ISSUES ===
        flow.append(Paragraph("Identified Issues", self.H2))

        # Create paragraph style for text wrapping in table cells
        issue_cell_style = ParagraphStyle(
            "ISSUE_CELL", parent=self.base_styles["Normal"],
            fontName="Helvetica", fontSize=10, leading=12,
            textColor=Colors.TEXT_DARK
        )

        issues_data = [["Severity", "Issue", "Scope"]]
        for rc in root_causes:
            issue_text = sanitize_text(rc.get("issue", "Unknown issue"))
            confidence_val = rc.get("confidence", 50)
            priority = rc.get("priority", "MEDIUM").upper()
            severity = "MAJOR" if priority in ["CRITICAL", "HIGH"] else "MINOR"
            scope = f"Confidence: {confidence_val:.0f}%"
            # Wrap text in Paragraph for proper word wrapping
            issues_data.append([
                severity,
                Paragraph(issue_text, issue_cell_style),
                Paragraph(scope, issue_cell_style)
            ])

        if len(issues_data) > 1:
            issue_tbl = Table(issues_data, repeatRows=1, colWidths=[28*mm, 100*mm, content_width-128*mm])

            issue_style = [
                ("BACKGROUND", (0,0), (-1,0), Colors.BG_HEADER),
                ("BACKGROUND", (0,1), (-1,-1), colors.white),  # White background for all data rows
                ("TEXTCOLOR", (0,0), (-1,0), Colors.TEXT_GRAY),
                ("FONTNAME", (0,0), (-1,0), "Helvetica"),
                ("FONTSIZE", (0,0), (-1,-1), 10),
                ("TEXTCOLOR", (0,1), (-1,-1), Colors.TEXT_DARK),  # Default dark text for all data rows
                ("GRID", (0,0), (-1,-1), 0.75, Colors.BORDER_LIGHT),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 10),
                ("RIGHTPADDING", (0,0), (-1,-1), 10),
                ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ]

            for r in range(1, len(issues_data)):
                sev = issues_data[r][0].upper()
                if sev == "MAJOR":
                    bg, fg = Colors.FAIL_BG, Colors.FAIL_DARK
                else:
                    bg, fg = Colors.PASS_BG, Colors.PASS_TEXT
                # Only color the severity badge column
                issue_style.extend([
                    ("BACKGROUND", (0,r), (0,r), bg),
                    ("TEXTCOLOR", (0,r), (0,r), fg),
                    ("FONTNAME", (0,r), (0,r), "Helvetica-Bold"),
                    ("ALIGN", (0,r), (0,r), "CENTER"),
                ])

            issue_tbl.setStyle(TableStyle(issue_style))
            flow.append(KeepTogether([issue_tbl]))
        else:
            flow.append(Paragraph("No significant issues identified.", self.BODY))

        # === ROOT CAUSE ANALYSIS ===
        flow.append(Paragraph("Root Cause Analysis", self.H2))

        for rc in root_causes:
            issue_text = sanitize_text(rc.get("issue", "Unknown"))
            cause = sanitize_text(rc.get("likely_cause", "Unknown cause"))
            explanation = sanitize_text(rc.get("explanation", ""))
            confidence_val = rc.get("confidence", 50)

            rc_content = (
                f"<b>{issue_text}</b><br/><br/>"
                f"<font color='#64748B'>Likely cause:</font> {cause}<br/><br/>"
                f"<font color='#64748B'>Explanation:</font> {explanation}<br/><br/>"
                f"<font color='#64748B'>Analysis confidence: {confidence_val:.0f}%</font>"
            )

            box = Table([[Paragraph(rc_content, self.BODY)]], colWidths=[content_width])
            box.setStyle(TableStyle([
                ("BOX", (0,0), (-1,-1), 1, Colors.BORDER),
                ("BACKGROUND", (0,0), (-1,-1), Colors.BG_LIGHT),
                ("LEFTPADDING", (0,0), (-1,-1), 12),
                ("RIGHTPADDING", (0,0), (-1,-1), 12),
                ("TOPPADDING", (0,0), (-1,-1), 12),
                ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ]))
            flow.append(KeepTogether([box]))
            flow.append(Spacer(1, 6*mm))

        if not root_causes:
            flow.append(Paragraph("No root causes identified. Part appears to meet specifications.", self.BODY))

        # === RECOMMENDED ACTIONS ===
        flow.append(Paragraph("Recommended Actions", self.H2))

        if recommendations:
            # Create paragraph style for text wrapping in table cells
            cell_style = ParagraphStyle(
                "CELL", parent=self.base_styles["Normal"],
                fontName="Helvetica", fontSize=10, leading=12,
                textColor=Colors.TEXT_DARK
            )

            rec_data = [["Priority", "Action", "Expected Result"]]

            for rec in recommendations:
                priority = rec.get("priority", "MEDIUM").upper()
                action = sanitize_text(rec.get("action", ""))
                expected = sanitize_text(rec.get("expected_improvement", ""))
                # Wrap text in Paragraph for proper word wrapping
                rec_data.append([
                    priority,
                    Paragraph(action, cell_style),
                    Paragraph(expected, cell_style)
                ])

            rec_tbl = Table(rec_data, repeatRows=1, colWidths=[28*mm, 80*mm, content_width-108*mm])

            rec_style = [
                ("BACKGROUND", (0,0), (-1,0), Colors.BG_HEADER),
                ("BACKGROUND", (0,1), (-1,-1), colors.white),  # White background for all data rows
                ("TEXTCOLOR", (0,0), (-1,0), Colors.TEXT_GRAY),
                ("FONTNAME", (0,0), (-1,0), "Helvetica"),
                ("FONTSIZE", (0,0), (-1,-1), 10),
                ("TEXTCOLOR", (0,1), (-1,-1), Colors.TEXT_DARK),  # Default dark text for all data rows
                ("GRID", (0,0), (-1,-1), 0.75, Colors.BORDER_LIGHT),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 10),
                ("RIGHTPADDING", (0,0), (-1,-1), 10),
                ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ]

            for r in range(1, len(rec_data)):
                pr = rec_data[r][0].upper()
                if pr in ["HIGH", "CRITICAL"]:
                    bg, fg = Colors.FAIL_BG, Colors.FAIL_DARK
                elif pr == "MEDIUM":
                    bg, fg = Colors.MEDIUM_BG, Colors.MEDIUM_TEXT
                else:
                    bg, fg = Colors.PASS_BG, Colors.PASS_TEXT
                # Only color the priority badge column
                rec_style.extend([
                    ("BACKGROUND", (0,r), (0,r), bg),
                    ("TEXTCOLOR", (0,r), (0,r), fg),
                    ("FONTNAME", (0,r), (0,r), "Helvetica-Bold"),
                    ("ALIGN", (0,r), (0,r), "CENTER"),
                ])

            rec_tbl.setStyle(TableStyle(rec_style))
            flow.append(rec_tbl)
        else:
            flow.append(Paragraph("No specific actions required. Continue standard quality monitoring.", self.BODY))

        # Build document
        doc.build(flow)
        return output_path

    def _draw_footer(self, canvas, doc_):
        """Draw footer with page number and report date."""
        canvas.saveState()
        canvas.setStrokeColor(Colors.BORDER)
        canvas.setLineWidth(1)
        canvas.line(self.margin, 25*mm, self.page_w - self.margin, 25*mm)
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(Colors.TEXT_GRAY)
        canvas.drawString(self.margin, 16*mm, "Generated by Sherman Scan QC System | AI-Powered Quality Control")
        canvas.drawCentredString(self.page_w / 2, 16*mm, datetime.now().strftime('%Y-%m-%d %H:%M'))
        canvas.drawRightString(self.page_w - self.margin, 16*mm, f"Page {doc_.page}")
        canvas.restoreState()

    def _first_page(self, canvas, doc_):
        """Draw first page with dark header band."""
        canvas.saveState()
        canvas.setFillColor(Colors.HEADER_BG)
        canvas.rect(self.margin, self.page_h - self.margin - 55*mm,
                    self.page_w - 2*self.margin, 55*mm, stroke=0, fill=1)
        canvas.restoreState()
        self._draw_footer(canvas, doc_)

    def _later_pages(self, canvas, doc_):
        """Draw subsequent pages with running header."""
        canvas.saveState()
        canvas.setFillColor(Colors.TEXT_GRAY)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(self.margin, self.page_h - self.margin + 4*mm,
                          f"{self._part_name}  |  Part No. {self._part_id}")
        canvas.setStrokeColor(Colors.BORDER)
        canvas.setLineWidth(1)
        canvas.line(self.margin, self.page_h - self.margin, self.page_w - self.margin, self.page_h - self.margin)
        canvas.restoreState()
        self._draw_footer(canvas, doc_)


def generate_pdf_report(report_data: Dict[str, Any], output_path: str) -> str:
    """
    Generate a professional PDF report from QCReport data.

    Args:
        report_data: Dictionary from QCReport.to_dict()
        output_path: Path to save the PDF file

    Returns:
        Path to the generated PDF file
    """
    generator = PDFReportGenerator()
    return generator.generate(report_data, output_path)


if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        "part_id": "TEST-001",
        "part_name": "Test Part",
        "material": "Al-5053-H32",
        "tolerance": 0.5,
        "timestamp": datetime.now().isoformat(),
        "overall_result": "FAIL",
        "quality_score": 65.5,
        "confidence": 92.0,
        "statistics": {
            "total_points": 175000,
            "pass_rate": 90.2,
            "mean_deviation_mm": 0.031,
            "max_deviation_mm": 5.37,
        },
        "regions": [
            {"name": "Top Surface", "point_count": 72000, "mean_deviation_mm": 0.084,
             "max_deviation_mm": 0.24, "pass_rate": 57.4, "status": "ATTENTION"},
        ],
        "root_causes": [
            {"issue": "Systematic positive offset detected", "likely_cause": "Tooling setup error",
             "explanation": "A uniform offset indicates die gap issues.", "confidence": 75, "priority": "HIGH"},
        ],
        "recommendations": [
            {"priority": "HIGH", "action": "Verify material thickness", "expected_improvement": "Eliminate offset"},
        ],
        "ai_summary": "Part FAILS inspection. 90.2% of points within tolerance.",
    }

    output = generate_pdf_report(sample_data, "/tmp/test_qc_report.pdf")
    print(f"Generated: {output}")
