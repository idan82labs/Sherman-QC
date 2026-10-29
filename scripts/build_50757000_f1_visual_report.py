#!/usr/bin/env python3
"""Build the visual PDF report for 50757000_02 good V versus defective X."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "output/research/f1-email-intake-20260804/50757000_02"
SOURCE = CASE / "source"
SURFACE_V1 = CASE / "surface_comparison"
SURFACE = CASE / "surface_comparison_v2"
F1 = CASE / "f1_v23"
F1_COMPARE = CASE / "f1_bend_compare"
OUTPUT = ROOT / "output/pdf/ShermanQC_50757000_02_Good_vs_Defective_F1_Report_20260804.pdf"
TMP = ROOT / "tmp/pdfs/50757000_02_report"

PAGE = landscape(A4)
NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
LINE = colors.HexColor("#D9E2EC")
PANEL = colors.HexColor("#F4F7FA")
CYAN = colors.HexColor("#00A6C8")
GREEN = colors.HexColor("#159447")
RED = colors.HexColor("#D7263D")
AMBER = colors.HexColor("#D97706")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _register_fonts() -> tuple[str, str]:
    candidates = (
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ),
        (
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        ),
    )
    for regular, bold in candidates:
        if Path(regular).is_file() and Path(bold).is_file():
            pdfmetrics.registerFont(TTFont("ShermanSans", regular))
            pdfmetrics.registerFont(TTFont("ShermanSansBold", bold))
            return "ShermanSans", "ShermanSansBold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_fonts()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=24,
            leading=27,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=3 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=10.5,
            leading=14,
            textColor=MUTED,
            spaceAfter=3 * mm,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=21,
            textColor=NAVY,
            spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceAfter=1.5 * mm,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.4,
            leading=11.2,
            textColor=INK,
            spaceAfter=1.8 * mm,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.2,
            leading=9.2,
            textColor=MUTED,
        ),
        "card_value": ParagraphStyle(
            "card_value",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=17,
            leading=19,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "card_label": ParagraphStyle(
            "card_label",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=7.3,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=7.2,
            leading=9,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "table_head",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=7.1,
            leading=8.6,
            textColor=colors.white,
        ),
    }


STYLES = _styles()


def _p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def _scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with PILImage.open(path) as picture:
        width_px, height_px = picture.size
    scale = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * scale, height=height_px * scale)


def _card(value: str, label: str) -> Table:
    card = Table(
        [[_p(value, "card_value")], [_p(label, "card_label")]],
        colWidths=[60 * mm],
        rowHeights=[9 * mm, 8 * mm],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1 * mm),
            ]
        )
    )
    return card


def _data_table(rows: list[list], widths: list[float], *, header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2 * mm),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    for row in range(1 if header else 0, len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), PANEL))
    table.setStyle(TableStyle(commands))
    return table


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = PAGE
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(14 * mm, height - 10 * mm, width - 14 * mm, height - 10 * mm)
    canvas.setFont(FONT_BOLD, 7)
    canvas.setFillColor(NAVY)
    canvas.drawString(14 * mm, height - 7.5 * mm, "SHERMAN QC | 50757000_02")
    canvas.setFont(FONT, 7)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 14 * mm, height - 7.5 * mm, "GOOD V VS DEFECTIVE X | RESEARCH REPORT")
    canvas.line(14 * mm, 9 * mm, width - 14 * mm, 9 * mm)
    canvas.drawString(14 * mm, 5.5 * mm, "Corrected alignment v2 only - runtime/main promotion not authorized")
    canvas.drawRightString(width - 14 * mm, 5.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _drawing_png() -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    output = TMP / "drawing.png"
    if not output.is_file():
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "1",
                "-singlefile",
                "-png",
                "-r",
                "180",
                str(SOURCE / "MEC003744_02.pdf"),
                str(TMP / "drawing"),
            ],
            check=True,
        )
    return output


def build() -> Path:
    surface = _load_json(SURFACE / "comparison_summary.json")
    correlation = _load_json(F1_COMPARE / "f1_v23_heatmap_correlation.json")
    accepted = _load_json(CASE / "alignment_visual_review.json")["accepted_run"]
    x_to_v = surface["defective_to_good"]
    rows_by_id = {
        row["good_f1_id"]: row
        for row in correlation["good_axis_surface_corridors"]
    }
    drawing = _drawing_png()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=PAGE,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=13 * mm,
        title="Sherman QC 50757000_02 Good vs Defective F1 Report",
        author="Sherman QC research pipeline",
        subject="Corrected scan alignment, surface deviation, drawing evidence, and F1 bend localization",
    )
    story = []

    # Page 1 - executive result
    story += [
        Spacer(1, 2 * mm),
        _p("50757000_02", "subtitle"),
        _p("Good V vs defective X", "title"),
        _p(
            "Corrected rigid surface comparison and scan-only F1 bend localization",
            "subtitle",
        ),
    ]
    verdict = Table(
        [[_p("DEFECTIVE RELATIVE TO V BASELINE", "h2"), _p("Evidence: email label + inspection table + corrected surface heatmap + localized F1 displacement", "body")]],
        colWidths=[82 * mm, 173 * mm],
    )
    verdict.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#FDE8EC")),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FFF7F8")),
                ("BOX", (0, 0), (-1, -1), 0.8, RED),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
            ]
        )
    )
    story += [verdict, Spacer(1, 4 * mm)]
    cards = Table(
        [[
            _card(f"{x_to_v['percentiles_mm']['50']:.3f} mm", "X-to-V median"),
            _card(f"{x_to_v['percentiles_mm']['95']:.3f} mm", "X-to-V P95"),
            _card(f"{100*x_to_v['fraction_above']['2_mm']:.1f}%", "surface above 2 mm"),
            _card("5.652 mm", "localized F1 axis shift"),
        ]],
        colWidths=[64 * mm] * 4,
    )
    cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1.5 * mm), ("RIGHTPADDING", (0, 0), (-1, -1), 1.5 * mm)]))
    story += [cards, Spacer(1, 4 * mm)]
    story += [
        _scaled_image(SURFACE / "overlay_good_V_vs_defective_X.png", 258 * mm, 84 * mm),
        Spacer(1, 2 * mm),
        _p(
            "The accepted v2 overlay uses a proper rotation only. The four large holes, top-center notch, bottom cutout and lugs, upright flange, and one-sided return remain on the same sides. The earlier reversed pose is explicitly rejected and excluded from every downstream result.",
            "small",
        ),
        PageBreak(),
    ]

    # Page 2 - alignment audit
    story += [_p("Alignment correction and visual QA", "h1")]
    comparison = Table(
        [
            [_p("Rejected four-pose result", "h2"), _p("Accepted 28-pose result", "h2")],
            [
                _scaled_image(SURFACE_V1 / "overlay_good_V_vs_defective_X.png", 127 * mm, 48 * mm),
                _scaled_image(SURFACE / "overlay_good_V_vs_defective_X.png", 127 * mm, 48 * mm),
            ],
            [
                _p("Side profile reversed; symmetric P80 = 22.741 mm. Not used.", "small"),
                _p("Proper PCA-frame pose; symmetric P80 = 3.031 mm. Used downstream.", "small"),
            ],
        ],
        colWidths=[130 * mm, 130 * mm],
    )
    comparison.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (0, -1), 0.7, RED),
                ("BOX", (1, 0), (1, -1), 0.7, GREEN),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    story += [comparison, Spacer(1, 4 * mm)]
    audit_rows = [
        [_p("Audit item", "table_head"), _p("Accepted result", "table_head")],
        [_p("Pose search", "table"), _p("4 dominant-plane + all 24 proper PCA-frame orientations", "table")],
        [_p("Selected frame", "table"), _p(f"{accepted['selected_frame_family']}; permutation {accepted['selected_axis_permutation']}; signs {accepted['selected_axis_signs']}", "table")],
        [_p("Rotation integrity", "table"), _p("det(R) = 0.999999999999988; orthogonality error 2.22e-14; no reflection", "table")],
        [_p("Visual anchors", "table"), _p("4 holes, center notch, lower cutout/lugs, upright flange, and one-sided long return all preserve handedness", "table")],
        [_p("Accepted metric", "table"), _p("symmetric P80 3.031 mm; X-to-V median 0.728 mm; P95 6.116 mm", "table")],
    ]
    story += [
        _data_table(audit_rows, [48 * mm, 212 * mm]),
        Spacer(1, 3 * mm),
        _p("Alignment decision", "h2"),
        _p(
            "The first image was a registration failure, not evidence of a reversed physical part. The v2 alignment restores the asymmetric side profile and is the only transform referenced in this report.",
            "body",
        ),
        PageBreak(),
    ]

    # Page 3 - surface evidence
    story += [_p("Independent surface-deviation evidence", "h1")]
    story += [
        _scaled_image(SURFACE / "defective_X_deviation_heatmap.png", 260 * mm, 88 * mm),
        Spacer(1, 2 * mm),
    ]
    stats_rows = [
        [_p("Metric", "table_head"), _p("X to V", "table_head"), _p("Interpretation", "table_head")],
        [_p("Median", "table"), _p("0.728 mm", "table"), _p("stable surfaces overlap closely after registration", "table")],
        [_p("P80 / P95 / P99", "table"), _p("2.997 / 6.116 / 7.491 mm", "table"), _p("localized multi-millimeter residuals remain", "table")],
        [_p("Above 2 mm", "table"), _p("28.46%", "table"), _p("substantial formed-area displacement", "table")],
        [_p("Above 5 mm", "table"), _p("8.12%", "table"), _p("consistent with failed inspection dimensions near 5 mm", "table")],
        [_p("Maximum", "table"), _p("8.408 mm", "table"), _p("localized extremum; not a tolerance claim", "table")],
    ]
    lower = Table(
        [[
            _data_table(stats_rows, [38 * mm, 42 * mm, 80 * mm]),
            _scaled_image(SURFACE / "deviation_histogram.png", 93 * mm, 57 * mm),
        ]],
        colWidths=[165 * mm, 96 * mm],
    )
    lower.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm)]))
    story += [
        lower,
        Spacer(1, 2 * mm),
        _p(
            "The heatmap is independent of F1. It shows concentrated residuals on the upright/formed wall and the adjacent broad formed region, rather than a uniform registration offset.",
            "small",
        ),
        PageBreak(),
    ]

    # Page 4 - drawing and supplied metrology
    story += [_p("Drawing and supplied inspection evidence", "h1")]
    measurement_rows = [
        [_p("Nominal", "table_head"), _p("Measured", "table_head"), _p("Deviation", "table_head"), _p("Status", "table_head")],
        [_p("90 deg", "table"), _p("89.8866 deg", "table"), _p("-0.1134 deg", "table"), _p("pass", "table")],
        [_p("186 mm", "table"), _p("181.1264 mm", "table"), _p("-4.8736 mm", "table"), _p("fail", "table")],
        [_p("25 mm", "table"), _p("29.9231 mm", "table"), _p("+4.9231 mm", "table"), _p("fail", "table")],
        [_p("25 mm", "table"), _p("23.9104 mm", "table"), _p("-1.0896 mm", "table"), _p("fail", "table")],
        [_p("87 mm", "table"), _p("81.9690 mm", "table"), _p("-5.0310 mm", "table"), _p("fail", "table")],
        [_p("222 mm", "table"), _p("216.7917 mm", "table"), _p("-5.2083 mm", "table"), _p("fail", "table")],
    ]
    evidence = Table(
        [[
            _scaled_image(drawing, 169 * mm, 121 * mm),
            [
                _p("Supplied metrology snapshot", "h2"),
                _scaled_image(SOURCE / "inline_2.png", 88 * mm, 51 * mm),
                Spacer(1, 2 * mm),
                _data_table(measurement_rows, [21 * mm, 25 * mm, 25 * mm, 17 * mm]),
            ],
        ]],
        colWidths=[173 * mm, 91 * mm],
    )
    evidence.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm)]))
    story += [
        evidence,
        Spacer(1, 2 * mm),
        _p(
            "Drawing basis: 3 mm sheet metal, nominal 90 deg bends, general linear tolerance +/-0.2 mm, angular tolerance +/-1 deg. The failed dimensions are consistent in scale with the corrected scan-to-scan residuals.",
            "small",
        ),
        PageBreak(),
    ]

    # Page 5 - F1 limitations
    story += [_p("F1 execution and visual critique", "h1")]
    f1_views = Table(
        [
            [_p("V good - v23 selected 4 lines", "h2"), _p("X defective - v23 selected 7 lines", "h2")],
            [
                _scaled_image(F1 / "V/final_line_audit.png", 128 * mm, 47 * mm),
                _scaled_image(F1 / "X/final_line_audit.png", 128 * mm, 47 * mm),
            ],
        ],
        colWidths=[130 * mm, 130 * mm],
    )
    f1_views.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm), ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm)]))
    story += [f1_views, Spacer(1, 3 * mm)]
    story += [
        _scaled_image(F1_COMPARE / "f1_bend_by_bend_comparison.png", 258 * mm, 78 * mm),
        Spacer(1, 2 * mm),
    ]
    limitation_rows = [
        [_p("Observation", "table_head"), _p("Result", "table_head")],
        [_p("Raw support strips", "table"), _p("V = 3; X = 10; only 1 high-quality identity match", "table")],
        [_p("Frozen v23 lines", "table"), _p("V = 4; X = 7; only 2 frozen-contract matches", "table")],
        [_p("Visual failure mode", "table"), _p("X fragments long physical folds into multiple collinear pieces and retains outline-like paths", "table")],
        [_p("Count interpretation", "table"), _p("4 vs 7 is detector instability, not a valid manufacturing bend-count conclusion", "table")],
    ]
    story += [
        _data_table(limitation_rows, [58 * mm, 202 * mm]),
        PageBreak(),
    ]

    # Page 6 - bend-local evidence
    story += [_p("Bend-level F1 localization over the independent heatmap", "h1")]
    story += [
        _scaled_image(F1_COMPARE / "f1_v23_axes_over_surface_heatmap.png", 260 * mm, 88 * mm),
        Spacer(1, 2 * mm),
    ]
    bend_rows = [
        [_p("Good axis", "table_head"), _p("X counterpart", "table_head"), _p("Axis angle", "table_head"), _p("Offset", "table_head"), _p("Local P95", "table_head"), _p("Assessment", "table_head")],
        [_p("F1U2", "table"), _p("nearest X F1U3", "table"), _p("0.164 deg", "table"), _p("5.652 mm", "table"), _p("3.764 mm", "table"), _p("corroborated displaced formed boundary", "table")],
        [_p("F1U1", "table"), _p("X F1U2", "table"), _p("0.478 deg", "table"), _p("0.924 mm", "table"), _p("0.614 mm", "table"), _p("stable / cool", "table")],
        [_p("F1U3", "table"), _p("X F1U1", "table"), _p("0.287 deg", "table"), _p("0.608 mm", "table"), _p("0.584 mm", "table"), _p("stable / cool", "table")],
        [_p("F1U4", "table"), _p("no contract match", "table"), _p("-", "table"), _p("-", "table"), _p("0.921 mm", "table"), _p("identity unresolved; not confirmed", "table")],
    ]
    story += [
        _data_table(bend_rows, [28 * mm, 42 * mm, 30 * mm, 28 * mm, 30 * mm, 102 * mm]),
        Spacer(1, 3 * mm),
    ]
    conclusion = Table(
        [[
            _p("Defensible localized finding", "h2"),
            _p(
                "F1U2 is nearly parallel with almost full span overlap but is displaced 5.652 mm. Its local surface corridor has P95 3.764 mm and 37.7% of points above 2 mm. The F1 and independent heatmap evidence agree in both scale and location.",
                "body",
            ),
        ]],
        colWidths=[62 * mm, 193 * mm],
    )
    conclusion.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF5F6")), ("BOX", (0, 0), (-1, -1), 0.8, RED), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm), ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm), ("TOPPADDING", (0, 0), (-1, -1), 3 * mm), ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm)]))
    story += [
        conclusion,
        Spacer(1, 2 * mm),
        _p(
            "Scope note: F1U2 is a screening/localization result, not a calibrated bend-angle measurement. Production disposition should continue to rely on the drawing and approved metrology process.",
            "small",
        ),
    ]

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return OUTPUT


def main() -> int:
    output = build()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
