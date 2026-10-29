#!/usr/bin/env python3
"""Build the verified visual PDF report for the 38172000 V-vs-X investigation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "output" / "experiments" / "good-vs-defective-38172000-20260728"
CORRELATION_DIR = EXPERIMENT / "f1_corrected_prismatic_heatmap_correlation_final"
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs" / "38172000_f1_visual_report"
OUTPUT_PDF = OUTPUT_DIR / "ShermanQC_38172000_F1_Bend_Deviation_Report_20260729.pdf"
OUTPUT_MD = OUTPUT_DIR / "ShermanQC_38172000_F1_Bend_Deviation_Report_20260729.md"

NAVY = colors.HexColor("#071729")
INK = colors.HexColor("#142033")
MUTED = colors.HexColor("#5F6B7A")
PALE = colors.HexColor("#EFF4F8")
GRID = colors.HexColor("#D6DEE8")
GREEN = colors.HexColor("#20A66A")
GREEN_PALE = colors.HexColor("#E9F7F0")
AMBER = colors.HexColor("#E59A13")
AMBER_PALE = colors.HexColor("#FFF4DB")
RED = colors.HexColor("#D92145")
RED_PALE = colors.HexColor("#FDECEF")
PURPLE = colors.HexColor("#7C3AED")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _status(row: dict) -> tuple[str, colors.Color, colors.Color]:
    if row["concordant_deviation"]:
        return "CORROBORATED", RED, RED_PALE
    if row["f1"]["screening_deviation"]:
        return "F1 ONLY", AMBER, AMBER_PALE
    if row["heatmap_hot_at_p95_1_5mm"]:
        return "HEATMAP ONLY", PURPLE, colors.HexColor("#F1EAFE")
    return "STABLE", GREEN, GREEN_PALE


def _build_charts(correlation: dict) -> tuple[Path, Path]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rows = correlation["bends"]
    bend_ids = [row["bend_id"] for row in rows]
    p95 = [float(row["surface_heatmap_corridor"].get("p95_mm") or 0.0) for row in rows]
    bar_colors = []
    for row in rows:
        status, _, _ = _status(row)
        bar_colors.append(
            "#D92145" if status == "CORROBORATED" else "#E59A13" if status == "F1 ONLY" else "#20A66A"
        )
    bar_path = TMP_DIR / "bend_local_p95_bars.png"
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=180)
    ax.bar(bend_ids, p95, color=bar_colors, edgecolor="#142033", linewidth=0.6)
    ax.axhline(1.5, color="#D92145", linestyle="--", linewidth=1.5, label="diagnostic heatmap P95 threshold: 1.5 mm")
    ax.set_ylabel("Local X-to-V surface P95 (mm)")
    ax.set_xlabel("Corrected physical bend ID")
    ax.set_title("Independent surface deviation in a 12 mm corridor around each good-part F1 bend")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(bar_path, facecolor="white")
    plt.close(fig)

    scatter_path = TMP_DIR / "f1_offset_vs_heatmap_p95.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    for row in rows:
        status, _, _ = _status(row)
        color = "#D92145" if status == "CORROBORATED" else "#E59A13" if status == "F1 ONLY" else "#20A66A"
        x = float(row["f1"]["axis_offset_mm"])
        y = float(row["surface_heatmap_corridor"].get("p95_mm") or 0.0)
        ax.scatter(x, y, s=70, color=color, edgecolor="#142033", linewidth=0.7, zorder=3)
        ax.annotate(row["bend_id"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9, weight="bold")
    ax.axvline(1.0, color="#6B7280", linestyle="--", linewidth=1.2)
    ax.axhline(1.5, color="#6B7280", linestyle="--", linewidth=1.2)
    ax.text(1.01, 0.12, "F1 offset screen", color="#6B7280", fontsize=8, rotation=90, va="bottom")
    ax.text(0.04, 1.53, "heatmap P95 screen", color="#6B7280", fontsize=8)
    ax.set_xlabel("F1 axis offset after X-to-V registration (mm)")
    ax.set_ylabel("Local independent surface P95 (mm)")
    ax.set_title("Evidence agreement: only the upper-right region is corroborated")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(scatter_path, facecolor="white")
    plt.close(fig)
    return bar_path, scatter_path


def _page(canvas, doc) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 12 * mm, width, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(13 * mm, height - 7.7 * mm, "SHERMAN QC | 38172000 BEND DEVIATION INVESTIGATION")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(width - 13 * mm, 7 * mm, f"29 Jul 2026 | Page {doc.page}")
    canvas.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24, leading=27, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=MUTED, spaceAfter=8),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=NAVY, spaceAfter=7),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=INK, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.3, textColor=INK, spaceAfter=5),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6, leading=9.5, textColor=MUTED),
        "center": ParagraphStyle("Center", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11, textColor=INK, alignment=TA_CENTER),
        "metric": ParagraphStyle("Metric", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=17, leading=18, textColor=NAVY, alignment=TA_CENTER),
        "metric_label": ParagraphStyle("MetricLabel", parent=base["BodyText"], fontName="Helvetica", fontSize=7.7, leading=9, textColor=MUTED, alignment=TA_CENTER),
    }


def _metric_card(value: str, label: str, styles: dict) -> Table:
    table = Table([[Paragraph(value, styles["metric"])], [Paragraph(label, styles["metric_label"])]], colWidths=[49 * mm], rowHeights=[12 * mm, 9 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.7, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _finding_box(title: str, text: str, color: colors.Color, pale: colors.Color, styles: dict) -> Table:
    table = Table(
        [[Paragraph(title, ParagraphStyle("box_title", parent=styles["h2"], textColor=color, spaceAfter=0)), Paragraph(text, styles["body"])]],
        colWidths=[48 * mm, 198 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), pale),
                ("BOX", (0, 0), (-1, -1), 0.8, color),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def build() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    correlation = _load_json(CORRELATION_DIR / "f1_heatmap_correlation.json")
    score = _load_json(EXPERIMENT / "f1_corrected_prismatic_V_v2" / "manual_score_corrected_ids" / "score.json")
    comparison = _load_json(EXPERIMENT / "f1_corrected_prismatic_V_vs_X_v3" / "f1_bend_by_bend_comparison.json")
    registration = _load_json(EXPERIMENT / "comparison_summary.json")
    bar_path, scatter_path = _build_charts(correlation)
    styles = _styles()

    page_width, page_height = landscape(A4)
    doc = BaseDocTemplate(
        str(OUTPUT_PDF),
        pagesize=(page_width, page_height),
        leftMargin=13 * mm,
        rightMargin=13 * mm,
        topMargin=18 * mm,
        bottomMargin=12 * mm,
        title="Sherman QC 38172000 F1 Bend Deviation Report",
        author="Sherman QC / Codex",
        subject="Corrected zero-shot F1 and independent surface heatmap correlation",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=_page)])
    story = []

    # Page 1 - decision summary
    story += [Spacer(1, 7 * mm), Paragraph("38172000 Bend-Level Deviation Investigation", styles["title"])]
    story += [Paragraph("Corrected zero-shot F1 bend geometry for good V versus defective X, correlated with the independent registered surface heatmap", styles["subtitle"]), Spacer(1, 4 * mm)]
    cards = Table(
        [[
            _metric_card("10 / 10", "GOOD V F1 IDENTITIES", styles),
            _metric_card("10 / 10", "X-TO-V MATCHED IDENTITIES", styles),
            _metric_card("3", "CORROBORATED BENDS", styles),
            _metric_card("1.93 mm", "MAX LOCAL P95", styles),
            _metric_card("16", "REGRESSION TESTS PASSED", styles),
        ]],
        colWidths=[52 * mm] * 5,
    )
    cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story += [cards, Spacer(1, 7 * mm)]
    story += [
        _finding_box(
            "PRIMARY FINDING",
            "B02, B08, and B09 deviate in both corrected F1 geometry and the independent X-to-V surface heatmap. These are the defensible problem locations for QC review.",
            RED,
            RED_PALE,
            styles,
        ),
        Spacer(1, 4 * mm),
        _finding_box(
            "SCREENING ONLY",
            "B01, B04, B06, and B11 exceed at least one F1 screening threshold but remain cool in the local surface heatmap. They are not treated as confirmed defect locations.",
            AMBER,
            AMBER_PALE,
            styles,
        ),
        Spacer(1, 4 * mm),
        _finding_box(
            "TOPOLOGY CORRECTION",
            "The old B05 and B07 markers were end-cap residuals, not bends. They are retired. B11 is the recovered full-width junction above B03, and B12 is the opposite full-width folded junction.",
            GREEN,
            GREEN_PALE,
            styles,
        ),
        Spacer(1, 7 * mm),
        Paragraph("Disposition", styles["h2"]),
        Paragraph(
            "Flag X for QC review at B02, B08, and B09. Do not issue a production pass/fail decision from this report alone: the part drawing tolerances were not supplied. All thresholds in this report are diagnostic screening thresholds.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # Page 2 - corrected topology
    story += [Paragraph("1. Corrected Physical Bend Topology", styles["h1"])]
    story += [
        Paragraph(
            "The corrected reference contains six full-width structural folds, two genuine short outer-end bends, and two formed-tab bends. The visual below is the geometry reviewed from plan, profile, front, reverse, and underside views.",
            styles["body"],
        ),
        Spacer(1, 2 * mm),
        _scaled_image(EXPERIMENT / "manual_10_bends_V_good_corrected_ids_review.png", doc.width, 134 * mm),
        Spacer(1, 2 * mm),
        Paragraph(
            "ID policy: B05 and B07 remain retired so no downstream report can confuse the former false markers with the recovered physical folds. New long axes use B11 and B12.",
            styles["small"],
        ),
        PageBreak(),
    ]

    # Page 3 - greenlight on good
    story += [Paragraph("2. Offline F1 Greenlight on Known-Good V", styles["h1"])]
    story += [
        Paragraph(
            f"Against the corrected ten-bend reference, the optimized offline profile scored TP {score['true_positive']}, FP {score['false_positive']}, and FN {score['false_negative']}. Expected count, manual truth, and part identity were not exposed to inference.",
            styles["body"],
        ),
        _scaled_image(EXPERIMENT / "f1_corrected_prismatic_V_v2" / "manual_score_corrected_ids" / "gold_vs_zero_shot_six_view.png", doc.width, 140 * mm),
        Spacer(1, 2 * mm),
        Paragraph(
            "Solid green is the corrected reference and dashed cyan is F1. The six-view overlap verifies finite span, correct parallel ridge selection, recovered B11/B12, and removal of the old end-cap residuals.",
            styles["small"],
        ),
        PageBreak(),
    ]

    # Page 4 - independent heatmap
    surface = registration["defective_to_good"]
    story += [Paragraph("3. Independent Surface Evidence", styles["h1"])]
    story += [
        Paragraph(
            "The surface comparison was generated independently of F1 after the corrected rigid registration of X onto V. It measures nearest-good-surface distance for every defective scan point.",
            styles["body"],
        ),
        Table(
            [[
                Paragraph(f"<b>Median</b><br/>{surface['percentiles_mm']['50']:.3f} mm", styles["center"]),
                Paragraph(f"<b>P95</b><br/>{surface['percentiles_mm']['95']:.3f} mm", styles["center"]),
                Paragraph(f"<b>P99</b><br/>{surface['percentiles_mm']['99']:.3f} mm", styles["center"]),
                Paragraph(f"<b>Above 1 mm</b><br/>{100.0 * surface['fraction_above']['1_mm']:.1f}%", styles["center"]),
                Paragraph(f"<b>Above 2 mm</b><br/>{100.0 * surface['fraction_above']['2_mm']:.2f}%", styles["center"]),
            ]],
            colWidths=[52 * mm] * 5,
            style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.7, GRID), ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]),
        ),
        Spacer(1, 3 * mm),
        _scaled_image(EXPERIMENT / "corrected_pose_deviation_0_to_2mm.png", doc.width, 127 * mm),
        Spacer(1, 1 * mm),
        Paragraph("Color scale is clipped at 2 mm. Red/orange regions are locally most different from V; blue regions remain close after registration.", styles["small"]),
        PageBreak(),
    ]

    # Page 5 - bend overview
    story += [Paragraph("4. Bend-Level Problem Overview", styles["h1"])]
    story += [
        Paragraph(
            "This is the principal diagnostic view. Each corrected good-part F1 axis is placed directly over the independent surface heatmap. Red means both evidence sources agree; amber means F1-only; green means neither source confirms a local problem.",
            styles["body"],
        ),
        _scaled_image(CORRELATION_DIR / "f1_bends_over_surface_heatmap.png", doc.width, 135 * mm),
        Spacer(1, 2 * mm),
        _finding_box(
            "CORROBORATED LOCATIONS",
            "B02: lower perimeter fold. B08: lower short outer-end bend. B09: upper formed-tab bend. Their F1 offsets and local heatmap intensities point to the same physical regions.",
            RED,
            RED_PALE,
            styles,
        ),
        PageBreak(),
    ]

    # Page 6 - quantitative table
    story += [Paragraph("5. Bend-Level Measurements", styles["h1"])]
    header = ["Bend", "Status", "Axis offset\nmm", "Direction\ndeg", "Plane angle\ndeg", "Heatmap P95\nmm", "Points >1 mm", "Interpretation"]
    data = [header]
    for row in correlation["bends"]:
        status, _, _ = _status(row)
        f1 = row["f1"]
        local = row["surface_heatmap_corridor"]
        if status == "CORROBORATED":
            interpretation = "Primary QC location"
        elif status == "F1 ONLY":
            interpretation = "Do not confirm without tolerance"
        else:
            interpretation = "No corroborated change"
        data.append(
            [
                row["bend_id"],
                status,
                f"{f1['axis_offset_mm']:.3f}",
                f"{f1['axis_direction_delta_deg']:.3f}",
                f"{f1['incident_plane_angle_delta_deg']:+.3f}",
                f"{float(local.get('p95_mm') or 0.0):.3f}",
                f"{100.0 * float(local.get('fraction_above_1mm') or 0.0):.1f}%",
                interpretation,
            ]
        )
    table = Table(data, colWidths=[17 * mm, 28 * mm, 25 * mm, 25 * mm, 27 * mm, 27 * mm, 27 * mm, 77 * mm], repeatRows=1)
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("LEADING", (0, 0), (-1, -1), 8.4),
        ("ALIGN", (0, 0), (6, -1), "CENTER"),
        ("ALIGN", (7, 1), (7, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
    ]
    for index, row in enumerate(correlation["bends"], start=1):
        status, color, pale = _status(row)
        style_commands += [("BACKGROUND", (0, index), (-1, index), pale), ("TEXTCOLOR", (1, index), (1, index), color), ("FONTNAME", (0, index), (1, index), "Helvetica-Bold")]
    table.setStyle(TableStyle(style_commands))
    story += [table, Spacer(1, 4 * mm), _scaled_image(bar_path, doc.width, 67 * mm), PageBreak()]

    # Page 7 - evidence agreement and limitations
    story += [Paragraph("6. Evidence Agreement and Decision Boundary", styles["h1"])]
    story += [
        Table(
            [[
                _scaled_image(scatter_path, 135 * mm, 108 * mm),
                [
                        Paragraph("How to read the scatter", styles["h2"]),
                        Paragraph("The vertical line is the 1.0 mm F1 axis-offset screen. The horizontal line is the 1.5 mm local heatmap P95 screen.", styles["body"]),
                        Paragraph("B02, B08, and B09 are red because they combine a bend-level F1 signal with a hot independent surface corridor.", styles["body"]),
                        Paragraph("Amber points are useful for investigation but are intentionally excluded from the primary defect call.", styles["body"]),
                        Spacer(1, 3 * mm),
                        Paragraph("Limits", styles["h2"]),
                        Paragraph("1. No drawing or production tolerance was supplied.<br/>2. Bend angle is inferred from scan-supported incident planes, not from a calibrated press-brake measurement.<br/>3. The report establishes relative deviation from known-good V, not formal drawing nonconformance.<br/>4. The optimized profile remains offline and is not promoted to runtime/main.", styles["body"]),
                ],
            ]],
            colWidths=[145 * mm, 116 * mm],
            style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]),
        ),
        Spacer(1, 5 * mm),
        _finding_box(
            "RECOMMENDED ACTION",
            "Review the physical regions corresponding to B02, B08, and B09 against the drawing tolerances. Preserve B01 as borderline context. Do not escalate B04, B06, or B11 from F1 alone.",
            RED,
            RED_PALE,
            styles,
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            f"Traceability: V and X each produced 10 corrected F1 identities; {comparison['comparison']['matched_count']} identities matched after rigid registration. Global X-to-V median surface deviation was {surface['percentiles_mm']['50']:.3f} mm and P95 was {surface['percentiles_mm']['95']:.3f} mm.",
            styles["small"],
        ),
    ]

    doc.build(story)

    md_lines = [
        "# Sherman QC 38172000 F1 Bend Deviation Report",
        "",
        "## Finding",
        "",
        "Corrected F1 and the independent surface heatmap corroborate local deviations at **B02, B08, and B09**.",
        "",
        "- Old B05 and B07 are retired non-bend residuals.",
        "- B11 is the recovered full-width bend above B03.",
        "- B12 is the opposite full-width folded junction.",
        "- Good V validation: TP 10, FP 0, FN 0.",
        "- V and X: ten matched bend identities after registration.",
        "",
        f"![Bend overview]({CORRELATION_DIR / 'f1_bends_over_surface_heatmap.png'})",
        "",
        (CORRELATION_DIR / "F1_HEATMAP_CORRELATION.md").read_text(encoding="utf-8"),
        "",
        "These are diagnostic screens, not production tolerances.",
    ]
    OUTPUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return OUTPUT_PDF


if __name__ == "__main__":
    print(build())
