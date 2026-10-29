#!/usr/bin/env python3
"""Build the visual 38172000 nominal-CAD versus defective-X F1 report."""

from __future__ import annotations

import hashlib
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
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "output" / "experiments" / "38172000-X-vs-CAD-F1-20260730"
DATA_PATH = EXPERIMENT / "f1_defective_X_vs_CAD_bend_by_bend.json"
OVERLAY_PATH = EXPERIMENT / "f1_defective_X_vs_CAD_overlay.png"
BARS_PATH = EXPERIMENT / "f1_defective_X_vs_CAD_surface_p95.png"
BLIND_X_PATH = EXPERIMENT / "blind_cad_ply_case.json"
BLIND_V_PATH = EXPERIMENT / "blind_cad_ply_good_control.json"
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs" / "38172000_f1_cad_visual_report_20260802"
OUTPUT_PDF = OUTPUT_DIR / "ShermanQC_38172000_F1_CAD_vs_Defective_X_Report_20260802.pdf"
OUTPUT_MD = OUTPUT_DIR / "ShermanQC_38172000_F1_CAD_vs_Defective_X_Report_20260802.md"

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
CYAN = colors.HexColor("#0891B2")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24,
            leading=27, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11,
            leading=15, textColor=MUTED, spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=17,
            leading=20, textColor=NAVY, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12,
            leading=15, textColor=INK, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=12.3, textColor=INK, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.5,
            leading=9.4, textColor=MUTED,
        ),
        "tiny": ParagraphStyle(
            "Tiny", parent=base["BodyText"], fontName="Helvetica", fontSize=6.6,
            leading=8.0, textColor=INK,
        ),
        "center": ParagraphStyle(
            "Center", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5,
            leading=11, textColor=INK, alignment=TA_CENTER,
        ),
        "metric": ParagraphStyle(
            "Metric", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=17,
            leading=18, textColor=NAVY, alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6,
            leading=9, textColor=MUTED, alignment=TA_CENTER,
        ),
    }


def _page(canvas, doc) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 12 * mm, width, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(13 * mm, height - 7.7 * mm, "SHERMAN QC | 38172000 CAD DEVIATION INVESTIGATION")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(width - 13 * mm, 7 * mm, f"2 Aug 2026 | Page {doc.page}")
    canvas.restoreState()


def _metric_card(value: str, label: str, styles: dict) -> Table:
    table = Table(
        [[Paragraph(value, styles["metric"])], [Paragraph(label, styles["metric_label"])]],
        colWidths=[49 * mm], rowHeights=[12 * mm, 9 * mm],
    )
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


def _finding_box(title: str, text: str, color, pale, styles: dict) -> Table:
    title_style = ParagraphStyle(
        f"box_{title}", parent=styles["h2"], textColor=color, spaceAfter=0,
    )
    table = Table(
        [[Paragraph(title, title_style), Paragraph(text, styles["body"])]],
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


def _build_evidence_charts(data: dict) -> tuple[Path, Path]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rows = data["bends"]

    agreement_path = TMP_DIR / "three_channel_evidence_agreement.png"
    fig, axis = plt.subplots(figsize=(9.5, 5.8), dpi=180)
    for row in rows:
        x_value = float(row["independent_x_to_v_surface_corridor"].get("p95_mm") or 0.0)
        y_value = float(row["cad_surface_corridor"]["defective_minus_good_p95_mm"])
        if row["primary_defect_location"]:
            color, marker, size = "#D92145", "o", 100
        elif row["cad_only_secondary_screen"]:
            color, marker, size = "#E59A13", "s", 90
        else:
            color, marker, size = "#20A66A", "o", 65
        axis.scatter(x_value, y_value, s=size, color=color, marker=marker, edgecolor="#142033", linewidth=0.7, zorder=3)
        axis.annotate(row["bend_id"], (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=9, weight="bold")
    axis.axvline(1.5, color="#6B7280", linestyle="--", linewidth=1.2)
    axis.axhline(0.35, color="#6B7280", linestyle="--", linewidth=1.2)
    axis.set_xlabel("independent X-to-V local P95 (mm)")
    axis.set_ylabel("X-to-CAD P95 excess over good V (mm)")
    axis.set_title("Three-channel agreement: upper-right points are spatially corroborated")
    axis.grid(alpha=0.18)
    axis.text(1.52, axis.get_ylim()[0] + 0.05, "X-to-V screen", color="#6B7280", fontsize=8, rotation=90, va="bottom")
    axis.text(axis.get_xlim()[0] + 0.02, 0.38, "CAD excess screen", color="#6B7280", fontsize=8)
    fig.tight_layout()
    fig.savefig(agreement_path, facecolor="white")
    plt.close(fig)

    differential_path = TMP_DIR / "f1_differential_by_bend.png"
    ids = [row["bend_id"] for row in rows]
    offsets = [float(row["x_minus_v_differential"]["axis_transverse_offset_mm"]) for row in rows]
    angles = [abs(float(row["x_minus_v_differential"]["incident_plane_angle_delta_deg"])) for row in rows]
    positions = list(range(len(rows)))
    fig, left = plt.subplots(figsize=(12.5, 4.8), dpi=180)
    left.bar([value - 0.18 for value in positions], offsets, width=0.36, color="#0891B2", label="axis offset (mm)")
    right = left.twinx()
    right.bar([value + 0.18 for value in positions], angles, width=0.36, color="#E59A13", label="absolute angle delta (deg)")
    left.axhline(1.0, color="#0891B2", linestyle="--", linewidth=1.0, alpha=0.8)
    right.axhline(0.75, color="#E59A13", linestyle="--", linewidth=1.0, alpha=0.8)
    left.set_xticks(positions, ids)
    left.set_ylabel("X-to-V F1 axis offset (mm)", color="#06768F")
    right.set_ylabel("absolute F1 incident-plane angle delta (deg)", color="#A86808")
    left.set_title("Corrected F1 differential by physical bend identity")
    left.grid(axis="y", alpha=0.16)
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper center", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(differential_path, facecolor="white")
    plt.close(fig)
    return agreement_path, differential_path


def _status(row: dict) -> tuple[str, colors.Color, colors.Color]:
    if row["primary_defect_location"]:
        return "PRIMARY", RED, RED_PALE
    if row["cad_only_secondary_screen"]:
        return "CAD ONLY", AMBER, AMBER_PALE
    return "NOT CONFIRMED", GREEN, GREEN_PALE


def _bend_table(rows: list[dict], styles: dict) -> Table:
    header = [
        "Bend", "CAD axis", "Match", "CAD angle", "V F1", "X F1", "X-V angle",
        "Axis offset", "V-CAD P95", "X-CAD P95", "Excess", "X-V P95", "Finding",
    ]
    data = [header]
    status_colors = []
    for row in rows:
        status, color, pale = _status(row)
        status_colors.append((color, pale))
        data.append(
            [
                row["bend_id"], row["cad_bend_id"], row["cad_axis_match_quality"],
                f"{row['nominal_cad']['bend_angle_deg']:.2f} deg",
                f"{row['good_v_control']['f1_angle_deg']:.2f}",
                f"{row['defective_x']['f1_angle_deg']:.2f}",
                f"{row['x_minus_v_differential']['incident_plane_angle_delta_deg']:+.2f}",
                f"{row['x_minus_v_differential']['axis_transverse_offset_mm']:.2f} mm",
                f"{row['cad_surface_corridor']['good_v_control']['p95_mm']:.2f}",
                f"{row['cad_surface_corridor']['defective_x']['p95_mm']:.2f}",
                f"{row['cad_surface_corridor']['defective_minus_good_p95_mm']:+.2f}",
                f"{float(row['independent_x_to_v_surface_corridor'].get('p95_mm') or 0.0):.2f}",
                status,
            ]
        )
    widths = [12, 16, 16, 17, 13, 13, 17, 17, 17, 17, 14, 15, 22]
    table = Table(data, colWidths=[value * mm for value in widths], repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.4),
        ("LEADING", (0, 0), (-1, -1), 7.7),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (3, 1), (-2, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
    ]
    for index, (color, pale) in enumerate(status_colors, start=1):
        commands.extend(
            [
                ("BACKGROUND", (0, index), (-1, index), pale if index % 2 else colors.white),
                ("TEXTCOLOR", (-1, index), (-1, index), color),
                ("FONTNAME", (-1, index), (-1, index), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def _write_markdown(data: dict) -> None:
    lines = [
        "# 38172000 nominal CAD versus defective X - F1 bend-level report",
        "",
        "## Decision",
        "",
        "Primary defect locations: **B02, B08, and B09**.",
        "",
        "B01 is retained as a CAD-only secondary screen and is not promoted because the independent X-to-V heatmap is below its localization threshold.",
        "",
        "## Bend-level data",
        "",
        "| Bend | CAD axis | X-V angle delta deg | X-V axis offset mm | V-CAD P95 mm | X-CAD P95 mm | CAD excess mm | X-V P95 mm | Finding |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in data["bends"]:
        status, _, _ = _status(row)
        lines.append(
            f"| {row['bend_id']} | {row['cad_bend_id']} | "
            f"{row['x_minus_v_differential']['incident_plane_angle_delta_deg']:+.2f} | "
            f"{row['x_minus_v_differential']['axis_transverse_offset_mm']:.2f} | "
            f"{row['cad_surface_corridor']['good_v_control']['p95_mm']:.2f} | "
            f"{row['cad_surface_corridor']['defective_x']['p95_mm']:.2f} | "
            f"{row['cad_surface_corridor']['defective_minus_good_p95_mm']:+.2f} | "
            f"{float(row['independent_x_to_v_surface_corridor'].get('p95_mm') or 0.0):.2f} | {status} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The STEP extractor returns 12 candidate axes while drawing-backed physical truth contains 10 bends.",
            "- Direct X-only CAD alignment attempted a 0.9306 scale correction, so the accepted comparison rigidly registers X to good V and then applies the V-to-CAD transform to both.",
            "- B08 has a low-quality direct CAD-axis match; its location is retained because corrected F1 and both surface comparisons agree.",
            "- Thresholds are diagnostic screens, not drawing tolerances or a production disposition.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    data = _load(DATA_PATH)
    blind_x = _load(BLIND_X_PATH)
    blind_v = _load(BLIND_V_PATH)
    agreement_path, differential_path = _build_evidence_charts(data)
    styles = _styles()

    page_width, page_height = landscape(A4)
    doc = BaseDocTemplate(
        str(OUTPUT_PDF), pagesize=(page_width, page_height),
        leftMargin=13 * mm, rightMargin=13 * mm, topMargin=18 * mm, bottomMargin=12 * mm,
        title="Sherman QC 38172000 F1 CAD versus Defective X Report",
        author="Sherman QC / Codex",
        subject="Corrected F1 bend geometry and CAD-control surface deviation",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=_page)])
    story = []

    # Page 1 - decision summary.
    story += [Spacer(1, 7 * mm), Paragraph("38172000 CAD vs Defective X", styles["title"])]
    story += [
        Paragraph(
            "Corrected zero-shot F1 bend identities measured against nominal STEP geometry, "
            "with known-good V as the registration and systematic-deviation control",
            styles["subtitle"],
        ),
        Spacer(1, 4 * mm),
    ]
    cards = Table(
        [[
            _metric_card("10 / 10", "F1 PHYSICAL IDENTITIES", styles),
            _metric_card("3", "PRIMARY DEFECT BENDS", styles),
            _metric_card("1", "CAD-ONLY SECONDARY", styles),
            _metric_card("1.68 mm", "MAX CAD P95 EXCESS", styles),
            _metric_card("93.38%", "X-CAD BLIND FITNESS", styles),
        ]],
        colWidths=[52 * mm] * 5,
    )
    cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story += [cards, Spacer(1, 7 * mm)]
    story += [
        _finding_box(
            "PRIMARY FINDING",
            "B02, B08, and B09 depart in corrected F1 geometry, exceed the good-control CAD surface baseline, and are independently hot in the X-to-V heatmap. These are the defensible QC review locations.",
            RED, RED_PALE, styles,
        ),
        Spacer(1, 4 * mm),
        _finding_box(
            "CAD-ONLY SCREEN",
            "B01 has 1.24 mm additional CAD-relative P95 deviation, but its independent X-to-V P95 is 1.41 mm and remains below the 1.5 mm localization screen. It is secondary, not confirmed.",
            AMBER, AMBER_PALE, styles,
        ),
        Spacer(1, 4 * mm),
        _finding_box(
            "CONTROL RESULT",
            "Known-good V also differs systematically from this STEP surface. X is therefore registered rigidly to V and both scans use the same V-to-CAD transform; the unsafe X-only 0.9306 scale correction is excluded.",
            GREEN, GREEN_PALE, styles,
        ),
        Spacer(1, 7 * mm),
        Paragraph("Disposition", styles["h2"]),
        Paragraph(
            "Flag defective X for QC review at B02, B08, and B09. Do not issue a production pass/fail decision from this report alone: drawing tolerances were not supplied and all thresholds are diagnostic screens.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # Page 2 - overlay.
    story += [Paragraph("Nominal CAD Overlay with Corrected F1 Axes", styles["h1"])]
    story += [
        Paragraph(
            "Defective X is shown in the good-control CAD frame. Red axes are confirmed by all three evidence channels; amber is CAD-only; green is not corroborated.",
            styles["body"],
        ),
        Spacer(1, 2 * mm),
        _scaled_image(OVERLAY_PATH, 266 * mm, 150 * mm),
        PageBreak(),
    ]

    # Page 3 - surface bars.
    story += [Paragraph("CAD-Relative Surface Deviation by F1 Bend Corridor", styles["h1"])]
    story += [
        Paragraph(
            "Each 12 mm corridor is centered on the corrected good-control F1 bend line. The blue bar is the V-to-CAD systematic baseline; the red bar is defective X in the same frame.",
            styles["body"],
        ),
        Spacer(1, 2 * mm),
        _scaled_image(BARS_PATH, 255 * mm, 148 * mm),
        PageBreak(),
    ]

    # Page 4 - agreement and F1 differential.
    story += [Paragraph("Evidence Agreement and F1 Differential", styles["h1"])]
    figures = Table(
        [[
            _scaled_image(agreement_path, 127 * mm, 103 * mm),
            _scaled_image(differential_path, 127 * mm, 103 * mm),
        ]],
        colWidths=[131 * mm, 131 * mm],
    )
    figures.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 1)]))
    story += [
        Paragraph(
            "The agreement plot separates primary locations from B01: primary bends exceed both the independent X-to-V surface screen and the CAD excess-over-control screen. The right chart shows why angle-only changes at B04 and B11 are not sufficient.",
            styles["body"],
        ),
        figures,
        Spacer(1, 3 * mm),
        _finding_box(
            "READING",
            "B02 is the strongest mixed angle-and-position finding. B08 and B09 are primarily positional and surface deviations rather than bend-angle failures.",
            RED, RED_PALE, styles,
        ),
        PageBreak(),
    ]

    # Page 5 - bend table.
    story += [Paragraph("Bend-by-Bend Measurement Table", styles["h1"])]
    story += [
        Paragraph(
            "CAD values are extractor geometry; V and X values are corrected F1 measurements. CAD P95 columns are unsigned nearest-surface distances in the shared control frame.",
            styles["body"],
        ),
        Spacer(1, 2 * mm),
        _bend_table(data["bends"], styles),
        Spacer(1, 4 * mm),
        Paragraph(
            "B08 has a low-quality direct CAD-axis match because the STEP extractor oversegments the short-end topology. Its primary location status comes from corrected F1 correspondence plus both independent surface channels, not from the direct CAD axis alone.",
            styles["small"],
        ),
        PageBreak(),
    ]

    # Page 6 - controls and generic pipeline audit.
    x_summary = blind_x["summary"]
    v_summary = blind_v["summary"]
    control_data = [
        ["Control item", "Defective X", "Known-good V", "Interpretation"],
        ["Blind CAD total", str(x_summary["total_bends"]), str(v_summary["total_bends"]), "Legacy route reports 11; raw STEP extractor reports 12"],
        ["Detected", str(x_summary["detected"]), str(v_summary["detected"]), "Both report 10, matching physical F1 truth"],
        ["Generic pipeline fail", str(x_summary["failed"]), str(v_summary["failed"]), "Good V also fails one CAD bend; generic status is not dispositive"],
        ["Alignment fitness", "93.38%", "91.46%", "Both globally register, but X-only scaling is inconsistent"],
        ["Alignment RMSE", "1.025 mm", "1.059 mm", "Comparable global residuals"],
        ["F1 physical identity", "10 / 10", "10 / 10", "Corrected profile is the bend identity source"],
    ]
    control_table = Table(control_data, colWidths=[45 * mm, 35 * mm, 35 * mm, 135 * mm])
    control_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.45, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story += [Paragraph("Registration Controls and Fail-Closed Interpretation", styles["h1"])]
    story += [
        Paragraph(
            "The control run is essential: the generic CAD pipeline reports a fail on both defective X and known-good V, and its CAD count disagrees with both the raw extractor and drawing-backed physical topology.",
            styles["body"],
        ),
        Spacer(1, 3 * mm),
        control_table,
        Spacer(1, 7 * mm),
        _finding_box(
            "WHY THE CONTROL FRAME",
            "Direct X-to-CAD alignment applied a 0.9306 uniform scale while the V control skipped scaling. Comparing those outputs bend by bend would mix registration policy with manufacturing deviation. The accepted analysis applies one V-derived CAD frame to both scans.",
            AMBER, AMBER_PALE, styles,
        ),
        Spacer(1, 5 * mm),
        Paragraph("CAD topology warning", styles["h2"]),
        Paragraph(
            "The STEP mesh extractor returns 12 axes. CAD_B6 and CAD_B12 remain unmatched after assigning ten corrected physical F1 identities. They are retained as extractor candidates, not silently counted as production bends.",
            styles["body"],
        ),
        PageBreak(),
    ]

    # Page 7 - method, limitations, provenance.
    alignment = data["alignment"]
    story += [Paragraph("Method, Thresholds, and Provenance", styles["h1"])]
    method_rows = [
        ["Step", "Operation", "Leakage / integrity control"],
        ["1", "Run corrected F1 on X and V", "Ten physical identities; no CAD or expected count used by F1 inference"],
        ["2", "Register X rigidly to good V", "No scale; deterministic trimmed ICP"],
        ["3", "Transform V and X into CAD frame", "Same V-to-CAD affine applied to both; recovered residual below 0.00001 mm"],
        ["4", "Match V physical axes to STEP candidates", "One-to-one geometric assignment; low-quality matches exposed"],
        ["5", "Measure CAD surface distance", "Exact triangle-mesh distance in 12 mm F1 corridors"],
        ["6", "Require three-channel agreement", "F1 differential + CAD excess over V + independent X-to-V heatmap"],
    ]
    method_table = Table(method_rows, colWidths=[14 * mm, 92 * mm, 154 * mm])
    method_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    thresholds = data["method"]["screening_thresholds_not_production_tolerances"]
    story += [method_table, Spacer(1, 4 * mm)]
    story += [Paragraph("Diagnostic thresholds", styles["h2"])]
    story += [
        Paragraph(
            f"F1 axis offset: {thresholds['f1_axis_offset_mm']:.2f} mm; F1 direction delta: "
            f"{thresholds['f1_axis_direction_delta_deg']:.2f} deg; F1 incident-plane angle delta: "
            f"{thresholds['f1_incident_plane_angle_delta_deg']:.2f} deg; CAD local P95: "
            f"{thresholds['cad_surface_p95_mm']:.2f} mm; CAD P95 excess over good V: "
            f"{thresholds['cad_surface_p95_excess_over_good_mm']:.2f} mm.",
            styles["body"],
        ),
        Paragraph("These are screening thresholds, not drawing tolerances.", styles["body"]),
        Spacer(1, 2 * mm),
        Paragraph("Limitations", styles["h2"]),
    ]
    for limitation in data["limitations"]:
        story.append(Paragraph(f"- {limitation}", styles["body"]))
    story += [Spacer(1, 2 * mm), Paragraph("Source integrity", styles["h2"])]
    provenance_rows = [
        ["Artifact", "SHA-256"],
        ["Nominal STEP", _sha256(Path(data["cad_source"]))],
        ["Bend-level JSON", _sha256(DATA_PATH)],
        ["CAD overlay", _sha256(OVERLAY_PATH)],
        ["CAD corridor chart", _sha256(BARS_PATH)],
    ]
    provenance_table = Table(provenance_rows, colWidths=[55 * mm, 205 * mm])
    provenance_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (1, 1), (1, -1), "Courier"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story += [provenance_table]

    doc.build(story)
    _write_markdown(data)
    print(
        json.dumps(
            {
                "pdf": str(OUTPUT_PDF),
                "markdown": str(OUTPUT_MD),
                "pages": 7,
                "primary_defect_locations": data["primary_defect_locations"],
                "secondary": data["cad_only_secondary_screens"],
                "good_to_cad_affine_residual_max_mm": alignment["good_v_to_cad_affine_diagnostics"]["residual_max_mm"],
            },
            indent=2,
        )
    )
    return OUTPUT_PDF


if __name__ == "__main__":
    build()
