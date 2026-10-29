#!/usr/bin/env python3
"""Build the send-ready Sherman QC F1 multimodal GPT adjudication package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
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
PDF_PATH = ROOT / "output/pdf/Sherman_QC_F1_GPT_Adjudication_20260715.pdf"
PACKAGE_DIR = ROOT / "output/professor_package/Sherman_QC_F1_GPT_Submission_20260715"
ARCHIVE_PATH = ROOT / "output/professor_package/Sherman_QC_F1_GPT_Submission_20260715.zip"
PACKET_PATH = ROOT / "output/experiments/f1-10of13/professor-hard-cases-20260715/adjudication_packet.json"

CASE_IMAGES = {
    "06_38172000": ROOT / "output/experiments/f1-10of13/line-audits/06_38172000_plane_lattice.png",
    "07_38172000": ROOT / "output/experiments/f1-11of13/case07-same-pair-span-alias/07_38172000_render_admissible8.png",
    "08_40500000": ROOT / "output/experiments/f1-10of13/line-audits/08_40500000_plane_lattice.png",
}

NAVY = colors.HexColor("#12304A")
BLUE = colors.HexColor("#2563EB")
TEAL = colors.HexColor("#0F766E")
AMBER = colors.HexColor("#B45309")
RED = colors.HexColor("#B91C1C")
PALE_BLUE = colors.HexColor("#EFF6FF")
PALE_AMBER = colors.HexColor("#FFF7ED")
PALE_GRAY = colors.HexColor("#F8FAFC")
MID_GRAY = colors.HexColor("#64748B")
LINE_GRAY = colors.HexColor("#CBD5E1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PackageTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=25, leading=29, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=7 * mm,
        ),
        "subtitle": ParagraphStyle(
            "PackageSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=12, leading=17, textColor=MID_GRAY, spaceAfter=6 * mm,
        ),
        "h1": ParagraphStyle(
            "PackageH1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=19, leading=23, textColor=NAVY, spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "PackageH2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=TEAL, spaceBefore=2 * mm,
            spaceAfter=1.5 * mm,
        ),
        "body": ParagraphStyle(
            "PackageBody", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.4, leading=13, textColor=colors.HexColor("#1E293B"),
            spaceAfter=2.2 * mm,
        ),
        "small": ParagraphStyle(
            "PackageSmall", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.8, leading=10, textColor=MID_GRAY,
        ),
        "callout": ParagraphStyle(
            "PackageCallout", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=10.2, leading=14, textColor=NAVY,
        ),
        "center": ParagraphStyle(
            "PackageCenter", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=10, leading=13, textColor=NAVY, alignment=TA_CENTER,
        ),
        "form": ParagraphStyle(
            "PackageForm", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=colors.HexColor("#1E293B"),
        ),
    }


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setStrokeColor(LINE_GRAY)
    canvas.setLineWidth(0.5)
    canvas.line(15 * mm, height - 12 * mm, width - 15 * mm, height - 12 * mm)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(15 * mm, height - 9 * mm, "SHERMAN QC - F1 GPT ADJUDICATION")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawRightString(width - 15 * mm, height - 9 * mm, "Prepared 15 July 2026")
    canvas.line(15 * mm, 11 * mm, width - 15 * mm, 11 * mm)
    canvas.drawString(15 * mm, 7 * mm, "Research evidence only - do not promote to runtime from this package")
    canvas.drawRightString(width - 15 * mm, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _metric_table(styles: dict[str, ParagraphStyle]) -> Table:
    data = [
        [Paragraph("Verified scans", styles["small"]), Paragraph("Count-exact", styles["small"]), Paragraph("Hard visual", styles["small"]), Paragraph("Release accepted", styles["small"])],
        [Paragraph("13", styles["center"]), Paragraph("11 / 13", styles["center"]), Paragraph("10 / 13", styles["center"]), Paragraph("0 / 13", styles["center"])],
    ]
    table = Table(data, colWidths=[43 * mm] * 4, rowHeights=[8 * mm, 13 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE_GRAY),
        ("BACKGROUND", (0, 1), (1, 1), PALE_BLUE),
        ("BACKGROUND", (2, 1), (2, 1), PALE_AMBER),
        ("BACKGROUND", (3, 1), (3, 1), colors.HexColor("#FEF2F2")),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE_GRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return table


def _case_image(path: Path) -> Image:
    image = Image(str(path), width=267 * mm, height=89.3 * mm)
    image.hAlign = "CENTER"
    return image


def _callout(text: str, styles: dict[str, ParagraphStyle], background=PALE_BLUE) -> Table:
    table = Table([[Paragraph(text, styles["callout"])]], colWidths=[267 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE if background == PALE_BLUE else AMBER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _case_page(
    *,
    case_id: str,
    title: str,
    status: str,
    image_path: Path,
    findings: list[str],
    question: str,
    response_fields: list[str],
    styles: dict[str, ParagraphStyle],
) -> list:
    rows = [[Paragraph("Observed", styles["small"]), Paragraph(status, styles["body"])]]
    for finding in findings:
        rows.append([Paragraph("Evidence", styles["small"]), Paragraph(finding, styles["body"])])
    evidence_table = Table(rows, colWidths=[25 * mm, 238 * mm])
    evidence_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE_GRAY),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    required = "<b>Return:</b> " + "; ".join(response_fields) + "."
    return [
        Paragraph(f"{case_id} - {title}", styles["h1"]),
        _case_image(image_path),
        Spacer(1, 2 * mm),
        evidence_table,
        Spacer(1, 2.5 * mm),
        _callout(f"<b>GPT adjudication task:</b> {question}", styles, PALE_AMBER),
        Spacer(1, 2 * mm),
        Paragraph(required, styles["small"]),
    ]


def _build_pdf(packet: dict) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=landscape(A4),
        rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=17 * mm, bottomMargin=15 * mm,
        title="Sherman QC F1 GPT Adjudication - 15 July 2026",
        author="Sherman QC engineering",
        subject="Multimodal GPT submission for three semantic bend-inspection adjudications",
    )
    story = [
        Spacer(1, 4 * mm),
        Paragraph("Sherman QC F1 bend semantics", styles["title"]),
        Paragraph("Multimodal GPT submission - three structured decisions required", styles["subtitle"]),
        _metric_table(styles),
        Spacer(1, 7 * mm),
        _callout(
            "The detector is zero-shot and deterministic. Counts, CAD, part-specific targets, and per-line truth were excluded from inference. Adjudicate the physical bend objects from the attached evidence; do not tune the answer to the known count.",
            styles,
        ),
        Spacer(1, 6 * mm),
        Paragraph("What changed", styles["h2"]),
        Paragraph(
            "A truth-free containment rule removed case-07 OB8, a short same-pair alias fully overpainting the stronger OB4 strip. Automated marker-count equality rose from 10/13 to 11/13. Hard visual acceptance remains 10/13 because case 07 still contains an unresolved cross-pair collinear ambiguity.",
            styles["body"],
        ),
        Paragraph("Task for the reviewing GPT", styles["h2"]),
        Paragraph(
            "For cases 06-08, classify the visible candidate objects and state the semantic rule that rejects nearby ridges, perimeter edges, segmentation aliases, or gap fragments. Return only schema-valid JSON. Never invent 3-D coordinates: use null plus needs_3d_followup when the supplied views are insufficient.",
            styles["body"],
        ),
        Spacer(1, 3 * mm),
        Table(
            [[Paragraph("Decision boundary", styles["small"]), Paragraph("F1 remains research-only. Enterprise promotion is blocked at 4/16 gates; this package cannot authorize runtime promotion.", styles["body"])]],
            colWidths=[35 * mm, 228 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#FEF2F2")),
                ("BOX", (0, 0), (-1, -1), 0.7, RED),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]),
        ),
        PageBreak(),
    ]

    cases = {item["case"]: item for item in packet["cases"]}
    story.extend(_case_page(
        case_id="Case 06 / 38172000 scan 1",
        title="two missing internal bends",
        status="Evaluation count 6; current final markers 2; strongest defensible scan-supported candidate 4.",
        image_path=CASE_IMAGES["06_38172000"],
        findings=[
            "Four long two-sided plane intersections attach to visible surface transitions.",
            "Additional ridge and junction proposals lie on face interiors and fail hard visual review.",
        ],
        question=cases["06_38172000"]["professor_response_required"]["question"],
        response_fields=cases["06_38172000"]["professor_response_required"]["required_fields"],
        styles=styles,
    ))
    story.append(PageBreak())
    story.extend(_case_page(
        case_id="Case 07 / 38172000 scan 2",
        title="count exact, object identity unresolved",
        status="Evaluation count 8; current final markers 8; not credited as a hard visual pass.",
        image_path=CASE_IMAGES["07_38172000"],
        findings=[
            "Resolved: OB8 is fully contained by same-pair OB4 (2.063 mm line offset; 6.91x support-mass dominance) and is suppressed without truth.",
            "Unresolved: OB1 and OB3 are collinear and fully nested at 2.681 mm offset, but select different inferred second flanges. Merging them yields seven and exposes another missing bend.",
        ],
        question=cases["07_38172000"]["professor_response_required"]["question"],
        response_fields=cases["07_38172000"]["professor_response_required"]["required_fields"],
        styles=styles,
    ))
    story.append(PageBreak())
    story.extend(_case_page(
        case_id="Case 08 / 40500000",
        title="six supported transitions versus seven",
        status="Evaluation count 7; current final marker 1; strongest defensible scan-supported candidate 6.",
        image_path=CASE_IMAGES["08_40500000"],
        findings=[
            "Six distinct surface-attached plane intersections survive two-sided support and duplicate checks.",
            "No seventh transition remains defensible; a split across an observed support gap is the only unresolved alternative.",
        ],
        question=cases["08_40500000"]["professor_response_required"]["question"],
        response_fields=cases["08_40500000"]["professor_response_required"]["required_fields"],
        styles=styles,
    ))
    story.append(PageBreak())

    submission_rows = [
        [Paragraph("Step", styles["small"]), Paragraph("GPT submission action", styles["small"])],
        [Paragraph("1", styles["center"]), Paragraph("Attach this PDF, the three full-resolution orthographic PNGs, and gpt_response_schema.json.", styles["form"])],
        [Paragraph("2", styles["center"]), Paragraph("Paste GPT_SUBMISSION_PROMPT.md verbatim as the user message. Do not add the expected counts as optimization instructions.", styles["form"])],
        [Paragraph("3", styles["center"]), Paragraph("Require JSON only, without Markdown fences. Validate it against gpt_response_schema.json before using any conclusion.", styles["form"])],
        [Paragraph("4", styles["center"]), Paragraph("Treat null coordinates and needs_3d_followup=true as correct abstention when the 2-D views cannot support a 3-D localization.", styles["form"])],
    ]
    submission_table = Table(submission_rows, colWidths=[18 * mm, 245 * mm], rowHeights=[7 * mm, 12 * mm, 13 * mm, 13 * mm, 13 * mm])
    submission_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), PALE_GRAY),
        ("BOX", (0, 0), (-1, -1), 0.7, LINE_GRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    output_rows = [
        [Paragraph("Case 06", styles["center"]), Paragraph("Classify the four rendered lines; identify whether two additional bends are visually localizable; define the ridge rejection rule.", styles["form"])],
        [Paragraph("Case 07", styles["center"]), Paragraph("Decide whether OB1/OB3 are distinct or an alias; preserve the resolved OB4/OB8 dominance fact; identify any missing eighth bend only when directly supported.", styles["form"])],
        [Paragraph("Case 08", styles["center"]), Paragraph("Decide whether a seventh visible bend exists or whether one of the six instances should split across a support gap.", styles["form"])],
    ]
    output_table = Table(output_rows, colWidths=[28 * mm, 235 * mm], rowHeights=[14 * mm] * 3)
    output_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE_GRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([
        Paragraph("GPT submission protocol", styles["h1"]),
        Paragraph(
            "This is a constrained multimodal adjudication, not a request to improve F1 from the known counts. The reviewing GPT must distinguish direct visual observations from inference and unknowns.",
            styles["body"],
        ),
        submission_table,
        Spacer(1, 2.5 * mm),
        Paragraph("Required case outputs", styles["h2"]),
        output_table,
        Spacer(1, 2.5 * mm),
        _callout(
            "Safety rule: the GPT must not invent reference_mesh_raw coordinates from the screenshots. Unsupported coordinates must be null. Returned semantics remain evaluation-only until independently confirmed.",
            styles,
        ),
    ])

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


def _write_text_files(packet: dict) -> None:
    response_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "sherman-qc-f1-gpt-adjudication-response-v1",
        "title": "Sherman QC F1 GPT adjudication response",
        "type": "object",
        "additionalProperties": False,
        "required": ["submission_id", "global_assessment", "cases", "semantic_rules", "final_recommendation"],
        "properties": {
            "submission_id": {"const": "sherman-qc-f1-hard-cases-20260715-v1"},
            "global_assessment": {
                "type": "object",
                "additionalProperties": False,
                "required": ["visual_evidence_sufficient", "count_forcing_avoided", "limitations"],
                "properties": {
                    "visual_evidence_sufficient": {"type": "boolean"},
                    "count_forcing_avoided": {"const": True},
                    "limitations": {"type": "array", "items": {"type": "string"}},
                },
            },
            "cases": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": [
                        "case_id", "direct_observations", "decision", "rationale",
                        "confidence", "axis_endpoints_reference_mesh_raw", "needs_3d_followup"
                    ],
                    "properties": {
                        "case_id": {"enum": ["06_38172000", "07_38172000", "08_40500000"]},
                        "direct_observations": {
                            "type": "array", "minItems": 1,
                            "items": {
                                "type": "object", "additionalProperties": False,
                                "required": ["attachment", "panel", "observation"],
                                "properties": {
                                    "attachment": {"type": "string"},
                                    "panel": {"enum": ["PC1_PC2", "PC1_PC3", "PC2_PC3", "cross_panel"]},
                                    "observation": {"type": "string"},
                                },
                            },
                        },
                        "decision": {
                            "enum": [
                                "accept_current_objects", "merge_segmentation_alias",
                                "missing_bend_visually_localized", "split_maximal_instance",
                                "insufficient_visual_evidence", "other"
                            ]
                        },
                        "objects_to_keep": {"type": "array", "items": {"type": "string"}},
                        "objects_to_reject": {
                            "type": "array",
                            "items": {
                                "type": "object", "additionalProperties": False,
                                "required": ["object_id", "reason"],
                                "properties": {"object_id": {"type": "string"}, "reason": {"type": "string"}},
                            },
                        },
                        "missing_or_split_object": {
                            "anyOf": [
                                {"type": "null"},
                                {
                                    "type": "object", "additionalProperties": False,
                                    "required": ["description", "panel", "approximate_2d_location"],
                                    "properties": {
                                        "description": {"type": "string"},
                                        "panel": {"enum": ["PC1_PC2", "PC1_PC3", "PC2_PC3", "cross_panel"]},
                                        "approximate_2d_location": {"type": "string"},
                                    },
                                },
                            ]
                        },
                        "rationale": {"type": "string", "minLength": 20},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "axis_endpoints_reference_mesh_raw": {
                            "anyOf": [
                                {"type": "null"},
                                {
                                    "type": "array", "minItems": 1,
                                    "items": {
                                        "type": "array", "minItems": 2, "maxItems": 2,
                                        "items": {
                                            "type": "array", "minItems": 3, "maxItems": 3,
                                            "items": {"type": "number"},
                                        },
                                    },
                                },
                            ]
                        },
                        "needs_3d_followup": {"type": "boolean"},
                    },
                },
            },
            "semantic_rules": {
                "type": "object", "additionalProperties": False,
                "required": ["countable_bend", "reject_face_ridge", "merge_alias", "split_across_gap"],
                "properties": {
                    "countable_bend": {"type": "string"},
                    "reject_face_ridge": {"type": "string"},
                    "merge_alias": {"type": "string"},
                    "split_across_gap": {"type": "string"},
                },
            },
            "final_recommendation": {
                "type": "object", "additionalProperties": False,
                "required": ["safe_to_change_f1", "reason", "required_next_evidence"],
                "properties": {
                    "safe_to_change_f1": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "required_next_evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }
    schema_path = PACKAGE_DIR / "gpt_response_schema.json"
    schema_path.write_text(json.dumps(response_schema, indent=2) + "\n", encoding="utf-8")

    prompt = """# Sherman QC F1 multimodal bend-object adjudication

You are an expert reviewer of sheet-metal bend geometry, 3-D scan interpretation, and quality-control inspection systems.

## Attached evidence

1. `Sherman_QC_F1_GPT_Adjudication_20260715.pdf` - review context and all three cases.
2. `06_38172000_orthographic_audit.png` - full-resolution case-06 audit.
3. `07_38172000_orthographic_audit.png` - full-resolution case-07 audit.
4. `08_40500000_orthographic_audit.png` - full-resolution case-08 audit.
5. `gpt_response_schema.json` - mandatory response contract.

## Objective

Adjudicate the physical bend objects in cases 06-08. This is a semantic and visual review, not a request to tune F1 to the known evaluation counts.

Before using any discrepancy count, independently enumerate what the three orthographic views support. Distinguish direct visual observations from inference and unknowns.

## Case questions

- Case 06: Are two additional conventional bends visually identifiable beyond the four rendered plane-lattice lines? State the positive bend criterion and the rule rejecting face-interior ridges.
- Case 07: OB8 has already been rejected as a contained same-pair alias of OB4. Decide whether OB1 and OB3 are two physical bends separated by a real intermediate flange or one segmentation alias. If they are an alias, identify another eighth bend only if the images directly support it.
- Case 08: Decide whether a seventh visible conventional bend exists or whether one of the six rendered transitions should split into two maximal instances across a support gap.

## Hard constraints

- Do not select objects merely to match the known count.
- Do not use CAD, filenames, part identity, or unstated manufacturing history as evidence.
- Cite the attachment filename and panel (`PC1_PC2`, `PC1_PC3`, `PC2_PC3`, or `cross_panel`) for every direct observation.
- Do not invent 3-D coordinates from 2-D screenshots. If exact `reference_mesh_raw` endpoints are not derivable, return `axis_endpoints_reference_mesh_raw: null` and `needs_3d_followup: true`.
- A count-exact case is not automatically visually correct.
- Prefer `insufficient_visual_evidence` over a speculative answer.
- Returned conclusions are evaluation-only until independently confirmed.

## Response

Return exactly one JSON object valid against `gpt_response_schema.json`. Do not wrap the JSON in Markdown fences and do not add prose outside it.
"""
    (PACKAGE_DIR / "GPT_SUBMISSION_PROMPT.md").write_text(prompt, encoding="utf-8")

    submission = {
        "schema_version": 1,
        "submission_id": "sherman-qc-f1-hard-cases-20260715-v1",
        "mode": "multimodal_gpt_adjudication",
        "prompt_file": "GPT_SUBMISSION_PROMPT.md",
        "response_schema_file": "gpt_response_schema.json",
        "attachment_order": [
            PDF_PATH.name,
            "06_38172000_orthographic_audit.png",
            "07_38172000_orthographic_audit.png",
            "08_40500000_orthographic_audit.png",
            "gpt_response_schema.json",
        ],
        "optional_machine_context": "adjudication_packet.json",
        "expected_response_mime_type": "application/json",
        "runtime_promotion_authorized": False,
    }
    (PACKAGE_DIR / "gpt_submission.json").write_text(
        json.dumps(submission, indent=2) + "\n", encoding="utf-8"
    )
    (PACKAGE_DIR / "README.md").write_text(
        """# Sherman QC F1 GPT submission - 15 July 2026

This directory is designed for direct multimodal GPT submission.

Submission steps:

1. Attach the files listed in `gpt_submission.json` in order.
2. Paste `GPT_SUBMISSION_PROMPT.md` verbatim.
3. Require JSON-only output.
4. Validate the response against `gpt_response_schema.json`.

The PDF and three PNGs are visual evidence. `adjudication_packet.json` is optional machine context with provenance hashes. Source scans are intentionally omitted.

The model must abstain from invented 3-D coordinates and must not count-force. This package is evaluation-only and does not authorize runtime promotion.
""",
        encoding="utf-8",
    )


def _build_package(packet: dict) -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF_PATH, PACKAGE_DIR / PDF_PATH.name)
    shutil.copy2(PACKET_PATH, PACKAGE_DIR / "adjudication_packet.json")
    for case_id, source in CASE_IMAGES.items():
        shutil.copy2(source, PACKAGE_DIR / f"{case_id}_orthographic_audit.png")
    _write_text_files(packet)

    files = sorted(path for path in PACKAGE_DIR.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "package_name": PACKAGE_DIR.name,
        "generated_at": "2026-07-15",
        "purpose": "Multimodal GPT adjudication of Sherman QC F1 hard cases 06-08",
        "runtime_promotion_authorized": False,
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
    }
    manifest_path = PACKAGE_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checksum_files = sorted(path for path in PACKAGE_DIR.iterdir() if path.is_file())
    (PACKAGE_DIR / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_files),
        encoding="utf-8",
    )

    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(PACKAGE_DIR.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"{PACKAGE_DIR.name}/{path.name}")


def main() -> None:
    required = [PACKET_PATH, *CASE_IMAGES.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing package inputs: " + ", ".join(missing))
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    _build_pdf(packet)
    _build_package(packet)
    print(json.dumps({
        "pdf": str(PDF_PATH),
        "package_dir": str(PACKAGE_DIR),
        "archive": str(ARCHIVE_PATH),
        "pdf_sha256": _sha256(PDF_PATH),
        "archive_sha256": _sha256(ARCHIVE_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
